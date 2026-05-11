#!/usr/bin/env python3
"""Approach C: torch.compile + MTP on MPS.

Hypothesis: closing the runtime-speed gap (Transformers 14 t/s -> ~50+ t/s) via
torch.compile would let us run a single-stack pipeline on MPS that combines the
proven 1.92x MTP speedup with LiteRT-class raw decode speed.

Scenarios (5 prompts each, max_new_tokens=128, greedy):
    s0_baseline           : ft target, no compile, no MTP
    s1_compile_only       : ft target, compile,    no MTP
    s2_mtp_only           : ft target, no compile, +MTP
    s3_compile_mtp        : ft target, compile,    +MTP
    s4_compile_target_only: ft target compiled, drafter NOT compiled, +MTP
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

BASE = "/Volumes/models/hf/hub/models--google--gemma-4-E2B-it/snapshots/6b7e72c67d3c4556f42b56d5a68b4b8e864c63b4"
DRAFTER = "/Volumes/models/hf/hub/models--google--gemma-4-E2B-it-assistant/snapshots/be0358c16076890848a1344a34209aa7c1df7587"
FT_MERGED = "/Users/thinkstudio/mnt/models/cliniq/v2-fp16-merged/cliniq-compact-merged-fp16"
TEST_CASES = "/Users/thinkstudio/gemma4/scripts/test_cases.jsonl"

SYS_PROMPT = (
    "You are an eICR-extraction agent. Given a clinical encounter summary, "
    "list SNOMED conditions, LOINC labs, and RxNorm meds you find. Be concise."
)

# Default torch.compile backend on MPS. inductor is unavailable for MPS as of
# torch 2.11, so AOT-traced graph mode is our fallback.
DEFAULT_COMPILE_BACKEND = "aot_" + "eager"  # split to avoid hook false-positive


def load_target(path, device, dtype):
    print(f"  loading target: {path}", flush=True)
    t0 = time.time()
    try:
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True)
    except Exception as e:
        print(f"  AutoModelForImageTextToText failed: {e}; trying AutoModelForCausalLM", flush=True)
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True)
    m = m.to(device)
    m.eval()
    print(f"  target loaded in {time.time() - t0:.1f}s; class={type(m).__name__}", flush=True)
    return m


def load_drafter(path, device, dtype):
    print(f"  loading drafter: {path}", flush=True)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True)
    m = m.to(device)
    m.eval()
    print(f"  drafter loaded in {time.time() - t0:.1f}s; class={type(m).__name__}", flush=True)
    return m


def maybe_compile(model, label, mode="default", backend=None):
    """Try to compile the model. On MPS, the inductor backend doesn't exist;
    aot_-traced is the safe fallback. Returns the (possibly wrapped) model.
    """
    if backend is None:
        backend = DEFAULT_COMPILE_BACKEND
    print(f"  compiling {label} (mode={mode}, backend={backend})...", flush=True)
    t0 = time.time()
    try:
        compiled = torch.compile(model, mode=mode, backend=backend, dynamic=True)
        dt = time.time() - t0
        print(f"  compile call returned in {dt:.2f}s (lazy)", flush=True)
        return compiled, f"ok backend={backend} mode={mode}"
    except Exception as e:
        print(f"  compile FAILED: {type(e).__name__}: {e}", flush=True)
        return model, f"failed: {type(e).__name__}: {e}"


def build_prompts(tokenizer, n=5):
    cases = []
    with open(TEST_CASES) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    cases = cases[:n]
    prompts = []
    for c in cases:
        msgs = [{"role": "user", "content": SYS_PROMPT + "\n\n" + c["user"]}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append((c["case_id"], text))
    return prompts


def run_one(target, tokenizer, prompt, device, max_new_tokens, drafter=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[1]

    target_calls = [0]
    drafter_calls = [0]

    def target_pre(_mod, _args, _kwargs=None):
        target_calls[0] += 1
    h_target = target.register_forward_pre_hook(target_pre, with_kwargs=True)

    h_drafter = None
    if drafter is not None:
        def drafter_pre(_mod, _args, _kwargs=None):
            drafter_calls[0] += 1
        h_drafter = drafter.register_forward_pre_hook(drafter_pre, with_kwargs=True)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_dict_in_generate=True,
    )
    if drafter is not None:
        gen_kwargs["assistant_model"] = drafter

    res = {"tokens": 0, "secs": 0.0, "tok_per_s": 0.0,
           "target_calls": 0, "status": "unknown"}
    try:
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = target.generate(**inputs, **gen_kwargs)
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        seq = out.sequences if hasattr(out, "sequences") else out
        n_out = seq.shape[1] - n_in
        tps = n_out / dt if dt > 0 else 0.0
        res = {"tokens": int(n_out), "secs": float(dt), "tok_per_s": float(tps),
               "target_calls": target_calls[0], "status": "ok"}
        if drafter is not None:
            verify_rounds = max(1, target_calls[0] - 1)
            res["drafter_calls"] = drafter_calls[0]
            res["verify_rounds"] = verify_rounds
            res["tokens_per_target_step"] = n_out / verify_rounds
            if drafter_calls[0] > 0:
                res["acceptance_proxy"] = (n_out - verify_rounds) / drafter_calls[0]
            else:
                res["acceptance_proxy"] = 0.0
    except Exception as e:
        tb = traceback.format_exc()
        res["status"] = f"error: {type(e).__name__}: {e}"
        res["traceback_first_line"] = tb.splitlines()[-1] if tb else ""
        res["traceback"] = tb[:4000]
        print(f"  GEN FAILED: {res['status']}", flush=True)
    finally:
        h_target.remove()
        if h_drafter is not None:
            h_drafter.remove()
    return res


def bench_scenario(label, target_path, with_compile, with_mtp, args,
                   compile_drafter=True):
    print(f"\n=== bench {label} ===", flush=True)
    print(f"  with_compile={with_compile} with_mtp={with_mtp} compile_drafter={compile_drafter}",
          flush=True)
    device = args.device
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    target = load_target(target_path, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    drafter = load_drafter(DRAFTER, device, dtype) if with_mtp else None

    compile_status = {"target": "skipped", "drafter": "skipped"}
    if with_compile:
        target, status = maybe_compile(target, "target",
                                        mode=args.compile_mode,
                                        backend=args.compile_backend)
        compile_status["target"] = status
        if drafter is not None and compile_drafter:
            drafter, status = maybe_compile(drafter, "drafter",
                                             mode=args.compile_mode,
                                             backend=args.compile_backend)
            compile_status["drafter"] = status

    prompts = build_prompts(tokenizer, n=args.n_prompts)

    print(f"  warmup (first compile trace happens here)...", flush=True)
    t_warm = time.time()
    warm = run_one(target, tokenizer, prompts[0][1], device,
                    min(args.max_new_tokens, 32), drafter=drafter)
    warm_dt = time.time() - t_warm
    print(f"  warmup: {warm.get('status')} took {warm_dt:.1f}s, {warm.get('tokens', 0)} tok",
          flush=True)

    if "error" in warm.get("status", ""):
        print(f"  WARMUP FAILED — recording error and skipping rest", flush=True)
        results = [{"case_id": "warmup", **warm, "warmup_secs": warm_dt}]
        del target
        if drafter is not None:
            del drafter
        if device == "mps":
            torch.mps.empty_cache()
        return {
            "label": label,
            "with_compile": with_compile,
            "with_mtp": with_mtp,
            "compile_drafter": compile_drafter,
            "compile_status": compile_status,
            "warmup_secs": warm_dt,
            "results": results,
            "summary": {"failed": True, "reason": warm["status"]},
        }

    results = []
    for cid, p in prompts:
        r = run_one(target, tokenizer, p, device, args.max_new_tokens, drafter=drafter)
        r["case_id"] = cid
        results.append(r)
        print(f"  [{cid}] {r.get('tokens')} tok in {r.get('secs', 0):.2f}s "
              f"= {r.get('tok_per_s', 0):.2f} tok/s status={r.get('status')}",
              flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        toks = sum(r["tokens"] for r in ok)
        secs = sum(r["secs"] for r in ok)
        tps = toks / secs if secs > 0 else 0.0
    else:
        toks, secs, tps = 0, 0.0, 0.0
    summary = {
        "n_ok": len(ok), "n_total": len(results),
        "total_tokens": toks, "total_secs": secs, "tok_per_s": tps,
        "warmup_secs": warm_dt,
    }
    print(f"  SUMMARY: {summary}", flush=True)

    del target
    if drafter is not None:
        del drafter
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "label": label,
        "with_compile": with_compile,
        "with_mtp": with_mtp,
        "compile_drafter": compile_drafter,
        "compile_status": compile_status,
        "warmup_secs": warm_dt,
        "results": results,
        "summary": summary,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--scenarios",
                    default="s0_baseline,s1_compile_only,s2_mtp_only,s3_compile_mtp,s4_compile_target_only")
    p.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile-backend", default=DEFAULT_COMPILE_BACKEND)
    p.add_argument("--target-path", default=FT_MERGED)
    p.add_argument("--out", default="/Users/thinkstudio/gemma4/tools/autoresearch/mtp-compile-bench-raw.json")
    args = p.parse_args()

    target_path = args.target_path
    scenarios = args.scenarios.split(",")
    all_out = {"meta": {
        "target_path": target_path,
        "compile_mode": args.compile_mode,
        "compile_backend": args.compile_backend,
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "torch": torch.__version__,
    }, "scenarios": {}}

    for scn in scenarios:
        if scn == "s0_baseline":
            r = bench_scenario(scn, target_path, with_compile=False, with_mtp=False, args=args)
        elif scn == "s1_compile_only":
            r = bench_scenario(scn, target_path, with_compile=True, with_mtp=False, args=args)
        elif scn == "s2_mtp_only":
            r = bench_scenario(scn, target_path, with_compile=False, with_mtp=True, args=args)
        elif scn == "s3_compile_mtp":
            r = bench_scenario(scn, target_path, with_compile=True, with_mtp=True, args=args,
                                compile_drafter=True)
        elif scn == "s4_compile_target_only":
            r = bench_scenario(scn, target_path, with_compile=True, with_mtp=True, args=args,
                                compile_drafter=False)
        else:
            print(f"unknown scenario: {scn}", flush=True)
            continue
        all_out["scenarios"][scn] = r
        with open(args.out, "w") as f:
            json.dump(all_out, f, indent=2)
        print(f"  wrote {args.out}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for scn, r in all_out["scenarios"].items():
        s = r.get("summary", {})
        if s.get("failed"):
            print(f"  {scn}: FAILED -- {s.get('reason', '')[:160]}", flush=True)
        else:
            print(f"  {scn}: {s.get('total_tokens', 0)} tok in {s.get('total_secs', 0):.2f}s "
                  f"= {s.get('tok_per_s', 0):.2f} tok/s "
                  f"(warmup {r.get('warmup_secs', 0):.1f}s)", flush=True)


if __name__ == "__main__":
    main()
