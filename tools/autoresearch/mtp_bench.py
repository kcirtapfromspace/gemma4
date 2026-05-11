#!/usr/bin/env python3
"""MTP drafter acceptance bench: base vs c9 fine-tune.

Loads gemma-4-E2B-it (base or fine-tuned merged) as target and
gemma-4-E2B-it-assistant (78M drafter) as assistant, runs assisted
generation on ~9 prompts, measures tokens/sec and acceptance rate.
"""
import argparse
import json
import sys
import time
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


def build_prompts(tokenizer, n=9):
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

    # Instrument via forward pre-hook (doesn't replace .forward, avoids signature issues)
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

    try:
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = target.generate(**inputs, **gen_kwargs)
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
    finally:
        h_target.remove()
        if h_drafter is not None:
            h_drafter.remove()

    seq = out.sequences if hasattr(out, "sequences") else out
    n_out = seq.shape[1] - n_in
    tps = n_out / dt if dt > 0 else 0.0
    # Acceptance: with MTP, target_calls == number of verify rounds.
    # tokens generated = n_out; each target call produces >=1 accepted token.
    # accepted_per_round = n_out / target_calls (excludes the prefill call).
    # Subtract 1 for prefill (first target call is full prefill of n_in tokens).
    res = {"tokens": int(n_out), "secs": float(dt), "tok_per_s": float(tps),
           "target_calls": target_calls[0]}
    if drafter is not None:
        verify_rounds = max(1, target_calls[0] - 1)  # subtract prefill
        # tokens generated per verify round = n_out / verify_rounds
        # In assisted decoding, each verify produces 1 (rejection) to N+1 (all accept) tokens.
        # acceptance rate ~= (n_out - verify_rounds) / (drafter_proposals_total)
        # But drafter_calls counts each drafter step: total drafted = drafter_calls
        # Accepted tokens = n_out - verify_rounds (target's "free" bonus token per round)
        # is approximate; cleaner = n_out / verify_rounds = avg tokens per target step.
        res["drafter_calls"] = drafter_calls[0]
        res["verify_rounds"] = verify_rounds
        res["tokens_per_target_step"] = n_out / verify_rounds
        # Acceptance ratio: accepted drafted tokens / proposed drafted tokens
        # Proposed = drafter_calls (each drafter call proposes ~1 token in HF impl,
        # but really num_assistant_tokens per round; here we approximate)
        if drafter_calls[0] > 0:
            res["acceptance_proxy"] = (n_out - verify_rounds) / drafter_calls[0]
        else:
            res["acceptance_proxy"] = 0.0
    return res


def bench(target_path, label, device, dtype, max_new_tokens, n_prompts, with_mtp):
    print(f"\n=== bench {label} (mtp={with_mtp}) ===", flush=True)
    target = load_target(target_path, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    drafter = load_drafter(DRAFTER, device, dtype) if with_mtp else None

    prompts = build_prompts(tokenizer, n=n_prompts)
    results = []
    print(f"  warmup...", flush=True)
    _ = run_one(target, tokenizer, prompts[0][1], device, 8, drafter=drafter)
    print(f"  warmup done", flush=True)

    for cid, p in prompts:
        r = run_one(target, tokenizer, p, device, max_new_tokens, drafter=drafter)
        r["case_id"] = cid
        results.append(r)
        print(f"  [{cid}] {r['tokens']} tok in {r['secs']:.2f}s = {r['tok_per_s']:.2f} tok/s", flush=True)

    del target
    if drafter is not None:
        del drafter
    if device == "mps":
        torch.mps.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--n-prompts", type=int, default=9)
    p.add_argument("--scenarios", default="base_no_mtp,base_mtp,ft_no_mtp,ft_mtp")
    p.add_argument("--out", default="/Users/thinkstudio/gemma4/tools/autoresearch/mtp-mlx-bench-raw.json")
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    scenarios = args.scenarios.split(",")
    all_results = {}
    for scn in scenarios:
        if scn == "base_no_mtp":
            r = bench(BASE, "base", args.device, dtype, args.max_new_tokens, args.n_prompts, with_mtp=False)
        elif scn == "base_mtp":
            r = bench(BASE, "base", args.device, dtype, args.max_new_tokens, args.n_prompts, with_mtp=True)
        elif scn == "ft_no_mtp":
            r = bench(FT_MERGED, "ft", args.device, dtype, args.max_new_tokens, args.n_prompts, with_mtp=False)
        elif scn == "ft_mtp":
            r = bench(FT_MERGED, "ft", args.device, dtype, args.max_new_tokens, args.n_prompts, with_mtp=True)
        else:
            print(f"unknown scenario: {scn}", flush=True)
            continue
        all_results[scn] = r
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  wrote {args.out}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for scn, rs in all_results.items():
        toks = sum(r["tokens"] for r in rs)
        secs = sum(r["secs"] for r in rs)
        tps = toks / secs if secs > 0 else 0.0
        print(f"  {scn}: total {toks} tok in {secs:.2f}s = {tps:.2f} tok/s avg", flush=True)


if __name__ == "__main__":
    main()
