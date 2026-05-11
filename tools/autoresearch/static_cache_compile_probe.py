#!/usr/bin/env python3
"""Probe: HF Transformers static cache + torch.compile on MPS.

This is the documented HF path for fast inference. Uses
generation_config.cache_implementation='static' plus a compiled forward.

Goal: see whether this path gives any meaningful tok/s lift on MPS for Gemma 4.
"""
from __future__ import annotations

import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    GenerationConfig,
)

FT = "/Users/thinkstudio/mnt/models/cliniq/v2-fp16-merged/cliniq-compact-merged-fp16"
DRAFTER = "/Volumes/models/hf/hub/models--google--gemma-4-E2B-it-assistant/snapshots/be0358c16076890848a1344a34209aa7c1df7587"

PROMPT = "List three signs of measles in pediatric patients. Be concise, one line each."


def load(path, device="mps", dtype=torch.float16):
    try:
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True)
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True)
    return m.to(device).eval()


def main():
    device = "mps"
    dtype = torch.float16
    print("loading target...", flush=True)
    t0 = time.time()
    target = load(FT, device, dtype)
    tok = AutoTokenizer.from_pretrained(FT)
    print(f"target loaded {time.time() - t0:.1f}s", flush=True)

    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)

    base_kwargs = dict(
        max_new_tokens=64, do_sample=False, temperature=None, top_p=None,
        return_dict_in_generate=True,
    )

    print("\n--- A: dynamic cache, no compile ---", flush=True)
    for i in range(3):
        torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = target.generate(**inputs, **base_kwargs)
        torch.mps.synchronize()
        dt = time.time() - t0
        n = out.sequences.shape[1] - inputs.input_ids.shape[1]
        print(f"  iter {i}: {n} tok / {dt:.2f}s = {n/dt:.2f} tok/s", flush=True)

    print("\n--- B: static cache, no compile ---", flush=True)
    static_kwargs = dict(base_kwargs, cache_implementation="static")
    for i in range(3):
        torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = target.generate(**inputs, **static_kwargs)
        torch.mps.synchronize()
        dt = time.time() - t0
        n = out.sequences.shape[1] - inputs.input_ids.shape[1]
        print(f"  iter {i}: {n} tok / {dt:.2f}s = {n/dt:.2f} tok/s", flush=True)

    print("\n--- C: static cache + compiled forward ---", flush=True)
    target.forward = torch.compile(
        target.forward, backend="inductor", mode="default", dynamic=False,
    )
    for i in range(3):
        torch.mps.synchronize()
        t0 = time.time()
        try:
            with torch.no_grad():
                out = target.generate(**inputs, **static_kwargs)
            torch.mps.synchronize()
            dt = time.time() - t0
            n = out.sequences.shape[1] - inputs.input_ids.shape[1]
            print(f"  iter {i}: {n} tok / {dt:.2f}s = {n/dt:.2f} tok/s", flush=True)
        except Exception as e:
            print(f"  iter {i}: ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
            break


if __name__ == "__main__":
    main()
