#!/usr/bin/env python3
"""Tiny perf sanity check: run the COVID case 3x and report mean decode tok/s.

This is a Mac-CPU number, NOT a demo/phone-GPU number.

Usage:
    python perf_covid_3x.py build/litertlm/*.litertlm --backend cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are a clinical NLP assistant. Given an eICR summary, extract the "
    "primary conditions, labs, and medications as JSON with keys "
    "'conditions' (SNOMED codes), 'loincs' (LOINC codes), and 'rxnorms' "
    "(RxNorm codes). Return only JSON."
)

TURN_SYS_OPEN = "<|turn>system\n"
TURN_USER_OPEN = "<|turn>user\n"
TURN_MODEL_OPEN = "<|turn>model\n"
TURN_CLOSE = "<turn|>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--case-file", required=True)
    ap.add_argument("--case-id", default="bench_typical_covid")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import litert_lm

    backend = {
        "cpu": litert_lm.Backend.CPU,
        "gpu": litert_lm.Backend.GPU,
    }[args.backend]

    # Load target case
    case = None
    for ln in Path(args.case_file).read_text().splitlines():
        ln = ln.strip()
        if ln:
            c = json.loads(ln)
            if c.get("case_id") == args.case_id:
                case = c
                break
    if case is None:
        print(f"case {args.case_id} not found in {args.case_file}", file=sys.stderr)
        sys.exit(2)

    prompt = (
        f"{TURN_SYS_OPEN}{DEFAULT_SYSTEM}{TURN_CLOSE}"
        f"{TURN_USER_OPEN}{case['user']}{TURN_CLOSE}"
        f"{TURN_MODEL_OPEN}"
    )

    def _build_engine():
        return litert_lm.Engine(
            model_path=args.model,
            backend=backend,
            max_num_tokens=args.max_tokens,
        )

    print(f"Warming {args.model} on {args.backend}...")
    t0 = time.time()
    _warm = _build_engine()
    print(f"  warm-load in {time.time()-t0:.2f}s")
    del _warm
    import gc as _gc

    _gc.collect()

    tok_rates = []
    engine = None
    session = None
    for r in range(args.runs):
        if session is not None:
            try:
                del session
            except Exception:
                pass
        if engine is not None:
            try:
                del engine
            except Exception:
                pass
            _gc.collect()

        engine = _build_engine()
        session = engine.create_session()
        t0 = time.time()
        session.run_prefill(contents=[prompt])
        responses = session.run_decode()
        gen_s = time.time() - t0
        out_text = "".join(getattr(responses, "texts", []) or [])
        token_len = (
            sum(getattr(responses, "token_lengths", []) or [])
            or max(1, len(out_text.split()))
        )
        rate = token_len / gen_s if gen_s > 0 else 0.0
        tok_rates.append(rate)
        print(
            f"run {r+1}/{args.runs}: gen_s={gen_s:.2f} "
            f"tok={token_len} tok_s={rate:.2f}"
        )

    mean = statistics.mean(tok_rates)
    stdev = statistics.stdev(tok_rates) if len(tok_rates) > 1 else 0.0
    print(f"\nmean {mean:.2f} tok/s (stdev {stdev:.2f}) across {args.runs} runs (Mac CPU)")


if __name__ == "__main__":
    main()
