#!/usr/bin/env python3
"""Run all test cases from test_cases.jsonl through the .litertlm bundle.

Writes:
    - build/validation/VALIDATION-<N>-<case_id>.md  (per case)
    - build/validation/SUMMARY.json                 (score matrix)

For each case, computes extraction_score = matched_codes / expected_codes
across conditions (SNOMED), loincs, and rxnorms.

Usage:
    python validate_all_cases.py build/litertlm/*.litertlm \
        --case-file /Users/thinkstudio/gemma4/scripts/test_cases.jsonl \
        --backend cpu --max-cases 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are a clinical NLP assistant. Given an eICR summary, extract the "
    "primary conditions, labs, and medications as JSON with keys "
    "'conditions' (SNOMED codes), 'loincs' (LOINC codes), and 'rxnorms' "
    "(RxNorm codes). Return only JSON."
)

# Unsloth "gemma-4" chat template turn delimiters (see validate_litertlm.py
# and commit 78520b8 for rationale).
TURN_SYS_OPEN = "<|turn>system\n"
TURN_USER_OPEN = "<|turn>user\n"
TURN_MODEL_OPEN = "<|turn>model\n"
TURN_CLOSE = "<turn|>\n"


def build_prompt(case_user: str) -> str:
    return (
        f"{TURN_SYS_OPEN}{DEFAULT_SYSTEM}{TURN_CLOSE}"
        f"{TURN_USER_OPEN}{case_user}{TURN_CLOSE}"
        f"{TURN_MODEL_OPEN}"
    )


def score_extraction(output_text: str, case_meta: dict) -> dict:
    """Count how many of the expected codes appear verbatim in the output.

    Returns a dict with per-category matches and overall score.
    """
    expected_c = case_meta.get("expected_conditions", []) or []
    expected_l = case_meta.get("expected_loincs", []) or []
    expected_r = case_meta.get("expected_rxnorms", []) or []

    def _matches(codes: list[str]) -> list[bool]:
        return [bool(re.search(re.escape(c), output_text)) for c in codes]

    m_c = _matches(expected_c)
    m_l = _matches(expected_l)
    m_r = _matches(expected_r)

    total_expected = len(expected_c) + len(expected_l) + len(expected_r)
    total_matched = sum(m_c) + sum(m_l) + sum(m_r)
    return {
        "expected_conditions": expected_c,
        "expected_loincs": expected_l,
        "expected_rxnorms": expected_r,
        "matched_conditions": [c for c, ok in zip(expected_c, m_c) if ok],
        "matched_loincs": [c for c, ok in zip(expected_l, m_l) if ok],
        "matched_rxnorms": [c for c, ok in zip(expected_r, m_r) if ok],
        "total_expected": total_expected,
        "total_matched": total_matched,
        "extraction_score": (total_matched / total_expected) if total_expected else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to .litertlm")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--case-file", required=True, help="JSONL with test cases")
    ap.add_argument("--max-cases", type=int, default=0, help="Run first N cases only; 0 = all")
    ap.add_argument(
        "--out-dir",
        default="build/validation",
        help="Where to write per-case VALIDATION-N.md + SUMMARY.json",
    )
    args = ap.parse_args()

    import litert_lm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = {
        "cpu": litert_lm.Backend.CPU,
        "gpu": litert_lm.Backend.GPU,
    }[args.backend]

    # We build a fresh Engine per case because litert_lm only permits one
    # active session per Engine instance at a time, and Session has no
    # public close() method in this release (0.10.1). Load is ~0.5s on warm
    # mmap so the overhead is small enough compared to generation time.
    def _build_engine():
        return litert_lm.Engine(
            model_path=args.model,
            backend=backend,
            max_num_tokens=args.max_tokens,
        )

    print(f"Pre-loading {args.model} on {args.backend} for warmup...", flush=True)
    t0 = time.time()
    _warm = _build_engine()
    load_s = time.time() - t0
    del _warm
    import gc as _gc

    _gc.collect()
    print(f"  warmup loaded in {load_s:.2f}s", flush=True)

    cases = []
    for ln in Path(args.case_file).read_text().splitlines():
        ln = ln.strip()
        if ln:
            cases.append(json.loads(ln))

    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    print(f"Running {len(cases)} case(s) from {args.case_file}", flush=True)

    summary_rows = []
    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{idx}")
        description = case.get("description", "")
        case_user = case["user"]
        prompt = build_prompt(case_user)

        print(f"\n---- case {idx}/{len(cases)}: {case_id} ----", flush=True)
        # Tear down any previous engine/session. litert_lm only supports one
        # active session per engine and exposes no close() method, so we
        # rebuild the engine each case.
        if "engine" in locals():
            try:
                del session
            except Exception:
                pass
            try:
                del engine
            except Exception:
                pass
            _gc.collect()

        engine = _build_engine()
        session = engine.create_session()
        t_g = time.time()
        session.run_prefill(contents=[prompt])
        responses = session.run_decode()
        gen_s = time.time() - t_g

        out_text = "".join(getattr(responses, "texts", []) or [])
        token_len = (
            sum(getattr(responses, "token_lengths", []) or [])
            or max(1, len(out_text.split()))
        )
        tok_s = token_len / gen_s if gen_s > 0 else 0.0

        score = score_extraction(out_text, case)
        print(
            f"   score {score['total_matched']}/{score['total_expected']} "
            f"(tok/s={tok_s:.1f}, gen_s={gen_s:.2f}, out_tok={token_len})",
            flush=True,
        )
        print("   output:", out_text[:400].replace("\n", "\\n"), flush=True)

        # Write per-case file
        md = f"""# Validation case {idx} — {case_id}

- Model: `{args.model}` ({Path(args.model).stat().st_size/1e6:.1f} MB)
- Backend: `{args.backend}`
- Description: {description}

## Prompt

System:
```
{DEFAULT_SYSTEM}
```

User:
```
{case_user}
```

Turn-wrapped prompt (unsloth gemma-4 delimiters):
```
{prompt}
```

## Output

```
{out_text}
```

## Expected vs. matched

| category | expected | matched |
|---|---|---|
| conditions (SNOMED) | {score['expected_conditions']} | {score['matched_conditions']} |
| loincs | {score['expected_loincs']} | {score['matched_loincs']} |
| rxnorms | {score['expected_rxnorms']} | {score['matched_rxnorms']} |

**extraction_score: {score['total_matched']}/{score['total_expected']} = {score['extraction_score']:.2f}**

## Stats

| metric | value |
|---|---|
| load time (shared across cases) | {load_s:.2f} s |
| generate time | {gen_s:.2f} s |
| approx tokens | {token_len} |
| approx tok/s | {tok_s:.1f} |
"""
        per_case = out_dir / f"VALIDATION-{idx}-{case_id}.md"
        per_case.write_text(md)
        print(f"   wrote {per_case}")

        summary_rows.append(
            {
                "index": idx,
                "case_id": case_id,
                "description": description,
                "gen_s": gen_s,
                "token_len": token_len,
                "tok_s": tok_s,
                "output": out_text,
                **score,
            }
        )

    # Write summary
    totals_expected = sum(r["total_expected"] for r in summary_rows)
    totals_matched = sum(r["total_matched"] for r in summary_rows)
    summary = {
        "model": str(args.model),
        "model_size_mb": Path(args.model).stat().st_size / 1e6,
        "backend": args.backend,
        "case_file": str(args.case_file),
        "load_s": load_s,
        "cases": summary_rows,
        "totals": {
            "total_expected": totals_expected,
            "total_matched": totals_matched,
            "aggregate_extraction_score": (
                totals_matched / totals_expected if totals_expected else 0.0
            ),
        },
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'SUMMARY.json'}")
    print(
        f"\nAGGREGATE: {totals_matched}/{totals_expected} = "
        f"{summary['totals']['aggregate_extraction_score']:.2f}"
    )


if __name__ == "__main__":
    main()
