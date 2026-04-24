#!/usr/bin/env python3
"""Load a ``.litertlm`` bundle and run a single clinical extraction case.

Prints:
- load time
- output text
- decode time
- estimated tokens/sec (CPU on Mac is *not* phone GPU — this is a smoke
  test that the file works, not a benchmark of mobile speed).

Usage:
    python validate_litertlm.py build/litertlm/cliniq-gemma4-e2b.litertlm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are a clinical NLP assistant. Given an eICR summary, extract the "
    "primary conditions, labs, and medications as JSON with keys "
    "'conditions' (SNOMED codes), 'loincs' (LOINC codes), and 'rxnorms' "
    "(RxNorm codes). Return only JSON."
)

# Use the minimal COVID case from scripts/test_cases.jsonl so output can be
# easily compared by eye to an expected SNOMED=840539006 / LOINC=94500-6 /
# RxNorm=2599543 answer.
DEFAULT_CASE = (
    "Patient: Maria Garcia\n"
    "Gender: F\n"
    "DOB: 1985-06-14\n"
    "Encounter: 2026-03-15\n"
    "Reason: fever (39.2C), dry cough for 5 days, fatigue, shortness of breath\n"
    "Dx: COVID-19 (SNOMED 840539006)\n"
    "Lab: SARS-CoV-2 RNA NAA+probe Ql Resp (LOINC 94500-6) - Detected\n"
    "Vitals: Temp 39.2C, HR 98, RR 22, SpO2 94%\n"
    "Meds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to .litertlm")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--case-file", default=None, help="JSONL with test cases (uses first)")
    ap.add_argument("--out", default="VALIDATION.md")
    args = ap.parse_args()

    import litert_lm

    backend = {
        "cpu": litert_lm.Backend.CPU,
        "gpu": litert_lm.Backend.GPU,
    }[args.backend]

    print(f"Loading {args.model} on {args.backend}...", flush=True)
    t0 = time.time()
    engine = litert_lm.Engine(
        model_path=args.model,
        backend=backend,
        max_num_tokens=args.max_tokens,
    )
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.2f}s", flush=True)

    if args.case_file:
        first = next(
            (json.loads(ln) for ln in Path(args.case_file).read_text().splitlines() if ln.strip()),
            None,
        )
        case_user = first["user"] if first else DEFAULT_CASE
        case_meta = first or {}
    else:
        case_user = DEFAULT_CASE
        case_meta = {}

    # The baked-in Gemma 4 chat template relies on Jinja `message.get(...)`
    # which the embedded template engine inside LiteRT-LM does NOT
    # implement (fails with "unknown method: map has no method named get").
    # Work-around: bypass the high-level `Conversation` API and drive the
    # `Session` prefill/decode directly with a hand-rolled prompt. This is
    # exactly what a mobile integrator would do if they hit the same bug;
    # the fix is upstream in the LiteRT-LM Jinja vendored library.
    # LiteRT-LM rejects literal control tokens like <bos> in the raw prefill
    # input (the runtime inserts the BOS token itself based on
    # LlmMetadata.start_token). Send plain text; LiteRT-LM handles
    # turn-tagging through its own turn-transcript state machine, even in
    # raw session mode.
    plain_prompt = f"{DEFAULT_SYSTEM}\n\n{case_user}\n\nReturn JSON only.\n"

    session = engine.create_session()

    print("Sending user message via raw session prefill/decode...", flush=True)
    t0 = time.time()
    session.run_prefill(contents=[plain_prompt])
    responses = session.run_decode()
    gen_s = time.time() - t0

    # `Responses` is a dataclass with texts / scores / token_lengths lists.
    out_text = "".join(getattr(responses, "texts", []) or [])
    token_len = (
        sum(getattr(responses, "token_lengths", []) or [])
        or max(1, len(out_text.split()))
    )

    approx_tokens = token_len
    tok_s = approx_tokens / gen_s if gen_s > 0 else 0.0

    print("\n==== output ====")
    print(out_text)
    print("\n==== stats ====")
    print(f"load_s:       {load_s:.2f}")
    print(f"gen_s:        {gen_s:.2f}")
    print(f"approx_tok:   {approx_tokens}")
    print(f"tok_s (est):  {tok_s:.1f}")

    # Write VALIDATION.md
    md = f"""# Validation — {Path(args.model).name}

Harness: `validate_litertlm.py` on macOS CPU (arm64). This is a smoke test of
the `.litertlm` bundle, not a mobile benchmark — decoded tok/s on the phone
GPU is separate (~52-56 tok/s per the LiteRT-LM docs for this model).

## Setup

- Model: `{args.model}` ({Path(args.model).stat().st_size/1e6:.1f} MB)
- Backend: `{args.backend}` (LiteRT-LM {litert_lm.__version__ if hasattr(litert_lm, '__version__') else 'unknown'})
- Max tokens: {args.max_tokens}

## Prompt

System:
```
{DEFAULT_SYSTEM}
```

User:
```
{case_user}
```

## Output

```
{out_text}
```

## Stats

| metric | value |
|---|---|
| load time | {load_s:.2f} s |
| generate time | {gen_s:.2f} s |
| approx tokens | {approx_tokens} |
| approx tok/s | {tok_s:.1f} |

## Expected extraction (ground truth from test_cases.jsonl)

- case_id:    `{case_meta.get("case_id", "?")}`
- description: {case_meta.get("description", "?")}
- conditions: {case_meta.get("expected_conditions", [])}
- loincs:     {case_meta.get("expected_loincs", [])}
- rxnorms:    {case_meta.get("expected_rxnorms", [])}
"""
    Path(args.out).write_text(md)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
