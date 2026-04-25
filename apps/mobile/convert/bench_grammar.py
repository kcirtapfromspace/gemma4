"""Bench llama-server completion with and without GBNF grammar constraint.

Targets the 6 cases that exercise the residual LLM path:
- 3 originals where C16 litertlm failed (typical_covid, complex_multi, negative_lab)
- 3 adversarials that the regex parser cannot solve alone (zika, chickenpox, strep)

For each case, runs:
  (a) baseline   — no grammar
  (b) grammar    — gbnf forces { conditions: [digit-strings], loincs: [...], rxnorms: [...] }

Scoring uses the same re.search code matching as validate_all_cases.py.

Server is expected at http://127.0.0.1:8090 (llama-server). The /completion
endpoint accepts a `grammar` field (raw GBNF text); we POST the gbnf file's
contents directly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


SYSTEM = (
    "You are a clinical NLP assistant. Given an eICR summary, extract the "
    "primary conditions, labs, and medications as JSON with keys "
    "'conditions' (SNOMED codes), 'loincs' (LOINC codes), and 'rxnorms' "
    "(RxNorm codes). Return only JSON."
)


def build_prompt(user: str) -> str:
    """Match the unsloth gemma-4 turn-wrapping used in the C16 validator."""
    return (
        f"<|turn>system\n{SYSTEM}<turn|>\n"
        f"<|turn>user\n{user}<turn|>\n"
        f"<|turn>model\n"
    )


def score(out_text: str, case: dict) -> tuple[int, int]:
    expected = (
        list(case.get("expected_conditions") or [])
        + list(case.get("expected_loincs") or [])
        + list(case.get("expected_rxnorms") or [])
    )
    matched = sum(1 for c in expected if re.search(re.escape(c), out_text))
    return matched, len(expected)


def call(
    endpoint: str,
    prompt: str,
    grammar: str | None,
    n_predict: int,
    timeout: float,
) -> tuple[str, float, int]:
    """POST to llama-server /completion. Returns (text, gen_seconds, tokens)."""
    payload: dict = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.1,
        "cache_prompt": False,
        "stream": False,
    }
    if grammar:
        payload["grammar"] = grammar
    req = Request(
        f"{endpoint}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    text = body.get("content", "")
    tokens = body.get("tokens_predicted") or body.get("timings", {}).get(
        "predicted_n", 0
    )
    return text, elapsed, int(tokens or 0)


def load_cases(filter_ids: set[str] | None = None) -> list[dict]:
    cases: list[dict] = []
    for fname in (
        "scripts/test_cases.jsonl",
        "scripts/test_cases_adversarial.jsonl",
    ):
        for ln in Path(fname).read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            if filter_ids is None or row.get("case_id") in filter_ids:
                cases.append(row)
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090")
    ap.add_argument("--grammar-file", default="apps/mobile/convert/cliniq_extraction.gbnf")
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--out-json",
        default="apps/mobile/convert/build/grammar_bench.json",
    )
    args = ap.parse_args()

    grammar = Path(args.grammar_file).read_text()
    cases = load_cases()
    print(f"Loaded {len(cases)} cases\n", flush=True)
    n_cases = len(cases)

    rows: list[dict] = []
    for case in cases:
        case_id = case["case_id"]
        prompt = build_prompt(case["user"])
        for label, gbnf in (("baseline", None), ("grammar", grammar)):
            try:
                text, secs, toks = call(
                    args.endpoint, prompt, gbnf, args.n_predict, args.timeout
                )
                m, e = score(text, case)
                tok_s = toks / secs if secs > 0 else 0.0
                first = text.replace("\n", "\\n")[:160]
                print(
                    f"  {case_id:34s} {label:8s} "
                    f"{m}/{e} tok={toks:3d} t={secs:5.1f}s "
                    f"{tok_s:4.1f} t/s | {first}",
                    flush=True,
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "mode": label,
                        "matched": m,
                        "expected": e,
                        "tokens": toks,
                        "seconds": round(secs, 2),
                        "tok_per_s": round(tok_s, 2),
                        "output": text,
                    }
                )
            except URLError as exc:
                print(f"  {case_id} {label}: ERR {exc}", flush=True)
                rows.append(
                    {"case_id": case_id, "mode": label, "error": str(exc)}
                )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, indent=2))

    def _agg(mode: str) -> tuple[int, int, int]:
        sel = [r for r in rows if r.get("mode") == mode and "error" not in r]
        m = sum(r["matched"] for r in sel)
        e = sum(r["expected"] for r in sel)
        perfect = sum(1 for r in sel if r["matched"] == r["expected"])
        return m, e, perfect

    print()
    for mode in ("baseline", "grammar"):
        m, e, p = _agg(mode)
        score_v = m / e if e else 0.0
        print(
            f"AGGREGATE {mode:8s}: {m}/{e} = {score_v:.3f} "
            f"({p}/{n_cases} cases perfect)"
        )
    print(f"\nFull results: {args.out_json}")


if __name__ == "__main__":
    main()
