"""Deterministic-only bench for the external CDC eICR cases.

Mirrors the inline harness used to produce e.g. adv6_deterministic_only.json:
runs `regex_preparser.extract` on each case's `user` text, scores against
the `expected_*` keys (set-based F1 / precision / recall), and writes a
JSON artifact compatible with score_fhir.py --from-agent-bench.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add the convert/ package to sys.path so the import line below works
# regardless of cwd. Resolved relative to this script's location so the
# harness ports across worktrees / clones without edits.
ROOT = Path(__file__).resolve().parent.parent
CONVERT = ROOT / "apps" / "mobile" / "convert"
sys.path.insert(0, str(CONVERT))

from regex_preparser import extract as deterministic_extract  # type: ignore[import-not-found]


def score_extraction(extraction: dict, expected: dict) -> dict:
    matched: list[str] = []
    spurious: list[str] = []
    missing: list[str] = []

    for bucket in ("conditions", "loincs", "rxnorms"):
        got = set(str(c) for c in (extraction.get(bucket) or []))
        exp_key = "expected_" + ("conditions" if bucket == "conditions" else bucket)
        want = set(str(c) for c in (expected.get(exp_key) or []))
        for c in got & want:
            matched.append(f"{bucket}:{c}")
        for c in got - want:
            spurious.append(f"{bucket}:{c}")
        for c in want - got:
            missing.append(f"{bucket}:{c}")
    return {
        "matched": matched,
        "missing": missing,
        "spurious": spurious,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    cases: list[dict] = []
    for path in args.cases:
        for ln in Path(path).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))

    print(f"Deterministic-only bench on {len(cases)} cases")
    total_matched = 0
    total_expected = 0
    total_fp = 0
    perfect = 0
    for i, case in enumerate(cases, 1):
        cid = case["case_id"]
        text = case["user"]
        t0 = time.perf_counter()
        result = deterministic_extract(text)
        elapsed = time.perf_counter() - t0
        # `regex_preparser.extract` returns a ParsedExtraction-like obj — get
        # the conditions/loincs/rxnorms.
        extraction = {
            "conditions": list(getattr(result, "conditions", []) or result.get("conditions", []) if hasattr(result, "get") else getattr(result, "conditions", [])),
            "loincs":     list(getattr(result, "loincs", []) or result.get("loincs", []) if hasattr(result, "get") else getattr(result, "loincs", [])),
            "rxnorms":    list(getattr(result, "rxnorms", []) or result.get("rxnorms", []) if hasattr(result, "get") else getattr(result, "rxnorms", [])),
        }
        scoring = score_extraction(extraction, case)
        n_matched = len(scoring["matched"])
        n_expected = (
            len(case.get("expected_conditions") or [])
            + len(case.get("expected_loincs") or [])
            + len(case.get("expected_rxnorms") or [])
        )
        n_fp = len(scoring["spurious"])
        is_perfect = (n_matched == n_expected) and n_fp == 0
        if is_perfect:
            perfect += 1
        total_matched += n_matched
        total_expected += n_expected
        total_fp += n_fp
        marker = "OK  " if is_perfect else "    "
        fp_tag = f" fp={n_fp}" if n_fp else ""
        print(
            f"  {marker} {i:2d}/{len(cases)} {cid:42s} "
            f"{n_matched}/{n_expected}{fp_tag}  {elapsed*1000:.0f} ms"
        )
        rows.append({
            "case_id": cid,
            "matched": n_matched,
            "expected": n_expected,
            "false_positives": n_fp,
            "extraction": extraction,
            "expected_codes": {
                "conditions": list(case.get("expected_conditions") or []),
                "loincs": list(case.get("expected_loincs") or []),
                "rxnorms": list(case.get("expected_rxnorms") or []),
            },
            "missing_codes": scoring["missing"],
            "spurious_codes": scoring["spurious"],
            "matches": scoring["matched"],
            "elapsed_ms": elapsed * 1000.0,
        })

    n = len(cases)
    precision = (total_matched) / max(total_matched + total_fp, 1)
    recall = (total_matched) / max(total_expected, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    print(
        f"\nAggregate: matched={total_matched}/{total_expected} "
        f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}; "
        f"FP={total_fp}; perfect={perfect}/{n}"
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
