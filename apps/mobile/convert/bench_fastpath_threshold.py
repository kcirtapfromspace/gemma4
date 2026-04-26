"""Fast-path RAG threshold sweep — Candidate A from the 2026-04-25 LLM-tuning plan.

Sweeps `FAST_PATH_THRESHOLD` across {0.5, 0.6, 0.7, 0.8, 0.9} on
combined-27 + adv4 = 35 cases. At each threshold we measure:

  - precision / recall / F1 / perfect-rate (quality side)
  - fast-path hit rate (% of cases that skipped the agent loop)
  - median + p95 latency per case (speed side)
  - n_agent_invocations (how many cases fell to the slow path)

The c19 default of 0.70 was picked from a single eyeball on adv3. This
sweeps the tradeoff explicitly so we can pick the operating point with
data, not vibes. Honest answer: lower threshold buys latency, higher
threshold buys precision against false-positive RAG hits.

Requires llama-server on http://127.0.0.1:8090 with --jinja.

Usage:
    scripts/.venv/bin/python apps/mobile/convert/bench_fastpath_threshold.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_pipeline import run_agent, try_fast_path, score  # noqa: E402
from regex_preparser import extract as deterministic_extract  # noqa: E402


def _det_short_circuits_llm(det) -> bool:
    """Mirror of `ParsedExtraction.shortCircuitsLLM` in
    apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/EicrPreparser.swift
    (c20 Candidate D — see tools/autoresearch/c20-llm-tuning-2026-04-25.md).

    c20 final cleanup (Cand D refinement): short-circuit ONLY when the det
    result contains at least one explicit-assertion tier (inline `(SNOMED
    12345)` or CDA `<code .../>`). Lookup-tier-only matches — including
    multi-bucket lookup-only — fall through to fast-path / agent so the
    LLM can verify them against the surrounding context.

    Why drop the old `bucket_count >= 2` clause? Bug 5 from adv6:
    `adv6_long_form_admission_note` had two lookup-tier FPs (varicella
    SNOMED from "no varicella series", CBC LOINC from "CBC: WBC...")
    spanning 2 buckets, which short-circuited before the agent / RAG
    fast-path could surface the actual diagnosis (measles). Lookup-tier
    matches are inherently ambiguous — alias→code mapping carries no
    contextual confidence — so they should not gate the LLM out.

    For combined-45 cases that previously short-circuited via the bucket
    rule (e.g. adv2_h5n1_avian_flu, adv2_mpox), the fast-path's `try_fast_path`
    merge logic preserves the lookup matches and adds the RAG hit, so the
    extraction stays correct without regressing to the agent.
    """
    return any(m.tier in ("inline", "cda") for m in det.matches)


def run_one(case: dict, threshold: float, endpoint: str) -> dict:
    """Run one case with the given fast-path threshold."""
    narrative = case["user"]
    t0 = time.time()
    # Mirror agent_pipeline.main path: deterministic → fast-path → agent.
    # Gate: c20 Candidate D — short-circuit only when det has codes in
    # >=2 buckets OR an explicit-assertion (inline/cda) tier hit; lookup-
    # only single-bucket results fall through to fast-path / agent.
    det = deterministic_extract(narrative)
    det_dict = det.to_provenance_dict()
    if _det_short_circuits_llm(det):
        extraction = {
            "conditions": det_dict["conditions"],
            "loincs": det_dict["loincs"],
            "rxnorms": det_dict["rxnorms"],
        }
        path = "deterministic"
    else:
        fp = try_fast_path(narrative, threshold=threshold)
        if fp is not None:
            extraction, _ = fp
            path = "fast_path"
        else:
            extraction, _ = run_agent(narrative, endpoint=endpoint)
            path = "agent"
    elapsed = time.time() - t0
    m, e, fp_count = score(extraction, case)
    return {
        "case_id": case["case_id"],
        "threshold": threshold,
        "path": path,
        "matched": m,
        "expected": e,
        "false_positives": fp_count,
        "elapsed_s": round(elapsed, 3),
    }


def aggregate(rows: list[dict], threshold: float) -> dict:
    rows_at = [r for r in rows if r["threshold"] == threshold]
    n = len(rows_at)
    total_m = sum(r["matched"] for r in rows_at)
    total_e = sum(r["expected"] for r in rows_at)
    total_fp = sum(r["false_positives"] for r in rows_at)
    perfect = sum(1 for r in rows_at if r["expected"] and r["matched"] == r["expected"] and r["false_positives"] == 0)
    extracted = total_m + total_fp
    precision = total_m / extracted if extracted else 0.0
    recall = total_m / total_e if total_e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    paths = {p: sum(1 for r in rows_at if r["path"] == p) for p in ("deterministic", "fast_path", "agent")}
    elapsed = sorted(r["elapsed_s"] for r in rows_at)
    p50 = statistics.median(elapsed) if elapsed else 0
    p95 = elapsed[int(0.95 * (len(elapsed) - 1))] if elapsed else 0
    return {
        "threshold": threshold,
        "n_cases": n,
        "perfect": perfect,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positives": total_fp,
        "deterministic_hits": paths["deterministic"],
        "fast_path_hits": paths["fast_path"],
        "agent_invocations": paths["agent"],
        "p50_s": round(p50, 3),
        "p95_s": round(p95, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090")
    ap.add_argument(
        "--cases",
        nargs="+",
        default=[
            "scripts/test_cases.jsonl",
            "scripts/test_cases_adversarial.jsonl",
            "scripts/test_cases_adversarial2.jsonl",
            "scripts/test_cases_adversarial3.jsonl",
            "scripts/test_cases_adversarial4.jsonl",
        ],
    )
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--out-json", default="apps/mobile/convert/build/fastpath_threshold_sweep.json")
    args = ap.parse_args()

    cases: list[dict] = []
    for p in args.cases:
        for ln in Path(p).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    print(f"Sweep: {len(cases)} cases × {len(args.thresholds)} thresholds = {len(cases) * len(args.thresholds)} runs", flush=True)

    rows: list[dict] = []
    for threshold in args.thresholds:
        print(f"\n--- threshold = {threshold}", flush=True)
        for idx, case in enumerate(cases, 1):
            row = run_one(case, threshold, args.endpoint)
            rows.append(row)
            tag = "OK " if (row["matched"] == row["expected"] and row["false_positives"] == 0) else "MISS"
            print(f"  {tag} {idx:2d}/{len(cases)} {row['case_id']:42s} {row['matched']}/{row['expected']} fp={row['false_positives']} [{row['path']:13s}] {row['elapsed_s']:.2f}s", flush=True)

    summary = [aggregate(rows, t) for t in args.thresholds]

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n" + "=" * 96)
    print(f"{'threshold':>9} {'F1':>6} {'prec':>6} {'recall':>7} {'perf':>6} {'det':>4} {'fp_hit':>7} {'agent':>6} {'p50':>6} {'p95':>6}")
    print("=" * 96)
    for s in summary:
        print(f"{s['threshold']:>9.2f} {s['f1']:>6.3f} {s['precision']:>6.3f} {s['recall']:>7.3f} {s['perfect']:>3d}/{s['n_cases']:<2d} {s['deterministic_hits']:>4d} {s['fast_path_hits']:>7d} {s['agent_invocations']:>6d} {s['p50_s']:>5.2f}s {s['p95_s']:>5.2f}s")
    print(f"\nTrace: {out_path}")


if __name__ == "__main__":
    main()
