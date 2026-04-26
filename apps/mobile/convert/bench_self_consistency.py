"""Self-consistency / temperature sweep — Candidate B from the LLM-tuning plan.

Hypothesis: the 2/27 combined-bench misses are knowledge-coverage failures.
Self-consistency *might* help if the model sometimes surfaces a correct code
in one rollout that the deterministic temp=0 rollout never explores.

Method per case:
  - Run agent_pipeline.run_agent N times at temperature T
  - Code-level majority vote: emit a code only if ≥ ceil(N/2) rollouts agree
  - Score against expected; track precision (we cannot afford FP regression)

Sweeps {temp 0.0, 0.2, 0.4} × {n=1, 3, 5}. n=1 is the baseline. The win
condition is recall improvement on previously-missed cases AND precision
unchanged (== 1.000).

The bench is gated to a user-supplied list of case_ids by default — point
it at the misses surfaced by the latest combined-27 agent run rather than
re-running the full 27-case bench (which is what the grammar stability
bench already did and is expensive at higher n × T).

Usage:
    scripts/.venv/bin/python apps/mobile/convert/bench_self_consistency.py \
        --cases-by-id adv2_drug_dose_variant adv3_rmsf_rag \
        --temps 0.0 0.2 0.4 --n-rollouts 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_pipeline import run_agent, score  # noqa: E402


def majority_vote(extractions: list[dict], threshold: int) -> dict:
    """Code-level majority vote across N extractions.

    Each bucket (conditions/loincs/rxnorms) counts code appearances; emit a
    code only if it appears in `threshold` or more extractions. This is a
    precision-preserving aggregator: a code emitted by 1-of-3 rollouts at
    temp>0 is suppressed unless backed up.
    """
    out = {"conditions": [], "loincs": [], "rxnorms": []}
    for bucket in out:
        counter: Counter[str] = Counter()
        for ext in extractions:
            for code in (ext.get(bucket) or []):
                counter[code] += 1
        out[bucket] = sorted(c for c, n in counter.items() if n >= threshold)
    return out


def run_case(case: dict, *, n: int, temp: float, endpoint: str) -> dict:
    """Run one case n times at temperature temp; majority-vote and score."""
    rollouts: list[dict] = []
    elapsed = []
    parse_errs = 0
    for _ in range(n):
        t0 = time.time()
        try:
            ext, trace = run_agent(case["user"], endpoint=endpoint, temperature=temp)
        except Exception as exc:  # noqa: BLE001
            return {
                "case_id": case["case_id"],
                "temp": temp,
                "n": n,
                "error": repr(exc),
            }
        elapsed.append(time.time() - t0)
        rollouts.append(ext)
        for evt in trace:
            r = evt.get("result")
            if isinstance(r, dict) and isinstance(r.get("error"), str) and "argument JSON decode failed" in r["error"]:
                parse_errs += 1

    threshold = (n // 2) + 1 if n > 1 else 1  # n=1 → 1, n=3 → 2, n=5 → 3
    voted = majority_vote(rollouts, threshold)
    m, e, fp = score(voted, case)
    return {
        "case_id": case["case_id"],
        "temp": temp,
        "n": n,
        "vote_threshold": threshold,
        "voted_extraction": voted,
        "rollouts": rollouts,
        "matched": m,
        "expected": e,
        "false_positives": fp,
        "parse_errors": parse_errs,
        "median_elapsed_s": round(statistics.median(elapsed), 2),
    }


def aggregate(rows: list[dict], *, temp: float, n: int) -> dict:
    rs = [r for r in rows if r["temp"] == temp and r["n"] == n and "matched" in r]
    if not rs:
        return {"temp": temp, "n": n, "n_cases": 0}
    total_m = sum(r["matched"] for r in rs)
    total_e = sum(r["expected"] for r in rs)
    total_fp = sum(r["false_positives"] for r in rs)
    perfect = sum(1 for r in rs if r["expected"] and r["matched"] == r["expected"] and r["false_positives"] == 0)
    extracted = total_m + total_fp
    precision = total_m / extracted if extracted else 0.0
    recall = total_m / total_e if total_e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "temp": temp,
        "n": n,
        "n_cases": len(rs),
        "perfect": perfect,
        "matched": total_m,
        "expected": total_e,
        "false_positives": total_fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "median_elapsed_s": round(statistics.median(r["median_elapsed_s"] for r in rs), 2),
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
        ],
    )
    ap.add_argument("--cases-by-id", nargs="*", default=None,
                    help="Restrict to these case_ids (e.g., the 2/27 misses). Default: all loaded cases.")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    ap.add_argument("--n-rollouts", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--out-json", default="apps/mobile/convert/build/self_consistency_bench.json")
    args = ap.parse_args()

    all_cases: list[dict] = []
    for p in args.cases:
        for ln in Path(p).read_text().splitlines():
            ln = ln.strip()
            if ln:
                all_cases.append(json.loads(ln))
    if args.cases_by_id:
        wanted = set(args.cases_by_id)
        cases = [c for c in all_cases if c["case_id"] in wanted]
        missing = wanted - {c["case_id"] for c in cases}
        if missing:
            print(f"WARNING: missing case_ids: {missing}", flush=True)
    else:
        cases = all_cases
    total_runs = len(cases) * sum(n for n in args.n_rollouts) * len(args.temps)
    print(f"Self-consistency bench: {len(cases)} cases × {len(args.temps)} temps × n_rollouts={args.n_rollouts} → {total_runs} agent runs", flush=True)

    rows: list[dict] = []
    for temp in args.temps:
        for n in args.n_rollouts:
            print(f"\n--- temp={temp} n={n} (vote threshold={(n//2)+1 if n>1 else 1})", flush=True)
            for idx, case in enumerate(cases, 1):
                row = run_case(case, n=n, temp=temp, endpoint=args.endpoint)
                rows.append(row)
                if "error" in row:
                    print(f"  ERR {idx}/{len(cases)} {row['case_id']}: {row['error']}", flush=True)
                else:
                    tag = "OK " if (row["matched"] == row["expected"] and row["false_positives"] == 0) else "MISS"
                    print(f"  {tag} {idx}/{len(cases)} {row['case_id']:38s} {row['matched']}/{row['expected']} fp={row['false_positives']} [{row['median_elapsed_s']:.1f}s]", flush=True)

    summary = []
    for temp in args.temps:
        for n in args.n_rollouts:
            summary.append(aggregate(rows, temp=temp, n=n))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n" + "=" * 88)
    print(f"{'temp':>5} {'n':>3} {'F1':>6} {'prec':>6} {'recall':>7} {'perfect':>9} {'fp':>4} {'median_s':>9}")
    print("=" * 88)
    for s in summary:
        if s["n_cases"] == 0:
            continue
        print(f"{s['temp']:>5.2f} {s['n']:>3d} {s['f1']:>6.3f} {s['precision']:>6.3f} {s['recall']:>7.3f} {s['perfect']:>4d}/{s['n_cases']:<4d} {s['false_positives']:>4d} {s['median_elapsed_s']:>8.2f}s")
    print(f"\nTrace: {out_path}")


if __name__ == "__main__":
    main()
