"""Multi-temperature robustness bench on combined-54.

Confirms that the F1=1.000 result observed on combined-54 (53/54 perfect,
the SJS gap honestly missed) is robust to sampling variance by re-running
the FULL agent loop at temperature ∈ {0.0, 0.1, 0.2}.

Crucially, this bench bypasses the Cand D deterministic short-circuit and
the c19 fast-path: every case goes through ``agent_pipeline.run_agent``
directly. This isolates the LLM-side variance — the thing temperature
actually affects — from the deterministic tier whose behavior is
sampling-invariant. If F1 stays at 1.000 across all three temps and no
case flips, the agent path is robust under non-zero sampling.

Usage:
    scripts/.venv/bin/python apps/mobile/convert/bench_multi_temp.py \\
        --temps 0.0 0.1 0.2 \\
        --out-json apps/mobile/convert/build/multi_temp_bench.json

Companion to tools/autoresearch/c20-llm-tuning-2026-04-25.md (adv7
stress-bench section).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parent))
from agent_pipeline import run_agent, score  # noqa: E402

DEFAULT_CASES = [
    "scripts/test_cases.jsonl",
    "scripts/test_cases_adversarial.jsonl",
    "scripts/test_cases_adversarial2.jsonl",
    "scripts/test_cases_adversarial3.jsonl",
    "scripts/test_cases_adversarial4.jsonl",
    "scripts/test_cases_adversarial5.jsonl",
    "scripts/test_cases_adversarial6.jsonl",
]


def run_one(case: dict, temperature: float, endpoint: str) -> dict:
    """Run one case via the agent path at the given temperature.

    No Cand D short-circuit, no fast-path — ``run_agent`` only. Captures
    elapsed time and the parsed extraction. Errors are stashed on the row.
    """
    t0 = time.time()
    try:
        extraction, trace = run_agent(
            case["user"],
            endpoint=endpoint,
            temperature=temperature,
        )
        err = None
    except (URLError, TimeoutError, OSError) as exc:
        extraction, trace = {"conditions": [], "loincs": [], "rxnorms": []}, []
        err = str(exc)
    elapsed = time.time() - t0
    m, e, fpc = score(extraction, case)
    return {
        "case_id": case["case_id"],
        "temperature": temperature,
        "matched": m,
        "expected": e,
        "false_positives": fpc,
        "extraction": extraction,
        "elapsed_s": round(elapsed, 3),
        "n_turns": len(trace),
        "error": err,
    }


def aggregate(rows: list[dict], temperature: float) -> dict:
    rows_at = [r for r in rows if r["temperature"] == temperature and r.get("error") is None]
    n = len(rows_at)
    total_m = sum(r["matched"] for r in rows_at)
    total_e = sum(r["expected"] for r in rows_at)
    total_fp = sum(r["false_positives"] for r in rows_at)
    perfect = sum(
        1 for r in rows_at
        if r["expected"] and r["matched"] == r["expected"] and r["false_positives"] == 0
    )
    extracted = total_m + total_fp
    precision = total_m / extracted if extracted else 0.0
    recall = total_m / total_e if total_e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    elapsed = sorted(r["elapsed_s"] for r in rows_at)
    p50 = statistics.median(elapsed) if elapsed else 0
    p95 = elapsed[int(0.95 * (len(elapsed) - 1))] if elapsed else 0
    errs = sum(
        1 for r in rows
        if r["temperature"] == temperature and r.get("error") is not None
    )
    return {
        "temperature": temperature,
        "n_cases": n,
        "errors": errs,
        "perfect": perfect,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": total_m,
        "expected": total_e,
        "false_positives": total_fp,
        "p50_s": round(p50, 3),
        "p95_s": round(p95, 3),
    }


def detect_flips(rows: list[dict], temps: list[float]) -> list[dict]:
    """Find cases where the (matched, false_positives) tuple varies across temps.

    Returns one record per case_id that flipped. Useful for pinpointing
    sampling-sensitive cases without needing to scan the full per-row table.
    """
    by_case: dict[str, dict[float, dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["temperature"]] = r
    flips: list[dict] = []
    for case_id, by_t in by_case.items():
        signatures = set()
        for t in temps:
            r = by_t.get(t)
            if r is None or r.get("error"):
                signatures.add(("error",))
                continue
            signatures.add((r["matched"], r["false_positives"]))
        if len(signatures) > 1:
            flips.append({
                "case_id": case_id,
                "per_temp": {
                    str(t): {
                        "matched": by_t.get(t, {}).get("matched"),
                        "expected": by_t.get(t, {}).get("expected"),
                        "false_positives": by_t.get(t, {}).get("false_positives"),
                        "error": by_t.get(t, {}).get("error"),
                        "extraction": by_t.get(t, {}).get("extraction"),
                    } for t in temps
                },
            })
    return flips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    ap.add_argument(
        "--out-json",
        default="apps/mobile/convert/build/multi_temp_bench.json",
    )
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()

    cases: list[dict] = []
    for p in args.cases:
        for ln in Path(p).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(
        f"Multi-temp bench: {len(cases)} cases × {len(args.temps)} temps = "
        f"{len(cases) * len(args.temps)} agent runs",
        flush=True,
    )

    rows: list[dict] = []
    for temperature in args.temps:
        print(f"\n--- temperature = {temperature}", flush=True)
        for idx, case in enumerate(cases, 1):
            row = run_one(case, temperature, args.endpoint)
            rows.append(row)
            if row.get("error"):
                tag = "ERR "
            elif row["matched"] == row["expected"] and row["false_positives"] == 0:
                tag = "OK  "
            else:
                tag = "MISS"
            err_tag = f" err={row['error']}" if row.get("error") else ""
            print(
                f"  {tag} {idx:2d}/{len(cases)} {row['case_id']:50s} "
                f"{row['matched']}/{row['expected']} fp={row['false_positives']} "
                f"{row['elapsed_s']:5.1f}s turns={row['n_turns']}{err_tag}",
                flush=True,
            )

    summary = [aggregate(rows, t) for t in args.temps]
    flips = detect_flips(rows, args.temps)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": summary,
        "flipped_cases": flips,
        "rows": rows,
        "config": {
            "endpoint": args.endpoint,
            "cases": args.cases,
            "temps": args.temps,
            "n_cases": len(cases),
        },
    }, indent=2))

    # ---- pretty summary
    print("\n" + "=" * 96)
    print(
        f"{'temp':>5} {'F1':>6} {'prec':>6} {'recall':>7} "
        f"{'perf':>7} {'fp':>4} {'errs':>5} {'p50':>7} {'p95':>7}"
    )
    print("=" * 96)
    for s in summary:
        print(
            f"{s['temperature']:>5.2f} {s['f1']:>6.3f} {s['precision']:>6.3f} "
            f"{s['recall']:>7.3f} {s['perfect']:>3d}/{s['n_cases']:<3d} "
            f"{s['false_positives']:>4d} {s['errors']:>5d} "
            f"{s['p50_s']:>6.2f}s {s['p95_s']:>6.2f}s"
        )
    print(f"\nFlipped cases (matched/fp differs across temps): {len(flips)}")
    for f in flips:
        per_t = ", ".join(
            f"t={t}:m={v['matched']}/fp={v['false_positives']}"
            for t, v in f["per_temp"].items()
        )
        print(f"  {f['case_id']}: {per_t}")
    print(f"\nTrace: {out_path}")


if __name__ == "__main__":
    main()
