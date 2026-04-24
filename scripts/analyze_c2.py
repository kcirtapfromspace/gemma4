#!/usr/bin/env python3
"""Analysis for C2 sweep: print results table and identify winner."""

import duckdb
import sys

DB = "/Users/thinkstudio/gemma4/scripts/benchmarks.duckdb"


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else DB
    c = duckdb.connect(db, read_only=True)

    rows = c.execute(
        """SELECT experiment_name,
                  avg_gen_tok_s,
                  p50_gen_tok_s,
                  avg_extraction_score,
                  success_rate,
                  total_runs,
                  avg_ttft_ms,
                  p95_total_ms,
                  avg_prompt_tok_s,
                  notes
           FROM experiments
           WHERE experiment_name LIKE 'c2-%'
           ORDER BY avg_gen_tok_s DESC NULLS LAST"""
    ).fetchall()

    if not rows:
        print("No C2 experiments found.")
        return

    baseline = None
    for r in rows:
        if r[0] == "c2-baseline":
            baseline = r[1]
            break

    print("=" * 110)
    print(f"  C2 llama-server flag sweep — experiments={len(rows)}")
    print(f"  baseline (c2-baseline) tok/s: {baseline:.2f}" if baseline else "  baseline not found")
    print("=" * 110)
    header = f"  {'name':<26} {'tok/s':>7} {'p50':>6} {'score':>6} {'valid%':>6} {'runs':>5} {'ptok/s':>7} {'delta_%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        name, tok, p50, score, succ, runs, ttft, p95, ptok, notes = r
        if tok is None:
            print(f"  {name:<26} {'N/A':>7} {'':>6} {'':>6} {'':>6} {runs or 0:>5} {'':>7} {'':>8}")
            continue
        delta = (tok - baseline) / baseline * 100 if baseline else None
        print(
            f"  {name:<26} {tok:>7.2f} {p50 or 0:>6.2f} {score or 0:>6.2f} "
            f"{(succ or 0)*100:>5.0f}% {runs or 0:>5} {ptok or 0:>7.1f} "
            f"{delta:>7.1f}%" if delta is not None else
            f"  {name:<26} {tok:>7.2f} {p50 or 0:>6.2f} {score or 0:>6.2f} "
            f"{(succ or 0)*100:>5.0f}% {runs or 0:>5} {ptok or 0:>7.1f} {'':>8}"
        )
    print()

    # Top-3 by tok/s that maintain score >= 0.95
    good = [r for r in rows if r[1] is not None and (r[3] or 0) >= 0.95]
    good.sort(key=lambda x: -x[1])
    print("  TOP-3 with score >= 0.95:")
    for r in good[:3]:
        print(f"    - {r[0]:<25} {r[1]:.2f} tok/s (score {r[3]:.2f})")
    print()

    c.close()


if __name__ == "__main__":
    main()
