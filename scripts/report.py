#!/usr/bin/env python3
"""Generate comparison reports from the benchmark experiment tracker.

Usage:
    python report.py                          # Full report
    python report.py --db benchmarks.duckdb   # Custom DB path
    python report.py --csv results.csv        # Export to CSV
    python report.py --latest 5               # Last 5 experiments only
"""

import argparse
import sys
from pathlib import Path

import duckdb
from tabulate import tabulate

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DB = SCRIPT_DIR / "benchmarks.duckdb"


def experiment_summary(conn: duckdb.DuckDBPyConnection, limit: int | None = None) -> list:
    query = """
        SELECT
            experiment_name,
            created_at,
            COALESCE(quantization, '-') AS quant,
            COALESCE(cuda_enabled::VARCHAR, '-') AS cuda,
            COALESCE(flash_attn::VARCHAR, '-') AS flash,
            COALESCE(ctx_size::VARCHAR, '-') AS ctx,
            COALESCE(cache_type_k, '-') AS kv_cache,
            COALESCE(ROUND(avg_gen_tok_s, 1)::VARCHAR, '-') AS avg_tok_s,
            COALESCE(ROUND(p50_gen_tok_s, 1)::VARCHAR, '-') AS p50_tok_s,
            COALESCE(ROUND(p95_total_ms, 0)::VARCHAR, '-') AS p95_ms,
            COALESCE(ROUND(avg_ttft_ms, 0)::VARCHAR, '-') AS ttft,
            COALESCE(ROUND(success_rate * 100, 0)::VARCHAR || '%', '-') AS valid,
            COALESCE(ROUND(avg_extraction_score, 2)::VARCHAR, '-') AS quality,
            total_runs,
            CASE
                WHEN speedup_pct IS NOT NULL THEN
                    CASE WHEN speedup_pct >= 0
                        THEN '+' || ROUND(speedup_pct, 1)::VARCHAR || '%'
                        ELSE ROUND(speedup_pct, 1)::VARCHAR || '%'
                    END
                ELSE '--'
            END AS speedup,
            COALESCE(notes, '') AS notes
        FROM experiments
        ORDER BY created_at
    """
    if limit:
        query += f" LIMIT {limit}"
    return conn.execute(query).fetchall()


def per_case_detail(conn: duckdb.DuckDBPyConnection, experiment_name: str) -> list:
    return conn.execute("""
        SELECT
            case_id,
            COUNT(*) AS runs,
            ROUND(AVG(total_ms), 0) AS avg_ms,
            ROUND(AVG(gen_tok_per_sec), 1) AS avg_tok_s,
            ROUND(AVG(ttft_ms), 0) AS avg_ttft,
            ROUND(AVG(CASE WHEN valid_json THEN 1.0 ELSE 0.0 END) * 100, 0) AS valid_pct
        FROM benchmark_runs br
        JOIN experiments e ON br.experiment_id = e.experiment_id
        WHERE e.experiment_name = ?
          AND br.total_ms IS NOT NULL
        GROUP BY case_id
        ORDER BY case_id
    """, [experiment_name]).fetchall()


def waterfall(conn: duckdb.DuckDBPyConnection) -> list:
    """Show cumulative improvement from first to latest experiment."""
    rows = conn.execute("""
        SELECT
            experiment_name,
            avg_gen_tok_s,
            created_at
        FROM experiments
        WHERE avg_gen_tok_s IS NOT NULL
        ORDER BY created_at
    """).fetchall()

    if not rows:
        return []

    baseline = rows[0][1]
    result = []
    for name, tok_s, ts in rows:
        cumulative = ((tok_s - baseline) / baseline * 100) if baseline else 0
        result.append((name, tok_s, cumulative, ts))
    return result


def main():
    parser = argparse.ArgumentParser(description="ClinIQ benchmark report generator")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="DuckDB file path")
    parser.add_argument("--csv", default=None, help="Export summary to CSV")
    parser.add_argument("--latest", type=int, default=None, help="Show only N latest experiments")
    parser.add_argument("--detail", default=None, help="Show per-case detail for experiment name")
    parser.add_argument("--waterfall", action="store_true", help="Show improvement waterfall")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"No benchmark database found at {args.db}")
        print("Run benchmark.py first to generate data.")
        sys.exit(1)

    conn = duckdb.connect(args.db, read_only=True)

    # Summary table
    rows = experiment_summary(conn, args.latest)
    if not rows:
        print("No experiments found.")
        conn.close()
        return

    headers = [
        "Experiment", "Date", "Quant", "CUDA", "Flash", "Ctx",
        "KV$", "Avg tok/s", "P50 tok/s", "P95 ms", "TTFT ms",
        "Valid", "Quality", "Runs", "Speedup", "Notes",
    ]

    print("\n## Experiment Comparison\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # CSV export
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\nExported to {args.csv}")

    # Per-case detail
    if args.detail:
        detail = per_case_detail(conn, args.detail)
        if detail:
            print(f"\n## Per-Case Detail: {args.detail}\n")
            print(tabulate(
                detail,
                headers=["Case", "Runs", "Avg ms", "Tok/s", "TTFT ms", "Valid%"],
                tablefmt="github",
            ))
        else:
            print(f"\nNo runs found for experiment '{args.detail}'")

    # Waterfall
    if args.waterfall:
        wf = waterfall(conn)
        if wf:
            print("\n## Improvement Waterfall\n")
            print(tabulate(
                [(name, f"{tok_s:.1f}", f"{cum:+.1f}%") for name, tok_s, cum, _ in wf],
                headers=["Experiment", "Tok/s", "Cumulative vs Baseline"],
                tablefmt="github",
            ))

    # Quick stats
    best = conn.execute("""
        SELECT experiment_name, avg_gen_tok_s
        FROM experiments
        WHERE avg_gen_tok_s IS NOT NULL
        ORDER BY avg_gen_tok_s DESC
        LIMIT 1
    """).fetchone()
    worst = conn.execute("""
        SELECT experiment_name, avg_gen_tok_s
        FROM experiments
        WHERE avg_gen_tok_s IS NOT NULL
        ORDER BY avg_gen_tok_s ASC
        LIMIT 1
    """).fetchone()

    if best and worst:
        print(f"\nBest:  {best[0]} ({best[1]:.1f} tok/s)")
        print(f"Worst: {worst[0]} ({worst[1]:.1f} tok/s)")
        if worst[1] > 0:
            print(f"Range: {best[1] / worst[1]:.1f}x improvement\n")

    conn.close()


if __name__ == "__main__":
    main()
