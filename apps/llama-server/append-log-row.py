#!/usr/bin/env python3
"""Append a row to SPEC_DECODE_LOG.md from the most recent spec_decode.duckdb row.

Usage:
    python append-log-row.py <exp_id> <config_summary> <target> <draft> <draft_n> <ctk_ctv> [accept_rate]
"""
import sys
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB = REPO_ROOT / "apps" / "llama-server" / "spec_decode.duckdb"
LOG = REPO_ROOT / "apps" / "llama-server" / "SPEC_DECODE_LOG.md"


def main():
    if len(sys.argv) < 7:
        print("usage: append-log-row.py <id> <config> <target> <draft> <N> <ctk_ctv> [accept_rate]", file=sys.stderr)
        sys.exit(1)
    exp_id, config, target, draft, draft_n, ctk_ctv = sys.argv[1:7]
    accept = sys.argv[7] if len(sys.argv) > 7 else "n/a"

    conn = duckdb.connect(str(DB), read_only=True)
    row = conn.execute(
        """
        SELECT avg_gen_tok_s, avg_extraction_score, total_runs, notes
        FROM experiments
        WHERE experiment_name LIKE ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [f"c4-{exp_id}-%"],
    ).fetchone()
    conn.close()

    if not row:
        print(f"No experiments row for c4-{exp_id}-*", file=sys.stderr)
        sys.exit(2)

    tok_s, score, runs, notes = row
    tok_s_s = f"{tok_s:.2f}" if tok_s else "N/A"
    score_s = f"{score:.2f}" if score is not None else "N/A"

    md_row = (
        f"| {exp_id} | {target} | {draft} | {draft_n} | {ctk_ctv} | "
        f"{tok_s_s} (n={runs}) | {score_s} | {accept} | {config} |"
    )
    print(md_row)

    # Append to log.
    with open(LOG, "a") as f:
        f.write(md_row + "\n")


if __name__ == "__main__":
    main()
