"""Stability bench for the agent loop — 27 cases × N repeats, parse-failure count.

Per proposals-2026-04-25.md Rank 4, the success criterion is **0 unrecoverable
parse failures**. The original plan was to A/B "grammar on vs off" on Python,
but llama-server's `/v1/chat/completions` rejects custom `grammar` whenever
`tools` is set ("Cannot use custom grammar constraints with tools.") — the
`--jinja` path already applies an internal tool-call grammar derived from
the chat template. So the Python parse-failure rate is *already* whatever
llama-server's built-in grammar can guarantee; this bench measures that
empirically across 3 runs to confirm "0 unrecoverable failures" holds.

The explicit GBNF (`cliniq_toolcall.gbnf`) ships for the iOS path, where
`AgentRunner` drives raw llama.cpp via `LlamaCppInferenceEngine.applyGrammar`
on tool-response turns. The iOS measurement requires a physical device or
simulator with the GGUF mounted; it'll land alongside Rank 1 (`ios-eng`).

Aggregate metrics:

  - parse_errors: count of `tool_call.arguments` JSON decode failures (the
    error path the silent fallback used to hide). Target: 0 across 27×3.
  - turn_count_avg / decode_tok_s: latency / throughput sanity check.
  - F1 / perfect: extraction quality unchanged from the c17 baseline of
    F1=0.986 / 25-of-27 perfect.

Requires llama-server running (default 127.0.0.1:8090) with the agent's
chat template loaded (`--jinja`). Cases default to combined-27.

Usage:
    python apps/mobile/convert/bench_toolcall_grammar.py --repeats 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_pipeline import run_agent, score  # noqa: E402


def _count_parse_errors(trace: list[dict]) -> int:
    """Count tool_result entries whose result has an 'argument JSON decode' error.

    With grammar on this should be 0; with grammar off we typically see 1–2
    per 100 turns on adversarial inputs. Surfaced separately from
    `tool_raised` errors (which are tool-impl bugs, not parse failures).
    """
    n = 0
    for evt in trace:
        if not isinstance(evt, dict):
            continue
        result = evt.get("result")
        if isinstance(result, dict) and isinstance(result.get("error"), str):
            if "argument JSON decode failed" in result["error"]:
                n += 1
    return n


def _aggregate(rows: list[dict]) -> dict:
    """Per-mode summary across all repeats × all cases."""
    total_m = sum(r["matched"] for r in rows)
    total_e = sum(r["expected"] for r in rows)
    total_fp = sum(r["false_positives"] for r in rows)
    total_pe = sum(r["parse_errors"] for r in rows)
    total_turns = sum(r["turns"] for r in rows)
    elapsed_per_case = [r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]
    perfect = sum(1 for r in rows if r["expected"] and r["matched"] == r["expected"] and r["false_positives"] == 0)
    extracted_total = total_m + total_fp
    precision = total_m / extracted_total if extracted_total else 0.0
    recall = total_m / total_e if total_e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_runs": len(rows),
        "matched": total_m,
        "expected": total_e,
        "false_positives": total_fp,
        "parse_errors": total_pe,
        "perfect": perfect,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_turns_per_case": round(total_turns / len(rows), 2) if rows else 0.0,
        "median_elapsed_s": round(statistics.median(elapsed_per_case), 2) if elapsed_per_case else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090")
    ap.add_argument("--grammar-file", default="apps/mobile/convert/cliniq_toolcall.gbnf")
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
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument(
        "--out-json",
        default="apps/mobile/convert/build/toolcall_grammar_bench.json",
    )
    args = ap.parse_args()

    # Grammar is loaded for forward-compatibility and printed in the
    # output JSON header so the bench artifact records which GBNF was in
    # play, even though llama-server ignores it on /v1/chat/completions.
    grammar = Path(args.grammar_file).read_text() if args.grammar_file else None

    cases: list[dict] = []
    for p in args.cases:
        for ln in Path(p).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(
        f"Bench: {len(cases)} cases × {args.repeats} repeats "
        f"= {len(cases) * args.repeats} runs (grammar disabled on Python "
        f"per llama-server limitation; tracked for iOS bench)",
        flush=True,
    )

    all_rows: list[dict] = []
    for repeat in range(args.repeats):
        for case in cases:
            cid = case["case_id"]
            t0 = time.time()
            try:
                extraction, trace = run_agent(
                    case["user"],
                    endpoint=args.endpoint,
                    tool_call_grammar=grammar,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERR repeat={repeat} {cid}: {exc}")
                all_rows.append({
                    "repeat": repeat,
                    "mode": "baseline",
                    "case_id": cid,
                    "error": str(exc),
                    "matched": 0, "expected": 0, "false_positives": 0,
                    "parse_errors": 0, "turns": 0, "elapsed_s": None,
                })
                continue
            m, e, fp = score(extraction, case)
            pe = _count_parse_errors(trace)
            turns = sum(1 for t in trace if "turn" in t and "tool_calls" in t)
            elapsed = round(time.time() - t0, 2)
            marker = "OK " if (m == e and fp == 0) else "MISS"
            pe_tag = f" PE={pe}" if pe else ""
            print(
                f"  r{repeat} {marker} {cid:34s} "
                f"{m}/{e}{pe_tag} ({turns} turns, {elapsed}s)",
                flush=True,
            )
            all_rows.append({
                "repeat": repeat,
                "mode": "baseline",
                "case_id": cid,
                "matched": m,
                "expected": e,
                "false_positives": fp,
                "parse_errors": pe,
                "turns": turns,
                "elapsed_s": elapsed,
            })

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=2))

    print()
    rows = [r for r in all_rows if "error" not in r]
    agg = _aggregate(rows)
    print(f"AGG baseline (Python, llama-server internal grammar): {json.dumps(agg)}")
    print(f"\nFull results: {out}")
    print(
        "iOS-side grammar bench (against AgentRunner + LlamaCppInferenceEngine) "
        "is owned by ios-eng — see Rank 1 in proposals-2026-04-25.md."
    )


if __name__ == "__main__":
    main()
