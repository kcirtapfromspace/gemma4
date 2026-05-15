#!/usr/bin/env python3
"""Aggregate non-iOS benchmark evidence into Markdown or JSON.

This reads one or more agent-bench JSON outputs plus optional FHIR score JSON
outputs and emits a compact competition-reporting summary. It is intentionally
standalone so it does not change the benchmark or mobile conversion pipelines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_BUCKETS = ("conditions", "loincs", "rxnorms")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return default


def safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[max(0, min(len(ordered) - 1, idx))]


def fmt_rate(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def extract_trace_elapsed(row: dict[str, Any]) -> float | None:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return None
    elapsed = [as_float(step.get("elapsed_s")) for step in trace if isinstance(step, dict) and step.get("elapsed_s") is not None]
    return sum(elapsed) if elapsed else None


def extract_trace_tokens(row: dict[str, Any]) -> int:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return 0
    return sum(int(as_float(step.get("tokens"))) for step in trace if isinstance(step, dict))


def extract_trace_tok_s(row: dict[str, Any]) -> list[float]:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return []
    rates: list[float] = []
    for step in trace:
        if not isinstance(step, dict):
            continue
        timings = step.get("timings")
        if not isinstance(timings, dict):
            continue
        rate = as_float(timings.get("predicted_per_second"), default=-1.0)
        if rate >= 0:
            rates.append(rate)
    return rates


def summarize_agent_bench(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} is not an agent-bench array or object with rows[]")

    total_matched = 0
    total_expected = 0
    total_fp = 0
    exact = 0
    path_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    missing_cases: list[str] = []
    fp_cases: list[str] = []
    elapsed_values: list[float] = []
    tool_calls: list[int] = []
    token_counts: list[int] = []
    tok_s_values: list[float] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or f"row_{index}")
        matched = int(as_float(row.get("matched")))
        expected = int(as_float(row.get("expected")))
        false_positives = int(as_float(row.get("false_positives")))
        total_matched += matched
        total_expected += expected
        total_fp += false_positives
        if expected == matched and false_positives == 0:
            exact += 1
        if matched < expected:
            missing_cases.append(case_id)
        if false_positives:
            fp_cases.append(case_id)

        path_counts[str(row.get("path") or "unknown")] += 1
        tool_calls.append(int(as_float(row.get("n_tool_calls"))))

        extraction = row.get("extraction")
        if isinstance(extraction, dict):
            for bucket in CODE_BUCKETS:
                values = extraction.get(bucket)
                if isinstance(values, list):
                    code_counts[bucket] += len(values)

        elapsed = extract_trace_elapsed(row)
        if elapsed is not None:
            elapsed_values.append(elapsed)
        tokens = extract_trace_tokens(row)
        if tokens:
            token_counts.append(tokens)
        tok_s_values.extend(extract_trace_tok_s(row))

    precision = safe_rate(total_matched, total_matched + total_fp)
    recall = safe_rate(total_matched, total_expected)
    return {
        "source": str(path),
        "suite": path.stem,
        "n_cases": len(rows),
        "matched": total_matched,
        "expected": total_expected,
        "false_positives": total_fp,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "exact_match_rate": safe_rate(exact, len(rows)),
        "exact_match_cases": exact,
        "path_counts": dict(sorted(path_counts.items())),
        "code_counts": dict(code_counts),
        "tool_calls_total": sum(tool_calls),
        "tool_calls_mean": safe_rate(sum(tool_calls), len(tool_calls)),
        "elapsed_s_sum": sum(elapsed_values) if elapsed_values else None,
        "elapsed_s_mean": safe_rate(sum(elapsed_values), len(elapsed_values)),
        "elapsed_s_p50": percentile(elapsed_values, 0.50),
        "elapsed_s_p95": percentile(elapsed_values, 0.95),
        "tokens_mean": safe_rate(sum(token_counts), len(token_counts)),
        "tok_s_mean": safe_rate(sum(tok_s_values), len(tok_s_values)),
        "cases_with_missing_expected": missing_cases[:20],
        "cases_with_false_positives": fp_cases[:20],
        "n_cases_with_missing_expected": len(missing_cases),
        "n_cases_with_false_positives": len(fp_cases),
    }


def summarize_fhir_score(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a FHIR score object")
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    invalid_rows = [
        str(row.get("case_id") or f"row_{index}")
        for index, row in enumerate(rows)
        if isinstance(row, dict) and row.get("valid") is not True
    ]
    n_validated = int(as_float(payload.get("n_validated"), len(rows)))
    n_pass = int(as_float(payload.get("n_pass"), n_validated - len(invalid_rows)))
    pass_rate = payload.get("fhir_r4_pass_rate")
    return {
        "source": str(path),
        "agent_source": payload.get("source"),
        "backend": payload.get("backend"),
        "n_pass": n_pass,
        "n_validated": n_validated,
        "fhir_r4_pass_rate": as_float(pass_rate) if pass_rate is not None else safe_rate(n_pass, n_validated),
        "invalid_cases": invalid_rows[:20],
        "n_invalid_cases": len(invalid_rows),
    }


def build_summary(agent_paths: list[Path], fhir_paths: list[Path]) -> dict[str, Any]:
    suites = [summarize_agent_bench(path) for path in agent_paths]
    fhir_scores = [summarize_fhir_score(path) for path in fhir_paths]

    matched = sum(s["matched"] for s in suites)
    expected = sum(s["expected"] for s in suites)
    false_positives = sum(s["false_positives"] for s in suites)
    precision = safe_rate(matched, matched + false_positives)
    recall = safe_rate(matched, expected)
    path_counts: Counter[str] = Counter()
    for suite in suites:
        path_counts.update(suite["path_counts"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_bench": {
            "n_suites": len(suites),
            "n_cases": sum(s["n_cases"] for s in suites),
            "matched": matched,
            "expected": expected,
            "false_positives": false_positives,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "path_counts": dict(sorted(path_counts.items())),
        },
        "suites": suites,
        "fhir_scores": fhir_scores,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    agent = summary["agent_bench"]
    lines = [
        "# Non-iOS Benchmark Evidence Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Headline",
        "",
        (
            f"- Agent bench: **{agent['matched']}/{agent['expected']} matched**, "
            f"**{agent['false_positives']} false positives**, "
            f"micro-F1 **{fmt_rate(agent['f1'])}** "
            f"(precision {fmt_rate(agent['precision'])}, recall {fmt_rate(agent['recall'])}) "
            f"across **{agent['n_cases']} cases** in **{agent['n_suites']} suite(s)**."
        ),
        f"- Execution paths: `{json.dumps(agent['path_counts'], sort_keys=True)}`.",
    ]
    if summary["fhir_scores"]:
        total_pass = sum(score["n_pass"] for score in summary["fhir_scores"])
        total_validated = sum(score["n_validated"] for score in summary["fhir_scores"])
        lines.append(
            f"- FHIR R4 structural validation: **{total_pass}/{total_validated} pass** "
            f"({fmt_pct(safe_rate(total_pass, total_validated))}) across "
            f"**{len(summary['fhir_scores'])} score file(s)**."
        )
    else:
        lines.append("- FHIR R4 structural validation: not provided.")

    lines += [
        "",
        "## Agent Suites",
        "",
        "| Suite | Cases | Matched/Expected | FP | Precision | Recall | F1 | Exact | Paths | Mean tools | p50 s | p95 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for suite in summary["suites"]:
        lines.append(
            "| {suite} | {cases} | {matched}/{expected} | {fp} | {precision} | {recall} | {f1v} | {exact} | `{paths}` | {tools} | {p50} | {p95} |".format(
                suite=suite["suite"],
                cases=suite["n_cases"],
                matched=suite["matched"],
                expected=suite["expected"],
                fp=suite["false_positives"],
                precision=fmt_rate(suite["precision"]),
                recall=fmt_rate(suite["recall"]),
                f1v=fmt_rate(suite["f1"]),
                exact=fmt_pct(suite["exact_match_rate"], digits=0),
                paths=json.dumps(suite["path_counts"], sort_keys=True),
                tools=fmt_rate(suite["tool_calls_mean"], digits=2),
                p50=fmt_rate(suite["elapsed_s_p50"], digits=2),
                p95=fmt_rate(suite["elapsed_s_p95"], digits=2),
            )
        )

    if summary["fhir_scores"]:
        lines += [
            "",
            "## FHIR Scores",
            "",
            "| Score file | Backend | Agent source | Pass | Pass rate | Invalid cases |",
            "|---|---|---|---:|---:|---|",
        ]
        for score in summary["fhir_scores"]:
            invalid = ", ".join(score["invalid_cases"]) if score["invalid_cases"] else "-"
            lines.append(
                f"| {Path(score['source']).name} | {score.get('backend') or '-'} | "
                f"{score.get('agent_source') or '-'} | {score['n_pass']}/{score['n_validated']} | "
                f"{fmt_pct(score['fhir_r4_pass_rate'])} | {invalid} |"
            )

    issue_suites = [
        suite
        for suite in summary["suites"]
        if suite["n_cases_with_missing_expected"] or suite["n_cases_with_false_positives"]
    ]
    if issue_suites:
        lines += ["", "## Review Notes", ""]
        for suite in issue_suites:
            if suite["n_cases_with_missing_expected"]:
                lines.append(
                    f"- `{suite['suite']}` missing expected codes in {suite['n_cases_with_missing_expected']} case(s): "
                    + ", ".join(suite["cases_with_missing_expected"])
                )
            if suite["n_cases_with_false_positives"]:
                lines.append(
                    f"- `{suite['suite']}` has false positives in {suite['n_cases_with_false_positives']} case(s): "
                    + ", ".join(suite["cases_with_false_positives"])
                )

    lines += [
        "",
        "## Inputs",
        "",
    ]
    for suite in summary["suites"]:
        lines.append(f"- Agent bench: `{suite['source']}`")
    for score in summary["fhir_scores"]:
        lines.append(f"- FHIR score: `{score['source']}`")
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate non-iOS agent-bench and optional FHIR score JSON evidence.",
    )
    parser.add_argument(
        "agent_bench",
        nargs="+",
        type=Path,
        help="Agent-bench JSON output(s), usually arrays of per-case rows.",
    )
    parser.add_argument(
        "--fhir-score",
        action="append",
        type=Path,
        default=[],
        help="Optional FHIR score JSON output. May be repeated.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Primary stdout/output format. Default: markdown.",
    )
    parser.add_argument("--out", type=Path, help="Write primary output to this path instead of stdout.")
    parser.add_argument("--markdown-out", type=Path, help="Also write Markdown summary to this path.")
    parser.add_argument("--json-out", type=Path, help="Also write JSON summary to this path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        summary = build_summary(args.agent_bench, args.fhir_score)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    markdown = render_markdown(summary)
    json_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    primary = markdown if args.format == "markdown" else json_text

    if args.out:
        write_text(args.out, primary)
    else:
        print(primary, end="")
    if args.markdown_out:
        write_text(args.markdown_out, markdown)
    if args.json_out:
        write_text(args.json_out, json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
