#!/usr/bin/env python3
"""Summarize ClinIQ evaluation corpus sizes and broad benchmark results.

This is deliberately stdlib-only so it can run in CI, Kaggle notebooks, and
local worktrees without the benchmark DuckDB environment.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CODE_CASES = [
    "scripts/test_cases.jsonl",
    "scripts/test_cases_adversarial.jsonl",
    "scripts/test_cases_adversarial2.jsonl",
    "scripts/test_cases_adversarial3.jsonl",
    "scripts/test_cases_adversarial4.jsonl",
    "scripts/test_cases_adversarial5.jsonl",
    "scripts/test_cases_adversarial6.jsonl",
    "scripts/test_cases_adversarial7.jsonl",
    "scripts/test_cases_adversarial8.jsonl",
    "scripts/test_cases_external.jsonl",
    "scripts/test_cases_longitudinal.jsonl",
]
CODE_KEYS = ("expected_conditions", "expected_loincs", "expected_rxnorms")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = round((len(values) - 1) * q)
    return values[max(0, min(len(values) - 1, idx))]


def wilson_lower_bound(successes: int, n: int, z: float = 1.959963984540054) -> float:
    """95% Wilson score lower bound for a binomial proportion."""
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def summarize_code_cases(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    duplicate_case_ids: list[str] = []

    for path in paths:
        file_rows = load_jsonl(path)
        file_expected = 0
        file_hard_negatives = 0
        for row in file_rows:
            cid = str(row.get("case_id") or "")
            if cid in seen_case_ids:
                duplicate_case_ids.append(cid)
            seen_case_ids.add(cid)
            n_expected = sum(len(row.get(k) or []) for k in CODE_KEYS)
            file_expected += n_expected
            file_hard_negatives += 1 if n_expected == 0 else 0
            rows.append(row)
        per_file.append(
            {
                "path": str(path),
                "cases": len(file_rows),
                "expected_codes": file_expected,
                "hard_negatives": file_hard_negatives,
            }
        )

    counts = {
        "conditions": sum(len(row.get("expected_conditions") or []) for row in rows),
        "loincs": sum(len(row.get("expected_loincs") or []) for row in rows),
        "rxnorms": sum(len(row.get("expected_rxnorms") or []) for row in rows),
    }
    narrative_lengths = [len(str(row.get("user") or "")) for row in rows]
    return {
        "cases": len(rows),
        "unique_case_ids": len(seen_case_ids),
        "duplicate_case_ids": duplicate_case_ids,
        "expected_codes": sum(counts.values()),
        "expected_by_bucket": counts,
        "hard_negatives": sum(1 for row in rows if sum(len(row.get(k) or []) for k in CODE_KEYS) == 0),
        "narrative_chars_p50": percentile(narrative_lengths, 0.50),
        "narrative_chars_p95": percentile(narrative_lengths, 0.95),
        "narrative_chars_max": max(narrative_lengths) if narrative_lengths else 0,
        "per_file": per_file,
    }


def parse_compact_assistant(row: dict[str, Any]) -> dict[str, Any]:
    conversations = row.get("conversations") or []
    assistant = next(
        (item.get("content") for item in conversations if item.get("role") == "assistant"),
        "{}",
    )
    try:
        parsed = json.loads(assistant)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compact_code_count(record: dict[str, Any]) -> int:
    total = 0
    for section in ("conditions", "labs", "meds"):
        for item in record.get(section) or []:
            if not isinstance(item, dict):
                continue
            for key in ("snomed", "icd10", "loinc", "rxnorm"):
                if item.get(key):
                    total += 1
    vitals = record.get("vitals") or {}
    if isinstance(vitals, dict):
        total += sum(1 for value in vitals.values() if value not in (None, ""))
    return total


def summarize_compact(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    schema = {"patient", "conditions", "labs", "meds", "vitals"}
    output_lengths: list[int] = []
    code_counts: list[int] = []
    complete = 0
    section_rows = Counter()
    for row in rows:
        parsed = parse_compact_assistant(row)
        if schema.issubset(parsed.keys()):
            complete += 1
        assistant = next(
            (
                item.get("content") or ""
                for item in row.get("conversations") or []
                if item.get("role") == "assistant"
            ),
            "",
        )
        output_lengths.append(len(assistant))
        code_counts.append(compact_code_count(parsed))
        for section in ("conditions", "labs", "meds"):
            if parsed.get(section):
                section_rows[section] += 1
        if parsed.get("vitals"):
            section_rows["vitals"] += 1
    return {
        "path": str(path),
        "rows": len(rows),
        "schema_complete_rows": complete,
        "schema_complete_rate": complete / len(rows) if rows else 0.0,
        "gold_code_like_items": sum(code_counts),
        "gold_code_like_items_p50": percentile(code_counts, 0.50),
        "gold_code_like_items_p95": percentile(code_counts, 0.95),
        "output_chars_p50": percentile(output_lengths, 0.50),
        "output_chars_p95": percentile(output_lengths, 0.95),
        "output_chars_max": max(output_lengths) if output_lengths else 0,
        "rows_with_section": dict(section_rows),
    }


def summarize_agent_bench(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    matched = sum(int(row.get("matched") or 0) for row in rows)
    expected = sum(int(row.get("expected") or 0) for row in rows)
    fp = sum(int(row.get("false_positives") or 0) for row in rows)
    perfect = sum(
        1
        for row in rows
        if int(row.get("matched") or 0) == int(row.get("expected") or 0)
        and int(row.get("false_positives") or 0) == 0
    )
    path_counts = Counter(str(row.get("path") or "unknown") for row in rows)
    precision = matched / (matched + fp) if matched + fp else 0.0
    recall = matched / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "path": str(path),
        "cases": len(rows),
        "matched_codes": matched,
        "expected_codes": expected,
        "false_positives": fp,
        "perfect_cases": perfect,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "case_pass_wilson95_lower": wilson_lower_bound(perfect, len(rows)),
        "code_recall_wilson95_lower": wilson_lower_bound(matched, expected),
        "path_counts": dict(path_counts),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    code = summary["code_cases"]
    lines = [
        "# ClinIQ Evaluation Corpus Summary",
        "",
        "## Code Extraction Regression Corpus",
        "",
        f"- Cases: {code['cases']} ({code['unique_case_ids']} unique case IDs)",
        f"- Expected codes: {code['expected_codes']} "
        f"(SNOMED {code['expected_by_bucket']['conditions']}, "
        f"LOINC {code['expected_by_bucket']['loincs']}, "
        f"RxNorm {code['expected_by_bucket']['rxnorms']})",
        f"- Hard-negative cases: {code['hard_negatives']}",
        f"- Narrative chars p50/p95/max: {code['narrative_chars_p50']} / "
        f"{code['narrative_chars_p95']} / {code['narrative_chars_max']}",
        "",
        "| File | Cases | Expected codes | Hard negatives |",
        "|---|---:|---:|---:|",
    ]
    for row in code["per_file"]:
        lines.append(
            f"| `{row['path']}` | {row['cases']} | "
            f"{row['expected_codes']} | {row['hard_negatives']} |"
        )

    if summary.get("agent_bench"):
        bench = summary["agent_bench"]
        lines.extend(
            [
                "",
                "## Broad Agent/Deterministic Result",
                "",
                f"- Cases: {bench['perfect_cases']}/{bench['cases']} perfect",
                f"- Codes: {bench['matched_codes']}/{bench['expected_codes']} matched",
                f"- False positives: {bench['false_positives']}",
                f"- Precision/recall/F1: {bench['precision']:.4f} / "
                f"{bench['recall']:.4f} / {bench['f1']:.4f}",
                f"- 95% Wilson lower bound, case pass rate: "
                f"{bench['case_pass_wilson95_lower']:.4f}",
                f"- 95% Wilson lower bound, code recall: "
                f"{bench['code_recall_wilson95_lower']:.4f}",
                f"- Path counts: {json.dumps(bench['path_counts'], sort_keys=True)}",
            ]
        )

    for label, compact in summary.get("compact", {}).items():
        lines.extend(
            [
                "",
                f"## Compact Model Dataset: {label}",
                "",
                f"- Rows: {compact['rows']}",
                f"- Schema-complete gold rows: {compact['schema_complete_rows']} "
                f"({compact['schema_complete_rate']:.1%})",
                f"- Gold code/vital-like items: {compact['gold_code_like_items']}",
                f"- Items per row p50/p95: {compact['gold_code_like_items_p50']} / "
                f"{compact['gold_code_like_items_p95']}",
                f"- Output chars p50/p95/max: {compact['output_chars_p50']} / "
                f"{compact['output_chars_p95']} / {compact['output_chars_max']}",
                f"- Rows with sections: {json.dumps(compact['rows_with_section'], sort_keys=True)}",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The 90-case code corpus is a curated regression suite, not an iid "
            "clinical prevalence sample. Use it to claim coverage of known edge "
            "classes and FHIR validity. Use the 400-row compact validation set "
            "for model-training generalization claims, and add a larger held-out "
            "eICR/code corpus before making population-level accuracy claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-cases", nargs="+", default=DEFAULT_CODE_CASES)
    parser.add_argument("--compact-train", default="kaggle-training/dataset/train_v2.jsonl")
    parser.add_argument("--compact-val", default="kaggle-training/dataset/val-compact.jsonl")
    parser.add_argument("--agent-bench")
    parser.add_argument("--out-md")
    parser.add_argument("--out-json")
    args = parser.parse_args()

    summary: dict[str, Any] = {
        "code_cases": summarize_code_cases([Path(p) for p in args.code_cases]),
        "compact": {},
    }
    for label, path in (("train_v2", args.compact_train), ("val_compact", args.compact_val)):
        p = Path(path)
        if p.exists():
            summary["compact"][label] = summarize_compact(p)
    if args.agent_bench:
        summary["agent_bench"] = summarize_agent_bench(Path(args.agent_bench))

    rendered = render_markdown(summary)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True))
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
