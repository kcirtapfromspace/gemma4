#!/usr/bin/env python3
"""Build a scalable held-out candidate corpus from compact ClinIQ JSONL rows.

The source compact datasets are synthetic chat examples. This helper converts
them into the benchmark case schema while preserving provenance, excluding
protected eval IDs, and refusing exact protected narrative reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = ["kaggle-training/dataset/val-compact.jsonl"]
DEFAULT_PROTECTED_GLOBS = [
    "scripts/test_cases*.jsonl",
]
DEFAULT_OUT_JSONL = "tools/autoresearch/generated/heldout-candidates.jsonl"
DEFAULT_OUT_MANIFEST = "tools/autoresearch/generated/heldout-candidates.manifest.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def conversation_content(row: dict[str, Any], role: str) -> str:
    for item in row.get("conversations") or []:
        if item.get("role") == role:
            return str(item.get("content") or "")
    return ""


def parse_assistant(row: dict[str, Any], path: Path, idx: int) -> dict[str, Any]:
    assistant = conversation_content(row, "assistant")
    try:
        parsed = json.loads(assistant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{idx + 1}: assistant content is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}:{idx + 1}: assistant content must decode to an object")
    return parsed


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def unique_codes(items: Any, key: str) -> list[str]:
    seen: set[str] = set()
    codes: list[str] = []
    if not isinstance(items, list):
        return codes
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get(key) or "").strip()
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def protected_eval_state(patterns: list[str]) -> tuple[set[str], set[str], list[str]]:
    case_ids: set[str] = set()
    user_hashes: set[str] = set()
    files: list[str] = []
    for pattern in patterns:
        for path in sorted(Path().glob(pattern)):
            if not path.is_file():
                continue
            files.append(str(path))
            for row in load_jsonl(path):
                case_id = row.get("case_id")
                if case_id:
                    case_ids.add(str(case_id))
                user = row.get("user")
                if isinstance(user, str) and user.strip():
                    user_hashes.add(stable_id(normalize_text(user), 32))
    return case_ids, user_hashes, files


def build_case(
    row: dict[str, Any],
    parsed: dict[str, Any],
    source_path: Path,
    source_index: int,
    prefix: str,
) -> dict[str, Any]:
    user = conversation_content(row, "user")
    if not user.strip():
        raise ValueError(f"{source_path}:{source_index + 1}: missing user content")

    source_key = f"{source_path}:{source_index}:{stable_id(normalize_text(user), 16)}"
    case_id = f"{prefix}_{stable_id(source_key)}"
    expected_conditions = unique_codes(parsed.get("conditions"), "snomed")
    expected_loincs = unique_codes(parsed.get("labs"), "loinc")
    expected_rxnorms = unique_codes(parsed.get("meds"), "rxnorm")

    return {
        "case_id": case_id,
        "description": (
            "Synthetic compact-row held-out candidate; source="
            f"{source_path}:{source_index + 1}"
        ),
        "source": {
            "path": str(source_path),
            "row": source_index + 1,
            "kind": "compact_chat_jsonl",
            "synthetic": True,
            "user_sha256": stable_id(normalize_text(user), 64),
        },
        "user": user,
        "expected_conditions": expected_conditions,
        "expected_loincs": expected_loincs,
        "expected_rxnorms": expected_rxnorms,
    }


def build_corpus(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protected_ids, protected_user_hashes, protected_files = protected_eval_state(
        args.protected_case_glob
    )
    selected: list[dict[str, Any]] = []
    skipped = Counter()
    source_counts: Counter[str] = Counter()
    code_counts = Counter()
    generated_ids: set[str] = set()

    for source in args.source:
        path = Path(source)
        rows = load_jsonl(path)
        for idx, row in enumerate(rows):
            parsed = parse_assistant(row, path, idx)
            user = conversation_content(row, "user")
            user_hash = stable_id(normalize_text(user), 32)
            if user_hash in protected_user_hashes:
                skipped["protected_user_text"] += 1
                continue
            case = build_case(row, parsed, path, idx, args.case_prefix)
            if case["case_id"] in protected_ids:
                skipped["protected_case_id"] += 1
                continue
            if case["case_id"] in generated_ids:
                skipped["duplicate_generated_case_id"] += 1
                continue
            if (
                args.require_codes
                and not case["expected_conditions"]
                and not case["expected_loincs"]
                and not case["expected_rxnorms"]
            ):
                skipped["no_expected_codes"] += 1
                continue
            generated_ids.add(case["case_id"])
            selected.append(case)
            source_counts[str(path)] += 1
            code_counts["conditions"] += len(case["expected_conditions"])
            code_counts["loincs"] += len(case["expected_loincs"])
            code_counts["rxnorms"] += len(case["expected_rxnorms"])
            if args.limit and len(selected) >= args.limit:
                break
        if args.limit and len(selected) >= args.limit:
            break

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/build_heldout_corpus.py",
        "sources": args.source,
        "protected_case_globs": args.protected_case_glob,
        "protected_case_files": protected_files,
        "protected_case_ids_seen": len(protected_ids),
        "case_prefix": args.case_prefix,
        "synthetic": True,
        "claim_boundary": (
            "Rows assembled from compact synthetic train/validation data are "
            "held out from protected regression IDs, but are not independent "
            "real-world eICR cases."
        ),
        "cases": len(selected),
        "source_counts": dict(source_counts),
        "skipped": dict(skipped),
        "expected_codes": {
            "conditions": code_counts["conditions"],
            "loincs": code_counts["loincs"],
            "rxnorms": code_counts["rxnorms"],
            "total": sum(code_counts.values()),
        },
        "case_id_sample": [case["case_id"] for case in selected[:10]],
    }
    return selected, manifest


def write_outputs(cases: list[dict[str, Any]], manifest: dict[str, Any], args: argparse.Namespace) -> None:
    out_jsonl = Path(args.out_jsonl)
    out_manifest = Path(args.out_manifest)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def render_summary(manifest: dict[str, Any], dry_run: bool) -> str:
    mode = "DRY RUN" if dry_run else "WROTE"
    lines = [
        f"{mode}: {manifest['cases']} held-out candidate cases",
        f"Expected codes: {json.dumps(manifest['expected_codes'], sort_keys=True)}",
        f"Protected case IDs checked: {manifest['protected_case_ids_seen']}",
        f"Skipped: {json.dumps(manifest['skipped'], sort_keys=True)}",
        f"Sources: {json.dumps(manifest['source_counts'], sort_keys=True)}",
        f"Case ID sample: {', '.join(manifest['case_id_sample'])}",
        f"Claim boundary: {manifest['claim_boundary']}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Compact chat JSONL source. Repeat for multiple files.",
    )
    parser.add_argument(
        "--protected-case-glob",
        action="append",
        default=[],
        help="Glob for protected regression/eval JSONL files.",
    )
    parser.add_argument("--case-prefix", default="heldout_synth")
    parser.add_argument("--limit", type=int, default=0, help="Maximum cases to emit; 0 means all.")
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-manifest", default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-codes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows without condition, LOINC, or RxNorm gold codes.",
    )
    args = parser.parse_args()
    if not args.source:
        args.source = DEFAULT_SOURCES
    if not args.protected_case_glob:
        args.protected_case_glob = DEFAULT_PROTECTED_GLOBS

    cases, manifest = build_corpus(args)
    if not args.dry_run:
        write_outputs(cases, manifest, args)
    print(render_summary(manifest, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
