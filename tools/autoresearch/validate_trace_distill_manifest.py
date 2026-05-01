#!/usr/bin/env python3
"""Validate trace-distillation manifests before training.

The distiller already excludes protected eval IDs when invoked correctly. This
helper is a fail-closed gate that verifies the emitted manifest and row files
still satisfy the contamination and shape contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CODE_KEYS = ("conditions", "loincs", "rxnorms")


def normalized_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_json_or_jsonl(path: Path) -> Any:
    raw = path.read_text()
    stripped = raw.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_case_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        rows = load_json_or_jsonl(path)
        if isinstance(rows, dict):
            rows = rows.get("rows") or rows.get("cases") or [rows]
        for row in rows:
            case_id = row.get("case_id")
            if case_id:
                ids.add(str(case_id))
    return ids


def row_user_text(row: dict[str, Any]) -> str | None:
    user = row.get("user")
    if isinstance(user, str) and user.strip():
        return user
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if turn.get("role") == "user" and isinstance(turn.get("content"), str):
                return turn["content"]
    return None


def load_narrative_hashes(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        rows = load_json_or_jsonl(path)
        if isinstance(rows, dict):
            rows = rows.get("rows") or rows.get("cases") or [rows]
        for row in rows:
            text = row_user_text(row)
            if text:
                hashes.add(normalized_text_hash(text))
    return hashes


def infer_row_paths(manifest: Path) -> tuple[Path, Path]:
    name = manifest.name
    if not name.endswith("-manifest.json"):
        raise ValueError(
            "cannot infer train/val paths; pass --train-jsonl and --val-jsonl"
        )
    prefix = name[: -len("-manifest.json")]
    return (
        manifest.with_name(f"{prefix}-train.jsonl"),
        manifest.with_name(f"{prefix}-val.jsonl"),
    )


def read_jsonl_count(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def validate_target(target: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(target, dict):
        return ["target is not an object"]
    if set(target) != set(CODE_KEYS):
        errors.append(f"target keys are {sorted(target)}, expected {list(CODE_KEYS)}")
    for key in CODE_KEYS:
        value = target.get(key)
        if not isinstance(value, list):
            errors.append(f"target.{key} is not a list")
        elif any(not isinstance(item, str) or not item.strip() for item in value):
            errors.append(f"target.{key} contains non-string or empty code")
    return errors


def validate_row_file(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            conversations = row.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 3:
                errors.append(f"{path}:{line_no}: expected 3 conversations")
                continue
            roles = [turn.get("role") for turn in conversations]
            if roles != ["system", "user", "assistant"]:
                errors.append(f"{path}:{line_no}: roles are {roles}")
            assistant = conversations[2].get("content")
            if not isinstance(assistant, str):
                errors.append(f"{path}:{line_no}: assistant content is not a string")
                continue
            try:
                target = json.loads(assistant)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: assistant JSON invalid: {exc}")
                continue
            errors.extend(f"{path}:{line_no}: {err}" for err in validate_target(target))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--val-jsonl", type=Path)
    parser.add_argument("--eval-case-files", nargs="*", type=Path, default=[])
    parser.add_argument("--min-admitted", type=int, default=0)
    parser.add_argument("--require-val", action="store_true")
    parser.add_argument(
        "--allow-eval-cases",
        action="store_true",
        help="Permit admitted protected eval IDs. Never use for training gates.",
    )
    args = parser.parse_args()

    train_path, val_path = (
        (args.train_jsonl, args.val_jsonl)
        if args.train_jsonl and args.val_jsonl
        else infer_row_paths(args.manifest)
    )

    manifest = json.loads(args.manifest.read_text())
    admitted = manifest.get("admitted")
    excluded = manifest.get("excluded")
    errors: list[str] = []

    if not isinstance(admitted, list):
        errors.append("manifest.admitted is not a list")
        admitted = []
    if not isinstance(excluded, list):
        errors.append("manifest.excluded is not a list")

    if manifest.get("allow_eval_cases") and not args.allow_eval_cases:
        errors.append("manifest was generated with allow_eval_cases=true")

    if len(admitted) < args.min_admitted:
        errors.append(f"admitted rows {len(admitted)} < required {args.min_admitted}")

    protected_ids = load_case_ids(args.eval_case_files) if args.eval_case_files else set()
    protected_hashes = (
        load_narrative_hashes(args.eval_case_files) if args.eval_case_files else set()
    )
    admitted_ids = [str(row.get("case_id")) for row in admitted if row.get("case_id")]
    protected_admitted = sorted(set(admitted_ids) & protected_ids)
    if protected_admitted and not args.allow_eval_cases:
        sample = ", ".join(protected_admitted[:10])
        errors.append(f"protected eval case_ids admitted: {sample}")

    admitted_hashes = {
        str(row.get("narrative_sha256"))
        for row in admitted
        if row.get("narrative_sha256")
    }
    protected_narratives_admitted = sorted(admitted_hashes & protected_hashes)
    if protected_narratives_admitted and not args.allow_eval_cases:
        sample = ", ".join(protected_narratives_admitted[:5])
        errors.append(f"protected eval narratives admitted: {sample}")

    duplicate_ids = sorted(cid for cid, count in Counter(admitted_ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate admitted case_ids: {', '.join(duplicate_ids[:10])}")

    hashes = [
        str(row.get("narrative_sha256"))
        for row in admitted
        if row.get("narrative_sha256")
    ]
    duplicate_hashes = sorted(h for h, count in Counter(hashes).items() if count > 1)
    if duplicate_hashes:
        errors.append(f"duplicate narrative hashes: {', '.join(duplicate_hashes[:10])}")

    split_counts = Counter(str(row.get("split")) for row in admitted)
    if set(split_counts) - {"train", "val"}:
        errors.append(f"unexpected splits: {sorted(set(split_counts) - {'train', 'val'})}")
    if args.require_val and split_counts.get("val", 0) == 0:
        errors.append("validation split is empty")

    for idx, row in enumerate(admitted):
        if row.get("split") not in {"train", "val"}:
            errors.append(f"admitted[{idx}] has invalid split {row.get('split')!r}")
        if not row.get("case_id"):
            errors.append(f"admitted[{idx}] missing case_id")
        if not row.get("narrative_sha256"):
            errors.append(f"admitted[{idx}] missing narrative_sha256")
        errors.extend(f"admitted[{idx}]: {err}" for err in validate_target(row.get("target")))

    if not train_path.exists():
        errors.append(f"missing train JSONL: {train_path}")
        n_train = None
    else:
        n_train = read_jsonl_count(train_path)
        errors.extend(validate_row_file(train_path))

    if not val_path.exists():
        errors.append(f"missing val JSONL: {val_path}")
        n_val = None
    else:
        n_val = read_jsonl_count(val_path)
        errors.extend(validate_row_file(val_path))

    if n_train is not None and n_train != split_counts.get("train", 0):
        errors.append(
            f"train rows {n_train} != manifest train split {split_counts.get('train', 0)}"
        )
    if n_val is not None and n_val != split_counts.get("val", 0):
        errors.append(f"val rows {n_val} != manifest val split {split_counts.get('val', 0)}")
    if manifest.get("n_train") != split_counts.get("train", 0):
        errors.append("manifest.n_train does not match admitted train split")
    if manifest.get("n_val") != split_counts.get("val", 0):
        errors.append("manifest.n_val does not match admitted val split")

    summary = {
        "manifest": str(args.manifest),
        "admitted": len(admitted),
        "train": split_counts.get("train", 0),
        "val": split_counts.get("val", 0),
        "excluded": len(excluded) if isinstance(excluded, list) else None,
        "protected_eval_ids_loaded": len(protected_ids),
        "protected_eval_admitted": len(protected_admitted),
        "protected_eval_narratives_loaded": len(protected_hashes),
        "protected_eval_narratives_admitted": len(protected_narratives_admitted),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("manifest gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
