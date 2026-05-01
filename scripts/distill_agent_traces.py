#!/usr/bin/env python3
"""Distill verified agent-bench traces into single-shot SFT rows.

Input:
  * agent_pipeline.py --out-json rows, each with case_id, extraction, matched,
    expected, false_positives, trace.
  * one or more source case JSONL files carrying case_id + user narrative.

Output:
  * Gemma chat JSONL rows: {"conversations": [system, user, assistant]}.
  * A split manifest documenting admitted/excluded rows and contamination
    controls.

The default target is intentionally smaller than the legacy compact dataset:
the single-shot student learns the final agent/RAG answer only, not tool calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SINGLE_SHOT_SYSTEM = (
    "Extract reportable clinical codes from this eICR narrative. Return ONLY "
    "minified JSON with exactly three keys: "
    '{"conditions":[],"loincs":[],"rxnorms":[]}. '
    "conditions are SNOMED codes, loincs are LOINC codes, rxnorms are RxNorm "
    "codes. Include asserted positives only. No markdown. No prose."
)

CODE_KEYS = ("conditions", "loincs", "rxnorms")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip()).lower()
    return stable_hash(normalized)


def load_json_or_jsonl(path: Path) -> Any:
    raw = path.read_text()
    stripped = raw.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return [json.loads(ln) for ln in raw.splitlines() if ln.strip()]


def load_cases(paths: list[Path]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_json_or_jsonl(path):
            cid = row.get("case_id")
            if not cid:
                continue
            cases[cid] = row
    return cases


def load_agent_rows(path: Path) -> list[dict[str, Any]]:
    data = load_json_or_jsonl(path)
    if isinstance(data, dict):
        if "rows" in data:
            data = data["rows"]
        elif "cases" in data:
            data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(f"{path} did not contain a JSON list of rows")
    return data


def normalize_codes(extraction: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in CODE_KEYS:
        values = extraction.get(key) or []
        if not isinstance(values, list):
            values = []
        clean = sorted({str(v).strip() for v in values if str(v).strip()})
        out[key] = clean
    return out


def make_conversation(user: str, target: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "conversations": [
            {"role": "system", "content": SINGLE_SHOT_SYSTEM},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": json.dumps(target, separators=(",", ":")),
            },
        ]
    }


def split_bucket(case_id: str, narrative_hash: str, val_fraction: float, seed: str) -> str:
    digest = stable_hash(f"{seed}:{case_id}:{narrative_hash}")
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if score < val_fraction else "train"


def is_perfect_agent_row(row: dict[str, Any]) -> bool:
    if "extraction" not in row:
        return False
    matched = row.get("matched")
    expected = row.get("expected")
    false_positives = row.get("false_positives")
    if matched is None or expected is None or false_positives is None:
        return False
    return int(matched) == int(expected) and int(false_positives) == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-json", required=True, type=Path)
    ap.add_argument("--case-files", required=True, nargs="+", type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--eval-case-files",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Protected eval JSONL files. Matching case_id rows are excluded "
            "from train/val unless --allow-eval-cases is set."
        ),
    )
    ap.add_argument("--allow-eval-cases", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--split-seed", default="v63-trace-distill-2026-04-30")
    ap.add_argument(
        "--prefix",
        default="trace-distill",
        help="Output filename prefix inside --out-dir.",
    )
    ap.add_argument(
        "--include-imperfect",
        action="store_true",
        help="Include non-perfect agent rows. Default keeps only exact bench passes.",
    )
    args = ap.parse_args()

    if not (0.0 <= args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be >=0 and <1")

    cases = load_cases(args.case_files)
    protected_eval_ids = set(load_cases(args.eval_case_files)) if args.eval_case_files else set()
    rows = load_agent_rows(args.agent_json)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "source_agent_json": str(args.agent_json),
        "case_files": [str(p) for p in args.case_files],
        "eval_case_files": [str(p) for p in args.eval_case_files],
        "allow_eval_cases": bool(args.allow_eval_cases),
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "system_prompt_sha256": stable_hash(SINGLE_SHOT_SYSTEM),
        "admitted": [],
        "excluded": [],
    }
    seen_narratives: set[str] = set()

    for row in rows:
        cid = row.get("case_id")
        reason = None
        if not cid:
            reason = "missing_case_id"
        elif cid not in cases:
            reason = "missing_source_case"
        elif not args.include_imperfect and not is_perfect_agent_row(row):
            reason = "not_perfect_agent_bench_row"
        elif not args.allow_eval_cases and cid in protected_eval_ids:
            reason = "protected_eval_case_id"

        if reason:
            manifest["excluded"].append({"case_id": cid, "reason": reason})
            continue

        case = cases[cid]
        user = str(case.get("user") or "")
        if not user.strip():
            manifest["excluded"].append({"case_id": cid, "reason": "empty_user"})
            continue

        narrative_hash = normalized_text_hash(user)
        if narrative_hash in seen_narratives:
            manifest["excluded"].append(
                {"case_id": cid, "reason": "duplicate_normalized_narrative"}
            )
            continue
        seen_narratives.add(narrative_hash)

        target = normalize_codes(row["extraction"])
        bucket = split_bucket(cid, narrative_hash, args.val_fraction, args.split_seed)
        item = make_conversation(user, target)
        (val if bucket == "val" else train).append(item)
        manifest["admitted"].append(
            {
                "case_id": cid,
                "split": bucket,
                "narrative_sha256": narrative_hash,
                "target": target,
                "path": row.get("path"),
                "n_tool_calls": row.get("n_tool_calls"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"{args.prefix}-train.jsonl"
    val_path = args.out_dir / f"{args.prefix}-val.jsonl"
    manifest_path = args.out_dir / f"{args.prefix}-manifest.json"

    with train_path.open("w") as f:
        for item in train:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    with val_path.open("w") as f:
        for item in val:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")

    manifest["n_train"] = len(train)
    manifest["n_val"] = len(val)
    manifest["n_excluded"] = len(manifest["excluded"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"wrote {len(train)} train rows -> {train_path}")
    print(f"wrote {len(val)} val rows -> {val_path}")
    print(f"wrote manifest -> {manifest_path}")
    if not train and not val:
        print("warning: no rows admitted; check eval exclusions and verification filters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
