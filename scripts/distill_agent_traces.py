#!/usr/bin/env python3
"""Distill verified agent-bench traces into SFT rows.

Input:
  * agent_pipeline.py --out-json rows, each with case_id, extraction, matched,
    expected, false_positives, trace.
  * one or more source case JSONL files carrying case_id + user narrative.

Output:
  * Gemma chat JSONL rows: {"conversations": [...]}.
  * A split manifest documenting admitted/excluded rows and contamination
    controls.

The default target is intentionally smaller than the legacy compact dataset:
the single-shot student learns the final agent/RAG answer only, not tool calls.
Pass --export-format tool-trace to export assistant tool calls and tool results
from agent traces for tool-use SFT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
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

TOOL_TRACE_SYSTEM = (
    "You are a clinical NLP agent. Given an eICR narrative, produce a JSON "
    "object with three keys: 'conditions' (SNOMED), 'loincs' (LOINC), and "
    "'rxnorms' (RxNorm).\n\n"
    "RAW CODE CONTRACT: every array value must be the raw code string only. "
    "Never include display names or prose in code arrays. Use tools to ground "
    "candidate codes, validate the final extraction, then reply with ONLY the "
    "validated minified JSON object."
)


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


def _case_signature(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "user": row.get("user"),
            "expected_conditions": row.get("expected_conditions") or [],
            "expected_loincs": row.get("expected_loincs") or [],
            "expected_rxnorms": row.get("expected_rxnorms") or [],
        },
        sort_keys=True,
    )


def iter_case_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = load_json_or_jsonl(path)
        if isinstance(data, dict):
            if "rows" in data:
                data = data["rows"]
            elif "cases" in data:
                data = data["cases"]
            else:
                data = [data]
        for row in data:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_cases(paths: list[Path]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    signatures: dict[str, str] = {}
    for row in iter_case_rows(paths):
        cid = row.get("case_id")
        if not cid:
            continue
        cid = str(cid)
        signature = _case_signature(row)
        if cid in cases and signatures[cid] != signature:
            raise ValueError(
                f"duplicate case_id {cid!r} has conflicting source rows"
            )
        cases[cid] = row
        signatures[cid] = signature
    return cases


def protected_eval_state(paths: list[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    narrative_hashes: set[str] = set()
    for row in iter_case_rows(paths):
        cid = row.get("case_id")
        if cid:
            ids.add(str(cid))
        user = str(row.get("user") or "")
        if user.strip():
            narrative_hashes.add(normalized_text_hash(user))
    return ids, narrative_hashes


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


def _is_fhir_id_safe(value: str) -> bool:
    return bool(value) and all(ch.isalnum() or ch in "-." for ch in value)


def normalize_code_value(bucket: str, value: object) -> str | None:
    """Conservatively reduce display-bearing values to raw code strings."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None

    if bucket == "conditions":
        if text.isdigit():
            return text
        match = re.search(
            r"\b(?:SNOMED(?:\s+CT)?\s*)?(\d{5,18})\b",
            text,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    if bucket == "loincs":
        match = re.search(r"\b\d{1,7}-\d\b", text)
        if match:
            return match.group(0)
        if _is_fhir_id_safe(text) and any(ch.isdigit() for ch in text):
            return text
        return None

    if bucket == "rxnorms":
        if text.isdigit():
            return text
        match = re.search(r"\b(?:RXNORM\s*)?(\d{3,18})\b", text, re.IGNORECASE)
        return match.group(1) if match else None

    return text if _is_fhir_id_safe(text) else None


def normalize_codes(extraction: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in CODE_KEYS:
        values = extraction.get(key) or []
        if not isinstance(values, list):
            values = []
        clean = sorted(
            {
                code
                for raw in values
                if (code := normalize_code_value(key, raw)) is not None
            }
        )
        out[key] = clean
    return out


def compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=False)


def make_conversation(user: str, target: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "conversations": [
            {"role": "system", "content": SINGLE_SHOT_SYSTEM},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": compact_json(target),
            },
        ]
    }


def _call_arguments(call: dict[str, Any]) -> str:
    raw = (call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, str):
        try:
            return compact_json(json.loads(raw))
        except json.JSONDecodeError:
            return raw
    return compact_json(raw)


def _tool_call_id(call: dict[str, Any], fallback: str) -> str:
    cid = call.get("id")
    return str(cid) if cid else fallback


def _tool_call_name(call: dict[str, Any]) -> str | None:
    name = (call.get("function") or {}).get("name")
    return str(name) if name else None


def _assistant_has_tool_calls(entry: dict[str, Any]) -> bool:
    calls = entry.get("tool_calls")
    return isinstance(calls, list) and bool(calls)


def make_tool_trace_conversation(
    case: dict[str, Any],
    row: dict[str, Any],
    target: dict[str, list[str]],
) -> dict[str, Any] | None:
    trace = row.get("trace")
    if not isinstance(trace, list) or not trace:
        return None

    user = str(case.get("user") or "")
    system = str(case.get("system") or TOOL_TRACE_SYSTEM)
    conversations: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    pending_tool_calls: list[tuple[str, str]] = []
    synthetic_tool_idx = 0
    emitted_tool_call = False

    for entry in trace:
        if not isinstance(entry, dict):
            continue

        tool_name = entry.get("tool_result")
        if tool_name:
            if pending_tool_calls:
                tool_call_id, pending_name = pending_tool_calls.pop(0)
                name = pending_name or str(tool_name)
            else:
                tool_call_id = (
                    "forced_extract_0"
                    if entry.get("forced")
                    else f"synthetic_tool_call_{synthetic_tool_idx}"
                )
                synthetic_tool_idx += 1
                name = str(tool_name)
                conversations.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": compact_json(entry.get("args") or {}),
                                },
                            }
                        ],
                    }
                )
                emitted_tool_call = True
            conversations.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": compact_json(entry.get("result")),
                }
            )
            continue

        if _assistant_has_tool_calls(entry):
            calls_out: list[dict[str, Any]] = []
            for idx, call in enumerate(entry.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fallback_id = f"trace_turn_{entry.get('turn', 'x')}_call_{idx}"
                call_id = _tool_call_id(call, fallback_id)
                name = _tool_call_name(call)
                if not name:
                    continue
                calls_out.append(
                    {
                        "id": call_id,
                        "type": call.get("type") or "function",
                        "function": {
                            "name": name,
                            "arguments": _call_arguments(call),
                        },
                    }
                )
                pending_tool_calls.append((call_id, name))
            if calls_out:
                conversations.append(
                    {
                        "role": "assistant",
                        "content": entry.get("content") or "",
                        "tool_calls": calls_out,
                    }
                )
                emitted_tool_call = True

    if not emitted_tool_call:
        return None
    conversations.append({"role": "assistant", "content": compact_json(target)})
    return {"conversations": conversations}


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


def is_validated_agent_row(row: dict[str, Any]) -> bool:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return False
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        if entry.get("tool_result") != "validate_fhir_extraction":
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        issues = result.get("issues") or []
        if result.get("valid") is True and not issues:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--agent-json",
        required=True,
        type=Path,
        help="Agent bench JSON/JSONL artifact containing rows with case_id, extraction, metrics, and optional trace.",
    )
    ap.add_argument(
        "--case-files",
        required=True,
        nargs="+",
        type=Path,
        help="Source case JSON/JSONL files carrying case_id and user narrative.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where train/val JSONL files and manifest JSON are written.",
    )
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
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Deterministic validation split fraction. Must be >=0 and <1.",
    )
    ap.add_argument(
        "--split-seed",
        default="v63-trace-distill-2026-04-30",
        help="Seed string used for deterministic case_id/narrative-hash splitting.",
    )
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
    ap.add_argument(
        "--require-validated",
        action="store_true",
        help=(
            "Admit only rows whose trace includes validate_fhir_extraction with "
            "valid=true and no issues. Useful for tool-trace SFT exports."
        ),
    )
    ap.add_argument(
        "--export-format",
        choices=("single-shot", "tool-trace"),
        default="single-shot",
        help=(
            "single-shot preserves the legacy final-answer SFT rows. tool-trace "
            "exports assistant tool calls, tool results, and a normalized final target."
        ),
    )
    args = ap.parse_args()

    if not (0.0 <= args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be >=0 and <1")

    cases = load_cases(args.case_files)
    protected_eval_ids, protected_eval_narrative_hashes = (
        protected_eval_state(args.eval_case_files)
        if args.eval_case_files
        else (set(), set())
    )
    rows = load_agent_rows(args.agent_json)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "source_agent_json": str(args.agent_json),
        "case_files": [str(p) for p in args.case_files],
        "eval_case_files": [str(p) for p in args.eval_case_files],
        "allow_eval_cases": bool(args.allow_eval_cases),
        "protected_eval_ids_seen": len(protected_eval_ids),
        "protected_eval_narratives_seen": len(protected_eval_narrative_hashes),
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "export_format": args.export_format,
        "include_imperfect": bool(args.include_imperfect),
        "require_validated": bool(args.require_validated),
        "target_normalization": "raw_code_sorted",
        "system_prompt_sha256": stable_hash(
            SINGLE_SHOT_SYSTEM
            if args.export_format == "single-shot"
            else TOOL_TRACE_SYSTEM
        ),
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
        elif args.require_validated and not is_validated_agent_row(row):
            reason = "not_validated_agent_trace"
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
        if (
            not args.allow_eval_cases
            and narrative_hash in protected_eval_narrative_hashes
        ):
            manifest["excluded"].append(
                {"case_id": cid, "reason": "protected_eval_narrative"}
            )
            continue
        seen_narratives.add(narrative_hash)

        target = normalize_codes(row["extraction"])
        bucket = split_bucket(cid, narrative_hash, args.val_fraction, args.split_seed)
        if args.export_format == "tool-trace":
            item = make_tool_trace_conversation(case, row, target)
            if item is None:
                manifest["excluded"].append(
                    {"case_id": cid, "reason": "missing_tool_trace"}
                )
                continue
        else:
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
                "validated": is_validated_agent_row(row),
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
    manifest["excluded_reasons"] = {
        reason: count
        for reason, count in sorted(
            Counter(
                item.get("reason", "unknown")
                for item in manifest["excluded"]
            ).items()
        )
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"wrote {len(train)} train rows -> {train_path}")
    print(f"wrote {len(val)} val rows -> {val_path}")
    print(f"wrote manifest -> {manifest_path}")
    if not train and not val:
        print("warning: no rows admitted; check eval exclusions and verification filters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
