#!/usr/bin/env python3
"""Build gold tool-trace SFT rows from labeled ClinIQ cases.

This is complementary to scripts/distill_agent_traces.py. Distillation admits
only traces the current model already got right. This script builds supervised
gold traces from labels, including corrective rows where the extractor returns
plausible but wrong candidates and the assistant must reject them before
validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONVERT_DIR = ROOT / "apps" / "mobile" / "convert"
sys.path.insert(0, str(CONVERT_DIR))

from agent_pipeline import (  # noqa: E402
    DEFAULT_SYSTEM,
    tool_extract_codes_from_text,
    tool_lookup_displayname,
    tool_lookup_reportable_conditions,
    tool_validate_fhir_extraction,
)
from fhir_bundle import normalize_extraction  # noqa: E402


CODE_KEYS = ("conditions", "loincs", "rxnorms")
EXPECTED_FIELDS = {
    "conditions": "expected_conditions",
    "loincs": "expected_loincs",
    "rxnorms": "expected_rxnorms",
}
EXTRACT_TOOL_NAMES = {
    "extract_codes_from_text",
    "extract_codes_from_current_chunk",
}


def compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=False)


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
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def iter_rows(paths: list[Path]) -> list[dict[str, Any]]:
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
        if not isinstance(data, list):
            raise ValueError(f"{path} did not contain a JSON object/list/JSONL rows")
        for row in data:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_cases(paths: list[Path]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    signatures: dict[str, str] = {}
    for row in iter_rows(paths):
        cid = row.get("case_id")
        if not cid:
            continue
        cid = str(cid)
        signature = compact_json({
            "user": row.get("user") or "",
            "expected_conditions": row.get("expected_conditions") or [],
            "expected_loincs": row.get("expected_loincs") or [],
            "expected_rxnorms": row.get("expected_rxnorms") or [],
        })
        if cid in cases and signatures[cid] != signature:
            raise ValueError(f"duplicate case_id {cid!r} has conflicting rows")
        cases[cid] = row
        signatures[cid] = signature
    return cases


def protected_eval_state(paths: list[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    narrative_hashes: set[str] = set()
    for row in iter_rows(paths):
        cid = row.get("case_id")
        if cid:
            ids.add(str(cid))
        user = str(row.get("user") or "")
        if user.strip():
            narrative_hashes.add(normalized_text_hash(user))
    return ids, narrative_hashes


def expected_extraction(case: dict[str, Any]) -> dict[str, list[str]]:
    return normalize_extraction({
        key: list(case.get(field) or [])
        for key, field in EXPECTED_FIELDS.items()
    })


def _code_sets(extraction: dict[str, Any]) -> dict[str, set[str]]:
    normalized = normalize_extraction(extraction)
    return {key: set(normalized.get(key) or []) for key in CODE_KEYS}


def diff_counts(
    candidate: dict[str, Any],
    target: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    candidate_sets = _code_sets(candidate)
    target_sets = _code_sets(target)
    missing = {
        key: len(target_sets[key] - candidate_sets[key])
        for key in CODE_KEYS
    }
    false_pos = {
        key: len(candidate_sets[key] - target_sets[key])
        for key in CODE_KEYS
    }
    return missing, false_pos


def split_bucket(case_id: str, narrative_hash: str, val_fraction: float, seed: str) -> str:
    digest = stable_hash(f"{seed}:{case_id}:{narrative_hash}")
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if score < val_fraction else "train"


def load_candidate_extracts(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = iter_rows([path])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row.get("case_id")
        trace = row.get("trace")
        if not cid or not isinstance(trace, list):
            continue
        for entry in trace:
            if not isinstance(entry, dict):
                continue
            if entry.get("tool_result") in EXTRACT_TOOL_NAMES:
                result = entry.get("result")
                if isinstance(result, dict):
                    out[str(cid)] = result
                break
    return out


def load_lookup_queries() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    lookup_path = CONVERT_DIR / "lookup_table.json"
    reportable_path = CONVERT_DIR / "reportable_conditions.json"
    lookup = json.loads(lookup_path.read_text())
    reportable = json.loads(reportable_path.read_text())

    condition_queries: dict[str, str] = {}
    for row in reportable.get("conditions", []):
        code = str(row.get("code") or "")
        display = str(row.get("display") or "").strip()
        if code and display:
            condition_queries.setdefault(code, display)
    for row in lookup.get("snomed", []):
        aliases = row.get("aliases") or []
        code = str(row.get("code") or "")
        if code and aliases:
            condition_queries.setdefault(code, str(aliases[0]))

    loinc_queries: dict[str, str] = {}
    for row in lookup.get("loincs", []):
        aliases = row.get("aliases") or []
        code = str(row.get("code") or "")
        if code and aliases:
            loinc_queries.setdefault(code, str(aliases[0]))

    rxnorm_queries: dict[str, str] = {}
    for row in lookup.get("rxnorms", []):
        aliases = row.get("aliases") or []
        code = str(row.get("code") or "")
        if code and aliases:
            rxnorm_queries.setdefault(code, str(aliases[0]))

    return condition_queries, loinc_queries, rxnorm_queries


def _assistant_tool_call(
    call_id: str,
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": compact_json(args),
                },
            }
        ],
    }


def _tool_result(call_id: str, name: str, result: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": compact_json(result),
    }


def missing_lookup_calls(
    candidate: dict[str, Any],
    target: dict[str, list[str]],
    lookup_queries: tuple[dict[str, str], dict[str, str], dict[str, str]],
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], list[str]]:
    candidate_sets = _code_sets(candidate)
    cond_queries, loinc_queries, rxnorm_queries = lookup_queries
    calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    missing_query_codes: list[str] = []

    for code in target["conditions"]:
        if code in candidate_sets["conditions"]:
            continue
        query = cond_queries.get(code)
        if not query:
            missing_query_codes.append(f"conditions:{code}")
            continue
        args = {"query": query, "top_k": 3}
        calls.append(("lookup_reportable_conditions", args, tool_lookup_reportable_conditions(args)))

    for code in target["loincs"]:
        if code in candidate_sets["loincs"]:
            continue
        query = loinc_queries.get(code)
        if not query:
            missing_query_codes.append(f"loincs:{code}")
            continue
        args = {"name": query, "codeset": "loinc"}
        calls.append(("lookup_displayname", args, tool_lookup_displayname(args)))

    for code in target["rxnorms"]:
        if code in candidate_sets["rxnorms"]:
            continue
        query = rxnorm_queries.get(code)
        if not query:
            missing_query_codes.append(f"rxnorms:{code}")
            continue
        args = {"name": query, "codeset": "rxnorm"}
        calls.append(("lookup_displayname", args, tool_lookup_displayname(args)))

    return calls, missing_query_codes


def make_conversation(
    case: dict[str, Any],
    target: dict[str, list[str]],
    candidate_result: dict[str, Any],
    lookup_queries: tuple[dict[str, str], dict[str, str], dict[str, str]],
    *,
    preserve_case_system: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    user = str(case.get("user") or "")
    system = str(case.get("system") or DEFAULT_SYSTEM) if preserve_case_system else DEFAULT_SYSTEM
    conversations: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    conversations.append(
        _assistant_tool_call(
            "gold_extract_0",
            "extract_codes_from_text",
            {"text": user},
        )
    )
    conversations.append(_tool_result("gold_extract_0", "extract_codes_from_text", candidate_result))

    lookup_calls, missing_query_codes = missing_lookup_calls(
        candidate_result,
        target,
        lookup_queries,
    )
    for idx, (name, args, result) in enumerate(lookup_calls):
        call_id = f"gold_lookup_{idx}"
        conversations.append(_assistant_tool_call(call_id, name, args))
        conversations.append(_tool_result(call_id, name, result))

    validate_args = {"extraction": target}
    validate_result = tool_validate_fhir_extraction(validate_args)
    conversations.append(
        _assistant_tool_call(
            "gold_validate_0",
            "validate_fhir_extraction",
            validate_args,
        )
    )
    conversations.append(_tool_result("gold_validate_0", "validate_fhir_extraction", validate_result))
    conversations.append({"role": "assistant", "content": compact_json(target)})

    metadata = {
        "lookup_calls": len(lookup_calls),
        "missing_lookup_queries": missing_query_codes,
        "candidate": normalize_extraction(candidate_result),
    }
    return {"conversations": conversations}, metadata


def include_for_mode(
    mode: str,
    case: dict[str, Any],
    candidate: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    if mode == "all":
        return True
    axis = str(case.get("axis") or "")
    if mode == "hard-negatives":
        return axis.startswith("hard_negative")
    if mode == "corrections":
        missing, false_pos = diff_counts(candidate, target)
        return sum(missing.values()) > 0 or sum(false_pos.values()) > 0
    raise ValueError(f"unknown mode {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-files", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--candidate-agent-json",
        type=Path,
        default=None,
        help=(
            "Optional agent bench artifact. If provided, the first recorded "
            "extract_codes_from_text result per case is used as the synthetic "
            "extractor tool result; otherwise the current extractor is run."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="gold-tool-trace")
    parser.add_argument(
        "--mode",
        choices=("all", "corrections", "hard-negatives"),
        default="all",
        help="Which labeled cases to export.",
    )
    parser.add_argument("--eval-case-files", nargs="*", type=Path, default=[])
    parser.add_argument("--allow-eval-cases", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", default="gold-tool-trace-2026-05-02")
    parser.add_argument("--preserve-case-system", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be >=0 and <1")

    cases = load_cases(args.case_files)
    candidate_extracts = load_candidate_extracts(args.candidate_agent_json)
    protected_ids, protected_hashes = (
        protected_eval_state(args.eval_case_files)
        if args.eval_case_files
        else (set(), set())
    )
    lookup_queries = load_lookup_queries()

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    seen_narratives: set[str] = set()
    excluded: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = {
        "axis": Counter(),
        "profile": Counter(),
        "missing": Counter(),
        "false_positive": Counter(),
    }
    target_code_counts: dict[str, Counter[str]] = {
        key: Counter() for key in CODE_KEYS
    }
    candidate_source_counts: Counter[str] = Counter()
    missing_lookup_query_counts: Counter[str] = Counter()

    for cid in sorted(cases):
        case = cases[cid]
        user = str(case.get("user") or "")
        if not user.strip():
            excluded.append({"case_id": cid, "reason": "empty_user"})
            continue
        narrative_hash = normalized_text_hash(user)
        if narrative_hash in seen_narratives:
            excluded.append({"case_id": cid, "reason": "duplicate_normalized_narrative"})
            continue
        if not args.allow_eval_cases and cid in protected_ids:
            excluded.append({"case_id": cid, "reason": "protected_eval_case_id"})
            continue
        if not args.allow_eval_cases and narrative_hash in protected_hashes:
            excluded.append({"case_id": cid, "reason": "protected_eval_narrative"})
            continue

        target = expected_extraction(case)
        candidate_result = candidate_extracts.get(cid)
        candidate_source = "agent_trace" if candidate_result is not None else "current_extractor"
        if candidate_result is None:
            candidate_result = tool_extract_codes_from_text({"text": user})
        if not include_for_mode(args.mode, case, candidate_result, target):
            excluded.append({"case_id": cid, "reason": f"mode_{args.mode}_filtered"})
            continue

        conversation, metadata = make_conversation(
            case,
            target,
            candidate_result,
            lookup_queries,
            preserve_case_system=args.preserve_case_system,
        )
        missing, false_pos = diff_counts(candidate_result, target)
        for key in CODE_KEYS:
            if missing[key]:
                counters["missing"][key] += missing[key]
            if false_pos[key]:
                counters["false_positive"][key] += false_pos[key]
            target_code_counts[key].update(target.get(key) or [])
        for missing_query in metadata["missing_lookup_queries"]:
            missing_lookup_query_counts[missing_query] += 1

        bucket = split_bucket(cid, narrative_hash, args.val_fraction, args.split_seed)
        if bucket == "val":
            val.append(conversation)
        else:
            train.append(conversation)
        seen_narratives.add(narrative_hash)
        candidate_source_counts[candidate_source] += 1
        counters["axis"][str(case.get("axis") or "unknown")] += 1
        counters["profile"][str(case.get("profile") or "unknown")] += 1
        admitted.append({
            "case_id": cid,
            "split": bucket,
            "axis": case.get("axis"),
            "profile": case.get("profile"),
            "candidate_source": candidate_source,
            "missing_from_candidate": missing,
            "false_positive_candidate": false_pos,
            "lookup_calls": metadata["lookup_calls"],
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"{args.prefix}-train.jsonl"
    val_path = args.out_dir / f"{args.prefix}-val.jsonl"
    manifest_path = args.out_dir / f"{args.prefix}-manifest.json"
    train_path.write_text("".join(compact_json(row) + "\n" for row in train))
    val_path.write_text("".join(compact_json(row) + "\n" for row in val))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/build_gold_tool_trace_sft.py",
        "case_files": [str(p) for p in args.case_files],
        "candidate_agent_json": str(args.candidate_agent_json) if args.candidate_agent_json else None,
        "candidate_source_counts": dict(sorted(candidate_source_counts.items())),
        "eval_case_files": [str(p) for p in args.eval_case_files],
        "allow_eval_cases": bool(args.allow_eval_cases),
        "protected_eval_ids_seen": len(protected_ids),
        "protected_eval_narratives_seen": len(protected_hashes),
        "mode": args.mode,
        "preserve_case_system": bool(args.preserve_case_system),
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_admitted": len(admitted),
        "n_excluded": len(excluded),
        "axis_counts": dict(sorted(counters["axis"].items())),
        "profile_counts": dict(sorted(counters["profile"].items())),
        "candidate_missing_counts": dict(sorted(counters["missing"].items())),
        "candidate_false_positive_counts": dict(sorted(counters["false_positive"].items())),
        "missing_lookup_query_counts": dict(sorted(missing_lookup_query_counts.items())),
        "target_code_counts": {
            key: dict(counter.most_common())
            for key, counter in target_code_counts.items()
        },
        "admitted": admitted,
        "excluded": excluded,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(
        f"Wrote {len(train)} train / {len(val)} val gold tool-trace rows "
        f"({len(excluded)} excluded) to {args.out_dir}"
    )
    print(f"Candidate sources: {json.dumps(manifest['candidate_source_counts'], sort_keys=True)}")
    print(f"Candidate missing: {json.dumps(manifest['candidate_missing_counts'], sort_keys=True)}")
    print(f"Candidate false positives: {json.dumps(manifest['candidate_false_positive_counts'], sort_keys=True)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
