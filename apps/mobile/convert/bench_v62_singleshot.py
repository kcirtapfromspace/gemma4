"""Single-shot bench of v62 Unsloth LoRA on val-compact.jsonl.

Sends each (system, user) pair to a local llama-server (default 127.0.0.1:8091),
parses the assistant's compact-JSON output, and scores it against the gold
assistant JSON on:
  * code-level F1 (union of (system, code) tuples across conditions / labs /
    meds / vitals)
  * JSON validity rate
  * schema completeness (all 6 top-level keys present)
  * latency p50 / p95

Comparison harness: pass --compare to run twice -- v62 (port 8091) and base
(port 8090) -- and emit both blocks into one JSON file.

Usage:
  python bench_v62_singleshot.py \
      --val /tmp/eicr-fhir-data/val-compact.jsonl \
      --out apps/mobile/convert/build/v62_val_compact_bench.json \
      --compare
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


# Top-level keys the prompt promises.
SCHEMA_KEYS = {"patient", "encounter", "conditions", "labs", "meds", "vitals"}

# Code systems we score on. (system_label, key_in_record)
CODE_SYSTEMS_LIST = [
    ("snomed", "snomed"),
    ("icd10", "icd10"),
    ("loinc", "loinc"),
    ("rxnorm", "rxnorm"),
]


def chat_completion(
    base_url: str,
    system: str,
    user: str,
    timeout: int = 180,
    max_tokens: int = 1024,
    grammar: str | None = None,
) -> tuple[str, float]:
    """POST to /v1/chat/completions and return (content, elapsed_seconds)."""
    payload: dict[str, Any] = {
        "model": "default",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if grammar:
        payload["grammar"] = grammar
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    elapsed = time.perf_counter() - t0
    obj = json.loads(raw)
    msg = obj["choices"][0]["message"]
    content = msg.get("content") or ""
    # Some servers (default Gemma chat parser) split thinking into
    # `reasoning_content` and leave `content` empty. Fall back so we still
    # score whatever the model actually produced.
    if not content.strip():
        content = msg.get("reasoning_content") or ""
    return content, elapsed


def try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to extract a JSON object from a model response."""
    if not text:
        return None
    s = text.strip()
    # Strip code fences if present.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Direct parse.
    try:
        return json.loads(s)
    except Exception:
        pass
    # Find the first balanced { ... } block.
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[start : i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    return None
    return None


def extract_code_tuples(record: dict[str, Any] | None) -> set[tuple[str, str]]:
    """Return the set of (system, code) tuples for conditions / labs / meds.

    Vitals are unit-less measurements so they don't contribute coded tuples;
    they're scored separately (presence-only) but rolled into the master F1
    so the score reflects the full schema, not just codes.
    """
    out: set[tuple[str, str]] = set()
    if not isinstance(record, dict):
        return out
    for top in ("conditions", "labs", "meds"):
        items = record.get(top) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for system_label, key in CODE_SYSTEMS_LIST:
                v = item.get(key)
                if v is None or v == "":
                    continue
                out.add((system_label, str(v)))
    # Vitals: include each present numeric field as its own pseudo-code so
    # missing vitals are penalised. (system="vital", code=field_name)
    vitals = record.get("vitals") or {}
    if isinstance(vitals, dict):
        for k, v in vitals.items():
            if v is None or v == "":
                continue
            out.add(("vital", str(k)))
    return out


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_case(gold_text: str, pred_text: str) -> dict[str, Any]:
    gold = try_parse_json(gold_text) or {}
    pred = try_parse_json(pred_text)
    json_valid = pred is not None
    schema_complete = (
        json_valid
        and isinstance(pred, dict)
        and SCHEMA_KEYS.issubset(set(pred.keys()))
    )
    gold_codes = extract_code_tuples(gold)
    pred_codes = extract_code_tuples(pred or {})
    tp = len(gold_codes & pred_codes)
    fp = len(pred_codes - gold_codes)
    fn = len(gold_codes - pred_codes)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1(p, r),
        "json_valid": json_valid,
        "schema_complete": bool(schema_complete),
        "gold_n": len(gold_codes),
        "pred_n": len(pred_codes),
    }


def aggregate(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_case)
    if n == 0:
        return {}
    tp = sum(c["score"]["tp"] for c in per_case)
    fp = sum(c["score"]["fp"] for c in per_case)
    fn = sum(c["score"]["fn"] for c in per_case)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = f1(micro_p, micro_r)
    macro_p = statistics.mean(c["score"]["precision"] for c in per_case)
    macro_r = statistics.mean(c["score"]["recall"] for c in per_case)
    macro_f1 = statistics.mean(c["score"]["f1"] for c in per_case)
    json_valid_rate = sum(1 for c in per_case if c["score"]["json_valid"]) / n
    schema_complete_rate = (
        sum(1 for c in per_case if c["score"]["schema_complete"]) / n
    )
    latencies = sorted(c["latency_s"] for c in per_case if c.get("latency_s"))

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        k = max(0, min(len(latencies) - 1, int(round(p * (len(latencies) - 1)))))
        return latencies[k]

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "json_valid_rate": round(json_valid_rate, 4),
        "schema_complete_rate": round(schema_complete_rate, 4),
        "latency_p50_s": round(pct(0.50), 3),
        "latency_p95_s": round(pct(0.95), 3),
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
    }


def run_bench(
    base_url: str,
    label: str,
    val_path: Path,
    limit: int | None,
    max_tokens: int,
    grammar: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    print(f"\n=== {label} @ {base_url} ===", flush=True)
    with val_path.open() as fh:
        lines = [ln for ln in fh if ln.strip()]
    if limit:
        lines = lines[:limit]
    for idx, ln in enumerate(lines):
        rec = json.loads(ln)
        convs = rec["conversations"]
        sys_msg = next(c["content"] for c in convs if c["role"] == "system")
        user_msg = next(c["content"] for c in convs if c["role"] == "user")
        gold_msg = next(c["content"] for c in convs if c["role"] == "assistant")
        try:
            content, elapsed = chat_completion(
                base_url,
                sys_msg,
                user_msg,
                max_tokens=max_tokens,
                grammar=grammar,
            )
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            print(f"  [{idx:3d}] ERROR: {exc}", flush=True)
            rows.append(
                {
                    "idx": idx,
                    "error": str(exc),
                    "latency_s": None,
                    "score": {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "json_valid": False,
                        "schema_complete": False,
                        "gold_n": 0,
                        "pred_n": 0,
                    },
                }
            )
            continue
        score = score_case(gold_msg, content)
        rows.append(
            {
                "idx": idx,
                "latency_s": round(elapsed, 3),
                "score": score,
                "pred_preview": content[:200],
            }
        )
        if idx % 10 == 0 or idx + 1 == len(lines):
            print(
                f"  [{idx:3d}/{len(lines)}] f1={score['f1']:.3f} "
                f"json={score['json_valid']} t={elapsed:.2f}s",
                flush=True,
            )
    agg = aggregate(rows)
    print(f"  -> micro_f1={agg.get('micro_f1')} json={agg.get('json_valid_rate')}")
    return {"label": label, "base_url": base_url, "aggregate": agg, "cases": rows}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="/tmp/eicr-fhir-data/val-compact.jsonl")
    p.add_argument(
        "--out",
        default="apps/mobile/convert/build/v62_val_compact_bench.json",
    )
    p.add_argument("--v62-url", default="http://127.0.0.1:8091")
    p.add_argument("--base-url", default="http://127.0.0.1:8090")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--compare",
        action="store_true",
        help="Also run base model (8090) for delta",
    )
    p.add_argument("--label", default="v62_lora")
    p.add_argument(
        "--grammar-file",
        default=None,
        help="Path to GBNF grammar file applied to v62 requests",
    )
    args = p.parse_args()

    val_path = Path(args.val)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grammar: str | None = None
    if args.grammar_file:
        grammar = Path(args.grammar_file).read_text()

    result: dict[str, Any] = {
        "val_path": str(val_path),
        "limit": args.limit,
        "max_tokens": args.max_tokens,
        "grammar_file": args.grammar_file,
        "ts": int(time.time()),
    }
    result["v62"] = run_bench(
        args.v62_url,
        args.label,
        val_path,
        args.limit,
        args.max_tokens,
        grammar=grammar,
    )
    if args.compare:
        result["base"] = run_bench(
            args.base_url,
            "base_q3km",
            val_path,
            args.limit,
            args.max_tokens,
        )
        # Delta block.
        v = result["v62"]["aggregate"]
        b = result["base"]["aggregate"]
        result["delta"] = {
            "f1": round(v.get("micro_f1", 0) - b.get("micro_f1", 0), 4),
            "precision": round(
                v.get("micro_precision", 0) - b.get("micro_precision", 0), 4
            ),
            "recall": round(v.get("micro_recall", 0) - b.get("micro_recall", 0), 4),
            "json_valid": round(
                v.get("json_valid_rate", 0) - b.get("json_valid_rate", 0), 4
            ),
        }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(result.get("v62", {}).get("aggregate", {}), indent=2))
    if args.compare:
        print("base:", json.dumps(result["base"]["aggregate"], indent=2))
        print("delta:", json.dumps(result["delta"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
