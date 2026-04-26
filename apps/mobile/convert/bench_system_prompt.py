"""System-prompt A/B — Candidate C from the LLM-tuning plan.

The current DEFAULT_SYSTEM in agent_pipeline.py is prescriptive ("Step 1, 2,
3, 4"). This bench runs combined-27 (or any case set) under three variants
to see whether prescription helps, hurts, or is neutral on the long-tail
cases.

Variants:
  A. baseline — current DEFAULT_SYSTEM (prescriptive workflow)
  B. think_first — same workflow + explicit "think aloud about which tool
     to call before calling it" prefix. Tests whether chain-of-thought ahead
     of tool calls reduces malformed-arg retries on adversarial inputs.
  C. terse — strip the prescriptive 5-step workflow, let the model figure
     it out from tool docstrings alone. Tests whether the workflow scaffolding
     is doing real work or just adding tokens.

Win condition: F1 >= baseline AND median latency not worse by >20%. A win
on B suggests we should ship the think-first prompt as default. A win on C
suggests the workflow is ceremony.

Usage:
    scripts/.venv/bin/python apps/mobile/convert/bench_system_prompt.py \
        --variants A B C
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_pipeline import DEFAULT_SYSTEM, run_agent, score  # noqa: E402

PROMPTS: dict[str, str] = {
    "A": DEFAULT_SYSTEM,
    "B": (
        "You are a clinical NLP agent. Given an eICR narrative, produce a JSON "
        "object with three keys: 'conditions' (SNOMED), 'loincs' (LOINC), and "
        "'rxnorms' (RxNorm).\n\n"
        "BEFORE calling any tool, briefly think aloud (one sentence) about "
        "what the narrative contains and which tool will move you forward. "
        "Then call the tool.\n\n"
        "MANDATORY workflow — execute steps in order:\n"
        "1. Call extract_codes_from_text(text) ONCE on the full narrative.\n"
        "2. If 'conditions' is EMPTY in the result AND the narrative mentions "
        "ANY disease name (in any phrasing — formal, colloquial, abbreviation), "
        "you MUST call lookup_reportable_conditions(query=<disease name>). "
        "Take the top result if score >= 0.4 and add its code to conditions.\n"
        "3. Same for 'loincs' (lookup_displayname for the lab name) and "
        "'rxnorms' (drug name).\n"
        "4. Call validate_fhir_extraction once on your final JSON.\n"
        "5. Reply with ONLY the validated JSON object — no extra prose.\n\n"
        "Do NOT call extract_codes_from_text more than once."
    ),
    "C": (
        "You are a clinical NLP agent. Given an eICR narrative, produce a JSON "
        "object with three keys: 'conditions' (SNOMED), 'loincs' (LOINC), and "
        "'rxnorms' (RxNorm). Use the available tools to extract codes from "
        "the narrative; validate your output before replying. Reply with ONLY "
        "the JSON object."
    ),
}


def run_variant(case: dict, *, prompt: str, endpoint: str) -> dict:
    t0 = time.time()
    try:
        ext, trace = run_agent(case["user"], endpoint=endpoint, system=prompt)
    except Exception as exc:  # noqa: BLE001
        return {"case_id": case["case_id"], "error": repr(exc)}
    elapsed = time.time() - t0
    m, e, fp = score(ext, case)
    n_calls = sum(1 for t in trace if t.get("tool_result"))
    n_turns = sum(1 for t in trace if "finish_reason" in t)
    parse_errs = sum(
        1 for t in trace
        if isinstance(t.get("result"), dict)
        and isinstance(t["result"].get("error"), str)
        and "argument JSON decode failed" in t["result"]["error"]
    )
    return {
        "case_id": case["case_id"],
        "matched": m,
        "expected": e,
        "false_positives": fp,
        "elapsed_s": round(elapsed, 2),
        "n_tool_calls": n_calls,
        "n_turns": n_turns,
        "parse_errors": parse_errs,
        "extraction": ext,
    }


def aggregate(rows: list[dict], variant: str) -> dict:
    rs = [r for r in rows if r.get("variant") == variant and "matched" in r]
    if not rs:
        return {"variant": variant, "n_cases": 0}
    total_m = sum(r["matched"] for r in rs)
    total_e = sum(r["expected"] for r in rs)
    total_fp = sum(r["false_positives"] for r in rs)
    perfect = sum(1 for r in rs if r["expected"] and r["matched"] == r["expected"] and r["false_positives"] == 0)
    extracted = total_m + total_fp
    precision = total_m / extracted if extracted else 0.0
    recall = total_m / total_e if total_e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "variant": variant,
        "n_cases": len(rs),
        "perfect": perfect,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positives": total_fp,
        "median_elapsed_s": round(statistics.median(r["elapsed_s"] for r in rs), 2),
        "median_turns": round(statistics.median(r["n_turns"] for r in rs), 2),
        "parse_errors": sum(r["parse_errors"] for r in rs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090")
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
    ap.add_argument("--variants", nargs="+", default=list(PROMPTS.keys()))
    ap.add_argument("--out-json", default="apps/mobile/convert/build/system_prompt_bench.json")
    args = ap.parse_args()

    cases: list[dict] = []
    for p in args.cases:
        for ln in Path(p).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    print(f"System-prompt A/B: {len(cases)} cases × {len(args.variants)} variants = {len(cases) * len(args.variants)} runs", flush=True)

    rows: list[dict] = []
    for variant in args.variants:
        prompt = PROMPTS[variant]
        print(f"\n--- variant {variant} ({len(prompt)} chars)", flush=True)
        for idx, case in enumerate(cases, 1):
            r = run_variant(case, prompt=prompt, endpoint=args.endpoint)
            r["variant"] = variant
            rows.append(r)
            if "error" in r:
                print(f"  ERR {idx}/{len(cases)} {r['case_id']}: {r['error']}", flush=True)
            else:
                tag = "OK " if (r["matched"] == r["expected"] and r["false_positives"] == 0) else "MISS"
                print(f"  {tag} {idx:2d}/{len(cases)} {r['case_id']:38s} {r['matched']}/{r['expected']} fp={r['false_positives']}  [{r['elapsed_s']:.1f}s, {r['n_tool_calls']} calls]", flush=True)

    summary = [aggregate(rows, v) for v in args.variants]

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n" + "=" * 92)
    print(f"{'variant':>8} {'F1':>6} {'prec':>6} {'recall':>7} {'perfect':>9} {'fp':>4} {'med_s':>7} {'turns':>6} {'parse_err':>10}")
    print("=" * 92)
    for s in summary:
        if s["n_cases"] == 0:
            continue
        print(f"{s['variant']:>8} {s['f1']:>6.3f} {s['precision']:>6.3f} {s['recall']:>7.3f} {s['perfect']:>4d}/{s['n_cases']:<4d} {s['false_positives']:>4d} {s['median_elapsed_s']:>6.2f}s {s['median_turns']:>6.1f} {s['parse_errors']:>10d}")
    print(f"\nTrace: {out_path}")


if __name__ == "__main__":
    main()
