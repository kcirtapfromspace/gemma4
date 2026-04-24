#!/usr/bin/env python3
"""Team C4 lightweight spec-decode benchmark.

Sends 1-2 requests per test case, captures server-reported tok/s and
extraction quality. Writes a JSON line per run to stdout and a rolled-up
summary line at the end. Much faster than benchmark.py (no duckdb, no
warmup, no per-test schema overhead).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_CASES = REPO_ROOT / "scripts" / "test_cases_val3.jsonl"
SYS_PROMPT = (
    "Extract clinical entities from this eICR summary. Output JSON with: "
    "patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), "
    "medications (RxNorm), vitals, and a case summary. Output valid JSON only."
)


def strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    while text and text[-1] not in ']}"0123456789trufalsen':
        text = text[:-1]
    return text


def score(content, case):
    text = strip_fences(content)
    try:
        parsed = json.loads(text)
    except Exception:
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0

    expected_keys = {"patient", "conditions", "labs", "meds", "vitals"}
    alt_map = {
        "patient": ["patient_demographics", "demographics"],
        "conditions": ["diagnoses", "diagnosis"],
        "labs": ["laboratory", "lab_results", "results"],
        "meds": ["medications", "medication"],
        "vitals": ["vital_signs"],
    }
    keys_found = []
    for k in expected_keys:
        if k in parsed:
            keys_found.append(k)
        else:
            for alt in alt_map.get(k, []):
                if alt in parsed:
                    keys_found.append(k)
                    break
    schema_score = len(keys_found) / len(expected_keys)

    def collect(code_key, section_key, alt_section_keys=None, alt_code_keys=None):
        found = []
        sections = [section_key] + (alt_section_keys or [])
        for sk in sections:
            for it in parsed.get(sk, []) or []:
                if not isinstance(it, dict):
                    continue
                code = str(it.get(code_key, ""))
                if not code and alt_code_keys:
                    for alt in alt_code_keys:
                        val = str(it.get(alt, ""))
                        if val:
                            parts = val.split()
                            code = parts[-1] if parts else ""
                            break
                if code:
                    found.append(code)
        return found

    conds = collect("snomed", "conditions", ["diagnoses", "diagnosis"], ["code"])
    loincs = collect("loinc", "labs", ["laboratory", "lab_results"], ["code", "test_code"])
    rxs = collect("rxnorm", "meds", ["medications", "medication"], ["code"])

    def bsc(found, expected):
        if not expected:
            return 1.0
        return len(set(found) & set(expected)) / len(expected)

    cond_s = bsc(conds, case.get("expected_conditions", []))
    loinc_s = bsc(loincs, case.get("expected_loincs", []))
    rx_s = bsc(rxs, case.get("expected_rxnorms", []))

    return 0.20 + 0.20 * schema_score + 0.30 * cond_s + 0.15 * loinc_s + 0.15 * rx_s


def infer(endpoint, user, max_tokens):
    payload = {
        "model": "gemma4-eicr-fhir",
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=900)
    t1 = time.perf_counter()
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    timings = body.get("timings", {})
    usage = body.get("usage", {})
    return {
        "content": content,
        "total_ms": (t1 - t0) * 1000,
        "gen_tok_per_sec": timings.get("predicted_per_second"),
        "prompt_tok_per_sec": timings.get("prompt_per_second"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "timings": timings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://192.168.150.41:30083")
    ap.add_argument("--cases", default=str(TEST_CASES))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    cases = []
    with open(args.cases) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    all_tok = []
    all_sc = []
    all_runs = []
    for case in cases:
        for r in range(args.runs):
            try:
                res = infer(args.endpoint, case["user"], args.max_tokens)
                sc = score(res["content"], case)
                tok = res.get("gen_tok_per_sec") or 0.0
                comp = res.get("completion_tokens") or 0
                print(
                    f"[{args.label}] case={case['case_id']} run={r+1} "
                    f"tok/s={tok:.2f} comp={comp} score={sc:.2f} "
                    f"total_ms={res['total_ms']:.0f}",
                    flush=True,
                )
                all_tok.append(tok)
                all_sc.append(sc)
                all_runs.append({"case": case["case_id"], "run": r + 1, "tok_s": tok, "score": sc, "timings": res.get("timings", {})})
            except Exception as e:
                print(f"[{args.label}] case={case['case_id']} run={r+1} ERROR: {e}", flush=True)

    if all_tok:
        avg_tok = sum(all_tok) / len(all_tok)
        avg_sc = sum(all_sc) / len(all_sc)
        print(
            f"[{args.label}] SUMMARY n={len(all_tok)} avg_tok/s={avg_tok:.2f} "
            f"avg_score={avg_sc:.2f}",
            flush=True,
        )
        print("SUMMARY_JSON " + json.dumps({
            "label": args.label,
            "n": len(all_tok),
            "avg_tok_s": avg_tok,
            "avg_score": avg_sc,
            "runs": all_runs,
        }), flush=True)


if __name__ == "__main__":
    main()
