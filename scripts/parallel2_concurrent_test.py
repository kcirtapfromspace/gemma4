#!/usr/bin/env python3
"""Test perceived throughput under parallel=2 + 2 concurrent requests.

Runs 2 concurrent requests against the llama-server (needs parallel=2 server
config) and measures:
  - per-request tok/s
  - aggregate requests/min
  - aggregate tok/s across both clients
"""

import concurrent.futures
import json
import sys
import time
import uuid

import requests

ENDPOINT = "http://192.168.150.41:30083"
PROMPT = (
    "Extract clinical entities from this eICR summary. Output JSON with: "
    "patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), "
    "medications (RxNorm), vitals, and a case summary. Output valid JSON only."
)


def one_request(case_user: str, max_tokens: int = 300) -> dict:
    payload = {
        "model": "g",
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": case_user},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t_start = time.perf_counter()
    try:
        r = requests.post(f"{ENDPOINT}/v1/chat/completions", json=payload, timeout=900)
        total = time.perf_counter() - t_start
        r.raise_for_status()
        body = r.json()
        timings = body.get("timings", {})
        usage = body.get("usage", {})
        return {
            "ok": True,
            "total_s": total,
            "tok_s": timings.get("predicted_per_second"),
            "prompt_tok_s": timings.get("prompt_per_second"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "total_s": time.perf_counter() - t_start}


def main():
    # Load cases
    case_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/thinkstudio/gemma4/scripts/test_cases_val3.jsonl"
    cases = []
    with open(case_file) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    # Run 2 concurrent
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(one_request, c["user"]) for c in cases[:2]]
        results = [f.result() for f in futs]
    t_total = time.perf_counter() - t_start

    total_tokens = sum(r.get("completion_tokens") or 0 for r in results if r["ok"])
    per_req_tok_s = [r["tok_s"] for r in results if r.get("tok_s")]

    print(f"Parallel 2-concurrent test")
    print(f"  total wall time:  {t_total:.1f} s")
    print(f"  total out tokens: {total_tokens}")
    print(f"  aggregate tok/s:  {total_tokens/t_total:.2f}")
    for i, r in enumerate(results):
        if r["ok"]:
            print(f"  req {i}: {r['total_s']:.1f}s, {r['tok_s']:.2f} tok/s, {r['completion_tokens']} tokens")
        else:
            print(f"  req {i}: FAILED {r['error']}")
    print(f"  mean per-req tok/s: {sum(per_req_tok_s)/len(per_req_tok_s) if per_req_tok_s else 0:.2f}")


if __name__ == "__main__":
    main()
