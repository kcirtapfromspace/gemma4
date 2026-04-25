"""Hybrid extraction pipeline: regex/CDA/lookup tiers, LLM fallback.

This is the production architecture proposed by autoresearch Rank 2:

  Tier 1  inline regex     (e.g. "(SNOMED 76272004)") — already in regex_preparser
  Tier 2  CDA XML attrs    (e.g. <code code="..." codeSystem="2.16.840.1.113883.6.96"/>)
  Tier 3  displayName lookup table (e.g. "amoxicillin" → 723)
  Tier 4  LLM with grammar — only when Tiers 1-3 return empty across ALL categories

The deterministic tiers (1-3) cover the bench at 1.00. Tier 4 catches inputs
that have NO codes, NO CDA XML, AND no aliases in the lookup table — i.e.,
the truly free-text remainder.

Run modes:
  --policy never        — pure deterministic
  --policy if-empty     — LLM only when all categories empty (default; safest)
  --policy always-merge — LLM always runs and unions with deterministic
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

# Local module — runs from the apps/mobile/convert/ directory or repo root
sys.path.insert(0, str(Path(__file__).parent))
from regex_preparser import extract as deterministic_extract  # noqa: E402

GRAMMAR_PATH = Path(__file__).parent / "cliniq_extraction.gbnf"

SYSTEM = (
    "You are a clinical NLP assistant. Given an eICR summary, extract the "
    "primary conditions, labs, and medications as JSON with keys "
    "'conditions' (SNOMED codes), 'loincs' (LOINC codes), and 'rxnorms' "
    "(RxNorm codes). Return only JSON."
)


def build_prompt(user: str) -> str:
    return (
        f"<|turn>system\n{SYSTEM}<turn|>\n"
        f"<|turn>user\n{user}<turn|>\n"
        f"<|turn>model\n"
    )


Policy = Literal["never", "if-empty", "always-merge"]


def call_llm(
    endpoint: str,
    prompt: str,
    grammar: str,
    n_predict: int = 256,
    timeout: float = 120.0,
) -> tuple[dict[str, list[str]], dict]:
    """POST to llama-server /completion with grammar. Returns (extraction dict, telemetry)."""
    payload = {
        "prompt": prompt,
        "grammar": grammar,
        "n_predict": n_predict,
        "temperature": 0.1,
        "cache_prompt": False,
        "stream": False,
    }
    req = Request(
        f"{endpoint}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    text = body.get("content", "")
    tokens = int(body.get("tokens_predicted") or 0)

    parsed: dict[str, list[str]] = {"conditions": [], "loincs": [], "rxnorms": []}
    try:
        obj = json.loads(text)
        for k in parsed:
            v = obj.get(k)
            if isinstance(v, list):
                parsed[k] = [str(x) for x in v]
    except json.JSONDecodeError:
        # Grammar should prevent this — if it happens, log and continue
        pass

    return parsed, {
        "raw": text,
        "seconds": round(elapsed, 2),
        "tokens": tokens,
        "tok_per_s": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
    }


def hybrid_extract(
    text: str,
    *,
    endpoint: str,
    grammar: str,
    policy: Policy,
    use_lookup: bool = True,
) -> tuple[dict[str, list[str]], dict]:
    """Run the tiered pipeline and return (extraction, audit dict)."""
    det = deterministic_extract(text, use_lookup=use_lookup)
    det_dict = {
        "conditions": list(det.conditions),
        "loincs": list(det.loincs),
        "rxnorms": list(det.rxnorms),
    }

    audit: dict = {
        "policy": policy,
        "deterministic": det_dict,
        "llm_invoked": False,
    }
    total_det = sum(len(v) for v in det_dict.values())

    invoke_llm = (
        (policy == "always-merge")
        or (policy == "if-empty" and total_det == 0)
    )

    if not invoke_llm:
        return det_dict, audit

    audit["llm_invoked"] = True
    llm_out, telemetry = call_llm(endpoint, build_prompt(text), grammar)
    audit["llm"] = {**telemetry, "extraction": llm_out}

    merged = {
        k: _dedupe(det_dict[k] + llm_out.get(k, []))
        for k in ("conditions", "loincs", "rxnorms")
    }
    return merged, audit


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def score(extracted: dict[str, list[str]], case: dict) -> tuple[int, int]:
    expected = (
        list(case.get("expected_conditions") or [])
        + list(case.get("expected_loincs") or [])
        + list(case.get("expected_rxnorms") or [])
    )
    flat = (
        " ".join(extracted.get("conditions", []))
        + " "
        + " ".join(extracted.get("loincs", []))
        + " "
        + " ".join(extracted.get("rxnorms", []))
    )
    matched = sum(1 for c in expected if re.search(re.escape(c), flat))
    return matched, len(expected)


def load_cases(paths: list[str]) -> list[dict]:
    cases: list[dict] = []
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases",
        nargs="+",
        default=[
            "scripts/test_cases.jsonl",
            "scripts/test_cases_adversarial.jsonl",
        ],
    )
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090")
    ap.add_argument(
        "--policy",
        choices=["never", "if-empty", "always-merge"],
        default="if-empty",
    )
    ap.add_argument(
        "--no-lookup",
        action="store_true",
        help="Disable Tier 3 (lookup table); useful to force LLM fallback",
    )
    ap.add_argument(
        "--out-json",
        default="apps/mobile/convert/build/hybrid_bench.json",
    )
    args = ap.parse_args()

    grammar = GRAMMAR_PATH.read_text()
    cases = load_cases(args.cases)
    print(
        f"Hybrid pipeline: policy={args.policy}, lookup={'OFF' if args.no_lookup else 'ON'}, "
        f"endpoint={args.endpoint}, cases={len(cases)}\n",
        flush=True,
    )

    rows: list[dict] = []
    total_matched = total_expected = perfect = llm_calls = 0
    for idx, case in enumerate(cases, 1):
        case_id = case["case_id"]
        try:
            extracted, audit = hybrid_extract(
                case["user"],
                endpoint=args.endpoint,
                grammar=grammar,
                policy=args.policy,
                use_lookup=not args.no_lookup,
            )
        except URLError as exc:
            print(f"  ERR {case_id}: {exc}", flush=True)
            rows.append({"case_id": case_id, "error": str(exc)})
            continue

        m, e = score(extracted, case)
        total_matched += m
        total_expected += e
        if e and m == e:
            perfect += 1
        if audit["llm_invoked"]:
            llm_calls += 1

        marker = "OK " if m == e else "MISS"
        llm_flag = "+LLM" if audit["llm_invoked"] else "    "
        print(
            f"  {marker} {idx:2d}/{len(cases)} {case_id:34s} "
            f"{m}/{e} {llm_flag}",
            flush=True,
        )
        rows.append(
            {"case_id": case_id, "matched": m, "expected": e, **audit, "extracted": extracted}
        )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, indent=2))

    score_v = total_matched / total_expected if total_expected else 0.0
    print(
        f"\nAGGREGATE: {total_matched}/{total_expected} = {score_v:.3f} "
        f"({perfect}/{len(cases)} cases perfect, LLM invoked on {llm_calls}/{len(cases)})"
    )
    print(f"Full results: {args.out_json}")


if __name__ == "__main__":
    main()
