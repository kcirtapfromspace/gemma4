"""LiteRT-LM agent runner — same agent contract as agent_pipeline.py but
runs against an on-device .litertlm bundle via litert_lm.Conversation, the
same path the iOS app will eventually call into.

Two purposes:
  1. Verify whether a given .litertlm bundle can actually do tool calling
     in agent mode (the existing C16 v2 fine-tune may have lost the base
     model's tool-call chat template behavior — this script proves it).
  2. Bench the int4 mobile bundle against the same 22-case combined bench
     so we can compare to the Mac llama-server agent (63/63 = 1.000) and
     decide whether we need to rebuild from base Gemma 4.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from regex_preparser import extract as deterministic_extract  # noqa: E402
from regex_preparser import _load_lookup  # noqa: E402

import litert_lm  # noqa: E402


# Tools — plain Python functions; LiteRT-LM derives the schema from the
# signature + docstring via inspect, so type hints + docstrings matter.

def extract_codes_from_text(text: str) -> dict:
    """Extract SNOMED, LOINC, and RxNorm codes from clinical narrative text.

    Runs a deterministic 3-tier extractor: inline parenthesized codes, CDA
    XML attribute parsing, and curated displayName lookup with NegEx negation
    suppression. Returns code arrays plus per-code provenance (source span,
    tier, confidence). Run this FIRST.

    Args:
        text: The full clinical narrative.

    Returns:
        {conditions, loincs, rxnorms, matches}
    """
    return deterministic_extract(text).to_provenance_dict()


def lookup_displayname(name: str, codeset: str) -> dict:
    """Look up a single displayName in the curated SNOMED/LOINC/RxNorm dict.

    Use this when extract_codes_from_text returns empty for a category but the
    narrative clearly mentions an entity in that category.

    Args:
        name: The disease, lab, or drug name.
        codeset: One of "snomed", "loinc", "rxnorm".

    Returns:
        {code, system, matched_alias} or {code: null, system, matched_alias: null}
    """
    table = _load_lookup()
    bucket = {
        "snomed": ("snomed", "SNOMED"),
        "loinc": ("loincs", "LOINC"),
        "rxnorm": ("rxnorms", "RxNorm"),
    }.get(codeset.lower())
    if bucket is None:
        return {"code": None, "system": None, "matched_alias": None,
                "error": f"unknown codeset '{codeset}'; use snomed|loinc|rxnorm"}
    cat, system = bucket
    for code, patterns in table.get(cat, []):
        for alias, p in patterns:
            if p.search(name):
                return {"code": code, "system": system, "matched_alias": alias}
    return {"code": None, "system": system, "matched_alias": None}


def validate_fhir_extraction(extraction: dict) -> dict:
    """Validate the structure of a final extraction object.

    Confirms keys conditions/loincs/rxnorms exist and are arrays. Run this
    LAST before producing the final answer.

    Args:
        extraction: The proposed final extraction dict.

    Returns:
        {valid: bool, issues: [str]}
    """
    issues: list[str] = []
    for key in ("conditions", "loincs", "rxnorms"):
        if key not in extraction:
            issues.append(f"missing key '{key}'")
        elif not isinstance(extraction.get(key), list):
            issues.append(f"'{key}' must be an array")
    return {"valid": not issues, "issues": issues}


SYSTEM = (
    "You are a clinical NLP agent. Given an eICR narrative, produce a JSON "
    "object with three keys: 'conditions' (SNOMED), 'loincs' (LOINC), "
    "'rxnorms' (RxNorm). ALWAYS call extract_codes_from_text first. If it "
    "returns empty arrays for a category but the narrative clearly mentions "
    "an entity in that category, call lookup_displayname for the specific "
    "name. Finally, call validate_fhir_extraction with your final JSON. "
    "Reply with ONLY the validated JSON object as your final message."
)


def run_one(model_path: str, narrative: str, *, verbose: bool = False) -> dict:
    """Run the agent loop for one narrative; return parsed extraction."""
    backend_choice = litert_lm.Backend.CPU
    engine = litert_lm.Engine(model_path=model_path, backend=backend_choice)

    trace: list[dict] = []
    handler_events: list[dict] = []

    class TraceHandler(litert_lm.ToolEventHandler):
        def approve_tool_call(self, tool_call):  # noqa: D401
            handler_events.append({"event": "call", "tool_call": tool_call})
            if verbose:
                print(f"    tool_call: {json.dumps(tool_call, default=str)[:160]}")
            return True

        def process_tool_response(self, tool_response):  # noqa: D401
            handler_events.append({"event": "response", "tool_response": tool_response})
            if verbose:
                print(f"    tool_resp: {json.dumps(tool_response, default=str)[:160]}")
            return tool_response

    try:
        # Conversations are spawned from the Engine — the Engine owns the
        # weight buffers; the Conversation owns the per-session KV state and
        # the chat-template wiring. Pass tools as plain Python callables;
        # LiteRT-LM derives the function-declaration JSON from inspect.
        with engine.create_conversation(
            messages=[{"role": "system", "content": SYSTEM}],
            tools=[
                extract_codes_from_text,
                lookup_displayname,
                validate_fhir_extraction,
            ],
            tool_event_handler=TraceHandler(),
        ) as conv:
            response = conv.send_message({"role": "user", "content": narrative})
    except Exception as exc:  # noqa: BLE001
        return {"error": f"conversation init/send failed: {exc!r}", "trace": handler_events}

    if verbose:
        print(f"    final: {str(response)[:200]}")

    return {
        "raw_response": str(response),
        "tool_events": handler_events,
        "extraction": _parse_final(str(response)),
    }


def _parse_final(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[5:]
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(text[s:e + 1])
            if isinstance(obj, dict):
                return {
                    "conditions": list(obj.get("conditions") or []),
                    "loincs": list(obj.get("loincs") or []),
                    "rxnorms": list(obj.get("rxnorms") or []),
                }
        except json.JSONDecodeError:
            pass
    return {"conditions": [], "loincs": [], "rxnorms": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="apps/mobile/convert/build/litertlm/cliniq-gemma4-e2b.litertlm",
        help="Path to .litertlm bundle",
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        default=[
            "scripts/test_cases.jsonl",
            "scripts/test_cases_adversarial.jsonl",
            "scripts/test_cases_adversarial2.jsonl",
        ],
    )
    ap.add_argument("--max-cases", type=int, default=2,
                    help="Smoke-test cap. Set 0 to run all.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases: list[dict] = []
    for p in args.cases:
        for ln in Path(p).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f"LiteRT-LM agent on {args.model}, {len(cases)} cases\n", flush=True)

    total_m = total_e = perfect = errors = 0
    for idx, case in enumerate(cases, 1):
        cid = case["case_id"]
        t0 = time.time()
        out = run_one(args.model, case["user"], verbose=args.verbose)
        secs = time.time() - t0
        if "error" in out:
            errors += 1
            print(f"  ERR  {idx:2d}/{len(cases)} {cid}: {out['error']} ({secs:.1f}s)")
            continue
        ext = out["extraction"]
        expected = (
            list(case.get("expected_conditions") or [])
            + list(case.get("expected_loincs") or [])
            + list(case.get("expected_rxnorms") or [])
        )
        got = (ext.get("conditions") or []) + (ext.get("loincs") or []) + (ext.get("rxnorms") or [])
        m = sum(1 for c in expected if c in got)
        total_m += m
        total_e += len(expected)
        if m == len(expected) and m > 0:
            perfect += 1
        n_calls = sum(1 for e in out["tool_events"] if e["event"] == "call")
        marker = "OK " if m == len(expected) else "MISS"
        print(f"  {marker} {idx:2d}/{len(cases)} {cid:34s} {m}/{len(expected)} "
              f"({n_calls} tool calls, {secs:.1f}s)")

    print()
    print(f"Aggregate: {total_m}/{total_e} matched, {errors} errors, "
          f"{perfect}/{len(cases)} perfect")


if __name__ == "__main__":
    main()
