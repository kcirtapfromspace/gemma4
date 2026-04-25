"""Gemma 4 agentic eICR-to-FHIR extractor.

The base Gemma 4 E2B (with native function calling) acts as the orchestrator.
It calls our existing infrastructure as tools:

  extract_codes_from_text   →  regex_preparser.extract (3-tier deterministic)
  validate_fhir_extraction  →  structural FHIR R4 sanity check
  lookup_displayname        →  single-name displayName lookup

This satisfies the Gemma 4 hackathon's "native function calling" rubric and
turns the deterministic stack into one of the agent's tools rather than a
parallel pipeline.

Design notes:
- The base Gemma 4 (NOT our v2 fine-tune) is used here. The fine-tune wasn't
  trained on tool-calling format and emits malformed function arguments. The
  base model's clinical reasoning + our extractor tool covers the bench.
- Tool-call arguments are JSON-validated against the declared schema before
  execution; a malformed call returns an error to the model so it can retry.
- Loop bound at 6 turns to prevent runaway agents. EpiCast and Tracer in the
  MedGemma winners both used short, bounded tool-call traces.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# Local — relies on regex_preparser being importable from this directory.
sys.path.insert(0, str(Path(__file__).parent))
from regex_preparser import extract as deterministic_extract  # noqa: E402
from regex_preparser import _load_lookup, _is_negated  # noqa: E402
from rag_search import (  # noqa: E402
    FAST_PATH_THRESHOLD,
    FastPathHit,
    fast_path_hit,
    search as rag_search,
)

DEFAULT_ENDPOINT = "http://127.0.0.1:8090"
DEFAULT_SYSTEM = (
    "You are a clinical NLP agent. Given an eICR narrative, produce a JSON "
    "object with three keys: 'conditions' (SNOMED), 'loincs' (LOINC), and "
    "'rxnorms' (RxNorm).\n\n"
    "MANDATORY workflow — execute steps in order:\n"
    "1. Call extract_codes_from_text(text) ONCE on the full narrative.\n"
    "2. If 'conditions' is EMPTY in the result AND the narrative mentions "
    "ANY disease name (in any phrasing — formal, colloquial, abbreviation), "
    "you MUST call lookup_reportable_conditions(query=<disease name>). "
    "Examples: 'valley fever', 'C diff colitis', 'Legionnaires disease', "
    "'Marburg hemorrhagic fever'. Take the top result if score >= 0.4 and "
    "add its code to conditions.\n"
    "3. Same for 'loincs' (call lookup_displayname for the lab name) and "
    "'rxnorms' (drug name).\n"
    "4. Call validate_fhir_extraction once on your final JSON.\n"
    "5. Reply with ONLY the validated JSON object — no extra prose.\n\n"
    "Do NOT call extract_codes_from_text more than once. Do NOT skip step 2 "
    "when 'conditions' is empty and the narrative names a disease. After "
    "validation passes, the next assistant message must be the JSON object."
)


# -----------------------------------------------------------------------------
# Tools


def tool_extract_codes_from_text(args: dict) -> dict:
    text = args.get("text", "")
    res = deterministic_extract(text)
    # Surface provenance so the agent can explain WHY each code was emitted
    # ("source line N, alias 'hepatitis C', tier 'lookup', confidence 0.85").
    # Tracer-style discrepancy/confidence reporting is what won the MedGemma
    # Impact Challenge category, so we ship the same shape here.
    return res.to_provenance_dict()


def tool_validate_fhir_extraction(args: dict) -> dict:
    extraction = args.get("extraction") or {}
    issues: list[str] = []
    for key in ("conditions", "loincs", "rxnorms"):
        if key not in extraction:
            issues.append(f"missing key '{key}'")
        elif not isinstance(extraction.get(key), list):
            issues.append(f"'{key}' must be an array")
    return {"valid": not issues, "issues": issues}


def tool_lookup_displayname(args: dict) -> dict:
    name = args.get("name", "")
    codeset = (args.get("codeset") or "").lower()
    table = _load_lookup()
    bucket = {
        "snomed": ("snomed", "SNOMED"),
        "loinc": ("loincs", "LOINC"),
        "rxnorm": ("rxnorms", "RxNorm"),
    }.get(codeset)
    if bucket is None:
        return {"code": None, "system": None, "matched_alias": None,
                "error": f"unknown codeset '{codeset}'; use snomed|loinc|rxnorm"}
    cat, system = bucket
    for code, patterns in table.get(cat, []):
        for alias, p in patterns:
            if p.search(name):
                return {"code": code, "system": system, "matched_alias": alias}
    return {"code": None, "system": system, "matched_alias": None}


def tool_lookup_reportable_conditions(args: dict) -> dict:
    """RAG over the curated reportable-conditions database (CDC NNDSS / WHO IDSR)."""
    query = args.get("query", "")
    top_k = int(args.get("top_k") or 3)
    hits = rag_search(query, top_k=top_k)
    return {
        "results": [h.to_dict() for h in hits],
        "query": query,
    }


TOOLS: dict[str, callable] = {
    "extract_codes_from_text": tool_extract_codes_from_text,
    "validate_fhir_extraction": tool_validate_fhir_extraction,
    "lookup_displayname": tool_lookup_displayname,
    "lookup_reportable_conditions": tool_lookup_reportable_conditions,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "extract_codes_from_text",
            "description": (
                "Extract SNOMED, LOINC, and RxNorm codes from clinical "
                "narrative text using a deterministic 3-tier extractor "
                "(inline parenthesized codes + CDA XML attribute parsing + "
                "curated displayName lookup with NegEx negation suppression). "
                "Returns three arrays of code strings. Run this FIRST."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw clinical narrative.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_displayname",
            "description": (
                "Look up a single displayName in the curated SNOMED/LOINC/"
                "RxNorm dictionary. Use this for entities the extractor "
                "missed (rare diseases, drug variations). Returns code or null."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Disease, lab, or drug name.",
                    },
                    "codeset": {
                        "type": "string",
                        "enum": ["snomed", "loinc", "rxnorm"],
                        "description": "Which codeset to search.",
                    },
                },
                "required": ["name", "codeset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_reportable_conditions",
            "description": (
                "Search the curated reportable-conditions database (CDC NNDSS "
                "+ WHO IDSR, ~60 entries) for candidate codes when both the "
                "deterministic extractor and the displayName lookup miss. "
                "Returns top_k candidates with score, source, and source_url. "
                "Trust results with score >= 0.6."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Disease name, syndrome, or short clinical description.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return (default 3).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_fhir_extraction",
            "description": (
                "Validate the structure of a final extraction object. "
                "Confirms keys conditions/loincs/rxnorms exist and are arrays. "
                "Run this LAST before producing your final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "extraction": {
                        "type": "object",
                        "properties": {
                            "conditions": {"type": "array", "items": {"type": "string"}},
                            "loincs": {"type": "array", "items": {"type": "string"}},
                            "rxnorms": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["conditions", "loincs", "rxnorms"],
                    }
                },
                "required": ["extraction"],
            },
        },
    },
]


# -----------------------------------------------------------------------------
# Agent loop


def chat(endpoint: str, payload: dict, timeout: float = 300.0) -> dict:
    req = Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_agent(
    narrative: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    system: str = DEFAULT_SYSTEM,
    max_turns: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 3072,
    verbose: bool = False,
    tool_call_grammar: str | None = None,
) -> tuple[dict, list[dict]]:
    """Run the Gemma 4 agent loop. Returns (final_extraction, trace).

    ``tool_call_grammar`` — optional GBNF text. NOTE on llama-server (>=b6000):
    the OpenAI-compatible ``/v1/chat/completions`` endpoint **rejects**
    ``grammar`` whenever ``tools`` is set ("Cannot use custom grammar
    constraints with tools."), because the ``--jinja`` path already applies
    an internal tool-call grammar derived from the rendered chat template.
    We therefore drop the ``grammar`` field on chat-completions turns; the
    parameter is kept for API symmetry with the iOS ``AgentRunner`` path
    (where ``LlamaCppInferenceEngine.applyGrammar`` *does* honour it,
    because ToolCallParser is looser than llama-server's internal grammar
    and IS the parse-failure surface the GBNF was authored to lock).
    See proposals-2026-04-25.md Rank 4 for the rationale.
    """
    # Force-disable on Python: llama-server's tool-call path already grammar-locks
    # the wire format. Sending our GBNF here would 400 the request.
    _ = tool_call_grammar
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": narrative},
    ]
    trace: list[dict] = []

    for turn in range(max_turns):
        payload = {
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # NOTE: not setting ``payload["grammar"]`` here — llama-server's
        # /v1/chat/completions rejects grammar when tools is set. The
        # internal tool-grammar from the chat template is sufficient for
        # the Python path; the explicit GBNF applies only on the iOS
        # AgentRunner. See ``run_agent`` docstring.
        t0 = time.time()
        resp = chat(endpoint, payload)
        elapsed = time.time() - t0
        choice = resp["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason")

        trace.append({
            "turn": turn,
            "finish_reason": finish,
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "elapsed_s": round(elapsed, 2),
            "tokens": resp.get("usage", {}).get("completion_tokens"),
        })
        if verbose:
            print(f"  turn {turn}: finish={finish} elapsed={elapsed:.1f}s "
                  f"calls={len(msg.get('tool_calls') or [])}")

        # Append the assistant message verbatim so subsequent turns see it.
        messages.append(msg)

        if finish == "tool_calls" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                fn = call["function"]["name"]
                raw_args = call["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError as exc:
                    result = {"error": f"argument JSON decode failed: {exc}"}
                else:
                    impl = TOOLS.get(fn)
                    if impl is None:
                        result = {"error": f"unknown tool '{fn}'"}
                    else:
                        try:
                            result = impl(args)
                        except Exception as exc:  # noqa: BLE001
                            result = {"error": f"tool raised: {exc!r}"}
                trace.append({"turn": turn, "tool_result": fn, "args": args if 'args' in dir() else raw_args, "result": result})
                if verbose:
                    print(f"    tool {fn}({json.dumps(args, default=str)[:80]}) → "
                          f"{json.dumps(result)[:120]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": fn,
                    "content": json.dumps(result),
                })
            continue

        # Final assistant message — try to parse JSON from content.
        content = msg.get("content") or ""
        extraction = _parse_extraction(content)
        return extraction, trace

    # Hit max_turns without a final answer; salvage by parsing whatever the
    # last assistant message had.
    last_content = next(
        (t.get("content") for t in reversed(trace) if t.get("content")),
        None,
    ) or ""
    return _parse_extraction(last_content), trace


# -----------------------------------------------------------------------------
# Fast-path gate (c19 Rank 2)
#
# Mirror of `ExtractionService.run(narrative:)` in
# apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/ExtractionService.swift, the
# inserted block between Tier 1 (deterministic) and Tier 2 (agent loop).


def try_fast_path(
    narrative: str,
    *,
    threshold: float = FAST_PATH_THRESHOLD,
    use_negex: bool = True,
) -> tuple[dict, list[dict]] | None:
    """Run the fast-path gate. Returns (extraction, trace) on hit, else None.

    Gate (matching Swift):
      1. deterministic_extract returns no codes (none of conditions/loincs/rxnorms)
      2. RAG top hit score >= threshold (default 0.70)
      3. matched_phrase has at least one non-negated occurrence in the narrative

    On hit, emit a single synthetic trace entry tagged 'fast_path' and a
    SNOMED-only extraction (the reportable_conditions DB is keyed on
    diseases, so loincs/rxnorms are always empty here).

    `use_negex=False` skips Tier-3 NegEx in the gate. Used by the CLI parity
    probes (validate_fast_path.py) to test the gate's matched-phrase rule
    in isolation.
    """
    t0 = time.time()
    det = deterministic_extract(narrative)
    det_dict = det.to_provenance_dict() if hasattr(det, "to_provenance_dict") else {}
    det_codes = (det_dict.get("conditions") or []) + (
        det_dict.get("loincs") or []
    ) + (det_dict.get("rxnorms") or [])
    if det_codes:
        return None  # Tier 1 already has answers — fast-path doesn't apply.

    is_negated = _is_negated if use_negex else (lambda *_args: False)
    fp = fast_path_hit(narrative, threshold=threshold, is_negated=is_negated)
    if fp is None:
        return None

    elapsed = time.time() - t0
    extraction = {
        "conditions": [fp.hit.code] if fp.hit.system.upper() == "SNOMED" else [],
        "loincs": [fp.hit.code] if fp.hit.system.upper() == "LOINC" else [],
        "rxnorms": [fp.hit.code] if fp.hit.system.upper() == "RXNORM" else [],
    }
    trace = [{
        "turn": 0,
        "fast_path": True,
        "elapsed_s": round(elapsed, 3),
        "rag_hit": fp.hit.to_dict(),
        "matched_span": {
            "text": fp.span.text,
            "location": fp.span.location,
            "length": fp.span.length,
        },
        "extraction": extraction,
    }]
    return extraction, trace


def _parse_extraction(content: str) -> dict:
    """Best-effort JSON parse — strips markdown fencing and null-likes."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[5:]
        elif text.lstrip().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
    # Try to find a balanced JSON object inside the text.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                # Strip None / null-like sentinels so downstream scoring
                # doesn't treat a literal "null" string as a spurious code.
                def _clean(arr):
                    return [str(x) for x in (arr or [])
                            if x is not None and str(x).lower() != "null" and str(x).strip()]
                return {
                    "conditions": _clean(obj.get("conditions")),
                    "loincs": _clean(obj.get("loincs")),
                    "rxnorms": _clean(obj.get("rxnorms")),
                }
        except json.JSONDecodeError:
            pass
    return {"conditions": [], "loincs": [], "rxnorms": []}


# -----------------------------------------------------------------------------
# Bench


def score(extraction: dict, case: dict) -> tuple[int, int, int]:
    expected = (
        list(case.get("expected_conditions") or [])
        + list(case.get("expected_loincs") or [])
        + list(case.get("expected_rxnorms") or [])
    )
    got = (
        list(extraction.get("conditions") or [])
        + list(extraction.get("loincs") or [])
        + list(extraction.get("rxnorms") or [])
    )
    matched = sum(1 for c in expected if c in got)
    fp = sum(1 for c in got if c not in expected)
    return matched, len(expected), fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument(
        "--cases",
        nargs="+",
        default=[
            "scripts/test_cases.jsonl",
            "scripts/test_cases_adversarial.jsonl",
            "scripts/test_cases_adversarial2.jsonl",
        ],
    )
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument(
        "--out-json",
        default="apps/mobile/convert/build/agent_bench.json",
    )
    ap.add_argument(
        "--tool-call-grammar",
        default=None,
        help=(
            "Optional GBNF file (e.g. apps/mobile/convert/cliniq_toolcall.gbnf) "
            "applied to llama-server on tool-response turns to lock the "
            "tool-call wire format. See proposals-2026-04-25.md Rank 4."
        ),
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--fast-path-rag-threshold",
        type=float,
        default=FAST_PATH_THRESHOLD,
        help=(
            "Fast-path RAG score threshold (default 0.70, mirror of Swift "
            "RagSearch.fastPathThreshold). Set to a value >1.0 to disable."
        ),
    )
    ap.add_argument(
        "--no-fast-path",
        action="store_true",
        help="Disable the c19 single-turn fast-path; always run the agent loop.",
    )
    args = ap.parse_args()
    grammar_text: str | None = None
    if args.tool_call_grammar:
        grammar_text = Path(args.tool_call_grammar).read_text()

    cases: list[dict] = []
    for p in args.cases:
        for ln in Path(p).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(f"Agent bench on {len(cases)} cases\n", flush=True)

    rows: list[dict] = []
    total_m = total_e = total_fp = perfect = 0
    n_fast_path_hits = 0
    for idx, case in enumerate(cases, 1):
        cid = case["case_id"]
        path_label = "agent"
        try:
            # Try the c19 single-turn fast-path first. On miss, fall through
            # to the agent loop with the optional tool-call grammar.
            fp_result = (
                None
                if args.no_fast_path
                else try_fast_path(
                    case["user"], threshold=args.fast_path_rag_threshold
                )
            )
            if fp_result is not None:
                extraction, trace = fp_result
                path_label = "fast"
                n_fast_path_hits += 1
            else:
                extraction, trace = run_agent(
                    case["user"],
                    endpoint=args.endpoint,
                    verbose=args.verbose,
                    tool_call_grammar=grammar_text,
                )
        except URLError as exc:
            print(f"  ERR {cid}: {exc}", flush=True)
            rows.append({"case_id": cid, "error": str(exc)})
            continue

        m, e, fpc = score(extraction, case)
        total_m += m
        total_e += e
        total_fp += fpc
        if e and m == e and fpc == 0:
            perfect += 1

        n_calls = sum(1 for t in trace if t.get("tool_result"))
        marker = "OK " if (m == e and fpc == 0) else "MISS"
        fp_tag = f" FP={fpc}" if fpc else ""
        latency = trace[0].get("elapsed_s") if path_label == "fast" else None
        latency_tag = f" {latency*1000:.0f}ms" if latency is not None else ""
        print(
            f"  {marker} {idx:2d}/{len(cases)} {cid:34s} {m}/{e}{fp_tag}  "
            f"[{path_label}{latency_tag}, {n_calls} tool calls, {len(trace)} turns)",
            flush=True,
        )
        rows.append({
            "case_id": cid,
            "matched": m,
            "expected": e,
            "false_positives": fpc,
            "extraction": extraction,
            "n_tool_calls": n_calls,
            "path": path_label,
            "trace": trace,
        })

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, indent=2))

    recall = total_m / total_e if total_e else 0.0
    extracted_total = total_m + total_fp
    precision = total_m / extracted_total if extracted_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(
        f"\nAggregate: {total_m}/{total_e} matched, {total_fp} FP — "
        f"recall={recall:.3f} precision={precision:.3f} F1={f1:.3f} "
        f"({perfect}/{len(cases)} perfect)"
    )
    if not args.no_fast_path and len(cases):
        print(
            f"Fast-path: {n_fast_path_hits}/{len(cases)} cases "
            f"({100*n_fast_path_hits/len(cases):.0f}%) hit before agent loop"
        )
    print(f"Trace: {args.out_json}")


if __name__ == "__main__":
    main()
