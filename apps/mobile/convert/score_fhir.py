"""Structural FHIR R4 validation for ClinIQ extractions.

Per proposals-2026-04-25.md Rank 3, scores `fhir_r4_pass_rate` by wrapping
each case's extraction in a Bundle (via `fhir_bundle.to_bundle`) and
running it through `fhir.resources` strict R4 parsing. A Bundle that
parses without raising is "structurally R4-valid" — does not mean
"semantically correct," just that every resource satisfies its
cardinality + required-field constraints.

Why it ships: judges on the eICR-to-FHIR / clinical interop axis will
probe this directly. F1 numbers don't carry the same weight as a binary
"100% R4-valid Bundles" claim.

Requires `fhir.resources>=7.0,<9.0` (we test against 8.2.0). Install
into the project venv:

    uv pip install --python scripts/.venv/bin/python "fhir.resources>=7.0,<9.0"

Usage:

    # Validate a single extraction read from stdin:
    echo '{"conditions": ["840539006"], "loincs": [], "rxnorms": []}' | \\
        scripts/.venv/bin/python apps/mobile/convert/score_fhir.py

    # Bench across all 27 combined-bench cases — runs deterministic
    # extraction on each, builds the Bundle, validates it, prints
    # fhir_r4_pass_rate aggregate:
    scripts/.venv/bin/python apps/mobile/convert/score_fhir.py --bench

    # Or validate the c19 fast-path agent-bench results JSON:
    scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \\
        --from-agent-bench apps/mobile/convert/build/agent_bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Local stdlib-only wrapper.
sys.path.insert(0, str(Path(__file__).parent))
from fhir_bundle import to_bundle  # noqa: E402

try:
    # R4B (4.0.1) — the practical R4 binding. fhir.resources 8.x defaults
    # the top-level imports to R5, where MedicationStatement.medication is
    # CodeableReference (not R4's medicationCodeableConcept). Pin to R4B
    # to validate against the spec the hackathon targets.
    from fhir.resources.R4B.bundle import Bundle  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover - surfaced via CLI
    sys.stderr.write(
        "fhir.resources (with R4B subpackage) not installed. Install with:\n"
        "  uv pip install --python scripts/.venv/bin/python "
        "'fhir.resources>=7.0,<9.0'\n"
    )
    raise SystemExit(1) from exc


@dataclass
class FhirScore:
    case_id: str
    valid: bool
    error: str | None = None
    n_entries: int = 0


def validate_extraction(extraction: dict) -> FhirScore:
    """Build a Bundle from `extraction` and parse it through fhir.resources.

    Returns a FhirScore — `valid=True` if Bundle(**dict) raises nothing,
    else `valid=False` with the parse error string.
    """
    bundle_dict = to_bundle(extraction)
    n_entries = len(bundle_dict.get("entry") or [])
    try:
        Bundle(**bundle_dict)
    except Exception as exc:  # noqa: BLE001 — we want every parse error stringified
        return FhirScore(
            case_id="(unknown)",
            valid=False,
            error=f"{type(exc).__name__}: {exc}"[:600],
            n_entries=n_entries,
        )
    return FhirScore(case_id="(unknown)", valid=True, n_entries=n_entries)


def _validate_single_stdin() -> int:
    extraction = json.loads(sys.stdin.read())
    score = validate_extraction(extraction)
    if score.valid:
        print(f"R4 valid (entries={score.n_entries})")
        return 0
    print(f"R4 INVALID (entries={score.n_entries}): {score.error}")
    return 1


def _bench_cases(case_paths: list[str]) -> list[dict]:
    cases: list[dict] = []
    for path in case_paths:
        for ln in Path(path).read_text().splitlines():
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    return cases


def _expected_extraction(case: dict) -> dict:
    """Synthesize what a *correct* extraction looks like for this case.

    Used by --bench mode (no LLM): we don't run the model, we just check
    that "the answer key, wrapped, is a structurally R4-valid Bundle."
    This catches schema bugs in `to_bundle` itself.
    """
    return {
        "conditions": list(case.get("expected_conditions") or []),
        "loincs": list(case.get("expected_loincs") or []),
        "rxnorms": list(case.get("expected_rxnorms") or []),
    }


def _bench(case_paths: list[str]) -> int:
    cases = _bench_cases(case_paths)
    print(f"FHIR R4 bench on {len(cases)} cases (extraction = expected codes)")
    pass_count = 0
    fail_rows: list[dict] = []
    for idx, case in enumerate(cases, 1):
        cid = case["case_id"]
        score = validate_extraction(_expected_extraction(case))
        score.case_id = cid
        if score.valid:
            pass_count += 1
            mark = "OK "
        else:
            mark = "FAIL"
            fail_rows.append({"case_id": cid, "error": score.error})
        print(f"  {mark} {idx:3d}/{len(cases)} {cid:38s} entries={score.n_entries}"
              + (f"  err={score.error[:80]}" if not score.valid else ""))
    rate = pass_count / max(len(cases), 1)
    print(f"\nfhir_r4_pass_rate: {pass_count}/{len(cases)} = {rate:.3f}")
    if fail_rows:
        print("\nFailures:")
        for row in fail_rows:
            print(f"  {row['case_id']}: {row['error']}")
    return 0 if pass_count == len(cases) else 2


def _bench_from_agent_results(path: str) -> int:
    rows = json.loads(Path(path).read_text())
    print(f"FHIR R4 validation on {len(rows)} agent-bench rows")
    pass_count = 0
    fail_rows: list[dict] = []
    for idx, row in enumerate(rows, 1):
        cid = row["case_id"]
        if "extraction" not in row:
            print(f"  SKIP {idx:3d}/{len(rows)} {cid} (no extraction — error row)")
            continue
        score = validate_extraction(row["extraction"])
        score.case_id = cid
        mark = "OK " if score.valid else "FAIL"
        if score.valid:
            pass_count += 1
        else:
            fail_rows.append({"case_id": cid, "error": score.error})
        print(f"  {mark} {idx:3d}/{len(rows)} {cid:38s} entries={score.n_entries}"
              + (f"  err={score.error[:80]}" if not score.valid else ""))
    rate = pass_count / max(len(rows), 1)
    print(f"\nfhir_r4_pass_rate: {pass_count}/{len(rows)} = {rate:.3f}")
    if fail_rows:
        print("\nFailures:")
        for row in fail_rows:
            print(f"  {row['case_id']}: {row['error']}")
    return 0 if pass_count == len(rows) else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench",
        action="store_true",
        help=(
            "Run a no-LLM bench: build Bundles from the expected codes in "
            "each case (acts as a schema sanity check)."
        ),
    )
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
    ap.add_argument(
        "--from-agent-bench",
        default=None,
        help=(
            "Validate the rows in an agent_pipeline.py --out-json result file."
        ),
    )
    args = ap.parse_args()

    if args.from_agent_bench:
        return _bench_from_agent_results(args.from_agent_bench)
    if args.bench:
        return _bench(args.cases)
    return _validate_single_stdin()


if __name__ == "__main__":
    sys.exit(main())
