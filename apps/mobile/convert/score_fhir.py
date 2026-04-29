"""Structural FHIR R4 validation for ClinIQ extractions.

Per proposals-2026-04-25.md Rank 3, scores `fhir_r4_pass_rate` by wrapping
each case's extraction in a Bundle (via `fhir_bundle.to_bundle`) and
running it through a strict R4 validator. A Bundle that validates without
errors is "structurally R4-valid" — does not mean "semantically correct,"
just that every resource satisfies its cardinality + required-field +
invariant constraints.

Why it ships: judges on the eICR-to-FHIR / clinical interop axis will
probe this directly. F1 numbers don't carry the same weight as a binary
"100% R4-valid Bundles" claim.

Two backends are available (selected via `--backend`):

* `python` (default): pydantic-based parse via `fhir.resources.R4B.bundle.Bundle`.
  Fast, embed-friendly, but only covers structural+cardinality constraints —
  it does NOT enforce Bundle.entry invariants like `bdl-7` (`fullUrl` required)
  or terminology display-name binding.

* `java`: HL7-published `validator_cli.jar` (org.hl7.fhir.core, the canonical
  reference implementation). Slower (~15 s warm-up + ~0.5 s/bundle), but enforces
  the FULL R4 spec including invariants and terminology service lookups.
  Pin to `-version 4.0.1` (R4) in invocations. Download separately:

      mkdir -p /tmp/fhir-validator
      curl -L -o /tmp/fhir-validator/validator_cli.jar \\
          https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar

  Override location via `CLINIQ_FHIR_VALIDATOR_JAR=/path/to/validator_cli.jar`.

Requires `fhir.resources>=7.0,<9.0` for the python backend (we test against
8.2.0). Install into the project venv:

    uv pip install --python scripts/.venv/bin/python "fhir.resources>=7.0,<9.0"

Usage:

    # Validate a single extraction read from stdin (python backend):
    echo '{"conditions": ["840539006"], "loincs": [], "rxnorms": []}' | \\
        scripts/.venv/bin/python apps/mobile/convert/score_fhir.py

    # Same, official HL7 Java validator:
    echo '{"conditions": ["840539006"], "loincs": [], "rxnorms": []}' | \\
        scripts/.venv/bin/python apps/mobile/convert/score_fhir.py --backend java

    # Bench across all combined-bench cases — runs deterministic
    # extraction on each, builds the Bundle, validates it, prints
    # fhir_r4_pass_rate aggregate:
    scripts/.venv/bin/python apps/mobile/convert/score_fhir.py --bench
    scripts/.venv/bin/python apps/mobile/convert/score_fhir.py --bench --backend java

    # Or validate the c19 fast-path agent-bench results JSON:
    scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \\
        --from-agent-bench apps/mobile/convert/build/agent_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Local stdlib-only wrapper.
sys.path.insert(0, str(Path(__file__).parent))
from fhir_bundle import to_bundle  # noqa: E402
from case_diff import emit_csv_from_manifest  # noqa: E402


@dataclass
class FhirScore:
    case_id: str
    valid: bool
    error: str | None = None
    n_entries: int = 0
    # Filled by the Java backend when available; left None by the Python
    # backend (pydantic raises on the first error, so a single message
    # captures the whole failure).
    n_errors: int | None = None
    n_warnings: int | None = None


# ---------------------------------------------------------------------------
# Backend: Python — fhir.resources.R4B (pydantic-based)
# ---------------------------------------------------------------------------

class PythonValidator:
    """fhir.resources.R4B.bundle.Bundle — pydantic structural parse.

    Covers cardinality + required-field + datatype constraints. Does NOT
    enforce Bundle.entry invariants (`bdl-7` fullUrl) or terminology
    binding. Use the JavaValidator backend for spec-complete validation.
    """

    name = "python"

    def __init__(self) -> None:
        try:
            # R4B (4.0.1) — the practical R4 binding. fhir.resources 8.x
            # defaults the top-level imports to R5, where
            # MedicationStatement.medication is CodeableReference (not R4's
            # medicationCodeableConcept). Pin to R4B to validate against the
            # spec the hackathon targets.
            from fhir.resources.R4B.bundle import Bundle  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - surfaced via CLI
            sys.stderr.write(
                "fhir.resources (with R4B subpackage) not installed. Install with:\n"
                "  uv pip install --python scripts/.venv/bin/python "
                "'fhir.resources>=7.0,<9.0'\n"
            )
            raise SystemExit(1) from exc
        self._Bundle = Bundle

    def validate_extraction(self, extraction: dict) -> FhirScore:
        bundle_dict = to_bundle(extraction)
        n_entries = len(bundle_dict.get("entry") or [])
        try:
            self._Bundle(**bundle_dict)
        except Exception as exc:  # noqa: BLE001
            return FhirScore(
                case_id="(unknown)",
                valid=False,
                error=f"{type(exc).__name__}: {exc}"[:600],
                n_entries=n_entries,
            )
        return FhirScore(case_id="(unknown)", valid=True, n_entries=n_entries)


# ---------------------------------------------------------------------------
# Backend: Java — org.hl7.fhir.validation.ValidatorCli (canonical HL7 ref)
# ---------------------------------------------------------------------------

class JavaValidator:
    """HL7-published `validator_cli.jar` invoked via subprocess.

    Pins to `-version 4.0.1` (FHIR R4). Pure subprocess; the validator
    self-loads its package cache (~15 s on first call, then warmed). We
    invoke once per bundle for simplicity — the validator can be batched
    by passing multiple files to one invocation; keep that for a future
    perf pass if Java validation becomes the default path.
    """

    name = "java"

    DEFAULT_JAR = "/tmp/fhir-validator/validator_cli.jar"

    # Matches the validator's per-source summary line, in any of the three
    # formats the tool emits:
    #   "*FAILURE*: 8 errors, 5 warnings, 0 notes"           (batch mode)
    #   "*SUCCESS*: 0 errors, 1 warnings, 0 notes"            (batch mode)
    #   "Success: 0 errors, 6 warnings, 0 notes"              (single-source mode)
    #   "Failure: 1 errors, 0 warnings, 0 notes"              (single-source mode)
    _SUMMARY_RE = re.compile(
        r"(?:\*(?P<verdict_starred>SUCCESS|FAILURE)\*|"
        r"(?P<verdict_plain>Success|Failure))[: ]\s*"
        r"(?P<errors>\d+)\s+errors?,\s+(?P<warnings>\d+)\s+warnings?,\s+\d+\s+notes?",
        re.IGNORECASE,
    )

    def __init__(
        self,
        jar_path: str | None = None,
        fhir_version: str = "4.0.1",
        tx_server: str | None = None,
    ) -> None:
        jar = jar_path or os.environ.get("CLINIQ_FHIR_VALIDATOR_JAR") or self.DEFAULT_JAR
        if not Path(jar).exists():
            sys.stderr.write(
                f"validator_cli.jar not found at {jar}.\n"
                "Download it with:\n"
                "  mkdir -p /tmp/fhir-validator && curl -L -o /tmp/fhir-validator/validator_cli.jar \\\n"
                "    https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar\n"
                "Or set CLINIQ_FHIR_VALIDATOR_JAR=/path/to/validator_cli.jar\n"
            )
            raise SystemExit(1)
        if not shutil.which("java"):  # pragma: no cover - env check
            sys.stderr.write("`java` not on PATH (need Java 11+). brew install openjdk@21\n")
            raise SystemExit(1)
        self._jar = jar
        self._version = fhir_version
        # `-tx n/a` disables terminology server lookups so validation is
        # purely structural (cardinality + invariants + datatype). With a
        # tx server, the validator additionally rejects codes the server
        # doesn't know — useful for spec compliance, but creates a moving
        # target as terminology snapshots evolve. Default: tx-disabled to
        # match the python backend's structural-only behaviour. Override
        # via env CLINIQ_FHIR_TX_SERVER (set "http://tx.fhir.org" to
        # re-enable terminology binding).
        self._tx = (
            tx_server
            or os.environ.get("CLINIQ_FHIR_TX_SERVER")
            or "n/a"
        )

    def _invoke(self, bundle_paths: list[str]) -> tuple[int, str]:
        """Run validator_cli on one or more files; return (exit_code, combined_output).

        validator_cli accepts multiple file paths in a single invocation and
        amortizes the ~15 s package-load cost across all of them. Pass a
        list to validate a batch.
        """
        proc = subprocess.run(
            [
                "java",
                "-jar",
                self._jar,
                *bundle_paths,
                "-version",
                self._version,
                "-tx",
                self._tx,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def validate_extraction(self, extraction: dict) -> FhirScore:
        bundle_dict = to_bundle(extraction)
        n_entries = len(bundle_dict.get("entry") or [])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(bundle_dict, f)
            tmp_path = f.name
        try:
            _, output = self._invoke([tmp_path])
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return self._parse_summary(output, n_entries)

    def _parse_summary(self, output: str, n_entries: int) -> FhirScore:
        # Find the summary line. The validator emits SUCCESS/FAILURE; we treat
        # anything other than `errors=0` as invalid even if exit code is 0
        # (it returns 0 with errors=0 only).
        m = self._SUMMARY_RE.search(output)
        if not m:
            return FhirScore(
                case_id="(unknown)",
                valid=False,
                error=f"java validator produced no summary; tail={output[-400:]!r}",
                n_entries=n_entries,
            )
        n_errors = int(m.group("errors"))
        n_warnings = int(m.group("warnings"))
        valid = n_errors == 0
        err_payload: str | None = None
        if not valid:
            # Capture the first few error lines verbatim.
            err_lines = [
                ln.strip()
                for ln in output.splitlines()
                if ln.lstrip().startswith("Error @")
            ][:6]
            err_payload = "; ".join(err_lines)[:600] or output[-500:]
        return FhirScore(
            case_id="(unknown)",
            valid=valid,
            error=err_payload,
            n_entries=n_entries,
            n_errors=n_errors,
            n_warnings=n_warnings,
        )

    def validate_extractions_batch(
        self, extractions: list[tuple[str, dict]]
    ) -> list[FhirScore]:
        """Validate many extractions in one Java invocation.

        ~30x faster than per-call when batch >= 10. The validator emits a
        per-source `Validate /path/to/file.json` block followed by error
        lines, terminated by a single `Done.` summary covering the whole
        batch. We split on the per-source headers and parse each block
        for its own SUCCESS/FAILURE summary.
        """
        if not extractions:
            return []
        # Write each bundle to a stable, ordered tempdir so we can map
        # validator output back to case ids.
        tmpdir = Path(tempfile.mkdtemp(prefix="cliniq-fhir-batch-"))
        paths: list[str] = []
        case_for_path: dict[str, tuple[str, int]] = {}
        try:
            for idx, (case_id, extraction) in enumerate(extractions):
                bundle_dict = to_bundle(extraction)
                n_entries = len(bundle_dict.get("entry") or [])
                # Filename includes a 4-digit prefix to keep ordering clean
                # in validator output and avoid collisions on duplicate ids.
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", case_id)[:60]
                p = tmpdir / f"{idx:04d}_{safe}.json"
                p.write_text(json.dumps(bundle_dict))
                paths.append(str(p))
                case_for_path[str(p)] = (case_id, n_entries)
            _, output = self._invoke(paths)

            # Split on per-source markers. validator_cli emits a block
            # per source after the global "Validate" pass:
            #
            #   -- /tmp/.../0000_x.json ----------------------------------------
            #   *FAILURE*: 4 errors, 2 warnings, 0 notes
            #   ...error/warning lines...
            #   ----------------------------------------------------------------
            #   -- /tmp/.../0001_y.json ----------------------------------------
            #   *SUCCESS*: 0 errors, 1 warnings, 0 notes
            #   ...
            #
            # We split on the `-- /path -----` header lines.
            block_header_re = re.compile(
                r"^--\s+(?P<path>/[^\s]+\.json)\s+-{3,}\s*$"
            )
            blocks: dict[str, str] = {}
            current_path: str | None = None
            buf: list[str] = []
            for line in output.splitlines():
                m_hdr = block_header_re.match(line)
                if m_hdr:
                    if current_path is not None:
                        blocks[current_path] = "\n".join(buf)
                    current_path = m_hdr.group("path")
                    buf = []
                else:
                    buf.append(line)
            if current_path is not None:
                blocks[current_path] = "\n".join(buf)

            # Parse each block.
            results: list[FhirScore] = []
            for p in paths:
                case_id, n_entries = case_for_path[p]
                block = blocks.get(p)
                if block is None:
                    # Validator skipped or errored out before reaching this
                    # file; record as invalid with the tail of the full
                    # output so the failure is debuggable.
                    score = FhirScore(
                        case_id=case_id,
                        valid=False,
                        error=f"no validator block for {p}; output tail={output[-300:]!r}",
                        n_entries=n_entries,
                    )
                else:
                    score = self._parse_summary(block, n_entries)
                    score.case_id = case_id
                results.append(score)
            return results
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except OSError:
                pass


# Module-level convenience used by other code (e.g. the spaces app or
# follow-up notebooks). Defaults to the python backend for backwards
# compat — all existing callers continue to work without any change.
_DEFAULT_BACKEND: "PythonValidator | JavaValidator | None" = None


def _get_default_backend() -> "PythonValidator | JavaValidator":
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = PythonValidator()
    return _DEFAULT_BACKEND


def validate_extraction(extraction: dict) -> FhirScore:
    """Backwards-compatible top-level entry point (python backend).

    Existing callers in `spaces/app.py` etc. import this name directly.
    For the Java backend, instantiate `JavaValidator()` explicitly.
    """
    return _get_default_backend().validate_extraction(extraction)


def _make_backend(name: str) -> "PythonValidator | JavaValidator":
    if name == "python":
        return PythonValidator()
    if name == "java":
        return JavaValidator()
    raise SystemExit(f"unknown --backend: {name!r} (choices: python, java)")


def _validate_single_stdin(backend: "PythonValidator | JavaValidator") -> int:
    extraction = json.loads(sys.stdin.read())
    score = backend.validate_extraction(extraction)
    suffix = ""
    if score.n_errors is not None:
        suffix = f" errors={score.n_errors} warnings={score.n_warnings}"
    if score.valid:
        print(f"R4 valid [{backend.name}] (entries={score.n_entries}){suffix}")
        return 0
    print(
        f"R4 INVALID [{backend.name}] (entries={score.n_entries}){suffix}: {score.error}"
    )
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


def _bench(
    case_paths: list[str],
    backend: "PythonValidator | JavaValidator",
    *,
    out_json: str | None = None,
) -> int:
    cases = _bench_cases(case_paths)
    print(
        f"FHIR R4 bench on {len(cases)} cases via [{backend.name}] backend "
        f"(extraction = expected codes)"
    )
    pass_count = 0
    fail_rows: list[dict] = []
    all_rows: list[dict] = []

    # Java backend amortizes the ~15s package-load over a single batched
    # call; per-case invocation is ~13 minutes for combined-54.
    if isinstance(backend, JavaValidator):
        extractions = [(case["case_id"], _expected_extraction(case)) for case in cases]
        scores = backend.validate_extractions_batch(extractions)
        score_iter = iter(scores)
    else:
        score_iter = None  # type: ignore[assignment]

    for idx, case in enumerate(cases, 1):
        cid = case["case_id"]
        if score_iter is not None:
            score = next(score_iter)
        else:
            score = backend.validate_extraction(_expected_extraction(case))
            score.case_id = cid
        if score.valid:
            pass_count += 1
            mark = "OK "
        else:
            mark = "FAIL"
            fail_rows.append({"case_id": cid, "error": score.error})
        suffix = ""
        if score.n_errors is not None:
            suffix = f" e={score.n_errors} w={score.n_warnings}"
        print(
            f"  {mark} {idx:3d}/{len(cases)} {cid:42s} entries={score.n_entries}{suffix}"
            + (f"  err={(score.error or '')[:80]}" if not score.valid else "")
        )
        all_rows.append(
            {
                "case_id": cid,
                "valid": score.valid,
                "n_entries": score.n_entries,
                "n_errors": score.n_errors,
                "n_warnings": score.n_warnings,
                "error": score.error,
            }
        )
    rate = pass_count / max(len(cases), 1)
    print(f"\nfhir_r4_pass_rate [{backend.name}]: {pass_count}/{len(cases)} = {rate:.3f}")
    if fail_rows:
        print("\nFailures:")
        for row in fail_rows:
            print(f"  {row['case_id']}: {row['error']}")
    if out_json:
        Path(out_json).write_text(
            json.dumps(
                {
                    "backend": backend.name,
                    "n_cases": len(cases),
                    "n_pass": pass_count,
                    "fhir_r4_pass_rate": rate,
                    "rows": all_rows,
                },
                indent=2,
            )
        )
        print(f"\nWrote {out_json}")
    return 0 if pass_count == len(cases) else 2


def _bench_from_agent_results(
    path: str,
    backend: "PythonValidator | JavaValidator",
    *,
    out_json: str | None = None,
) -> int:
    rows = json.loads(Path(path).read_text())
    # Some artifacts wrap rows in {"summary":..., "rows":[...]}.
    if isinstance(rows, dict) and "rows" in rows:
        rows = rows["rows"]
    print(
        f"FHIR R4 validation on {len(rows)} agent-bench rows via [{backend.name}] backend"
    )
    pass_count = 0
    fail_rows: list[dict] = []
    all_rows: list[dict] = []
    n_validated = 0

    # Pre-batch the rows that have extractions for the Java backend.
    if isinstance(backend, JavaValidator):
        eligible = [r for r in rows if "extraction" in r]
        scores = backend.validate_extractions_batch(
            [(r["case_id"], r["extraction"]) for r in eligible]
        )
        score_by_case = {s.case_id: s for s in scores}
    else:
        score_by_case = None  # type: ignore[assignment]

    for idx, row in enumerate(rows, 1):
        cid = row["case_id"]
        if "extraction" not in row:
            print(f"  SKIP {idx:3d}/{len(rows)} {cid} (no extraction — error row)")
            continue
        n_validated += 1
        if score_by_case is not None:
            score = score_by_case[cid]
        else:
            score = backend.validate_extraction(row["extraction"])
            score.case_id = cid
        mark = "OK " if score.valid else "FAIL"
        if score.valid:
            pass_count += 1
        else:
            fail_rows.append({"case_id": cid, "error": score.error})
        suffix = ""
        if score.n_errors is not None:
            suffix = f" e={score.n_errors} w={score.n_warnings}"
        print(
            f"  {mark} {idx:3d}/{len(rows)} {cid:42s} entries={score.n_entries}{suffix}"
            + (f"  err={(score.error or '')[:80]}" if not score.valid else "")
        )
        all_rows.append(
            {
                "case_id": cid,
                "valid": score.valid,
                "n_entries": score.n_entries,
                "n_errors": score.n_errors,
                "n_warnings": score.n_warnings,
                "error": score.error,
            }
        )
    rate = pass_count / max(n_validated, 1)
    print(
        f"\nfhir_r4_pass_rate [{backend.name}]: {pass_count}/{n_validated} = {rate:.3f}"
    )
    if fail_rows:
        print("\nFailures:")
        for row in fail_rows:
            print(f"  {row['case_id']}: {row['error']}")
    if out_json:
        Path(out_json).write_text(
            json.dumps(
                {
                    "backend": backend.name,
                    "source": path,
                    "n_validated": n_validated,
                    "n_pass": pass_count,
                    "fhir_r4_pass_rate": rate,
                    "rows": all_rows,
                },
                indent=2,
            )
        )
        print(f"\nWrote {out_json}")
    return 0 if pass_count == n_validated else 2


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
    ap.add_argument(
        "--backend",
        choices=("python", "java"),
        default="python",
        help=(
            "Validation backend. `python` (default) = fhir.resources.R4B "
            "(pydantic structural). `java` = HL7 validator_cli.jar (canonical "
            "spec-complete; requires Java 11+ and validator_cli.jar at "
            "/tmp/fhir-validator/validator_cli.jar or "
            "$CLINIQ_FHIR_VALIDATOR_JAR)."
        ),
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Write per-case results to this JSON path (bench / from-agent modes).",
    )
    ap.add_argument(
        "--diff-csv",
        default=None,
        help=(
            "Emit the EZeCR-style flat longitudinal CSV (one row per axis "
            "change across the case series) to this path. Requires "
            "`--manifest` pointing at the longitudinal manifest produced by "
            "`agent_pipeline.py --prior-bundle`. Skips the validator entirely "
            "— the manifest's bundles are assumed already R4-valid."
        ),
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help=(
            "Path to a longitudinal manifest JSON (case_id → patient_hash + "
            "case_dt + bundle_path). Required with `--diff-csv`. See "
            "`case_diff.load_manifest` for the shape."
        ),
    )
    args = ap.parse_args()

    if args.diff_csv:
        if not args.manifest:
            sys.stderr.write(
                "--diff-csv requires --manifest (path to the longitudinal "
                "manifest JSON).\n"
            )
            return 2
        n_rows = emit_csv_from_manifest(args.manifest, args.diff_csv)
        print(
            f"EZeCR flat CSV: {args.diff_csv} ({n_rows} rows, plus header) "
            f"from manifest {args.manifest}"
        )
        return 0

    backend = _make_backend(args.backend)

    if args.from_agent_bench:
        return _bench_from_agent_results(
            args.from_agent_bench, backend, out_json=args.out_json
        )
    if args.bench:
        return _bench(args.cases, backend, out_json=args.out_json)
    return _validate_single_stdin(backend)


if __name__ == "__main__":
    sys.exit(main())
