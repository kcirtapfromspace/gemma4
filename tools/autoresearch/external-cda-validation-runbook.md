# External CDA/XML Validation Runbook

Date: 2026-04-30

Scope: CDC/HL7 eICR CDA XML validation for the deterministic extractor,
FHIR Bundle validator, and long-context agent chunker.

## Current external set

`scripts/test_cases_external.jsonl` contains all 10 official HL7 eICR XML
samples currently exposed by the tracked HL7 sample directories:

- STU 1.1: `Sample.xml`, `SAMPLE_EXTERNAL_ENCOUNTER.xml`,
  `SAMPLE_MANUAL.xml`, `SAMPLE_MANUAL_EXTERNAL_ENCOUNTER.xml`
- STU 1.3.0: `D3_SAMPLE.xml`, `D3_SAMPLE_EXTERNAL_ENCOUNTER.xml`,
  `D3_SAMPLE_MANUAL.xml`
- STU 3.1.1: `SAMPLE.xml`, `SAMPLE_EXTERNAL_ENCOUNTER.xml`,
  `SAMPLE_MANUAL.xml`

Local deterministic check on 2026-04-30:

```bash
python3 scripts/run_det_external.py \
  --cases scripts/test_cases_external.jsonl \
  --out-json build/external/det_external_10.json \
  --fail-on-imperfect
```

Result: 10/10 perfect, 474/474 expected codes matched, 0 false positives.

FHIR structural validation on 2026-04-30:

- Python `fhir.resources.R4B`: 10/10 external, 80/80 broad adversarial set.
- HL7 `validator_cli.jar` with `CLINIQ_FHIR_TX_SERVER=n/a`: 10/10 external,
  80/80 broad adversarial set.
- Agent/deterministic broad bench:
  `apps/mobile/convert/agent_pipeline.py` on the 80-case broad set matched
  202/202 expected codes with 0 false positives. It used deterministic
  short-circuit for 68/80 cases and RAG fast-path for 9/80 cases; the local
  model endpoint was not required for a perfect pass.

The strict Java pass required profile-safe code-only vital-sign encoding in
`fhir_bundle.py`: BP components are emitted under a BP panel, oxygen saturation
and measured body weight include required canonical profile codings, and
value-required vitals such as BMI/head circumference are represented as
vital-sign panel components with `dataAbsentReason=unknown` instead of
invented measurements.

## Coverage Audit

The official HL7 eICR sample directories currently expose 10 XML files.
Audit local coverage with:

```bash
python3 scripts/run_det_external.py \
  --cases scripts/test_cases_external.jsonl \
  --audit-official-coverage
```

As of 2026-04-30, local metadata covers all 10 official samples.

## Hardening tests to add

1. Chunker merge dedup:
   Build a synthetic long CDA from a real official sample by repeating one
   structured section across two chunk boundaries. Expected output should keep
   one copy of each code and no lookup-tier cross-axis duplicates. This should
   exercise the agent chunker path, not only `run_det_external.py`.

2. Messy CDA attribute handling:
   Add executable cases for valid XML variants that the current regex parser
   is likely to miss: namespace-prefixed `cda:code` elements, attributes split
   across newlines, reversed `code` / `codeSystem` order, single-quoted
   attributes, and entity-escaped display names. Treat single quotes as a
   parser gap until fixed.

3. CDA display-name bleed:
   Include a `<code code="LOINC" displayName="Zika ...">` plus duplicated
   rendered `<td>` text. Expected result should include the LOINC and exclude
   the Zika SNOMED unless an actual diagnosis code or diagnosis narrative is
   present.

4. Malformed-but-recoverable CDA:
   Use a truncated document with complete `<code .../>` tags before the
   truncation point. Expected behavior: recover complete code tags and avoid
   crashing. This is realistic for copied EHR XML snippets.

5. FHIR validator evidence:
   Run `score_fhir.py --bench --backend python` on the expanded external set.
   For Java, keep `CLINIQ_FHIR_TX_SERVER=n/a` unless intentionally testing
   terminology snapshots; otherwise failures can be moving-target code-system
   age issues rather than structural Bundle bugs.

## Commands

Deterministic external bench:

```bash
python3 scripts/run_det_external.py \
  --cases scripts/test_cases_external.jsonl \
  --out-json apps/mobile/convert/build/external_eicr_deterministic_only.json \
  --fail-on-imperfect
```

FHIR structural bench over external expected codes:

```bash
python3 apps/mobile/convert/score_fhir.py \
  --bench \
  --backend python \
  --cases scripts/test_cases_external.jsonl \
  --out-json apps/mobile/convert/build/external_eicr_fhir_python.json
```

HL7 Java validator:

```bash
mkdir -p /tmp/fhir-validator
curl -L -o /tmp/fhir-validator/validator_cli.jar \
  https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar

CLINIQ_FHIR_TX_SERVER=n/a python3 apps/mobile/convert/score_fhir.py \
  --bench \
  --backend java \
  --cases scripts/test_cases_external.jsonl \
  --out-json apps/mobile/convert/build/external_eicr_fhir_java.json
```
