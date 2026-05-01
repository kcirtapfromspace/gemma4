# CDA Hardening Runbook

Date: 2026-04-30

Scope: executable CDA parser hardening cases for the deterministic
`regex_preparser.extract` path and code-only FHIR bundle validation.

## Fixture

`scripts/test_cases_cda_hardening.jsonl` contains five synthetic CDA snippets:

- `cda_hardening_namespace_prefixed_code_tags`: `cda:code` elements with
  explicit SNOMED, LOINC, and RxNorm attributes.
- `cda_hardening_reversed_split_attributes`: `codeSystem`, `displayName`, and
  `code` split across lines with `codeSystem` before `code`.
- `cda_hardening_single_quoted_attributes`: single-quoted XML attributes.
- `cda_hardening_displayname_bleed_lab_only_zika`: a Zika-containing LOINC
  `displayName` duplicated in rendered table text; expected output excludes
  Zika SNOMED because the text is only a lab name.
- `cda_hardening_truncated_recoverable_complete_tags`: a truncated CDA with
  complete code tags before the dangling tail.

These are intentionally small and deterministic. They should run in milliseconds
and should not require a model endpoint.

## Parser Note

The single-quoted attribute case exposed a narrow CDA parser gap. The CDA tag,
`code`, and `displayName` regexes now accept either single or double quotes and
optional whitespace around `=`. The same regexes are used by extraction and CDA
display-name masking so lookup-tier bleed behavior remains aligned with the CDA
tier.

## Commands

Validate JSONL line-by-line:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("scripts/test_cases_cda_hardening.jsonl")
for i, line in enumerate(p.read_text().splitlines(), 1):
    if line.strip():
        json.loads(line)
print(f"valid jsonl: {p} ({i} lines)")
PY
```

Run deterministic hardening bench:

```bash
python3 scripts/run_det_external.py \
  --cases scripts/test_cases_cda_hardening.jsonl \
  --out-json build/cda_hardening/det_cda_hardening.json \
  --fail-on-imperfect
```

Run external CDA regression bench:

```bash
python3 scripts/run_det_external.py \
  --cases scripts/test_cases_external.jsonl \
  --out-json build/cda_hardening/det_external_regression.json \
  --fail-on-imperfect
```

Run Python FHIR structural validation:

```bash
scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --bench \
  --backend python \
  --cases scripts/test_cases_cda_hardening.jsonl \
  --out-json build/cda_hardening/fhir_python_cda_hardening.json
```

Run HL7 Java validator:

```bash
CLINIQ_FHIR_TX_SERVER=n/a scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --bench \
  --backend java \
  --cases scripts/test_cases_cda_hardening.jsonl \
  --out-json build/cda_hardening/fhir_java_cda_hardening.json
```

Syntax smoke:

```bash
python3 -m py_compile apps/mobile/convert/regex_preparser.py scripts/run_det_external.py
```

## Evidence

Local results on 2026-04-30:

- JSONL validation: pass, 5 lines.
- CDA hardening deterministic bench: 5/5 perfect, 13/13 expected codes,
  precision 1.000, recall 1.000, F1 1.000, 0 false positives.
- Existing external CDA deterministic regression: 10/10 perfect, 474/474
  expected codes, precision 1.000, recall 1.000, F1 1.000, 0 false positives.
- Python FHIR bench over the hardening fixture: 5/5 pass.
- HL7 Java validator bench over the hardening fixture: 5/5 pass, 0 errors.
- `py_compile`: pass for `regex_preparser.py` and `run_det_external.py`.

## Remaining Gaps

- These cases verify complete CDA tags before truncation. They do not attempt to
  recover a code from a tag whose closing `>` or required attribute is itself
  truncated.
