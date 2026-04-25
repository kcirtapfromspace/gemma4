---
experiment: c19-fhir-r4-bundle
team_tag: c19
backend: fhir.resources
device: macbook-pro-m4
runtime: cpu
model_variant: n/a
model_format: json
data_source: measured
created_at: 2026-04-25 17:00:00
status: done
---

# c19-fhir-r4-bundle

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: —  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: 100.0%
- **avg_extraction_score**: —
- **success_rate (valid JSON)**: 100.0%
- **total_runs**: 35

> C19 FHIR R4 Bundle wrapper + structural validator. apps/mobile/convert/fhir_bundle.py — pure-stdlib to_bundle builds Patient + Condition (clinicalStatus=active) + Observation (status=final) + MedicationStatement (status=recorded) per code; displayName from lookup_table.json; provenance source URLs into Resource.meta.source. score_fhir.py validates via fhir.resources.R4B (8.2.0; pinned R4B because top-level R5 rejects medicationCodeableConcept). Bench: combined-27 = 27/27, combined-27 + adv4 = 35/35, fhir_r4_pass_rate=1.000. Demo claim '100% R4-valid Bundles, on-device, offline' defensible. Swift mirror BundleBuilder.swift + 'View FHIR Bundle' sheet on Review screen. Outbox payload now defaults to FHIR Bundle wire format (SyncConfig.useFhirBundlePayload=true).

<!-- METRICS:END -->

## Hypothesis
_TODO: what did we expect going in? What would make this experiment interesting?_

## Method
_TODO: hardware, workload, scorer, anything non-obvious about how this was measured._

## Result
_TODO: narrative interpretation of the Metrics block above. Surprises? Regressions?_

## Decision
_TODO: what are we doing with this? What's the next experiment?_

## Links
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c19-fhir-r4-bundle';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c19-fhir-r4-bundle';`
- commit: _TODO_
- raw artifacts: _TODO_
