---
experiment: c18-ios-demo-polish
team_tag: c18
backend: llama-cpp
device: iphone-17-pro-sim
runtime: cpu
model_variant: base-gemma-4-e2b-it-q3km
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-25 07:19:00
status: done
---

# c18-ios-demo-polish

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: —  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: —
- **avg_extraction_score**: —
- **success_rate (valid JSON)**: 100.0%
- **total_runs**: 0

> iOS demo polish — three commits: (1) ProvenanceLegend on Review screen, INLINE/CDA/LOOKUP/RAG chips with tap-to-expand explanations; (2) phase-aware running view that observes InferenceMetrics and renders LOAD/PREFILL/DECODE/FINAL chip + ETA copy that adapts past 60s decode; (3) Settings → Demo → Reset demo cases, one-tap wipe + DemoSeed.build() re-seed. xcodebuild green after each commit. Three new screenshots: review-with-legend / settings-reset / cases-list.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c18-ios-demo-polish';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c18-ios-demo-polish';`
- commit: _TODO_
- raw artifacts: _TODO_
