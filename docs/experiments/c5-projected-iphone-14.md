---
experiment: c5-projected-iphone-14
team_tag: c5
backend: litert-lm
device: iphone-14
runtime: gpu
model_variant: gemma-4-E2B-base
model_format: litertlm-int4
data_source: projected
created_at: 2026-04-23 12:00:00
status: done
---

# c5-projected-iphone-14

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: 12.00  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: —
- **avg_extraction_score**: —
- **success_rate (valid JSON)**: —
- **total_runs**: —

> Projected from modelfit.io A15 + AI Edge Gallery reports. ~46 s e2e on 700-prefill + 500-decode workload — inside the 60 s demo budget.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c5-projected-iphone-14';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c5-projected-iphone-14';`
- commit: _TODO_
- raw artifacts: _TODO_
