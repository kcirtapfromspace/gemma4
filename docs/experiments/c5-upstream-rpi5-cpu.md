---
experiment: c5-upstream-rpi5-cpu
team_tag: c5
backend: litert-lm
device: rpi5
runtime: cpu
model_variant: gemma-4-E2B-base
model_format: litertlm-int4
data_source: upstream-bench
created_at: 2026-04-23 12:00:00
status: done
---

# c5-upstream-rpi5-cpu

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: 8.00  •  **p50_gen_tok_s**: 8.00
- **avg_ttft_ms**: 7800  •  **avg_prompt_tok_s**: 133.00
- **extraction_pass_rate**: —
- **avg_extraction_score**: —
- **success_rate (valid JSON)**: —
- **total_runs**: —

> Upstream Google LiteRT-LM benchmark. Peak mem 1546 MB.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c5-upstream-rpi5-cpu';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c5-upstream-rpi5-cpu';`
- commit: _TODO_
- raw artifacts: _TODO_
