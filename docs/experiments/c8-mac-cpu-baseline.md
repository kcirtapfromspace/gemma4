---
experiment: c8-mac-cpu-baseline
team_tag: c8
backend: llama-cpp-server
device: macbook-pro-m4
runtime: cpu
model_variant: cliniq-compact-lora-v1
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-22 12:00:00
status: done
---

# c8-mac-cpu-baseline

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: —  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: 72.2%
- **avg_extraction_score**: 0.722
- **success_rate (valid JSON)**: —
- **total_runs**: 5

> C8 Mac-CPU baseline — fine-tuned LoRA + Gemma 4 E2B on Mac CPU. 13/18 on the 5 canonical bench_* cases. Referenced as apples-to-apples target by C12/C16.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c8-mac-cpu-baseline';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c8-mac-cpu-baseline';`
- commit: _TODO_
- raw artifacts: _TODO_
