---
experiment: c1-jetson-orin-nx-7w
team_tag: c1
backend: llama-cpp-server
device: jetson-orin-nx
runtime: gpu
model_variant: cliniq-compact-lora-v1
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-23 10:00:00
status: done
---

# c1-jetson-orin-nx-7w

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: 0.90  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: —
- **avg_extraction_score**: —
- **success_rate (valid JSON)**: —
- **total_runs**: —

> Jetson Orin NX 8GB on Talos @ 7W (EMC 2133 MHz). 15W mode blocked on Talos without image rebuild per team/c1 POWER_MODE.md. 0.9 tok/s — ~60x below mobile GPU.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c1-jetson-orin-nx-7w';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c1-jetson-orin-nx-7w';`
- commit: _TODO_
- raw artifacts: _TODO_
