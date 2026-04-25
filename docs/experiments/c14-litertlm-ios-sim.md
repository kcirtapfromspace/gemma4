---
experiment: c14-litertlm-ios-sim
team_tag: c14
backend: litert-lm
device: iphone-17-pro-sim
runtime: cpu
model_variant: gemma-4-E2B-base
model_format: litertlm-int4
data_source: measured
created_at: 2026-04-23 22:00:00
status: done
---

# c14-litertlm-ios-sim

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: 14.70  •  **p50_gen_tok_s**: 14.70
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: —
- **avg_extraction_score**: —
- **success_rate (valid JSON)**: 100.0%
- **total_runs**: 2

> C14 LiteRT-LM Swift decode — iPhone 17 Pro simulator CPU. 13.84-15.55 tok/s on smoke prompts (XCTest + CLI). Metal/GPU path is iPhone-hardware-only; expect ~56 tok/s per C5 upstream benchmark.

### Per-case runs

| case_id | tok/s | extraction_score | tokens | valid_json |
|---|---:|---:|---:|:---:|
| smoke_colors | 13.84 | — | 6 | ✗ |
| smoke_france_paris | 15.55 | — | 2 | ✗ |

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c14-litertlm-ios-sim';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c14-litertlm-ios-sim';`
- commit: _TODO_
- raw artifacts: _TODO_
