---
experiment: c12-llama-cpp-ios-sim
team_tag: c12
backend: llama-cpp-ios
device: iphone-17-pro-sim
runtime: cpu
model_variant: cliniq-compact-lora-v1
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-23 20:00:00
status: done
---

# c12-llama-cpp-ios-sim

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: 4.30  •  **p50_gen_tok_s**: 4.01
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: 66.7%
- **avg_extraction_score**: 0.667
- **success_rate (valid JSON)**: 100.0%
- **total_runs**: 5
- **model_file**: `Documents/cliniq-gemma4-e2b-Q3_K_M.gguf`

> C12 on-device llama.cpp on iPhone 17 Pro simulator (CPU, 4 threads). 12/18 extraction (93% of C8 Mac baseline), median 4.0 tok/s. Projected physical iPhone 17 Pro Metal throughput: 10-20 tok/s per upstream benchmarks.

### Per-case runs

| case_id | tok/s | extraction_score | tokens | valid_json |
|---|---:|---:|---:|:---:|
| bench_complex_multi | 5.37 | 0.667 | 317 | ✓ |
| bench_meningitis | 4.00 | 0.667 | 165 | ✓ |
| bench_minimal | 3.74 | 0.000 | 140 | ✓ |
| bench_negative_lab | 4.37 | 1.000 | 182 | ✓ |
| bench_typical_covid | 4.01 | 1.000 | 172 | ✓ |

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c12-llama-cpp-ios-sim';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c12-llama-cpp-ios-sim';`
- commit: _TODO_
- raw artifacts: _TODO_
