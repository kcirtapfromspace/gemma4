---
experiment: c16-litertlm-macos-cpu
team_tag: c16
backend: litert-lm
device: macbook-pro-m4
runtime: cpu
model_variant: c16-retrain
model_format: litertlm-int4
data_source: measured
created_at: 2026-04-24 14:00:00
status: done
---

# c16-litertlm-macos-cpu

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: 3.81  •  **p50_gen_tok_s**: 4.06
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: 62.1%
- **avg_extraction_score**: 0.694
- **success_rate (valid JSON)**: 88.9%
- **total_runs**: 9
- **model_file**: `build/litertlm/cliniq-gemma4-e2b.litertlm`

> C16 retry — freshly-quantized cliniq-gemma4-e2b.litertlm on macOS CPU. 18/29 = 0.621 across 9 cases. 5/9 perfect. Two cases hit known int4 degeneration (negative_lab, complex_multi) pending KV-sharing-aware Unsloth retrain.

### Per-case runs

| case_id | tok/s | extraction_score | tokens | valid_json |
|---|---:|---:|---:|:---:|
| bench_complex_multi | 3.40 | 0.167 | 20 | ✓ |
| bench_lyme | 4.51 | 1.000 | 38 | ✓ |
| bench_meningitis | 4.06 | 1.000 | 32 | ✓ |
| bench_minimal | 3.86 | 1.000 | 33 | ✓ |
| bench_multi_enteric | 4.46 | 1.000 | 41 | ✓ |
| bench_negative_lab | 0.49 | 0.000 | 7 | ✗ |
| bench_no_vitals_no_meds | 4.73 | 1.000 | 28 | ✓ |
| bench_tb_multi_med | 5.04 | 0.750 | 54 | ✓ |
| bench_typical_covid | 3.72 | 0.333 | 27 | ✓ |

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c16-litertlm-macos-cpu';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c16-litertlm-macos-cpu';`
- commit: _TODO_
- raw artifacts: _TODO_
