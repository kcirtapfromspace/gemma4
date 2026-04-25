---
experiment: c17-deterministic-agent-rag-swift
team_tag: c17
backend: llama-cpp
device: iphone-17-pro-sim
runtime: cpu
model_variant: base-gemma-4-e2b-it-q3km
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-24 20:00:00
status: done
---

# c17-deterministic-agent-rag-swift

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: 4.00  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: 100.0%
- **avg_extraction_score**: 1.000
- **success_rate (valid JSON)**: 100.0%
- **total_runs**: 14
- **model_file**: `gemma-4-E2B-it-Q3_K_M.gguf`

> Swift mirror — EicrPreparser + AgentRunner + RagSearch + ToolCallParser + GemmaToolTemplate. validate_rag.swift CLI: 11/11 top-k probes pass; xcodebuild green; iPhone17ProDemo simulator launches with 3-tier flow visible. Decode tok/s from C12 baseline — pending physical-device measurement.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c17-deterministic-agent-rag-swift';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c17-deterministic-agent-rag-swift';`
- commit: _TODO_
- raw artifacts: _TODO_
