---
experiment: c17-deterministic-agent-rag-py
team_tag: c17
backend: llama-cpp
device: macbook-pro-m4
runtime: cpu
model_variant: base-gemma-4-e2b-it-q3km
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-24 18:00:00
status: done
---

# c17-deterministic-agent-rag-py

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: —  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: 92.6%
- **avg_extraction_score**: 0.986
- **success_rate (valid JSON)**: 92.6%
- **total_runs**: 27
- **model_file**: `gemma-4-E2B-it-Q3_K_M.gguf`

> C17 deterministic preparser + Gemma 4 native tool-calling agent + RAG over CDC NNDSS / WHO IDSR. 27-case combined bench: 25/27 perfect, 0 false positives, F1=0.986 (recall 0.971, precision 1.000). Avg 13.0 s/case, 2.64 tool calls/case, 3.64 LLM turns. Two non-perfect cases are knowledge-coverage gaps (codes not in lookup or NNDSS RAG), not pipeline bugs.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c17-deterministic-agent-rag-py';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c17-deterministic-agent-rag-py';`
- commit: _TODO_
- raw artifacts: _TODO_
