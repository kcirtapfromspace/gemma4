---
experiment: c19-fast-path-py
team_tag: c19
backend: llama-cpp
device: macbook-pro-m4
runtime: cpu
model_variant: base-gemma-4-e2b-it-q3km
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-25 16:30:00
status: done
---

# c19-fast-path-py

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: —  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: —
- **avg_extraction_score**: —
- **success_rate (valid JSON)**: 100.0%
- **total_runs**: 8

> C19 fast-path Python mirror. apps/mobile/convert/rag_search.py FAST_PATH_THRESHOLD=0.70 + first_asserted_span (matched_phrase rule). agent_pipeline.py grew try_fast_path() before run_agent + --fast-path-rag-threshold / --no-fast-path CLI flags. validate_fast_path.py: 8/8 parity probes vs Swift, identical scores (Marburg 1.387, valley fever 1.218, C diff 1.225, Plasmodium malariae 1.514) and identical decline-cases. Adv4 projects 0/8 fast-path hits (every adv4 case has non-empty tier-1 extraction).

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c19-fast-path-py';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c19-fast-path-py';`
- commit: _TODO_
- raw artifacts: _TODO_
