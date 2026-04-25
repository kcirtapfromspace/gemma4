---
experiment: c19-fast-path-swift
team_tag: c19
backend: llama-cpp
device: iphone-17-pro-sim
runtime: cpu
model_variant: base-gemma-4-e2b-it-q3km
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-25 16:00:00
status: done
---

# c19-fast-path-swift

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: —  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: 100.0%
- **avg_extraction_score**: 1.000
- **success_rate (valid JSON)**: 100.0%
- **total_runs**: 19

> C19 single-turn fast-path Swift mirror. RagSearch.fastPathHit with threshold 0.70 + matched-phrase-only NegEx (avoids altName false-positive on 'Coccidioides' genus prefix). validate_rag.swift CLI: 11/11 top-k + 8/8 fast-path = 19/19. Demo seed +1 (Sofia Reyes / valley fever / draft) with no inline codes — exact long-tail target. New 'RAG · FAST' tier chip in ProvenanceBadge / ProvenanceLegend; UI confidence clamped to [0,1]. Latency: bypasses LLM, single-digit ms.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c19-fast-path-swift';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c19-fast-path-swift';`
- commit: _TODO_
- raw artifacts: _TODO_
