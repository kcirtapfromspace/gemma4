---
experiment: c19-toolcall-grammar
team_tag: c19
backend: llama-cpp
device: iphone-17-pro-sim
runtime: cpu
model_variant: base-gemma-4-e2b-it-q3km
model_format: gguf-q3km
data_source: measured
created_at: 2026-04-25 16:15:00
status: done
---

# c19-toolcall-grammar

<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->
## Metrics

- **avg_gen_tok_s**: —  •  **p50_gen_tok_s**: —
- **avg_ttft_ms**: —  •  **avg_prompt_tok_s**: —
- **extraction_pass_rate**: —
- **avg_extraction_score**: —
- **success_rate (valid JSON)**: 100.0%
- **total_runs**: 0

> C19 tool-call grammar lock. apps/mobile/convert/cliniq_toolcall.gbnf restricts tool-name to 4 registered tools and locks the bracket/quote syntax inside the tool-call payload. Wired through AgentRunner.engine.beginAgentTurn(grammar:) on tool-response turns only. Key finding: llama-server's /v1/chat/completions REJECTS custom grammar when tools is set — the --jinja path applies an internal tool-call grammar. So Python silently drops the grammar field; the explicit GBNF matters only on the iOS AgentRunner path. Stability bench (27×3, target 0 parse failures) gated on llama-server.

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
- duckdb: `SELECT * FROM experiments WHERE experiment_name = 'c19-toolcall-grammar';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = 'c19-toolcall-grammar';`
- commit: _TODO_
- raw artifacts: _TODO_
