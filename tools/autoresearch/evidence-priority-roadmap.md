# Evidence Priority Roadmap

Date: 2026-04-30

This ranks the remaining non-iOS work by exhibition impact. The current
Broad90 result is useful, but it is mostly deterministic and should not be the
only headline if we want to demonstrate Gemma/LLM value.

## Priority 0: LLM-Required Evidence

Goal: show cases where deterministic extraction cannot reasonably solve the
task alone, and the LLM/RAG path materially improves recall while preserving
zero false positives.

Done enough:

- A separate `llm_required` JSONL suite with no inline codes and minimal direct
  lookup-table aliases.
- Bench output reports path distribution and confirms actual model/tool usage,
  not only deterministic short-circuit.
- Summary separates this suite from Broad90 so the claim is honest:
  Broad90 = robust offline pipeline; LLM-required = model value.

## Priority 1: Larger Held-Out Corpus

Goal: address sample-size skepticism with a documented route from curated
regression cases to a held-out accuracy estimate.

Done enough:

- A reproducible manifest or generator that assembles hundreds of non-protected
  held-out examples.
- Explicit contamination controls against `scripts/test_cases*.jsonl`.
- A report that distinguishes synthetic/gold-template examples from real
  official CDA/XML or human-authored clinical narratives.

## Priority 2: External CDA Hardening

Goal: prove that the parser handles realistic CDA/XML variation, not just the
current official examples.

Done enough:

- Fixtures for namespace-prefixed tags, attribute order changes, split
  attributes, quote variants, rendered-table displayName bleed, and truncated
  recoverable documents.
- Deterministic bench result and FHIR validation result.
- Known unsupported CDA shapes documented as parser gaps.

## Priority 3: v63/v64 Training Gate

Goal: make model training decisions reproducible and resistant to contamination.

Done enough:

- Clear commands from teacher trace generation to distillation to Kaggle train.
- Keep/discard thresholds for JSON validity, F1, precision, and latency.
- Manifest validation proving protected eval cases were excluded.

## Priority 4: Benchmark Publication

Goal: turn local artifacts into a concise exhibit-ready benchmark table.

Done enough:

- `scripts/publish_benchmarks.py` includes the latest Broad90, LLM-required,
  compact-val, and physical-device results.
- The top-line table carries sample sizes and caveats in the same row as the
  metric, so we do not overstate curated-regression evidence.
