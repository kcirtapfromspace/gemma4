# v63/v64 Trace Distillation Plan

Worker E scope: turn verified agent/RAG traces into single-shot student pairs
without training on protected evaluation cases.

## Data Contract

Use `scripts/distill_agent_traces.py` on `agent_pipeline.py --out-json`
artifacts. The student target is the final verified extraction only:

```json
{"conditions":["..."],"loincs":["..."],"rxnorms":["..."]}
```

This avoids repeating the v31 failure mode where the fine-tune learned brittle
tool-call formatting. v63/v64 should be a direct extractor; the agent remains
the teacher and verifier.

## Controls

- Admit only rows with `matched == expected` and `false_positives == 0`.
- Join each trace row back to source case JSONL by `case_id`; rows without a
  source narrative are excluded.
- Deduplicate by normalized narrative SHA-256.
- Protect evaluation by passing held-out files with `--eval-case-files`; those
  `case_id`s are excluded unless `--allow-eval-cases` is explicitly set.
- Deterministic split by `sha256(split_seed + case_id + narrative_hash)`.
- Write a manifest containing admitted/excluded rows, target JSON, split, path,
  and narrative hash.

## Recommended Flow

Generate a teacher artifact on non-protected candidate cases:

```bash
python apps/mobile/convert/agent_pipeline.py \
  --cases scripts/test_cases_external.jsonl \
  --out-json apps/mobile/convert/build/agent_distill_external.json \
  --tool-call-grammar apps/mobile/convert/cliniq_toolcall.gbnf
```

Build v63/v64 SFT rows while excluding the regression/eval suite:

```bash
python scripts/distill_agent_traces.py \
  --agent-json apps/mobile/convert/build/agent_distill_external.json \
  --case-files scripts/test_cases_external.jsonl \
  --eval-case-files scripts/test_cases.jsonl scripts/test_cases_adversarial*.jsonl \
  --out-dir kaggle-training/dataset \
  --prefix trace-distill-v63
```

Train on `trace-distill-v63-train.jsonl` and validate first on
`trace-distill-v63-val.jsonl`, then run untouched combined-45/54 plus the v62
single-shot bench. Do not merge these rows into `train_v2.jsonl` until the
manifest confirms zero protected eval IDs.
