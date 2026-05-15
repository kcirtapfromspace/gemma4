# v63/v64 Trace Distillation Gate

**Goal:** turn verified broad90/teacher-trace evidence into an operational
single-shot training and evaluation gate without training on protected eval.

**Decision:** `apps/mobile/convert/build/broad90_agent_verify.json` is a
protected eval proof, not a default training source. Use it to prove the teacher
is clean and to define holdouts. Distill only from a separate candidate trace
set whose `case_id`s and normalized narratives are absent from broad90 and the
compact v62 validation set.

## Protected Inputs

Treat these as no-train by default:

```bash
PROTECTED_CASES=(
  scripts/test_cases.jsonl
  scripts/test_cases_adversarial.jsonl
  scripts/test_cases_adversarial2.jsonl
  scripts/test_cases_adversarial3.jsonl
  scripts/test_cases_adversarial4.jsonl
  scripts/test_cases_adversarial5.jsonl
  scripts/test_cases_adversarial6.jsonl
  scripts/test_cases_adversarial7.jsonl
  scripts/test_cases_adversarial8.jsonl
  scripts/test_cases_external.jsonl
  scripts/test_cases_longitudinal.jsonl
)

PROTECTED_COMPACT_VAL=kaggle-training/dataset/val-compact.jsonl
```

Do not include `kaggle-training/dataset/val-compact.jsonl` in a training
dataset. The current distiller protects by `case_id`; for compact validation,
keep it outside `--case-files` entirely and run the v62 single-shot bench only
after training. The manifest validator also hashes user turns, so pass compact
validation there to catch copied or renamed narratives.

## Checklist

1. Verify the teacher evidence is still clean.

```bash
python - <<'PY'
import json, pathlib
rows = json.loads(pathlib.Path("apps/mobile/convert/build/broad90_agent_verify.json").read_text())
assert len(rows) == 90, len(rows)
assert sum(r["matched"] for r in rows) == sum(r["expected"] for r in rows) == 676
assert sum(r["false_positives"] for r in rows) == 0
assert all(r["matched"] == r["expected"] for r in rows)
print("broad90 teacher gate: 90/90 perfect, 676/676 matched, 0 FP")
PY
```

2. Build a non-protected candidate case file.

Create a JSONL file outside the protected list, for example
`build/trace-distill/v63_candidate_cases.jsonl`. Each row must have `case_id`
and `user`. Prefer generated or newly collected synthetic cases with fresh
`case_id`s. Do not copy rows from `scripts/test_cases*.jsonl` or
`kaggle-training/dataset/val-compact.jsonl`.

3. Generate teacher traces for candidate cases.

```bash
mkdir -p build/trace-distill kaggle-training/dataset

python apps/mobile/convert/agent_pipeline.py \
  --cases build/trace-distill/v63_candidate_cases.jsonl \
  --out-json build/trace-distill/v63_teacher_verify.json \
  --tool-call-grammar apps/mobile/convert/cliniq_toolcall.gbnf \
  --chat-timeout 3 \
  --max-turns 2
```

4. Distill only perfect teacher rows and exclude protected eval IDs.

```bash
python scripts/distill_agent_traces.py \
  --agent-json build/trace-distill/v63_teacher_verify.json \
  --case-files build/trace-distill/v63_candidate_cases.jsonl \
  --eval-case-files "${PROTECTED_CASES[@]}" \
  --out-dir kaggle-training/dataset \
  --prefix trace-distill-v63
```

Expected artifacts:

- `kaggle-training/dataset/trace-distill-v63-train.jsonl`
- `kaggle-training/dataset/trace-distill-v63-val.jsonl`
- `kaggle-training/dataset/trace-distill-v63-manifest.json`

5. Validate the manifest before any training job can see the rows.

```bash
python tools/autoresearch/validate_trace_distill_manifest.py \
  --manifest kaggle-training/dataset/trace-distill-v63-manifest.json \
  --eval-case-files "${PROTECTED_CASES[@]}" "$PROTECTED_COMPACT_VAL" \
  --min-admitted 1 \
  --require-val
```

The validator must report `protected_eval_admitted: 0` and
`protected_eval_narratives_admitted: 0`. If it fails, delete the generated
`trace-distill-v63-*.jsonl` files and regenerate from a clean candidate source.

6. Train v63/v64 as a separate candidate dataset.

Keep trace-distilled rows separate from `train_v2.jsonl` until the manifest
passes. If merging is needed for the Kaggle trainer, make the merge command
write a new filename such as `train_v3_trace_distill.jsonl`; keep
`train_v2.jsonl` unchanged for rollback.

```bash
cat \
  kaggle-training/dataset/train_v2.jsonl \
  kaggle-training/dataset/trace-distill-v63-train.jsonl \
  > kaggle-training/dataset/train_v3_trace_distill.jsonl
```

Record the exact trainer input filename in the Kaggle run notes before pushing.

7. Evaluate without retraining on eval.

Run gates in this order:

```bash
python apps/mobile/convert/bench_v62_singleshot.py \
  --val kaggle-training/dataset/val-compact.jsonl \
  --out apps/mobile/convert/build/v63_trace_val_compact_bench.json \
  --label v63_trace_distill \
  --max-tokens 2048 \
  --compare

python apps/mobile/convert/agent_pipeline.py \
  --cases "${PROTECTED_CASES[@]}" \
  --out-json apps/mobile/convert/build/v63_trace_broad90_regression.json \
  --tool-call-grammar apps/mobile/convert/cliniq_toolcall.gbnf \
  --chat-timeout 3 \
  --max-turns 2
```

If a single-shot bench harness is added for broad90, run it against the
protected case list and write a separate artifact under
`apps/mobile/convert/build/`; do not route those cases back into the distiller.

## Fail / Keep Thresholds

Keep a v63 trace-distilled candidate only if all gates pass:

- Manifest validation: `protected_eval_admitted == 0`, no duplicate
  `case_id`s, `protected_eval_narratives_admitted == 0`, no duplicate
  normalized narratives, train/val row counts match the manifest.
- Teacher trace admission: every admitted row has `matched == expected` and
  `false_positives == 0`.
- v63 compact bench: JSON validity >= `0.98` at `--max-tokens 2048`.
- v63 compact bench: overall micro-F1 >= `0.88` and JSON-valid micro-F1 >=
  `0.895`.
- v63 compact bench: precision >= `0.97`.
- Latency: p50 <= `5.0s` on the same local llama-server setup used for v62.
- Protected broad90 regression: no loss relative to the checked-in teacher
  proof if the evaluated path is expected to preserve agent behavior.

Discard or rerun data generation if any protected eval row is admitted, if
precision drops below `0.97`, or if JSON validity improves only by losing
recall.

## v64 Escalation

Use v64 only after v63 has one clean run and the failure mode is understood.
Allowed v64 changes:

- More non-protected candidate traces.
- A higher `--val-fraction` for the trace-distilled split if the candidate
  corpus is large enough.
- Longer-output augmentation focused on multi-condition, multi-lab,
  multi-medication cases.

Do not use v64 to relax contamination controls. `--allow-eval-cases` is for
local debugging only and must not appear in a training run command.
