# v63 Single-Shot LoRA Runbook

**Owner:** Worker A, direction 2
**Goal:** recover v62's JSON-validity and recall gap without touching the
tool-calling agent path.

## Evidence

v62 is the right branch to continue, not v31/v27:

- v62 single-shot compact LoRA: micro-F1 `0.823`, precision `0.979`, p50
  `4.1s`, JSON validity `86%`.
- v62 invalid JSON cases are documented as length-limit truncations, not
  model confusion: 28 invalid cases in the reported 200-case bench.
- Valid-only v62 already reaches F1 `0.895`, so the highest-value next
  iteration is reducing truncation and improving recall, not changing the
  inference topology.
- v31/v27 agent fine-tunes targeted tool-calling and were discarded because
  tool-calling regressed. v63 must stay single-shot compact JSON only.

Current local data shape:

| File | Rows | Gold schema | Output chars p50 / p95 / max |
|---|---:|---|---:|
| `kaggle-training/dataset/train-compact.jsonl` | 1600 | `patient, conditions, labs, meds, vitals` | 459 / 667 / 762 |
| `kaggle-training/dataset/train_v2.jsonl` | 1650 | same | 459 / 669 / 762 |
| `kaggle-training/dataset/train_v2_additions.jsonl` | 50 | same | 469 / 703 / 721 |
| `kaggle-training/dataset/val-compact.jsonl` | 400 | same | 458 / 661 / 793 |

Note: v62 submission docs cite a 200-case `val-compact` bench. The checked-in
`val-compact.jsonl` currently has 400 rows. For v63, record whether the bench
uses all 400 rows or the exact 200-row v62 subset.

## v63 Training Target

Train a new adapter from `kaggle-training/train-compact.py` as the v63
candidate. This script already incorporates the post-v62 direction:

- `train_v2.jsonl` preferred over legacy compact data.
- Response-only loss masking on `<|turn>model\n`.
- No tools, no function-call schema, no agent traces.
- Compact JSON output only.
- LoRA rank `r=32`.
- KV-shared layers 15..34 skip `k_proj` / `v_proj` LoRA dead weight.
- `packing=False`, avoiding packed-example boundary artifacts.
- No explicit TRL `max_seq_length`; the tokenizer context is left high enough
  that the local p99 train examples are not truncated.

Do not add `encounter` to the output schema for this iteration. The gold data
and prompt train five top-level keys only:

```json
{"patient":{},"conditions":[],"labs":[],"meds":[],"vitals":[]}
```

## Data Augmentation

Before running v63, add only if time allows:

1. Add 40-60 new compact examples to `train_v2_additions.jsonl`, then rebuild
   `train_v2.jsonl`.
2. Bias the additions toward outputs longer than the current p95: multi-dx,
   multi-lab, multi-med cases with 2-3 conditions and 2-3 medications.
3. Include negative-lab variants (`Not detected`, `Negative`) so longer cases
   do not reintroduce sequence degeneration.
4. Keep gold outputs minified and five-key compact. Avoid prose summaries,
   tool calls, FHIR Bundles, and markdown.

If adding examples is skipped, still run the current trainer as v63 because it
already differs materially from v62 through response-only masking, v2 data,
rank `32`, and non-packed training.

## Kaggle Run

From `kaggle-training/`:

```bash
kaggle kernels push -p .
```

Kaggle settings:

- Accelerator: GPU T4 x2. The script forces one visible GPU and fails fast on
  P100.
- Dataset input must contain `train_v2.jsonl` and `val-compact.jsonl`.
- Expected outputs:
  - `/kaggle/working/cliniq-compact-lora/`
  - `/kaggle/working/cliniq-compact-merged/`

The merged model is saved as 1 GB shards to avoid the v31 Kaggle HTTP
truncation issue.

## Conversion

After downloading `cliniq-compact-merged/`:

```bash
PY=/Users/thinkstudio/gemma4/scripts/.venv/bin/python

${PY} /tmp/llama-cpp-tools/convert_hf_to_gguf.py \
  /tmp/c9-v63/cliniq-compact-merged \
  --outfile /tmp/c9-v63/cliniq-gemma4-e2b-v63.f16.gguf \
  --outtype f16

llama-quantize \
  /tmp/c9-v63/cliniq-gemma4-e2b-v63.f16.gguf \
  /Users/thinkstudio/gemma4/models/cliniq-gemma4-e2b-v63-Q3_K_M.gguf \
  Q3_K_M
```

## Bench

Run base and v63 side by side:

```bash
llama-server \
  --model /Users/thinkstudio/gemma4/models/gemma-4-E2B-it-Q3_K_M.gguf \
  --port 8090 --host 127.0.0.1 --jinja --ctx-size 32768 \
  --parallel 4 --n-gpu-layers 99 --threads 8 \
  > /tmp/llama-server-base.log 2>&1 &

llama-server \
  --model /Users/thinkstudio/gemma4/models/cliniq-gemma4-e2b-v63-Q3_K_M.gguf \
  --port 8091 --host 127.0.0.1 --jinja --ctx-size 32768 \
  --parallel 4 --n-gpu-layers 99 --threads 8 \
  > /tmp/llama-server-v63.log 2>&1 &

python apps/mobile/convert/bench_v62_singleshot.py \
  --val kaggle-training/dataset/val-compact.jsonl \
  --out apps/mobile/convert/build/v63_val_compact_bench.json \
  --label v63_singleshot_lora \
  --max-tokens 2048 \
  --compare
```

Optional stress rerun, only if JSON validity remains below 98%:

```bash
python apps/mobile/convert/bench_v62_singleshot.py \
  --val kaggle-training/dataset/val-compact.jsonl \
  --out apps/mobile/convert/build/v63_val_compact_bench_maxtok4096.json \
  --label v63_singleshot_lora_maxtok4096 \
  --max-tokens 4096 \
  --compare
```

Do not use `apps/mobile/convert/cliniq_v62_compact.gbnf` as the primary v63
bench path. v62 evidence showed grammar-constrained compact JSON worsened
F1 and validity by driving longer generations into the token cap.

## Keep / Discard Rule

Keep v63 only if it beats v62 on the single-shot compact path:

- Micro-F1 >= `0.895` on JSON-valid cases, and preferably >= `0.88` overall.
- JSON validity >= `0.98` at `--max-tokens 2048`; if only the 4096 run passes,
  record the latency tradeoff and do not ship silently.
- Precision remains >= `0.97`.
- p50 latency stays near v62, target <= `5.0s` on the same Mac/server setup.
- No tool-call regression is relevant because v63 is not used for tool calls.

Discard or rerun data augmentation if v63 improves JSON validity by producing
longer but lower-recall JSON, or if precision drops below `0.97`.
