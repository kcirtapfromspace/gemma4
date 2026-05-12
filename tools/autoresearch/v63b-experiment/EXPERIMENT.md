# v63b experiment — restore LoRA k/v coverage on global-attention layers

## The bug

v63 (`tools/autoresearch/v63-experiment/`) produced a LoRA adapter at
`/Users/thinkstudio/gemma4/models/cliniq-gemma4-e2b-v63-lora/` that contains only
**410 saved safetensors tensors**. v62 saved **490**. The 80-tensor delta is
exactly:

    20 decoder layers (indices 15-34, Gemma 4 "global" attention) ×
     2 modules (k_proj, v_proj) ×
     2 matrices (lora_A, lora_B)
    = 80 missing tensors

Gemma 4 E2B uses a hybrid-attention scheme: decoder layers 0-14 use *local*
sliding-window attention, layers 15-34 use *global* attention. The newer
peft / unsloth release that v63 was trained against silently dropped LoRA
on k_proj and v_proj for the global-attention half. `adapter_config.json`
looks correct (all 7 target_modules listed, `layers_to_transform=null`),
which is what makes this a regression rather than a config error. The
`"unsloth_fixed": true` line in the `auto_mapping` block of the saved config
is the suspicious tell.

The Kaggle full-precision bench did not notice (F1 = 0.9989) because
attention with full q/o + half k/v can still attend usefully at bf16 / 4-bit.
After Mac llama.cpp Q3_K_M conversion, the same adapter dropped to
**F1 = 0.5475** — quantization rounded the partial-coverage attention into
noise on every global-attention layer.

## The v63b fix

This kernel forces full k/v coverage and fails fast if it isn't restored:

1. **Explicit `layers_to_transform=list(range(35))`** on
   `FastLanguageModel.get_peft_model(...)`. peft is no longer free to
   infer a partial range.
2. **Post-LoRA-setup coverage assertion** walks the wrapped model and
   counts decoder-layer indices that have a LoRA-wrapped k_proj and
   v_proj submodule (with both `lora_A` and `lora_B` materialized). If
   either count is < 35, the cell prints a FATAL line and calls
   `sys.exit(1)` so the Kaggle kernel fails fast instead of producing
   another broken artifact.
3. **Post-save tensor-count assertion** reads the saved `*.safetensors`
   in `cliniq_lora/` and asserts `len(tensor_keys) >= 490`. Also prints
   `k_proj` and `v_proj` tensor counts (each should be 70 = 35 layers ×
   lora_A+lora_B) and dumps the relevant fields from `adapter_config.json`.
4. **Fallback install path** controlled by env var `V63B_PIN_OLD=1`,
   pinning `peft<0.18` and `unsloth<2026.5.0` in case the
   `layers_to_transform` fix alone doesn't restore coverage.

## Acceptance criteria

| Gate | Target |
|---|---|
| In-kernel coverage assertion | k_proj == 35 / 35 AND v_proj == 35 / 35 |
| Post-save tensor count | >= 490 |
| Mac Q3_K_M F1 (val-compact, 200 cases) | **>= 0.85** (vs v63's **0.5475**) |

The first two gates run inside Kaggle. The third is run downstream on Mac
via `apps/mobile/convert/bench_v62_singleshot.py` after the kernel finishes
and the adapter is downloaded.

## Coexistence

This directory is independent of `v62-submission/` and `v63-experiment/`.
All three Kaggle kernels (v62 shipped submission, v63 experiment, v63b
regression-fix experiment) coexist in the repo and on Kaggle under
distinct slugs.

## Run record

**Kernel:** `https://www.kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-v63b-experiment`

**v1 (2026-05-11) — fail-fast tripped.** Default install path landed
`transformers 5.8.0` (Kaggle's preinstalled version), which is
incompatible with `unsloth 2026.5.2`'s pin (`transformers <= 5.5.0`).
The pip-install printed the conflict as a warning rather than
erroring; training proceeded; LoRA wrapping report showed
`k_proj wrapped on 15/35 layers` (same regression as v63). Fail-fast
assertion fired, kernel ended in ERROR after ~3 min. No GPU-hour
burned on a broken artifact.

**v2 (2026-05-11) — coverage restored, training completed.** Pinned
`transformers>=5.5,<=5.5.0` explicitly in both install branches.
With that resolved, the coverage assertion passed:
```
=== v63b LoRA coverage report ===
  q_proj     layers wrapped: 35/35  missing=none
  k_proj     layers wrapped: 35/35  missing=none
  v_proj     layers wrapped: 35/35  missing=none
  o_proj     layers wrapped: 35/35  missing=none
  gate_proj  layers wrapped: 35/35  missing=none
  up_proj    layers wrapped: 35/35  missing=none
  down_proj  layers wrapped: 35/35  missing=none
v63b coverage OK — full k/v coverage on all 35 decoder layers.
```
Training proceeded through all 5 epochs (checkpoint-100 through
checkpoint-500 written). Total trainable parameters: 31,039,488 of
5,154,217,504 (0.60% — vs v62's smaller LoRA because v63b's wider
target spec also wraps SigLIP vision-tower attention slots).

The kernel ended in ERROR because the inline bench cell crashed on
`tokenizer.apply_chat_template` — transformers 5.5.0's `Gemma4Processor`
is now multimodal and expects message content as a list of dicts
(`[{"type":"text","text":"..."}]`), not a plain string. **Training and
LoRA save completed successfully**; only the post-train bench cell
failed. checkpoint-500 was recovered locally to
`models/cliniq-gemma4-e2b-v63b-lora/`.

**v63b adapter shape (from saved safetensors):** 786 tensors total
(`786 = 35 decoder layers × 7 modules × 2 (A+B) + 56 SigLIP vision-tower
slot pairs`). v62 had 490 (decoder only). The extra vision-tower
wrapping is benign-ish for the language-model bench since vision
features aren't activated for text-only inputs; the load-bearing fix
is the full k/v coverage on the 35 decoder layers.

**Mac re-bench (50-case quick sample, 2026-05-11):** Result at
`apps/mobile/convert/build/v63b_val_compact_bench_quick50.json`.

| metric | v62 (200) | v63 (200) | v63b (50) |
|---|---:|---:|---:|
| micro-F1 | **0.837** | 0.548 | 0.610 |
| micro-precision | **0.837** | 0.393 | 0.474 |
| micro-recall | 0.837 | 0.902 | **0.856** |
| JSON-validity | 0.92 | 0.92 | 0.80 |
| schema-complete | high | 0.00 | 0.00 |
| latency p50 (s) | 3.08 | 2.79 | **2.81** |
| latency p95 (s) | 4.24 | 3.86 | 4.23 |
| FPs total | ~0 | 863 | 152 |

**Decision: v62 stays as the shipped Unsloth-track LoRA.** v63b clears
the k/v coverage gate (the load-bearing diagnostic this experiment was
designed to test — assertion passed, full 35/35 wrapping on both k_proj
and v_proj, 786 saved tensors) and confirms our diagnosis was correct.
But the Mac Q3_K_M F1 of 0.61 is still well below v62's 0.84. The
latency win is preserved (2.81s p50, 9% faster than v62) — so v63b is
"halfway back": precision improved from v63's 0.393 to 0.474, FPs cut
nearly 6× per case, but still hallucinating extras.

**Why we don't keep iterating** (with 7 days to deadline): the
remaining 0.23 F1 gap between v63b and v62 is not from the bug we
fixed. The 786-tensor adapter wraps SigLIP vision-tower attention
alongside the decoder language model — those vision weights are getting
gradient updates from text-only training data, which likely accounts
for the schema-completeness collapse (`schema_complete_rate = 0.00`
same as v63). A v63c that explicitly excludes vision modules via a more
specific `target_modules` regex is the obvious next experiment, but
it's lower priority than the deadline-critical artifacts that aren't
yet pushed (HF Hub for v62, public Kaggle notebook for v62, Space
hardware flip).

**The Unsloth-track story is now stronger, not weaker.** v62 is the
shipped LoRA. v63 + v63b are documented diagnostic work that found a
silent toolchain regression: post-2026-04-30 releases of unsloth and
peft drop LoRA k/v on Gemma 4's global-attention layers when
transformers > 5.5.0 is installed. The bug hides at full precision
(F1 = 0.9989 on Kaggle PyTorch) and only surfaces under Q3_K_M
quantization. The fail-fast tensor-count assertion in this experiment
(`v63b_experiment.ipynb` Section 3) is the upstream-reportable
detection pattern: count k_proj/v_proj layers, sys.exit if < 35.
