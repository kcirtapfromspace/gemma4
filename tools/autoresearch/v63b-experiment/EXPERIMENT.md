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
