# Team C7 — Phase 1 Diagnosis: Are the "dropped 40 LoRA projections" the cause of quality loss?

**Date:** 2026-04-23
**Branch:** `team/c7-lora-retrain-2026-04-23`
**Artefact under test:** `models/cliniq-compact-lora.gguf` (48 MB, 490 tensors, base `unsloth/gemma-4-E2B-it`)
**Reproducer:** `apps/training/diagnose_lora.py` (run directly; writes `apps/training/diagnose_lora_output.json`)

## TL;DR

**Outcome (a). The "drop" is benign — the 40 allegedly-dropped projections were never trained.**
Skip Phase 2 (do NOT retrain). The clinical-quality regression (mangled SNOMED `76272004`, missed RxNorm `105220`) has a different root cause. Candidate suspects and next-step probes listed below.

---

## 1. Ground truth about Gemma 4 E2B's KV-sharing

Pulled from the public `unsloth/gemma-4-E2B-it/config.json`:

| Field                    | Value                                                |
| ------------------------ | ---------------------------------------------------- |
| `num_hidden_layers`      | **35**                                               |
| `num_kv_shared_layers`   | **20**                                               |
| `first_shared_layer_idx` | `35 - 20 = 15`                                       |
| `layer_types`            | `sliding_attention` × 28, `full_attention` × 7 (repeats per 5) |
| `num_attention_heads`    | 8                                                    |
| `num_key_value_heads`    | 1                                                    |
| `hidden_size` / `head_dim` | 1536 / 256                                         |

So layers **15..34** (20 layers) reuse K/V from earlier layers; layers **0..14** compute their own K/V.

**Important correction to C6's framing:** the base safetensors *does* physically contain `model.language_model.layers.{0..34}.self_attn.k_proj.weight` and `v_proj.weight` — all 35 layers have the parameters. KV-sharing is a **runtime forward-pass** trick (in HF `transformers.models.gemma3n.Gemma3nTextAttention.forward`):

```python
if self.is_kv_shared_layer and past_key_values is not None:
    key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
    # <<< k_proj / v_proj ARE NOT CALLED for shared layers
else:
    key_states  = self.k_proj(hidden_states) ...
    value_states = self.v_proj(hidden_states) ...
```

During SFT (`past_key_values is not None` is the normal path in modern HF training with `DynamicCache`), `self.k_proj`/`self.v_proj` are skipped for shared layers → no autograd graph → no gradient → LoRA B stays at its PEFT-initialised **zero**.

## 2. Empirical norms from the 48 MB GGUF adapter

Computed `||B @ A||_F` for every (layer, projection) pair. Full per-pair dump in `apps/training/diagnose_lora_output.json`.

### Summary (grouped)

| Bucket                                                           |   n | mean ||BA||ₓF | median | min | max |
| ---------------------------------------------------------------- |  -: | ------------: | -----: | --: | --: |
| layers **0-14** k/v\_proj (non-shared, *should be* trained)       |  30 | **0.0680** | 0.0626 | 0.045 | 0.105 |
| layers **15-34** k/v\_proj (shared-KV, allegedly dropped)         |  40 | **0.0000** | 0.0000 | 0.000 | 0.000 |
| layers 0-14 q/o\_proj (own attention, trained)                     |  30 | 0.1864 | 0.1748 | 0.132 | 0.345 |
| layers 15-34 q/o\_proj (own Q/O, trained)                          |  40 | 0.2700 | 0.2782 | 0.183 | 0.400 |
| layers 0-14 gate/up/down (MLP, trained)                            |  45 | 0.3076 | 0.3328 | 0.143 | 0.509 |
| layers 15-34 gate/up/down (MLP, trained)                           |  60 | 0.6834 | 0.7485 | 0.297 | 1.049 |

**Ratio shared-KV / non-shared-KV = 0.000.** Non-shared K/V, and Q/O and MLP *on the same shared layers*, all show normal trained magnitudes. Only K/V on shared layers are untouched.

### Per-layer K/V (the interesting case)

```
layer |     k_proj |     v_proj | shared-KV?
----------------------------------------------
    0 |    0.04956 |    0.05238 |
    1 |    0.04563 |    0.05119 |
   ... (0.04-0.11 through layer 14) ...
   14 |    0.10210 |    0.08888 |
   15 |    0.00000 |    0.00000 | YES
   16 |    0.00000 |    0.00000 | YES
   ...
   34 |    0.00000 |    0.00000 | YES
```

### Confirmation that B is *initial zero*, not numerical noise

| Tensor                        | abs-max  | std     | interpretation              |
| ----------------------------- | -------- | ------- | --------------------------- |
| `blk.0.attn_k.weight.lora_a`  | 0.03075  | 0.01483 | kaiming-uniform init        |
| `blk.0.attn_k.weight.lora_b`  | 0.00471  | 0.00131 | trained (B moved off zero)  |
| `blk.15.attn_k.weight.lora_a` | 0.02551  | 0.01476 | kaiming-uniform init        |
| `blk.15.attn_k.weight.lora_b` | **0.0**  | **0.0** | **exact zero — never updated** |

The lora_b tensors for all 40 shared-KV K/V projections are **bit-exact 0.0**, not near-zero. No gradient ever flowed.

## 3. What this means for C6's "drop"

The merge pipeline in `apps/mobile/convert` (not read — file paths didn't resolve on this worktree) apparently refuses to merge a LoRA delta when the target layer is flagged as KV-shared. That refusal drops tensors whose computed `B @ A` is `0 × A = 0`. Merging zero into `base_k_proj.weight` is a no-op either way, and those layers' `k_proj` aren't called at inference when KV-sharing is active. So the drop loses **no information**.

**The fine-tune is *not* secretly assuming non-shared KV.** The training pipeline (unsloth + HF `Gemma3nTextAttention`) correctly skips those projections' forward pass, which is why their LoRA deltas stayed zero.

### Bonus: the LoRA *does* influence shared layers indirectly

Layers 15-34 reuse the **K/V states** produced by the source non-shared layer (one of 0-14). The LoRA update to layer 14's `k_proj`/`v_proj` changes what's stored in the shared cache, which is consumed by layers 15+. So clinical task signal flows through the LoRA update on layers 0-14 and is automatically propagated into 15-34's attention. That's the architecturally correct behaviour.

## 4. So where is the quality regression coming from?

Because Phase 1 rules out (1) dropped projections, the real cause lives elsewhere. Ordered by likelihood, the suspects we couldn't rule out in this phase:

### (S1) Chat-template mangling on the mobile path — **highest prior**
C6's own notes flagged a **Jinja incompatibility**. Training used `unsloth.chat_templates.get_chat_template(tokenizer, "gemma-4")`, which wraps turns with `<|turn>user\n…<|turn>model\n…`. The LiteRT-LM mobile bundle uses its own template string; if it's even slightly different (`<start_of_turn>user\n` vs `<|turn>user\n`, an extra `<bos>`, a missing `\n`), the logits distribution at the first assistant token is wrong and digit sequences drift after a few tokens — exactly the failure mode (LOINC is short, SNOMED + RxNorm are longer and start to mangle).
**Probe:** run the exact tokenised prompt byte-for-byte through (a) the Kaggle-trained PEFT model and (b) the mobile bundle; compare the first 10 logits. Any divergence >1e-3 on the argmax token identifies template drift.

### (S2) Quantisation artefacts on low-frequency digit tokens
Long numeric SNOMED/RxNorm codes tokenise to many single-digit tokens. INT4-style quantisation of the `lm_head` and output-side layers can shift the logits for these rarely-used tokens more than it does for English wordpieces. `76272004` could easily flip `7 → 1` or `4 → 5` when only a 1-2% logit change is needed.
**Probe:** compare outputs from `cliniq-gemma4-e2b.gguf` (higher precision) vs the `Q2_K/Q3_K_S/Q4_K_M` files side by side on the same failing prompt. If precision matters, quality should climb with bit-width.

### (S3) Tokenizer drift between training & deployment
If the mobile path uses a stripped/fast tokenizer that normalises whitespace or digits differently, the training-time target like `76272004` may tokenise as `[7, 6, 27, 2004]` while inference input tokenises as `[76, 27, 2004]` (illustrative). Small difference, catastrophic for exact-match digit tasks.
**Probe:** round-trip each failing code through both tokenizers; require tokens to match bit-for-bit.

### (S4) Not the cause, but ruled-out: data-coverage
Grep counts for `76272004`, `105220`, and the working `20507-0` all give the same frequency (48 in train, 15 in val). Coverage is fine.

## 5. Recommended next steps (for the human / next agent)

1. Start with (S1). Diff the two chat templates in plain text.
2. Run (S2) as a regression: if unquantised output also fails, quantisation isn't the culprit; if it passes, pick the smallest quant that preserves the failing codes.
3. (S3) is cheap: a 20-line Python script on the two tokenizer JSONs.
4. If all three pass: revisit training data quality and check whether `76272004`/`105220` appear in `val-compact.jsonl` with correct JSON formatting on the **assistant** side (not just in the user prompt).

## 6. Reproduce

```bash
cd /Users/thinkstudio/gemma4/.claude/worktrees/agent-aa94a1cb
python3 apps/training/diagnose_lora.py
cat apps/training/diagnose_lora_output.json | python3 -m json.tool | head
```

Requires only `gguf` and `numpy`; runs in <5 s on the Mac.
