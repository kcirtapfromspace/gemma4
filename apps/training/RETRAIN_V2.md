# Retrain V2 — ClinIQ Gemma 4 E2B LoRA (Team C9)

**Date:** 2026-04-23
**Branch:** `team/c9-lora-retrain-v2-2026-04-23`
**Baseline under attack:** Team C8 validation (2026-04-23), aggregate
extraction score **13/18 = 0.72** across 5 clinical cases. Demo target **>= 0.9**.

---

## TL;DR

Two new training artefacts on Kaggle give the LoRA a fair shot at >= 0.9:

1. **Data** — `train_v2.jsonl` = 1650 rows = original 1600 + **50 targeted
   diversification examples** (20 negative-lab + 20 code-preservation +
   10 length-stress).
2. **Config** — `r=16 -> 32`, KV-shared `k_proj`/`v_proj` excluded from
   `target_modules` on layers 15..34 (per Team C7 `DIAGNOSIS.md`),
   `train_on_responses_only` loss mask added, `max_seq_length=768`.

Kernel to push: `kaggle-training/train-compact.py` (unchanged file layout,
updated contents). Expected runtime on Kaggle T4: **~90-110 minutes**
(+30% vs v1 due to r=32 and +50 examples).

---

## Failure modes (from Team C8)

| case_id | description | C8 score | failure mode |
|---|---|--:|---|
| `bench_minimal` | syphilis, no vitals | 3/3 | (pass) |
| `bench_typical_covid` | COVID + vitals + labs + meds | **1/3** | **code elision** |
| `bench_complex_multi` | HIV multi-lab multi-med | 6/6 | (pass) |
| `bench_meningitis` | meningococcal + CSF | 3/3 | (pass) |
| `bench_negative_lab` | negative Hep C lab | **0/3** | **sequence degeneration** |
| aggregate | | **13/18** | |

### Mode 1: code elision (COVID)
Model emits the right JSON schema and drug/dx display names, but strips
the SNOMED (`840539006`) and RxNorm (`2599543`) codes from the parenthesized
tail. LOINC is short (`94500-6`) and survives. This is not a template bug —
C8 confirmed the template fix is load-bearing for cases 1/3/4. The LoRA
has learned an inconsistent output-format policy where short codes survive
and long ones get truncated.

### Mode 2: sequence degeneration (negative Hep C)
Input contains `Hepatitis C ... - Not detected` and the model collapses
into a repeat loop `SNOMED: SNOMED: 42800000000000...`. Decoder tok/s drops
from 3.86 baseline to 0.25. The training distribution is thin on negative
labs at the presentation boundary, so the prior over the first few
assistant tokens drifts.

---

## Fix 1: data diversification

Full generator: `/tmp/gen_diversification.py` (ephemeral; the output lives
in the tree). Output:

- `kaggle-training/dataset/train_v2_additions.jsonl` — 50 rows, auditable
- `kaggle-training/dataset/train_v2.jsonl` — 1650 rows (original + additions)

| bucket | n | what it trains |
|---|--:|---|
| **negative-lab diversification** | 20 | `Not detected` / `Negative` outputs across 20 distinct conditions (HIV, HCV, HBV, HAV, syphilis, TB, dengue, Zika, measles, pertussis, mumps, COVID, Lyme, chlamydia, gonorrhea, malaria, WNV, flu A, RSV, rubella). Every example keeps SNOMED+LOINC+RxNorm codes verbatim in the JSON. Targets **mode 2**. |
| **code preservation** | 20 | Paraphrased clinical presentations (e.g. COVID-19 → "pneumonia presentation, bilateral infiltrates, hypoxia" or HIV → "acute retroviral syndrome — mono-like illness 3 weeks post-exposure"). Same codes as the existing corpus, but the surface form of the Reason line is heavily varied. Teaches: "regardless of prose, copy the parenthesized code verbatim into output". Targets **mode 1**. |
| **length stress** | 10 | 140+ word inputs with 2-3 labs and 2 meds. Dense realistic clinical narrative. Output runs ~170 tokens of JSON. Pushes the model past the ~20-30 token shoulder where negative-lab degeneration starts. Uses real extra-LOINC codes (`57021-8` CBC, `1742-6` ALT, `1975-2` bilirubin, etc.). Targets **mode 2**. |

**Code integrity check:** all 50 new examples round-trip the SNOMED+LOINC+RxNorm
codes from user input into assistant output (50/50 pass under a grep-based
verifier).

### Source of codes
All SNOMED / LOINC / RxNorm identifiers are drawn from the canonical catalog
already present in the 2000-row baseline (`data/train-compact.jsonl` +
`data/val-compact.jsonl`). Those codes were vetted by prior teams and
cross-check against NLM RxNav + BioPortal + LOINC (the generator
`apps/training/generate_training_data.py` sourced them originally). No
codes are invented.

---

## Fix 2: LoRA config

### Target modules (from Team C7 `DIAGNOSIS.md`)

Gemma 4 E2B has 35 hidden layers and `num_kv_shared_layers=20`. Layers
15..34 reuse K/V from earlier layers at runtime — their `k_proj`/`v_proj`
are skipped in the attention forward pass when `past_key_values` is live.
During SFT, those LoRA deltas therefore never receive gradient and stay at
their PEFT-initial bit-exact zero (C7 measured abs-max == 0.0, std == 0.0
on all 40 tensors). Training them is dead weight and wastes capacity.

The retrain kernel installs per-layer target selection via a PEFT regex:

```python
# layers 0..14: q|k|v|o_proj; layers 15..34: q|o_proj only; all layers mlp
LORA_TARGET_REGEX = (
    r".*\.language_model\.layers\.([0-9]|1[0-4])\.self_attn\.(q|k|v|o)_proj$"
    r"|"
    r".*\.language_model\.layers\.(1[5-9]|2[0-9]|3[0-4])\.self_attn\.(q|o)_proj$"
    r"|"
    r".*\.language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj$"
)
```

Expected match count on Gemma 4 E2B: **205 modules** (35 q + 15 k + 15 v +
35 o + 105 mlp), versus the v1 stock list that attached LoRA to 245 modules
(the 40 dead shared-KV k/v included). The kernel has a fallback path for
layouts missing the `language_model.` prefix and a second fallback that
post-attaches then freezes `requires_grad=False` on the dead projections.

### Rank and alpha
`r=16 -> r=32`, `lora_alpha=32` (1:1 with `r`). With 50 additional diverse
examples and a tighter target list, we've got ~12% headroom on trainable
params to absorb the new patterns.

### Loss masking
`train_on_responses_only` wraps the SFTTrainer so labels are `-100` on the
user+system tokens. Gradient only flows on the assistant JSON tokens. This
is directly aimed at the code-elision failure — input tokens already
contain the codes verbatim, so the model was previously learning
(from the full-sequence loss) to compress rather than emit them.

**Template boundary strings:** `<start_of_turn>user\n` and
`<start_of_turn>model\n`. Prior `train.py` used `<|turn>...` which is
stale — C8's validation confirmed the unsloth-generated template uses
`<start_of_turn>...` for Gemma 4.

### max_seq_length
`512 -> 768`. Length-stress examples reach p99 of ~605 tokens including
chat-template wrappers. 768 leaves headroom; no perf cost on T4 at
`batch_size=1 grad_accum=8`.

### Epochs
Kept at **3**. With 50 new examples and r=32, this is the sweet spot
between overfitting (5+ epochs would memorize the new 50) and
undertraining (1-2 epochs wouldn't consolidate the new patterns). If
validation loss starts to climb, dial back to 2.

---

## Kernel packaging

### Files modified (all inside scope)
- `kaggle-training/train-compact.py` — LoRA config, data path, SFTConfig
- `kaggle-training/dataset/train_v2.jsonl` — new (merged)
- `kaggle-training/dataset/train_v2_additions.jsonl` — new (50-row audit slice)
- `notebooks/gemma4_eicr_fhir_finetune.ipynb` — cells 3, 5, 6 updated

### Files NOT touched (out of scope per brief)
- `apps/mobile/convert/*` — mobile pipeline stays as-is
- `apps/training/train.py` — Jetson build, separate path
- anything under `scripts/`

### kernel-metadata.json
Pre-existing values are correct:
- kernel slug: `patrickdeutsch/cliniq-compact-lora-training`
- dataset source: `patrickdeutsch/cliniq-training-data`
- machine_shape: `NvidiaTeslaT4` (load-bearing; P100 will fail sm check)

The Kaggle dataset `patrickdeutsch/cliniq-training-data` will need to be
refreshed with the new `train_v2.jsonl` before the kernel will pick it up.
The kernel has a fallback that falls through to legacy `train-compact.jsonl`
if v2 isn't present.

### Sanity check
```bash
python3 -c "import ast; ast.parse(open('kaggle-training/train-compact.py').read())"   # OK
wc -l kaggle-training/dataset/train_v2.jsonl                                          # 1650
wc -l kaggle-training/dataset/train_v2_additions.jsonl                                # 50
```

---

## Human runbook: push & launch

### 1. Refresh the Kaggle dataset (if v2 not yet uploaded)
```bash
cd /Users/thinkstudio/gemma4/.claude/worktrees/agent-a3b92296/kaggle-training/dataset
# Creates a new version of patrickdeutsch/cliniq-training-data with the
# v2 files added alongside the legacy ones.
kaggle datasets version -p . -m "C9: add train_v2.jsonl (+50 diversified)"
```
Wait ~1 min for Kaggle to finalize the version (check `kaggle datasets list -u patrickdeutsch`).

### 2. Push the kernel
```bash
cd /Users/thinkstudio/gemma4/.claude/worktrees/agent-a3b92296
kaggle kernels push -p kaggle-training/
```

### 3. Monitor
```bash
kaggle kernels status patrickdeutsch/cliniq-compact-lora-training
# when "complete":
kaggle kernels output patrickdeutsch/cliniq-compact-lora-training -p /tmp/c9-output
```

Expected wall time on T4x2 Kaggle accelerator: **90-110 min** for 3 epochs
over 1650 rows at effective batch 8 (1 * 8 grad accum). v1 took ~70 min at
r=16 over 1600 rows, so roughly 1.3x longer.

### 4. Download artifacts
From the Kaggle kernel output:
- `cliniq-compact-lora/` — LoRA adapter (~50 MB, r=32 doubles vs r=16)
- `cliniq-compact-merged/` — merged fp16 safetensors (~2.5 GB)

### 5. Re-merge + re-validate on mobile pipeline
Use the existing team C6/C7 mobile convert pipeline:
```bash
cd apps/mobile/convert
# Convert merged safetensors to .litertlm
python merge_and_convert.py \
    --lora /tmp/c9-output/cliniq-compact-lora \
    --base unsloth/gemma-4-E2B-it \
    --out build/litertlm/model.litertlm \
    --quant dynamic_wi4_afp32
# Run the 5-case validator (expects build/validation/SUMMARY.json written)
python validate_all_cases.py --cases ../../../scripts/test_cases.jsonl
```
Target: `aggregate_score >= 0.9` in `build/validation/SUMMARY.json`. Specifically
watch `bench_typical_covid` (was 1/3) and `bench_negative_lab` (was 0/3).

### 6. If score still < 0.9

Priority debug order:
1. Inspect `train_v2.jsonl` samples rendered through the gemma-4 chat
   template — confirm no byte-level drift from unsloth.
2. Re-run with `num_train_epochs=4` (keep r=32, same data).
3. If `bench_negative_lab` still degenerates: add 20 more neg-lab
   examples focused on the specific failing Hep C / Measles shapes.
4. If `bench_typical_covid` still elides: bump `lora_alpha` to 64 while
   keeping `r=32` (alpha/r = 2 scales the delta contribution) or rerun with
   `r=64`.

---

## Changelog (C8 baseline -> C9 retrain)

| axis | before (C8) | after (C9) |
|---|---|---|
| training rows | 1600 | **1650** (+50 diverse) |
| LoRA rank | 16 | **32** |
| LoRA alpha | 16 | **32** |
| target_modules | 7 full list | **regex, 205 modules** (excludes 40 dead shared-KV k/v) |
| loss mask | all tokens | **assistant-only** |
| max_seq_length | 512 | **768** |
| epochs | 3 | 3 (unchanged) |
| eval during train | no | no (unchanged — OOM on 262K vocab) |
| base model | `unsloth/gemma-4-E2B-it` / `google/gemma-4-E2B-it` | same |
| chat template | unsloth `gemma-4` | same |

End.
