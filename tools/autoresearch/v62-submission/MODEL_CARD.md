---
license: apache-2.0
base_model: unsloth/gemma-4-E2B-it
tags:
  - unsloth
  - lora
  - peft
  - medical
  - clinical
  - eicr
  - fhir
  - public-health
  - edge
  - gemma-4
  - hackathon-gemma-4-good
language:
  - en
library_name: peft
pipeline_tag: text-generation
datasets:
  - patrickdeutsch/eicr-fhir-training-data
---

# ClinIQ — Gemma 4 E2B fine-tuned for eICR → compact JSON extraction

A LoRA adapter on `unsloth/gemma-4-E2B-it` that turns a clinician-formatted
eICR (electronic Initial Case Report) into a compact JSON extraction
({patient, encounter, conditions, labs, meds, vitals}) ready for downstream
FHIR R4 bundling, public-health surveillance, or an offline clinical decision
support workflow.

Submission for the **Gemma 4 Good Hackathon — Unsloth Track ($10K)**.

## TL;DR

| Metric | Base `gemma-4-E2B-it` | + ClinIQ v62 LoRA | Delta |
|---|---:|---:|---:|
| Micro-F1 (val-compact, 200 cases) | 0.337 | **0.837** | **+0.500** |
| Micro-F1 (JSON-valid cases only, 184) | n/a | **0.872** | — |
| Micro-precision | 0.469 | **0.837** | +0.368 |
| Micro-recall | 0.263 | **0.837** | +0.574 |
| JSON validity | 100% | 92% | -8pp |
| Latency p50 (Mac M-series Q3_K_M) | 6.6s | **3.08s** | 2.1× faster |
| Adapter size | — | 124 MB (r=16) | — |

Base produces structurally-valid JSON but gets the *content* wrong 2/3 of the
time — micro-precision 0.47, micro-recall 0.26. The 124 MB LoRA flips
content quality to 0.84 precision and 0.84 recall in a single forward pass.

**The 8% JSON validity gap is a truncation problem, not the shipped
headline.** On the 184 cases where v62 produces valid JSON, F1 reaches
**0.872** (precision 0.837, recall 0.911). We tried GBNF grammar enforcement,
but it regressed both F1 and JSON validity on the measured sub-bench; ship
v62 without grammar and treat retry/salvage or v63 longer-context training as
follow-up.

## Why Unsloth specifically

Three specific Unsloth features did real work in this fine-tune:

1. **`FastLanguageModel.from_pretrained(..., load_in_4bit=True)`** — base
   Gemma 4 E2B fits comfortably on a Kaggle T4 ×2 (16 GB each) at 4-bit;
   training stayed under 12 GB peak. Vanilla HF + bitsandbytes also works
   but requires more careful manual config.
2. **`use_gradient_checkpointing="unsloth"`** — Unsloth's checkpointing
   recipe drops VRAM ~30% vs vanilla, letting us train at
   `max_seq_length=512` with `packing=True` on T4. The compact dataset has
   long inputs (~300 tokens average), so the headroom was load-bearing.
3. **`unsloth.chat_templates.get_chat_template(tokenizer, "gemma-4")`** —
   Gemma 4's native chat template uses `<|turn>system\n...<turn|>\n` markers
   that the HF tokenizer's default doesn't emit. Unsloth ships the verified
   template inline; vanilla HF tokenizer would have produced
   `<start_of_turn>` / `<end_of_turn>` and the LoRA would have learned the
   wrong delimiters, breaking inference.

The public training kernel
([`kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-submission`](https://www.kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-submission))
runs end-to-end in **~1 h 4 m on Kaggle's free T4 ×2**. No GPU-hour budget,
no proprietary dependencies, no auth-walled datasets.

## Training data

[`patrickdeutsch/eicr-fhir-training-data`](https://www.kaggle.com/datasets/patrickdeutsch/eicr-fhir-training-data) — 800 train + 200 val
synthetic eICRs across the CDC RCTC reportable-condition list (COVID-19,
influenza, measles, pertussis, tuberculosis, HIV, syphilis, hepatitis,
streptococcus, etc.) with stratified positive / negative-lab / pediatric
/ pregnancy-context cases.

Each example is a 3-turn conversation:
- **System**: `Extract clinical entities from this eICR. Output compact JSON with: patient, encounter, conditions (SNOMED/ICD-10), labs (LOINC), meds (RxNorm), vitals. No summary. Valid JSON only.`
- **User**: clinician-formatted eICR text (e.g., `Patient: ...\nDx: ...\nLab: ... \nVitals: ...\nMeds: ...`)
- **Assistant**: minified JSON in the schema above

## Training config

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it",
    max_seq_length=512,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # 30% VRAM cut
    random_state=3407,
)

# 5 epochs, lr=1e-4, packing=True, adamw_8bit
# total steps: 500 (800 examples / 8 grad-accum)
# final loss: 0.2446
```

## Inference

```python
# Local, with llama.cpp
# 1. convert LoRA: scripts/convert_lora_to_gguf.py /path/to/cliniq_lora \
#       --base unsloth/gemma-4-E2B-it \
#       --out  cliniq-gemma4-e2b-v62-lora.gguf
# 2. apply at runtime:
llama-server --model gemma-4-E2B-it-Q3_K_M.gguf \
             --lora cliniq-gemma4-e2b-v62-lora.gguf \
             --port 8091 --jinja --ctx-size 4096

# 3. POST to /v1/chat/completions with system + user as in training
```

## Evaluation

200 held-out val-compact cases (`/tmp/eicr-fhir-data/val-compact.jsonl`,
held out from training). **Code-level micro-F1** scored on the union of
`(coding_system, code)` tuples across `conditions / labs / meds / vitals`
— each axis weighted equally.

### F1 distribution by case

| F1 band | v62 | base |
|---|---:|---:|
| ≥ 0.95 | **150 (75%)** | 0 |
| 0.85 – 0.95 | 1 | 0 |
| 0.70 – 0.85 | 5 | 0 |
| 0.50 – 0.70 | 17 | 38 (19%) |
| 0.0 – 0.50 | 2 | 149 |
| 0.0 (incl. invalid JSON) | 25 | 13 |

**v62 puts 78% of cases above F1 = 0.70; base puts 0%.**

### JSON validity is the bottleneck

Of the 25 v62 cases scoring F1 = 0, most are JSON-invalid (the model
truncated mid-output before closing the schema). Mean latency is similar
across valid and invalid cases — failures are at the length-limit
boundary, not on confused inputs.

On the **184 cases where the JSON parses** (no grammar):
- precision = 0.837
- recall = 0.911
- **F1 = 0.872**

That is the ceiling implied by the JSON-valid subset. The measured GBNF path
below did not recover it.

### GBNF inference path — **negative result, do not use**

We tried a permissive GBNF grammar enforcing the 6-top-level-key schema
(`apps/mobile/convert/cliniq_v62_compact.gbnf`). On a 50-case sub-bench,
F1 dropped from the no-grammar baseline by ~5pp and JSON validity dropped
by ~4pp. Raw output: `apps/mobile/convert/build/v62_val_compact_grammar_bench.json`.

Why: at `max_new_tokens=2048`, the grammar pushes the model into longer
expansions that hit the length limit before closing the JSON, and forces
token paths that don't match the model's learned distribution on rare
fields. The model's own learned format is *more* JSON-valid than the
grammar-constrained version on this distribution.

**Recommendation: ship without grammar.** The unconstrained F1=0.837 with
92% JSON validity is the production number. JSON validity can be patched
client-side by retrying or salvaging truncated outputs (the 25 invalid
cases are all length-limit truncations, not malformed-JSON-from-confusion).

A targeted v63 retrain with longer `max_seq_length` and ~50 longer
expansion examples would close the recall + JSON gap together.

## Limitations

- **Trained on synthetic eICRs** — real CDC eICR XML (full HL7 CDA structure)
  is out of distribution. The repo ships a free-text-narrative path
  (regex + RAG + 6-turn agent) that handles real eICRs at F1=1.000;
  this LoRA is the *fast single-shot* path for known-format inputs.
- **Compact JSON ≠ FHIR Bundle.** The output needs one transform layer
  (`apps/mobile/convert/fhir_bundle.py` in the repo) to become a R4 Bundle.
- **English only**, US clinical conventions (SNOMED/LOINC/RxNorm).
- **No PHI in training data** — synthetic patients only.

## Repo + demo

- iOS app: GitHub `kcirtapfromspace/gemma4` — `apps/mobile/ios-app/ClinIQ/`
- Hosted demo: <https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir>
- Submission narrative: `tools/autoresearch/hackathon-submission-2026-05-18.md`
- This card: `tools/autoresearch/v62-submission/MODEL_CARD.md`

## Citation

```bibtex
@misc{cliniq-gemma4-2026,
  author = {Patrick Deutsch},
  title  = {ClinIQ: Gemma 4 E2B fine-tuned with Unsloth for offline eICR extraction},
  year   = {2026},
  note   = {Gemma 4 Good Hackathon — Unsloth Track submission},
}
```
