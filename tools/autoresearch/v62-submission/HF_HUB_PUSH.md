# HuggingFace Hub push runbook — Unsloth Track LoRA

What this does: uploads the v62 LoRA adapter and the polished model card
to HuggingFace Hub at
`kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62`.

You run these commands with **your** HF auth — I do not push anything.

## 0. What gets uploaded

From `/tmp/v62-output/cliniq_lora/` (149 MB on disk):

```
adapter_config.json         1.2 KB
adapter_model.safetensors   124 MB   ← the LoRA weights
chat_template.jinja         1.5 KB
processor_config.json       1.7 KB
README.md                   5.2 KB   ← original training-time card (will be REPLACED)
tokenizer_config.json       2.7 KB
tokenizer.json              32 MB
```

Plus the polished model card from this submission:

```
tools/autoresearch/v62-submission/MODEL_CARD.md  ← becomes the Hub README.md
```

## 1. Authenticate

```bash
# Install the CLI if you don't have it.
pip install -U "huggingface_hub[cli]"

# Log in interactively. Generates a token at https://huggingface.co/settings/tokens
huggingface-cli login
# Paste a token with "write" scope.
```

Verify:

```bash
huggingface-cli whoami
# Should print:
#   user: kcirtapfromspace
#   ...
```

If `whoami` shows a different namespace, edit the repo path below to
match (e.g., `patrickdeutsch/cliniq-gemma4-e2b-unsloth-v62`).

## 2. Create the repo (one-time)

```bash
huggingface-cli repo create \
  cliniq-gemma4-e2b-unsloth-v62 \
  --type model \
  --organization kcirtapfromspace
```

If `kcirtapfromspace` is a personal namespace and not an org, drop
`--organization` and create under your user:

```bash
huggingface-cli repo create cliniq-gemma4-e2b-unsloth-v62 --type model
```

The CLI will print the canonical repo URL — confirm it matches what you
expect before continuing.

## 3. Upload the polished model card as README.md

The `cliniq_lora/` directory ships with an Unsloth-generated training-time
README that we want to replace with the curated submission card.

```bash
huggingface-cli upload \
  kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62 \
  tools/autoresearch/v62-submission/MODEL_CARD.md \
  README.md \
  --repo-type model \
  --commit-message "Add curated submission model card"
```

The third positional argument (`README.md`) is the **destination filename
on the Hub**. The CLI uploads the local MODEL_CARD.md to the Hub root as
`README.md`, which is what HF renders on the model page.

## 4. Upload the adapter + tokenizer files

```bash
huggingface-cli upload \
  kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62 \
  /tmp/v62-output/cliniq_lora \
  . \
  --repo-type model \
  --exclude README.md \
  --commit-message "Upload v62 LoRA adapter (124 MB) + tokenizer"
```

Notes:

- The `.` second-positional arg is the **destination prefix on the Hub** —
  files land at the repo root, preserving filenames.
- `--exclude README.md` keeps the curated card from step 3 from being
  overwritten by the training-time README inside `cliniq_lora/`.
- `adapter_model.safetensors` is 124 MB — well under the 5 GB single-file
  limit, no LFS gymnastics needed; HF auto-tracks `.safetensors` as LFS.

**One-liner (assumes step 3 already ran):**

```bash
huggingface-cli upload kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62 /tmp/v62-output/cliniq_lora . --repo-type model --exclude README.md --commit-message "Upload v62 LoRA adapter"
```

## 5. Verify

```bash
# Repo metadata
huggingface-cli repo info kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62

# In a browser (no login required for public repos):
open "https://huggingface.co/kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62"
```

Confirm:

- The model page renders the curated card (TL;DR table at the top).
- "Files and versions" lists `adapter_model.safetensors` (124 MB),
  `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `chat_template.jinja`, `processor_config.json`.
- The card's `base_model: unsloth/gemma-4-E2B-it` link resolves.
- The card's `datasets: patrickdeutsch/eicr-fhir-training-data` link
  resolves (this is a **Kaggle** dataset, not an HF dataset — fine, the
  link in the body of the card points to the right place).

## 6. Smoke test the adapter loads

```python
# In any environment with peft + transformers:
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/gemma-4-E2B-it")
tok = AutoTokenizer.from_pretrained("kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62")
model = PeftModel.from_pretrained(base, "kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62")

prompt = tok.apply_chat_template(
    [
        {"role": "system",
         "content": "Extract clinical entities from this eICR. Output compact JSON with: patient, encounter, conditions (SNOMED), labs (LOINC), meds (RxNorm), vitals. No summary. Valid JSON only."},
        {"role": "user",
         "content": "Patient: Maria Garcia\nGender: F\nDOB: 1985-06-14\nDx: COVID-19 (SNOMED 840539006)\nLab: SARS-CoV-2 RNA (LOINC 94500-6) - Detected"},
    ],
    tokenize=True, add_generation_prompt=True, return_tensors="pt",
)
out = model.generate(prompt, max_new_tokens=512, temperature=0.0)
print(tok.decode(out[0][prompt.shape[-1]:], skip_special_tokens=True))
```

Should produce a compact JSON with `patient`, `conditions`, `labs`. If it
returns long-form prose, the chat template didn't apply — check that
`chat_template.jinja` was uploaded.

## 7. Cross-link

Replace `<HF_HUB_URL>` in:

- `tools/autoresearch/hackathon-submission-2026-04-27.md`
- `tools/autoresearch/v62-submission/SUBMISSION_SECTION_DRAFT.md`
- The Kaggle public notebook's "Artifacts" section (cell 24, "## 11. Limitations…")
- The repo root `README.md`
- The hackathon submission form

## Known blockers / things to check

- **Quota / file count.** HF model repos default to ~50k files; we ship
  7. No issue.
- **`tokenizer.json` is 32 MB.** Auto-tracked as LFS, no manual config.
- **`base_model` field in card frontmatter.** Set to
  `unsloth/gemma-4-E2B-it`. If that repo gates require accepting Gemma
  Terms of Use, downstream loaders need the same — note this in the
  submission narrative if it bites.
- **License field.** Card declares `license: apache-2.0` for the LoRA
  weights; base model is governed by the Gemma Terms of Use. HF may
  prompt you to acknowledge the dual-license framing on first push.
