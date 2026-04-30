# Kaggle push runbook — Unsloth Track public submission

What this does: makes the training-data dataset public, then pushes
`tools/autoresearch/v62-submission/public_notebook.ipynb` as a **new public
kernel** under the slug `cliniq-gemma4-unsloth-submission`. Everything below
is run **by you** with **your** Kaggle auth — I do not push anything.

## 0. Prerequisites

```bash
# Confirm Kaggle CLI auth is your account, not a stale token.
kaggle config view
# Should print "username: patrickdeutsch" (or whichever account owns the
# eicr-fhir-training-data dataset and the cliniq-gemma4-finetune kernel).

# If not authed:
# - Generate a token at https://www.kaggle.com/settings/account → "Create New Token"
# - Drop kaggle.json at ~/.kaggle/kaggle.json
# - chmod 600 ~/.kaggle/kaggle.json
```

If you previously stored a KGAT token in `KAGGLE_API_TOKEN` instead of
`kaggle.json`, that's fine for the API but the CLI looks at `kaggle.json`
first; either is sufficient for the commands below.

## 1. Make `eicr-fhir-training-data` public

The dataset is currently `is_private: true`. The public notebook depends
on it being readable from a non-owner account.

```bash
mkdir -p /tmp/eicr-fhir-meta && cd /tmp/eicr-fhir-meta
kaggle datasets metadata patrickdeutsch/eicr-fhir-training-data
# Writes ./dataset-metadata.json
```

Open `dataset-metadata.json`, set:

```json
{
  "title": "eICR → FHIR Training Data",
  "id": "patrickdeutsch/eicr-fhir-training-data",
  "licenses": [{"name": "CC0-1.0"}],
  "isPrivate": false
}
```

Push the metadata change:

```bash
kaggle datasets metadata patrickdeutsch/eicr-fhir-training-data -p .
# OR if metadata-only update is rejected, push a no-op version:
kaggle datasets version -p . -m "Make public for hackathon submission"
```

Verify in an incognito window: <https://www.kaggle.com/datasets/patrickdeutsch/eicr-fhir-training-data>

If you see a 404 or "private" badge, the metadata edit didn't apply — go
to the dataset page in the UI → Settings → toggle "Public" directly.

## 2. Push the public notebook

The repo ships:

- `tools/autoresearch/v62-submission/public_notebook.ipynb` — the polished public version
- `tools/autoresearch/v62-submission/kernel-metadata.json` — see below

Create the kernel-metadata.json next to the notebook (run from the repo
root):

```bash
mkdir -p /tmp/cliniq-public-kernel
cp tools/autoresearch/v62-submission/public_notebook.ipynb \
   /tmp/cliniq-public-kernel/cliniq-gemma4-unsloth-submission.ipynb

cat > /tmp/cliniq-public-kernel/kernel-metadata.json <<'EOF'
{
  "id": "patrickdeutsch/cliniq-gemma4-unsloth-submission",
  "title": "cliniq-gemma4-unsloth-submission",
  "code_file": "cliniq-gemma4-unsloth-submission.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "patrickdeutsch/eicr-fhir-training-data"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF
```

Push it:

```bash
kaggle kernels push -p /tmp/cliniq-public-kernel
```

The first push creates the kernel as v1. To upload an updated draft later,
re-run the same command (v2, v3, ...). The kernel slug is fixed to
`patrickdeutsch/cliniq-gemma4-unsloth-submission`; rename the slug only by
changing `id` in `kernel-metadata.json` and pushing fresh.

**One-liner (after editing dataset to public):**

```bash
kaggle kernels push -p /tmp/cliniq-public-kernel
```

## 3. Run the notebook on Kaggle

Kaggle accepts the upload but **does not auto-run** it. You need one
clean run on Kaggle hardware so the Output tab has the LoRA adapter and
the inline bench numbers.

```bash
# Trigger a run via API (T4 x2, internet on, ~1h 4m + ~25 min for inline bench).
kaggle kernels status patrickdeutsch/cliniq-gemma4-unsloth-submission
# In the UI: open the kernel, click "Save Version" → "Save & Run All".
```

If you want to skip the inline bench during the first run (faster
turnaround), edit cell 19 (`# Inline bench`) and set `BENCH_LIMIT = 25`
before clicking Run All; bump it to `None` for the final submission run.

## 4. Verify public access

```bash
# In an incognito window:
open "https://www.kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-submission"
```

Should load without a Kaggle login. If the page says "private" or 404s,
re-check `is_private: false` in `kernel-metadata.json` and re-push.

## 5. Cross-link the public URL

Once the public URL is confirmed, replace `<PUBLIC_KAGGLE_URL>` in:

- `tools/autoresearch/hackathon-submission-2026-04-27.md`
- `tools/autoresearch/v62-submission/SUBMISSION_SECTION_DRAFT.md`
- The repo root `README.md` (if not yet linked)
- The HF model card "training notebook" section
- The hackathon submission form

## Known blockers / things to check

- **Unsloth API drift.** The notebook uses `FastLanguageModel.from_pretrained`,
  `FastLanguageModel.get_peft_model`, `FastLanguageModel.for_inference`, and
  `unsloth.chat_templates.get_chat_template`. As of 2026-04-15 these are
  stable; if Kaggle's `pip install unsloth` lands a newer release that
  removed any of them, pin the version in cell 4:
  `!pip install "unsloth==2026.4.x"`.
- **Kaggle T4 disk pressure.** The original kernel notes "GGUF export
  skipped — T4 doesn't have enough disk space for merge+export." We keep
  GGUF conversion off the kernel; do it locally with
  `scripts/convert_lora_to_gguf.py`.
- **`val-compact.jsonl` location.** The inline-bench cell looks for
  `{DATASET_PATH}/val-compact.jsonl` first, then falls back to
  `val.jsonl`. The Kaggle dataset must include both for the bench to
  match the offline numbers — verify before pushing the dataset to public.
