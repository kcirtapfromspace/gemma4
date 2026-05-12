# Final-week publishing runbook

User-side runbook for the artifacts the harness can't push autonomously
(HF Hub uploads need a user-held HF write token; Kaggle public-notebook
flips need the user's account context). Execute these as a single block
when you're ready.

Status of artifacts at write time (2026-05-11):

| Artifact | State |
|---|---|
| v62 LoRA GGUF | local at `models/cliniq-gemma4-e2b-v62-lora.gguf` (committed-out via gitignore) |
| v62 PEFT safetensors | **not on disk** — original training output was in /tmp and is gone |
| v63 LoRA | local at `models/cliniq-gemma4-e2b-v63-lora/` and `.gguf`; **known Mac Q3_K_M regression**, not shipped |
| v63b LoRA | local at `models/cliniq-gemma4-e2b-v63b-lora/` and `.gguf`; Mac re-bench pending (see `apps/mobile/convert/build/v63b_val_compact_bench.json`) |
| HF Hub — v62 | repo doesn't exist yet (the v62-submission/HF_HUB_PUSH.md runbook was never executed) |
| HF Hub — v63 | repo doesn't exist; do NOT push (broken at Q3_K_M) |
| HF Hub — v63b | repo doesn't exist; push iff v63b Mac bench F1 >= 0.85 |
| Public Kaggle notebook (v62) | `cliniq-gemma4-unsloth-submission` slug not used; v62-submission/public_notebook.ipynb is prepared but never pushed |
| Public Kaggle notebook (v63b) | analogous — sanitize + push as `cliniq-gemma4-unsloth-v63b-submission` |
| HF Space hardware | live Space is on `cpu-basic`, needs flip to `zero-a10g` |

---

## Step 0 — Prereqs

```bash
# HF auth (interactive — generates a token at
# https://huggingface.co/settings/tokens, scope = write)
pip install -U "huggingface_hub[cli]"
huggingface-cli login

# Kaggle auth (already in env per handoff-2026-04-27.md, KGAT_...)
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN?: set from handoff-2026-04-27.md}"
```

---

## Step 1 — Flip the live HF Space hardware to ZeroGPU

This is a settings-page action; no CLI. The audit on 2026-05-11 found
the Space silently running on `cpu-basic` even though the README
frontmatter requests `zero-a10g`. HF doesn't auto-upgrade.

1. Open <https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir/settings>.
2. Scroll to "Space hardware".
3. Change from "CPU basic" → "ZeroGPU".
4. The Space rebuilds and restarts; first model load takes ~60–90 s.
5. Verify by visiting the Space URL: the Advanced accordion should show
   `Gemma 4 (gemma-4-E2B-it, ...) running in-process on ZeroGPU H200`.

This is independent of the LoRA work below; do it as soon as you're at a
keyboard.

---

## Step 2 — Push v62 LoRA GGUF to HF Hub

v62 is the deadline-safe Unsloth-track LoRA. The published artifact is
the Mac-deployable GGUF (the PEFT safetensors aren't on disk anymore;
they got cleaned out of /tmp). HF Hub supports GGUF distribution; the
model card lives at `tools/autoresearch/v62-submission/MODEL_CARD.md`.

```bash
cd /Users/thinkstudio/gemma4

# Create the repo (idempotent — succeeds whether or not the repo exists)
huggingface-cli repo create cliniq-gemma4-e2b-unsloth-v62 \
  --type model --yes || true

# Stage the model card as README.md in a temp dir to avoid moving the
# tracked file
WORK=$(mktemp -d)
cp tools/autoresearch/v62-submission/MODEL_CARD.md "$WORK/README.md"
cp models/cliniq-gemma4-e2b-v62-lora.gguf "$WORK/"

# Upload
huggingface-cli upload kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62 \
  "$WORK" / \
  --repo-type model \
  --commit-message "v62 LoRA GGUF + model card (Unsloth-track shipped)"

rm -rf "$WORK"
echo "Pushed: https://huggingface.co/kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62"
```

---

## Step 3 — Push v62 public Kaggle notebook

Only run this step if you're OK with the notebook re-running on Kaggle
with the latest Unsloth/transformers — which is the SAME version chain
that broke v63. The original v62 was trained 2026-04-30 against an older
chain. A fresh re-run today may not reproduce the v62 F1 = 0.837 number
exactly.

Safer alternative: edit `tools/autoresearch/v62-submission/public_notebook.ipynb`
to add the same `transformers>=5.5,<=5.5.0` pin v63b uses, then push.

```bash
cd /Users/thinkstudio/gemma4/tools/autoresearch/v62-submission

# Match the metadata's code_file to the actual ipynb filename
python3 -c "
import json
m = json.load(open('kernel-metadata.json'))
m['code_file'] = 'public_notebook.ipynb'
json.dump(m, open('kernel-metadata.json','w'), indent=2)
"

kaggle kernels push -p .
echo "Public Kaggle notebook (v62) status: https://www.kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-submission"
```

---

## Step 4 — Conditional on v63b Mac-bench result

Check `apps/mobile/convert/build/v63b_val_compact_bench.json` for the
micro_f1 number.

### Path A — v63b F1 >= 0.85 → ship v63b as the new primary

```bash
cd /Users/thinkstudio/gemma4

# Create v63b HF Hub repo
huggingface-cli repo create cliniq-gemma4-e2b-unsloth-v63b \
  --type model --yes || true

WORK=$(mktemp -d)
# Generate a v63b model card by adapting v62's
cp tools/autoresearch/v62-submission/MODEL_CARD.md "$WORK/README.md"
sed -i '' 's/v62/v63b/g; s/max_seq_length=512/max_seq_length=1024/g' "$WORK/README.md"
# Add the v63b-specific section
cat tools/autoresearch/v63b-experiment/EXPERIMENT.md >> "$WORK/README.md"

cp models/cliniq-gemma4-e2b-v63b-lora.gguf "$WORK/"
cp -r models/cliniq-gemma4-e2b-v63b-lora "$WORK/peft-adapter"

huggingface-cli upload kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v63b \
  "$WORK" / \
  --repo-type model \
  --commit-message "v63b LoRA GGUF + PEFT adapter + model card"

rm -rf "$WORK"

# Public Kaggle notebook for v63b — sanitize the v63b kernel first
# (drop the failing inline-bench cell that crashed on the multimodal
# tokenizer API change; replace with the save cell + a note pointing at
# the Mac re-bench JSON)
# ... see v63b-experiment/PUBLIC_NOTEBOOK_PLAN.md (TODO if v63b ships)
```

### Path B — v63b F1 < 0.85 → v62 stays shipped, document v63b as research

```bash
# No additional pushing. Update the submission narrative to note v63b's
# Mac result and the decision to ship v62. The relevant files are:
#  - tools/autoresearch/hackathon-submission-2026-05-18.md (final doc)
#  - tools/autoresearch/v63b-experiment/EXPERIMENT.md (full record)
#  - README.md (headline table)
# Update these in a commit titled "v63b: Mac re-bench result + ship
# decision".
```

---

## Step 5 — Final commit + tag + push to GitHub

```bash
cd /Users/thinkstudio/gemma4

# All doc edits from steps 2/3/4 above should be staged + committed
git add -A  # OR specific files
git status  # review

git commit -m "$(cat <<'EOF'
final hackathon submission convergence

[content depends on whether v63b shipped — see step 4]
EOF
)"

# Tag
git tag -a v1.0-hackathon-submission -m "Gemma 4 Good hackathon submission — 2026-05-18"

# Push branch + tag
git push origin main
git push origin v1.0-hackathon-submission
```

---

## Verification checklist (post-push)

- [ ] `https://huggingface.co/kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62` returns 200, model card renders, GGUF downloadable
- [ ] `https://www.kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-submission` is public and runs to completion (or shows the v62 results in the published output)
- [ ] (if Path A) `https://huggingface.co/kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v63b` returns 200
- [ ] HF Space hardware shows ZeroGPU on the Settings page
- [ ] Live Space agent-tier returns a valid Bundle on the COVID-19 demo sample
- [ ] iOS app builds clean on `iPhone17ProDemo` simulator with the seeded GGUF
- [ ] `tools/autoresearch/hackathon-submission-2026-05-18.md` matches the actual shipped state
- [ ] `git log --oneline origin/main..HEAD` returns empty (everything pushed)
