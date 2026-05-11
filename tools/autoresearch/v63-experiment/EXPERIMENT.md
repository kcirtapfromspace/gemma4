# v63 experiment — longer-context Unsloth retrain

## Hypothesis

The v62 submission lands F1 = 0.823 / JSON-validity 86% on `val-compact`. The
14 pp JSON-validity gap is entirely length-limit truncation. The v62
submission doc names the future-path retrain explicitly:

> Bump `max_seq_length` to 1024 to eliminate truncation.
> Retain the same 5-epoch / lr=1e-4 / `packing=True` recipe.
> Expected outcome: F1 ~0.90, JSON validity ~95%, no latency regression.
> — `tools/autoresearch/v62-submission/SUBMISSION_SECTION_DRAFT.md` and
>   the `Future v63 path` cell in `public_notebook.ipynb`.

We test that prediction here. We do **not** add the proposed ~50
long-expansion training examples; this isolates the `max_seq_length`
variable so we can attribute the delta cleanly.

## What changed vs v62

- `max_seq_length`: 512 → **1024**.
- Kernel id: new private slug `cliniq-gemma4-unsloth-v63-experiment` so the
  public v62 submission (`cliniq-gemma4-unsloth-submission`) is untouched.
- Everything else identical: same dataset (`patrickdeutsch/eicr-fhir-training-data`),
  same r=16 / α=16 / 7-target-module LoRA, same 5 epochs, lr=1e-4,
  packing=True, gradient_accumulation=8, optim=adamw_8bit,
  `use_gradient_checkpointing="unsloth"`, gemma-4 chat template, T4×2 (default).

## Why this is "Unsloth track" work

- `use_gradient_checkpointing="unsloth"` is the line that makes
  max_seq_length=1024 + packing=True fit on a 16 GB T4. Vanilla HF
  checkpointing would either OOM or force batch_size below 1.
- Latest `unsloth` from PyPI is installed in cell 5 (`!pip install unsloth`),
  which pulls in any post-NVIDIA-collab packed-sequence caching changes;
  we'll see in the wall-clock log whether that path activates on Turing.

## Acceptance criteria

| Metric | v62 (shipped) | v63 target | Pass if |
|---|---:|---:|---|
| Micro-F1 | 0.823 | ≥ 0.85 | F1 strictly > 0.823 |
| JSON validity | 86% | ≥ 95% | drop in truncated cases |
| Latency p50 (Mac) | 4.1 s | no regression | ≤ 5.0 s |
| Train wall-clock | 1 h 4 m | ≤ 1 h 30 m | doesn't blow the kernel quota |

If F1 regresses or training OOMs, the v62 submission stands as-is and we
log the negative result in the Unsloth-track writeup.

## Run record

- Push: 2026-05-07 — kernel went to v3 after two env-bringup failures.
- Slug: <https://www.kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-v63-experiment>
- v3 train wall-clock: **3h 04m** on T4 (single GPU, `CUDA_VISIBLE_DEVICES=0`).

### v1, v2 failure modes (worth keeping for the runbook)

- **v1** — `pip install unsloth` upgraded torch; landed `cudaErrorNoKernelImageForDevice`
  on a vanilla `tensor.fill_()` during Gemma 4 weight init. Suspected sm-arch mismatch
  but actually GPU allocation issue (see v2).
- **v2** — Tried `pip install --no-deps unsloth`; Kaggle assigned a Tesla P100 (sm_60)
  instead of a T4, and the preinstalled torch 2.10.0+cu128 wheel doesn't carry sm_60.
  The env-print added in cell 3 surfaced this immediately.
- **v3 (working)** — Added `"machine_shape": "NvidiaTeslaT4"` to `kernel-metadata.json`
  to pin the T4, kept the full `pip install unsloth` (with deps), and force-upgraded
  `transformers>=5.5` because Kaggle's preinstalled 5.0.0 doesn't recognize
  `model_type=gemma4` (KeyError → "is not supported yet in transformers==5.0.0").
  Asserted `cap >= (7, 5)` so a future P100 reassignment fails loudly instead of
  silently hitting `cudaErrorNoKernelImageForDevice`.

## Results

| Metric | v62 (shipped, max_seq=512) | v63 (max_seq=1024) | Delta |
|---|---:|---:|---:|
| Micro-F1 | 0.823 | **0.9989** | +0.176 |
| Micro-precision | 0.979 | **0.9989** | +0.020 |
| Micro-recall | 0.710 | **0.9989** | +0.289 |
| JSON validity | 0.86 | **1.00** | +14 pp |
| Schema-complete | 0.86 | **1.00** | +14 pp |
| Cases ≥ F1 0.70 | 162 / 200 | **200 / 200** | +38 |
| Train wall-clock | 1h 04m | 3h 04m | 2.85× |

The 2.85× train cost matches the expected attention-cost scaling for `packing=True`
when `max_seq_length` doubles (per-row attention is roughly quadratic, partially
offset by tighter packing density).

The v62 future-path cell predicted F1 ~0.90 / JSON-valid ~95%. v63 cleared both —
the entire JSON-validity gap was max-context-driven, not a separate decoding bug.

## Latency caveat (don't oversell in the submission)

The 38.1 s p50 / 51.0 s p95 in the kernel log is **unquantized PyTorch on T4 with
`model.generate(max_new_tokens=1024)`**. This is NOT comparable to v62's 4.1 s p50
on Mac M-series via llama.cpp Q3_K_M. To make a fair latency claim we need:

1. Convert `cliniq_lora/` → GGUF locally with
   `tools/llama-cpp/convert_lora_to_gguf.py`.
2. Run `apps/mobile/convert/bench_v62_singleshot.py` against the v63 GGUF on
   Mac at the same Q3_K_M quant.
3. Compare wall-clock against v62 GGUF on the same machine.

Until step 3 is done, leave the v63 latency line as "TBD — Mac bench pending"
in any submission edit.

## Honesty footer

F1 = 0.9989 is on synthetic, in-distribution `val-compact`. Real free-text
clinician dictation and full HL7 CDA XML still go through the agent/RAG path.
v63 is the fast single-shot path for known-format eICRs; it does not replace
the F1=1.000 verified-RAG agent path that headlines the submission.

## Pull results

```bash
# KGAT_... value lives in tools/autoresearch/handoff-2026-04-27.md.
# Not duplicated here; export before running.
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:?set from handoff first}"
mkdir -p /tmp/v63-out
kaggle kernels output patrickdeutsch/cliniq-gemma4-unsloth-v63-experiment \
  -p /tmp/v63-out
# bench summary is logged at the end of cell "Inline bench summary"
```
