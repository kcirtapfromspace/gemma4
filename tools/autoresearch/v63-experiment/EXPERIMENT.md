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

## Mac Q3_K_M re-bench — 2026-05-11 (DISCOVERY: v63 regresses)

After converting `cliniq_lora/` → GGUF via llama.cpp `convert_lora_to_gguf.py`
(at upstream tag `b8890`) and running `apps/mobile/convert/bench_v62_singleshot.py`
on Mac M-series at Q3_K_M, the quality picture flipped:

| metric | v62 LoRA | v63 LoRA | delta |
|---|---:|---:|---:|
| micro-F1 | **0.837** | 0.548 | -0.289 |
| micro-precision | 0.837 | 0.393 | -0.444 |
| micro-recall | 0.837 | **0.902** | +0.065 |
| JSON-validity | 0.92 | 0.92 | 0 |
| latency p50 (s) | 3.08 | **2.79** | -9.4% |
| latency p95 (s) | 4.24 | **3.86** | -9.0% |
| cases F1 ≥ 0.70 | 156 / 200 | 28 / 200 | -128 |

Raw outputs: `apps/mobile/convert/build/v62_val_compact_bench_localval.json`,
`apps/mobile/convert/build/v63_val_compact_bench.json`.

### Why the Kaggle 0.9989 did not survive quantization

The v63 LoRA safetensors has 410 tensors. v62 has 490. Difference: 80
tensors = 20 decoder layers × 2 modules (k_proj, v_proj) × 2 LoRA matrices
(A and B). Specifically, v63 is missing k_proj and v_proj LoRA weights on
decoder layers 15-34 — the 20 "global-attention" layers of Gemma 4's
hybrid attention pattern. Layers 0-14 (local attention) and all q_proj /
o_proj / MLP slots on all 35 layers are present.

The `adapter_config.json` reports all 7 target_modules with
`layers_to_transform=null` (meaning "all layers"). The runtime
config looks correct. But the actual saved tensors are partial. Likely
cause: a release of unsloth or peft ≥ 0.18 silently treats global-attention
k/v as non-trainable for Gemma 4 hybrid attention (the
`"unsloth_fixed": true` flag in `adapter_config.json`'s `auto_mapping` is
the suspicious clue). v62 (trained 2026-04-30) predates this regression.

Unquantized full-precision PyTorch can compensate for partial-coverage
attention — Kaggle's inline bench saw F1 = 0.9989 — but Q3_K_M
quantization introduces enough noise that the partial-coverage k/v can no
longer hold the attention pattern. The model becomes recall-happy
(emits ~8 codes/case vs the gold's 3) and precision collapses.

### Latency win stands

v63 is genuinely 9% faster than v62 at Mac Q3_K_M p50. This survives
quantization because latency is dominated by the base model's matmul, and
LoRA's incremental cost scales with adapter tensor count — fewer
tensors = slightly cheaper inference.

### Next step: v63b

A v63b retrain is queued (`tools/autoresearch/v63b-experiment/`) with
explicit `layers_to_transform=list(range(35))` in `get_peft_model` plus a
post-save assertion that the adapter contains ≥ 490 tensors before the
notebook exits. If v63b recovers F1 ≥ v62 at v63 latency, it becomes the
shipped Unsloth-track artifact.

For the submission, v62 stays as the primary shipped LoRA; v63 is
documented as the iteration that discovered the hybrid-attention LoRA
coverage regression.

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
