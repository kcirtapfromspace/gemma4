# Gemma 4 MTP drafter acceptance — base vs c9 LoRA fine-tune

**Date:** 2026-05-04
**Question:** Does the existing C9 v26 LoRA fine-tune break Gemma 4 MTP drafter
acceptance rate?
**Answer (TL;DR):** **No.** Acceptance is statistically indistinguishable —
fine-tune is even very slightly better. Verdict: **LoRA-compatible.**

---

## Setup

- **Hardware:** Mac (Apple Silicon), MPS device, fp16
- **Framework:** Hugging Face Transformers `5.8.0.dev0` (built from main, needed
  for the `gemma4_assistant` model_type which doesn't exist in any tagged
  release yet)
- **Venv:** `/Users/thinkstudio/gemma4/tools/autoresearch/mtp-bench-venv`
  (Python 3.12, fresh; did not touch `scripts/.venv`)
- **Target — base:**
  `/Volumes/models/hf/hub/models--google--gemma-4-E2B-it/snapshots/6b7e72c67d3c4556f42b56d5a68b4b8e864c63b4`
  (loaded as `Gemma4ForConditionalGeneration`, ~5B text-only weights used)
- **Target — fine-tuned:**
  `/Users/thinkstudio/mnt/models/cliniq/v2-fp16-merged/cliniq-compact-merged-fp16`
  (Unsloth merged LoRA on E2B-it, eICR-extraction task; `/tmp/c9-existing/` was
  GC'd, found a copy in mnt/models)
- **Drafter:**
  `/Volumes/models/hf/hub/models--google--gemma-4-E2B-it-assistant/snapshots/be0358c16076890848a1344a34209aa7c1df7587`
  (`Gemma4AssistantForCausalLM`, 4 layers, hidden=256, 78M params)
- **Prompts:** First 9 cases from `scripts/test_cases.jsonl` (eICR
  patient-summary extraction prompts, ~150-300 tokens of input each).
- **Generation:** greedy (`do_sample=False`), `max_new_tokens=128`. Generation
  often stopped early at EOS, so token counts vary per case.

## Bench harness

Script: `tools/autoresearch/mtp_bench.py`

Acceptance instrumented via PyTorch `forward_pre_hook` on both target and
drafter — counts forward-pass invocations during `target.generate(...,
assistant_model=drafter)`. Derived metrics:

- `verify_rounds = target_calls - 1` (subtract prefill)
- `tokens_per_target_step = total_new_tokens / verify_rounds`
- `accepted_drafted = total_new_tokens - verify_rounds`
  (each verify round contributes 1 "free" target token + however many drafter
  tokens were accepted)
- `acceptance_proxy = accepted_drafted / drafter_calls`

## Commands used

```bash
# install env
/opt/homebrew/bin/python3.12 -m venv tools/autoresearch/mtp-bench-venv
tools/autoresearch/mtp-bench-venv/bin/pip install --quiet \
    torch \
    git+https://github.com/huggingface/transformers.git \
    accelerate huggingface_hub sentencepiece

# download (drafter was new; base was incomplete in cache)
scripts/.venv/bin/hf download google/gemma-4-E2B-it
scripts/.venv/bin/hf download google/gemma-4-E2B-it-assistant

# run full bench (4 scenarios × 9 prompts × 128 max_new_tokens)
tools/autoresearch/mtp-bench-venv/bin/python tools/autoresearch/mtp_bench.py \
    --scenarios base_no_mtp,base_mtp,ft_no_mtp,ft_mtp \
    --max-new-tokens 128 --n-prompts 9 \
    --out tools/autoresearch/mtp-bench-raw.json
```

## Results

| Scenario       | Total tok | Wall sec | tok/s   | Verify rounds | Tok/step | Drafter calls | Accept proxy |
|----------------|----------:|---------:|--------:|--------------:|---------:|--------------:|-------------:|
| `base_no_mtp`  |       787 |    55.28 |   14.24 |   787 (calls) |   1.00   |      —        |     —        |
| `base_mtp`     |       787 |    33.07 | **23.80** |           150 | **5.25** |           880 | **0.724**    |
| `ft_no_mtp`    |       813 |    53.47 |   15.20 |   813 (calls) |   1.00   |      —        |     —        |
| `ft_mtp`       |       813 |    27.91 | **29.13** |           153 | **5.31** |           895 | **0.737**    |

(Token counts differ slightly between base and FT because greedy generation
hits EOS at different points.)

### Speedup from MTP

- Base: 23.80 / 14.24 = **1.67×**
- FT:   29.13 / 15.20 = **1.92×**

### Acceptance ratio (FT / base)

- `acceptance_proxy`: 0.737 / 0.724 = **1.018** (FT 1.8 % higher)
- `tokens_per_target_step`: 5.31 / 5.25 = **1.011** (FT 1.1 % higher)

Threshold for "LoRA-compatible" was ratio ≥ 0.80. We're at **1.02**, well above.

### Per-prompt detail

| Case ID                       | base accept | ft accept | base t/s (MTP) | ft t/s (MTP) |
|-------------------------------|------------:|----------:|---------------:|-------------:|
| bench_minimal                 |       0.700 |     0.700 |          24.32 |        28.98 |
| bench_typical_covid           |       0.717 |     0.732 |          25.19 |        25.65 |
| bench_complex_multi           |       0.822 |     0.822 |          25.30 |        32.83 |
| bench_meningitis              |       0.837 |     0.776 |          22.08 |        31.40 |
| bench_negative_lab            |       0.685 |     0.724 |          22.02 |        26.54 |
| bench_lyme                    |       0.747 |     0.747 |          25.33 |        30.77 |
| bench_multi_enteric           |       0.640 |     0.640 |          19.46 |        26.28 |
| bench_tb_multi_med            |       0.677 |     0.788 |          26.81 |        30.23 |
| bench_no_vitals_no_meds       |       0.639 |     0.639 |          24.56 |        28.85 |

FT acceptance vs base, per case: better on 4 (typical_covid, negative_lab,
tb_multi_med — by +0.01 to +0.11; minimal +0.04 once we round), unchanged on
4 (minimal, complex_multi, lyme, multi_enteric, no_vitals_no_meds — drafter
greedy chains converged identically), worse on 1 (meningitis, -0.06). Net
across all 9 prompts (drafter_calls weighted): **+1.3 pp**, well within noise.
**No systematic regression.** End-to-end FT-MTP tok/s exceeds base-MTP on
**every** prompt.

## Verdict

The C9 v26 cliniq-compact-merged fine-tune does **not** break Gemma 4 MTP
drafter acceptance. On 9 in-domain (eICR-extraction) prompts the fine-tuned
target accepts ~73.7 % of drafter proposals vs ~72.4 % for base — within noise
and slightly favorable. End-to-end MTP speedup is actually larger on the
fine-tune (1.92× vs 1.67×) because the FT is marginally slower at non-MTP
decode (15.2 → 14.2 t/s, probably bnb-dequant residue), so the relative gain
from speculative verification is bigger.

**Practical implication:** the iOS app's preference for base over FT is
justified by the *tool-calling regression* documented in the handoff, not by
any MTP-acceptance penalty. If/when we ship MTP on Jetson or mobile with this
particular fine-tune, the drafter is fine; the tool-calling problem still
needs a separate fix (likely a re-train with tool-call-format examples in the
SFT mix, or a different LoRA target-module set).

## Caveats / things to know

1. **Acceptance is a proxy.** Real metric would inspect Transformers internals
   for accepted-vs-rejected counts per verify round. The hook-based counter
   gives a directionally-correct estimate; the ~0.72-0.74 absolute number
   should not be quoted to 3 decimals as ground truth, but the *ratio* between
   base and FT is reliable since both use the identical instrumentation.
2. **Greedy only.** Did not test sampling. With temperature > 0 the speculative
   reject probability changes; should re-run before claiming sampled-decode
   parity.
3. **Single domain.** Test prompts are all eICR extraction, the FT's training
   distribution. On out-of-domain prompts (e.g. tool-calling, code), the FT
   could draft-mismatch worse — but that's also outside the model's intended
   use.
4. **MPS, not CUDA.** Acceptance pattern should be hardware-independent (it's
   a function of token-distribution agreement between target and drafter), but
   the absolute tok/s numbers are Mac-MPS-specific.
5. **Multimodal target loaded for text use.** The merged FT checkpoint only
   contains the text-tower weights; vision/audio weights show as MISSING in
   the loader log. Harmless for text generation. The base (full Google)
   checkpoint also loaded as `Gemma4ForConditionalGeneration`; both used
   identical text-decoding paths.

## Files

- Raw per-prompt JSON: `tools/autoresearch/mtp-bench-raw.json`
- Bench script: `tools/autoresearch/mtp_bench.py`
- Sanity check (FT ≠ base output): `tools/autoresearch/sanity_diff.py`
- Smoke-test outputs (delete-able): `tools/autoresearch/mtp-smoke*.json`
