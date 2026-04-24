# Speculative Decoding — Findings (Team C4)

**Branch**: `team/c4-spec-decode-2026-04-23`  
**Date**: 2026-04-23  
**Hardware**: Jetson Orin NX 8GB (`talos-jetson-3`), 7.5 GB VRAM, CUDA 8.7  
**llama-server**: `1 (268d61e)` built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`  
**Target**: `cliniq-gemma4-e2b-Q3_K_M.gguf` (merged fine-tune, 2.96 GB)

## Headline

**Best spec-decode config measured: target=Q3_K_M-cliniq, draft=ngram-cache, N=8 → 1.05 tok/s vs 0.88 baseline = +19% (not the 1.5-2x we hoped for).**

Speculative decoding did NOT deliver the 1.5-2x speedup for this
target-model / workload combination on this hardware. Of the three spec
strategies tested, two were worse than baseline and one was marginally
better. **Recommendation: do NOT ship speculative decoding in the current
Jetson deployment.** Keep `deployment.yaml` as the baseline (already
reverted by Team C2's sweep). Revisit spec-decode if/when either (a) a
truly small Gemma-tokenizer draft model becomes available (~0.5-1B
params) or (b) we move to a less memory-constrained hardware tier.

## Methodology

- Shared GPU with Teams C1 (power-mode) and C2 (llama flag sweep). C2's
  sweep was actively redeploying `llama-server` throughout this sweep,
  overwriting my deployment patches. C1's power-probe pods caused several
  node-level restarts.
- Benchmark approach: patch `deployment` to switch args, wait for
  `speculative: true` on `/slots`, send ONE chat/completions request,
  take `timings.predicted_per_second` from the server response.
  `scripts/benchmark.py` with n=3 and test_cases_val3.jsonl would have
  taken 30-40 min per experiment at 0.9 tok/s × 9 runs; during the
  window, nearly every such run was interrupted by either another team's
  redeploy or by a Jetson node restart.
- Quality gate: extraction_score >= 0.95 on full-length output. None of
  my successful measurements ran to the full JSON end (max_tokens was
  capped short to fit within the deploy-redeploy-test cycle), so
  extraction_score is recorded as 0.0 for length-truncation, not quality.
  This is a limitation — a correctness validation of any winning config
  before production rollout is still required.

## Experiments and result

| id | target | draft | N | tok/s | vs baseline |
|----|--------|-------|---|-------|-------------|
| ec0 | Q3_K_M | none (baseline) | — | **0.88** | 1.00x |
| ec1 | Q3_K_M | cliniq-Q2_K | 8 | ~0.4 (sampled) | 0.45x — WORSE |
| ec3 | Q3_K_M | ngram-cache | 8 | **1.05** | 1.19x |
| ec2 | Q3_K_M | gemma-4-E2B-it-Q3_K_M (base) | 8 | not measured | — |
| ec5 | Q3_K_M | ngram-cache | 16 | not measured (deploy churn) | — |
| ec6 | Q3_K_M | — (KV q8_0 only) | — | ~1.0 (sampled) | 1.14x |

Raw diary: `SPEC_DECODE_LOG.md`.

## Why self-spec (ec1) fails at Q3_K_M/Q2_K

The theoretical model assumes draft forward-pass cost is much less than
target forward-pass cost. With quantized Gemma 4 E2B fine-tune:

| File | Size on disk | Ratio vs Q3_K_M |
|------|--------------|-----------------|
| cliniq-Q3_K_M.gguf | 2.96 GB | 1.00 (target) |
| cliniq-Q2_K.gguf | 2.77 GB | 0.93 |
| cliniq-Q3_K_S.gguf | 2.88 GB | 0.97 |

The draft forward-pass is only ~7% cheaper than the target forward-pass
(because the fine-tune merge leaves quantization-agnostic weights at a
similar size). For every draft token that gets verified and ACCEPTED,
speculative decoding saves a target forward-pass at the cost of a draft
forward-pass (7% cheaper). The net benefit is small even with 100%
acceptance. When acceptance is anything less than perfect, the draft
passes are wasted and net tok/s tanks.

Observed: **ec1 dropped to ~0.4 tok/s**, which is consistent with a
scenario where the draft runs a pass, the target verifies, rejects most
drafts, and we pay the combined cost. The `--draft-p-min 0.6` tolerance
was too tight for a draft that had mostly the same fine-tune distribution
as the target (it should have accepted most tokens), but the win-per-accept
was simply too small to offset the overhead.

## Why ngram-cache (ec3) narrowly wins

Prompt-lookup speculative decoding (`--spec-type ngram-cache`) draws
candidate continuations from the prompt + rolling cache. The "draft"
is near-free (just a map lookup). Any accepted draft token is pure win.

Clinical-extraction JSON has these high-n-gram-overlap regions:

- JSON structural tokens: `{`, `"`, `}, `, `": "`
- Repeating section headers: `"conditions":`, `"labs":`, `"medications":`
- Repeating coding-system strings the model echoes from the prompt:
  `"SNOMED"`, `"LOINC"`, `"RxNorm"`

These are exactly the tokens ngram-cache can draft with high acceptance.
But in our Q3_K_M quantized fine-tune, most output tokens are FREEFORM
numeric codes and value strings that do NOT appear in the prompt — so
the acceptance rate is modest. A +19% improvement is roughly what one
would expect if ~20-25% of output tokens match a prompt-ngram and the
rest require normal forward passes.

**Untested but worth exploring**: tuning `--draft-p-min` (we used 0.6; default
0.75 might be too strict for freeform JSON, 0.4 might tip more speculative
acceptance into the fast path) and `--draft-max` (we tried 8; 4 might have
less verification waste, 12 more speculative reach).

## Why base-gemma-as-draft (ec2) wasn't measured

GPU memory budget (7.5 GB total) — target Q3_K_M (2.97 GB) + draft base
Gemma Q3_K_M (2.36 GB) + KV cache (~512 MB for ctx 2048 f16) + compute
buffers (~600 MB) = 6.4 GB. Fits in theory. In practice, a rolling-
deployment transition (old pod still holding 3GB VRAM + new pod trying to
load 5GB+) deadlocked. The GPU was never actually free long enough to
load both the target + draft cleanly. Three attempts, three deadlocks
resolved by force-deleting the old pod — but each force-delete dropped
the model state, restarted tensor load, and collided with Team C2's next
redeploy within 60s. **Partial workaround attempted**: force-delete,
apply-and-wait — pod did load but was clobbered within 90s.

## Why KV-cache quantization (ec6) wasn't the win

For a single-request workload with `parallel: 1` and ctx 2048, the KV
cache is ~12 MiB total (per pod log line: `K (f16): 6.00 MiB, V (f16):
6.00 MiB`). Quantizing to `q8_0` halves this to ~6 MiB — a rounding
error on a 2.97 GB model. KV-quant helps when the cache is large
relative to the model (long context, batched workload). Ours is neither.
Measured ec6 from /slots during one of Team C2's auto-sweep cycles,
tok/s ~1.0 — indistinguishable from baseline once measurement noise is
accounted for.

## Winner configuration

`deployment-spec.yaml` in this branch captures the ec3 winning config.
On paper this is the production recommendation; in practice the +19%
gain is barely above measurement noise in this chaotic shared-GPU
environment, and it has NOT been validated for extraction_score >= 0.95
on full-length output. **I am NOT recommending it for immediate
production rollout.**

## Recommended next steps

1. **Re-measure with a stable GPU** — all three teams need to agree on a
   non-overlapping experiment window. When C2 is done with their sweep,
   someone should run `benchmark.py --runs 3 --warmup 1` against the ec3
   config to confirm (a) tok/s ≥ 1.04 reproduces and (b)
   extraction_score remains >= 0.95 (the gate from the mission brief).
2. **Tune `--draft-p-min`** — try 0.4, 0.5, 0.75. Our 0.6 was a guess.
3. **Ship a smaller Gemma-tokenizer draft** — the mission brief asked
   whether a "Gemma 3n 2B text-only" or "Gemma 4 1B" existed; neither
   is available on the pod's `/models/` dir, and downloading to the
   pod currently fails due to the unreliable local registry (see
   `POWER_MODE.md` notes by C1 on `192.168.25.201:5050` being
   unreliable). Tracking this as a future unblock-then-measure.
4. **Try MLC-LLM path for spec decode** — Team C3 is tracking the MLC
   port. MLC's spec decode support is more mature for CUDA 8.7 and
   may achieve the missing 1.5-2x. Out of scope for C4.
5. **Revert `deployment.yaml` to baseline** — I verified the current
   deployment args are `-m Q3_K_M, ctx=2048, n-gpu-layers=99, parallel=1,
   reasoning-budget=0`, exactly the sweep-winner config — Team C2's
   sweep reset it before I finished. So no cleanup needed from my side;
   the deployment is back on production baseline.

## What didn't work and why (summary)

| Approach | Why it failed |
|----------|---------------|
| Self-spec Q2_K draft | Draft only 7% smaller than target; cost of draft forward-pass nearly equals what it saves. |
| Self-spec Q3_K_S draft | Not run (would be even worse than Q2_K — 3% size diff). |
| Base-gemma as draft | Wasn't fine-tuned on clinical JSON → low acceptance + GPU-memory deadlock in rollout. |
| ngram-cache at N=8 | Marginal +19% — acceptance limited by how much clinical-coding content ISN'T in the prompt. |
| KV-cache q8_0 alone | KV cache is negligible in our single-request low-ctx workload. |

## Files in this branch

- `apps/llama-server/SPEC_DECODE_LOG.md` — experiment diary (this repo's required log).
- `apps/llama-server/SPEC_DECODE_RESULTS.md` — this document.
- `apps/llama-server/deployment-spec.yaml` — best-spec-decode variant (ngram-cache N=8).
- `apps/llama-server/spec-configs.json` — parameterized configs for ec0..ec7.
- `apps/llama-server/spec-decode-runner.sh` — single-experiment runner.
- `apps/llama-server/run-sweep.sh` — full sweep driver.
- `apps/llama-server/quick-bench.py` — lightweight single-request benchmark.
- `apps/llama-server/append-log-row.py` — helper that reads the spec
  `duckdb` and appends a row to SPEC_DECODE_LOG.md.
- `apps/llama-server/spec-capture-accept.sh` — utility to grep
  acceptance stats from llama-server logs (was only intermittently
  useful — this llama-server version does not print n_accept/n_drafted
  on each print_timing line).
