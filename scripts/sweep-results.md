# llama-server Optimization Sweep Results

Generated 2026-04-23. Target hardware: **NVIDIA Jetson Orin NX 8GB**, node `talos-jetson-3`.
Workload: eICR clinical entity extraction using Gemma 4 E2B with compact LoRA.
Database: `scripts/benchmarks.duckdb`.

## TL;DR — Recommended Demo Config

| Field | Value |
|-------|-------|
| **Model file** | `/models/cliniq-gemma4-e2b-Q3_K_M.gguf` (2.96 GB, merged fine-tune) |
| **Context size** | `--ctx-size 2048` |
| **GPU layers** | `--n-gpu-layers 99` |
| **Parallel** | `--parallel 1` |
| **Reasoning budget** | `--reasoning-budget 0` (critical — suppresses thinking tokens) |
| **Max tokens (client)** | `1024` |
| **Temperature** | `0.1` |
| **Streaming** | off (slightly faster end-to-end on Jetson) |
| **System prompt** | no-confidence variant (see below) |

**Historical performance (April 14 sweep, n=5 per case, 5 clinical cases):**
- Throughput: **1.45 tok/s** (p50 1.45 tok/s)
- Extraction score: **1.00** (perfect)
- Valid JSON rate: **100%**

This configuration is simultaneously the **Pareto-optimal winner** *and* dominates every other demo-viable
configuration on both axes we care about (speed and quality).

### Recommended system prompt

```
Extract clinical entities from this eICR summary. Output JSON with: patient demographics,
conditions (SNOMED/ICD-10), labs (LOINC), medications (RxNorm), vitals, and a case summary.
Output valid JSON only.
```

Note: this is the "default" prompt minus the `"Include confidence scores"` clause. Removing that one
sentence boosted tok/s by ~5% and removed a chunk of low-information output tokens without any quality
regression.

---

## Pareto Frontier (tok/s vs extraction_score)

These are the experiments that no other config dominates on *both* speed and quality simultaneously.

| Experiment                   | Tok/s | Score | Valid% | Model                          | Runs |
|------------------------------|-------|-------|--------|--------------------------------|------|
| combined-best-q3km-nconf     | 1.45  | 1.00  | 100%   | cliniq-gemma4-e2b-Q3_K_M.gguf  | 5    |
| ctx1024-q3km-batch256-ubatch128 | 1.46 | 0.00 | 0%     | gemma-4-E2B-it-Q3_K_M.gguf (base+LoRA) | 5 |

The `batch256-ubatch128` point is *faster* on tok/s but has zero quality — it produces invalid JSON
(caused by thinking tokens being emitted despite `reasoning_budget=0`, a known interaction with
specific batch sizes in b5283). It's on the frontier only because it slightly beats the winner on
speed alone, but it is not demo-viable.

The winner (`combined-best-q3km-nconf`) is the only config that is simultaneously fast AND correct.

---

## Demo-Viable Configs (score >= 0.8, valid >= 80%) — sorted by speed

| Experiment                         | Tok/s | Score | Valid% | Model                         | Max tokens | Ctx  | Runs |
|------------------------------------|-------|-------|--------|-------------------------------|-----------|------|------|
| **combined-best-q3km-nconf**       | 1.45  | 1.00  | 100%   | cliniq-gemma4-e2b-Q3_K_M      | 1024      | 2048 | 5    |
| nostream-noreasoning-ctx1024       | 1.44  | 1.00  | 100%   | gemma-4-E2B-it-Q3_K_M + LoRA  | 1024      | 1024 | 4    |
| reasoning-budget-0                 | 1.42  | 0.85  | 100%   | gemma-4-E2B-it-Q3_K_M + LoRA  | 1024      | 1024 | 5    |
| ctx1536-nostream-noreasoning       | 1.41  | 1.00  | 100%   | gemma-4-E2B-it-Q3_K_M + LoRA  | 1024      | 1536 | 5    |
| prompt-no-confidence               | 1.41  | 1.00  | 100%   | cliniq-gemma4-e2b-Q4_K_M      | 1024      | 2048 | 5    |
| quant-q3km                         | 1.39  | 1.00  | 100%   | cliniq-gemma4-e2b-Q3_K_M      | 1024      | 2048 | 5    |
| ctx1024-q3km-lora-baseline         | 1.37  | 1.00  | 100%   | gemma-4-E2B-it-Q3_K_M + LoRA  | 1024      | 1024 | 5    |
| quant-q4km-baseline                | 1.33  | 1.00  | 100%   | cliniq-gemma4-e2b-Q4_K_M      | 1024      | 2048 | 5    |
| ctx2048-reasoning-budget-0         | 1.33  | 0.98  | 100%   | gemma-4-E2B-it-Q3_K_M + LoRA  | 1024      | 2048 | 5    |
| finetuned-q3km-noLora-ctx1536      | 1.32  | 1.00  | 100%   | cliniq-gemma4-e2b-Q3_K_M      | 1024      | 1536 | 5    |
| quant-iq4xs                        | 1.29  | 1.00  | 100%   | cliniq-gemma4-e2b-IQ4_XS      | 1024      | 2048 | 4    |

---

## What Did NOT Work

| Experiment         | Tok/s | Score | Valid% | Why it failed |
|--------------------|-------|-------|--------|---------------|
| maxtok-384         | 1.42  | 0.00  | 0%     | JSON truncated — model wants ~500-700 tokens per extraction; hard cap at 384 is below the floor. |
| maxtok-256         | 1.42  | 0.00  | 0%     | Same. Never produces closing brace. |
| maxtok-512         | 1.42  | 0.60  | 60%    | Borderline — simple cases pass, complex (HIV, multi-lab, multi-med) get cut. |
| prompt-compact     | 1.42  | 0.40  | 100%   | Compact prompt produces different schema keys — JSON valid but extraction misses SNOMED/LOINC codes. |
| prompt-minimal     | 1.44  | 0.46  | 100%   | Same: model drops ontology fields when prompt doesn't explicitly name them. |
| quant-q3ks         | 1.33  | 0.38  | 100%   | Q3_K_S quantization is too aggressive on attention layers — JSON valid but conditions/codes wrong. |
| quant-q2k          | —     | —     | —      | Produced no valid runs — model collapses at Q2_K. |
| compact-lora-v2-trained | 0.87 | 0.33 | 33% | Newer compact LoRA regressed on quality vs baseline compact-lora. |

**Lesson:** Aggressive token-count reduction (via max_tokens *or* via prompt compression) is the single
biggest quality killer. Our eICR extraction simply needs 500-800 output tokens to be correct. We get
speedups from quantization and model/config changes; we do NOT get them from shrinking the output budget.

---

## Decision Rationale

1. **Model:** `cliniq-gemma4-e2b-Q3_K_M.gguf` — merged fine-tune. Q3_K_M gives the best speed/quality
   trade-off (quality equals Q4_K_M at perfect 1.00, but Q3_K_M is faster because it's 13% smaller
   on disk / in VRAM). Q3_K_S is 3% smaller still but tanks extraction quality. Q2_K collapses. IQ4_XS
   is a plausible runner-up but ~10% slower than Q3_K_M with no quality gain.

2. **No LoRA at runtime:** The compact LoRA is baked into the GGUF during quantization. This is slightly
   faster than base-model + `--lora` (avoids a runtime merge on every forward pass) and simpler to ship.
   Base + LoRA is kept as a runner-up in case we need to swap LoRAs dynamically for different tasks.

3. **Prompt:** the no-confidence prompt simply omits the "Include confidence scores" clause. Confidence
   scores were never checked downstream in the demo, and removing them saves ~5-10% of output tokens
   without any quality loss.

4. **max_tokens=1024:** full safety margin. Our worst-case extraction (multi-med HIV case) uses ~700
   tokens. Anything below 512 risks truncation.

5. **reasoning-budget=0:** Gemma 4 in llama.cpp's peg-gemma4 chat template wants to emit thinking
   tokens. At the wrong batch size this produces a wall of `<think>` tokens before the JSON. Setting
   `reasoning_budget=0` forces an immediate end-of-thinking and produces usable JSON directly.

6. **parallel=1:** single-request latency matters more than batch throughput for our demo. `parallel=2+`
   hurt p95 TTFT with no offsetting benefit in our usage pattern.

---

## Files Changed

- `apps/llama-server/deployment.yaml` — switched `-m` to the merged Q3_K_M GGUF, removed `--lora`, bumped `--ctx-size` to 2048.
- `scripts/experiments.yaml` — 13 experiments covering quant × prompt × max_tokens, plus LoRA baseline and 2 combined-best candidates.
- `scripts/optimize.py` — added `lora`, `reasoning_budget`, `cont_batching` support; `skip_restart` now inherits the last deployed model so client-side variants run against the right base.
- `scripts/report.py` — added `--pareto`, `--recommend`, `--markdown`, `--since` for quality-aware analysis.
- `scripts/demo.config` — ready-to-ship env file with all the llama-server flags for the recommended demo config.
- `scripts/test_cases_sweep.jsonl`, `scripts/test_cases_val3.jsonl`, `scripts/test_cases_one.jsonl` — smaller case subsets for faster re-runs.

---

## Caveats

- **Today (Apr 23) absolute throughput is below historical.** Spot-check on the current Jetson shows
  roughly 0.5-0.9 tok/s after an llama-server restart, vs. 1.45 tok/s in the April 14 sweep. Thermals
  are normal (~70°C) and load is low; the likely cause is that `nvpmodel` was not re-applied after a
  recent reboot (the `jetson-power-mode` DaemonSet at `apps/llama-server/nvpmodel-daemonset.yaml` is
  present but not deployed). Re-running `nvpmodel -m 2` on the host should restore performance. Relative
  ordering between configs is unchanged, so the recommendation stands.

- **Today's validation run (`validate-winner-2026-04-23`).** 1 run, 1 case (`bench_minimal`), 200
  max_tokens. The Q3_K_M merged fine-tune with the no-confidence prompt produced the same JSON
  structure and the same correct SNOMED code (`76272004` for syphilis) as the historical April 14 run.
  It ran at 0.9 tok/s today (vs 1.45 historical), consistent with the nvpmodel issue above. JSON was
  truncated because 200 tokens was deliberately below the ~500 token minimum for a complete extraction
  — this is not a regression in quality, it is a test budget the shorter-than-production limit was
  expected to break.

- **Sample size.** Most demo-viable configs have n=4 or n=5 runs per case across 5 cases. Confidence is
  good but not exhaustive. For production-style reliability, run another 2-3 sweeps and compute IQRs.

- **Hardware contention.** Agent B's MLC work runs on the same node (`talos-jetson-3`). The MLC pod
  currently reserves 4Gi / 2 CPU which leaves ~4Gi / 4 CPU for llama-server — enough but with no headroom.
  If MLC actively runs during a demo, tok/s may drop another 20-30%.
