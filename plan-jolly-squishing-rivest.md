# Plan: Systematic Speed vs Accuracy Optimization Sweep

## Status (updated 2026-04-23)

**RESOLVED**. The sweep has been executed (originally on April 14 across 13 experiments × 5 runs × 5 cases) and the winner has been revalidated today.

- **Recommended demo config** is documented in `scripts/demo.config`.
- **Full Pareto report** is in `scripts/sweep-results.md`.
- **Winner:** `combined-best-q3km-nconf` — merged-fine-tune Q3_K_M GGUF (`/models/cliniq-gemma4-e2b-Q3_K_M.gguf`), no-confidence system prompt, max_tokens=1024, ctx=2048, reasoning_budget=0, parallel=1. Historical 1.45 tok/s at extraction_score 1.0 (100% valid JSON, all codes correct).
- The deployment manifest at `apps/llama-server/deployment.yaml` has been updated to match this winner.
- Today's revalidation confirmed the same output format and same correct SNOMED code, at reduced throughput (0.9 tok/s) due to a Jetson nvpmodel regression unrelated to the config sweep itself — relative ordering between configs is unchanged.

See `scripts/sweep-results.md` for the full Pareto frontier, runner-ups, and what did NOT work.

---

## Context

ClinIQ runs Gemma 4 E2B (4.63B params, Q4_K_M) at **1.4 tok/s** on Jetson Orin Nano, producing ~350 tokens per case = **~250s per inference**. This is too slow for an acceptable demo. The hackathon requires Gemma 4 (no switching to Gemma 3n). We need to systematically test three strategies, track all results, and find the best speed/accuracy tradeoff.

**Three strategies:**
1. Heavier quantization (smaller model = less bandwidth per token)
2. Shorter output via prompt engineering (fewer tokens = less time)
3. max_tokens reduction (may help KV cache allocation)

---

## Phase 0: Generate Quantized Models (local, no code changes)

`llama-quantize` is at `/opt/homebrew/bin/llama-quantize`. Source: `models/cliniq-gemma4-e2b.gguf` (Q8_0, 4.6GB). Generate 4 new variants locally, then scp to Jetson.

```bash
llama-quantize models/cliniq-gemma4-e2b.gguf models/cliniq-gemma4-e2b-Q3_K_M.gguf Q3_K_M
llama-quantize models/cliniq-gemma4-e2b.gguf models/cliniq-gemma4-e2b-Q3_K_S.gguf Q3_K_S
llama-quantize models/cliniq-gemma4-e2b.gguf models/cliniq-gemma4-e2b-IQ4_XS.gguf IQ4_XS
llama-quantize models/cliniq-gemma4-e2b.gguf models/cliniq-gemma4-e2b-Q2_K.gguf Q2_K
```

Expected sizes: Q3_K_M ~2.5GB, Q3_K_S ~2.4GB, IQ4_XS ~3.0GB, Q2_K ~2.3GB

Then scp all to `talos-jetson-3:/var/lib/ollama/models/`

---

## Phase 1: Instrument benchmark.py for the sweep

**File: `scripts/benchmark.py`**

### 1a. Add CLI parameters

- `--system-prompt TEXT` — override system prompt (default: current prompt)
- `--max-tokens INT` — override max_tokens (default: 1024)

Thread these through `run_inference_streaming()` and `run_inference_sync()` instead of the hardcoded `SYSTEM_PROMPT` constant and `"max_tokens": 1024`.

### 1b. Enhance extraction quality scoring

Upgrade `validate_output()` to return an `extraction_score` (0.0-1.0):

- **JSON validity** (20%): parses as JSON dict
- **Schema completeness** (20%): fraction of expected keys present (`patient`, `conditions`, `labs`, `meds`, `vitals`)
- **Condition accuracy** (30%): fraction of expected SNOMED codes found
- **Lab accuracy** (15%): fraction of expected LOINC codes found
- **Med accuracy** (15%): fraction of expected RxNorm codes found

### 1c. Add DuckDB schema columns

In `init_db()`, add after CREATE TABLE blocks:
```sql
ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS extraction_score DOUBLE;
ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS completion_tokens_actual INTEGER;
```

Store extraction_score and actual completion tokens per run.

---

## Phase 2: Enrich test cases

**File: `scripts/test_cases.jsonl`**

Add `expected_loincs` and `expected_rxnorms` to all 5 cases (values already in the user text):

| Case | expected_conditions | expected_loincs | expected_rxnorms |
|------|-------------------|----------------|-----------------|
| bench_minimal | ["76272004"] | ["20507-0"] | ["105220"] |
| bench_typical_covid | ["840539006"] | ["94500-6"] | ["2599543"] |
| bench_complex_multi | ["86406008"] | ["75622-1"] | ["1999563"] |
| bench_meningitis | ["23511006"] | ["49672-8"] | ["1665021"] |
| bench_negative_lab | ["50711007"] | ["11259-9"] | ["1940261"] |

---

## Phase 3: Update orchestrator

**File: `scripts/optimize.py`**

- Read `system_prompt` and `max_tokens` from experiment config
- Pass `--system-prompt` and `--max-tokens` to `benchmark.py` subprocess
- Add `skip_restart: true` support — skip K8s patching for experiments that only change client-side params (prompt, max_tokens)

---

## Phase 4: Define all experiments

**File: `scripts/experiments.yaml`**

### Quantization experiments (require server restart + model swap)
| Name | Model | Expected Size |
|------|-------|------|
| quant-q4km-baseline | cliniq-gemma4-e2b-Q4_K_M.gguf | 3.2GB |
| quant-q3km | cliniq-gemma4-e2b-Q3_K_M.gguf | ~2.5GB |
| quant-q3ks | cliniq-gemma4-e2b-Q3_K_S.gguf | ~2.4GB |
| quant-iq4xs | cliniq-gemma4-e2b-IQ4_XS.gguf | ~3.0GB |
| quant-q2k | cliniq-gemma4-e2b-Q2_K.gguf | ~2.3GB |

### Prompt experiments (client-side only, skip_restart: true)
| Name | Prompt Change |
|------|------|
| prompt-no-confidence | Remove "Include confidence scores" |
| prompt-compact | "Extract entities... compact JSON... No summary." |
| prompt-minimal | Extreme compression, minimal instruction |

### max_tokens experiments (client-side only, skip_restart: true)
| Name | max_tokens |
|------|-----------|
| maxtok-512 | 512 |
| maxtok-384 | 384 |
| maxtok-256 | 256 |

### Combined best (after individual results)
| Name | Best quant + best prompt + best max_tokens |
|------|------|
| combined-best | TBD after sweep |

---

## Phase 5: Enhance reporting

**File: `scripts/report.py`**

- Add `extraction_score` and `avg completion tokens` columns to summary table
- Add "Speed vs Quality" section showing tok/s alongside extraction_score
- Highlight Pareto-optimal experiments (not dominated on both speed and quality)
- Decision criteria: extraction_score >= 0.8 AND valid_json >= 80%, then maximize tok/s

---

## Execution Order

1. Generate 4 quantized models locally (Phase 0)
2. Instrument benchmark.py (Phase 1)
3. Enrich test_cases.jsonl (Phase 2)
4. Update optimize.py (Phase 3)
5. Write experiments.yaml (Phase 4)
6. Update report.py (Phase 5)
7. Scp models to Jetson
8. Run full sweep via `optimize.py`
9. Generate final report with `report.py --waterfall`

---

## Verification

1. `llama-quantize` produces valid GGUF files (check with `llama-server` loading them)
2. `benchmark.py --system-prompt "test" --max-tokens 256` works with overrides
3. `extraction_score` correctly captures quality differences (Q2_K should score lower than Q4_K_M)
4. `optimize.py` skips restart for `skip_restart: true` experiments
5. Final report shows speed vs accuracy tradeoff across all 12 experiments
6. Combined-best config achieves the best demo-viable speed

## Critical Files
- `scripts/benchmark.py` — add --system-prompt, --max-tokens, extraction_score
- `scripts/test_cases.jsonl` — add expected_loincs, expected_rxnorms
- `scripts/optimize.py` — pass new params, skip_restart support
- `scripts/experiments.yaml` — 12 new experiments (5 quant + 3 prompt + 3 max_tokens + 1 combined)
- `scripts/report.py` — extraction_score in tables, Pareto analysis
- `models/` — generate Q3_K_M, Q3_K_S, IQ4_XS, Q2_K from Q8_0 source
