# Speculative Decoding Experiment Log — Team C4

Diary of every speculative-decoding experiment against the Jetson llama-server
deployment. Each row records one deploy+benchmark cycle.

## Environment

- **Target model** (default): `/models/cliniq-gemma4-e2b-Q3_K_M.gguf` (merged fine-tune, 2.96 GB on pod)
- **Hardware**: Jetson Orin NX 8GB (talos-jetson-3), CUDA compute 8.7, 7580 MiB VRAM
- **llama-server version**: `1 (268d61e)`, built GNU 11.4.0 aarch64, CUDA
- **Spec decode support**: `--model-draft` + `--draft N` (target/draft), plus
  `--spec-type ngram-*` (prompt-lookup / n-gram cache, no draft model required).
- **Benchmark**: `scripts/benchmark.py`, `scripts/test_cases_val3.jsonl` (3 cases x 3 runs typical).
- **Endpoint**: `http://192.168.150.41:30083`
- **Power mode** at time of run: Team C1 is attempting to unlock 15W mode; if on
  7W baseline is ~0.87 tok/s, if on 15W baseline is ~1.45 tok/s. All rows note
  the mode in effect.

## Columns

- `id` — experiment identifier (ec1..)
- `target` — target model quant
- `draft` — draft model (or `ngram-*` for prompt-lookup, or `—` for baseline)
- `draft-N` — `--draft-max` (max draft tokens per step)
- `ctk/ctv` — KV cache quantization for target (default f16)
- `tok/s` — avg generation tokens/sec from llama-server `timings.predicted_per_second`
- `score` — avg extraction_score (quality gate >= 0.95)
- `accept` — draft acceptance rate if reported (n/a otherwise)
- `notes` — anything unusual

## Results

## Sampling method

Because benchmark.py runs on this shared Jetson take 4-15 min per case at ~1
tok/s, and the node was power-cycled many times during the sweep (Team C1's
nvpmodel work + Team C2's flag sweep + shared-GPU contention), most rows
below are SAMPLED mid-generation from `/slots` (decoded-tokens / elapsed
seconds) or taken from a single completed request's `timings.predicted_per_second`.
The numbers are directionally correct (±15%) but not publication-grade. A
clean n=3 sweep would require 2-3 hours of uninterrupted node ownership
that we did NOT have.

| id | target | draft | draft-N | ctk/ctv | tok/s | comp_tok | score | notes |
|----|--------|-------|---------|---------|-------|----------|-------|-------|
| ec0 | Q3_K_M | — | — | f16/f16 | ~1.0-1.1 (sampled) | ~256 (partial) | n/a | Baseline. Sampled from /slots across multiple runs; rate computed from n_decoded / elapsed. Case 1 in one run: 256 tokens in 3:13 = 1.33 tok/s eval (post prompt-processing); steady-state ~1.0 tok/s. This is higher than the mission-brief 0.87; 15W mode may have partially engaged via C1's power-probe despite C1's infra-constraints doc saying nvpmodel is blocked. |
| ec1 | Q3_K_M | cliniq-Q2_K | 8 | f16/f16 | ~0.4 (sampled) | n/a | n/a | Self-spec with Q2_K draft: ~half of baseline throughput. Draft model compute overhead (Q2_K is only 4% smaller than Q3_K_M on disk so draft forward-pass cost is ~equal to target verification cost) > acceptance gains. Tokenizer matched by construction (both cliniq fine-tunes). spec=True confirmed via /slots. |
| ec3 | Q3_K_M | ngram-cache | 8 | f16/f16 | **1.05 (server)** | 96 | n/a* | Server-reported `timings.predicted_per_second` = 1.05. Generated 96 tokens in 91.6s, draft-p-min=0.6. *Score 0.0 because output truncated at max_tokens=96 before the JSON completed — length issue, not quality issue. |
| ec0-final | Q3_K_M | — | — | f16/f16 | **0.88 (server)** | 96 | n/a* | Clean baseline. Server-reported `timings.predicted_per_second` = 0.88. 96 tokens in 109.5s. Matches mission-brief baseline of 0.87 exactly. *Score 0.0 for same length-truncation reason. |
| ec1 abandoned | Q3_K_M | cliniq-Q2_K | 8 | f16/f16 | ~0.4 (sampled) | n/a | n/a | Abandoned: self-spec with Q2_K draft was HALF baseline. Draft forward-pass cost (~= target forward cost since Q2_K only 4% smaller on disk) + verification overhead > acceptance gains. spec=True confirmed. |
| ec5 attempted | Q3_K_M | ngram-cache | 16 | f16/f16 | blocked | — | — | Tried ngram-cache with N=16 to improve over ec3's N=8. Deployment patch was repeatedly overwritten by Team C2's parallel sweep (--flash-attn / --cache-type / --ubatch-size). New pod also deadlocked on GPU (old pod's 3GB model still resident + new pod's 3GB = exceeded 7.5 GB VRAM). Could not measure. |
| ec2 NOT run | Q3_K_M | gemma-base-Q3_K_M | 8 | f16/f16 | — | — | — | NOT run due to GPU contention (see above). Hypothesis: base model has LOWER acceptance on clinical JSON than self-spec because it wasn't fine-tuned, but draft is SMALLER (2.36 GB vs 2.97 GB) so draft-pass is faster. Unclear which effect dominates without measurement. |
| ec6 NOT run | Q3_K_M | — | — | q8_0/q8_0 | — | — | — | KV-cache q8_0 quantization. NOT run due to GPU contention. Team C2's sweep applied this flag at one point; brief measurement from /slots mid-generation suggested ~1.0 tok/s, similar to baseline — KV-quant mostly helps memory, not tok/s, for a single-request workload. |



