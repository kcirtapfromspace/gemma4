# autoresearch — ClinIQ Inference Optimization

This is an adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for **inference optimization** on NVIDIA Jetson Orin Nano (8GB).

## Goal

Maximize `gen_tok_per_sec` (generation tokens per second) for the ClinIQ clinical entity extraction pipeline, while maintaining `extraction_score >= 0.80`.

## Hardware

- **Device**: NVIDIA Jetson Orin Nano 8GB (unified LPDDR5, 68 GB/s bandwidth)
- **GPU**: 1024 CUDA cores, Ampere, compute capability 8.7
- **Memory**: 8GB shared CPU+GPU (unified memory architecture)
- **Storage**: NVMe, 948GB free at `/var/lib/ollama/models`
- **Network**: Jetson is at `192.168.150.41`, llama-server exposed on NodePort `30083`

## Current Baseline

| Metric | Value |
|---|---|
| Model | Gemma 4 E2B, Q4_K_M GGUF (3.2GB) |
| Prompt tok/s | ~44 |
| **Gen tok/s** | **~1.4** (this is what we're optimizing) |
| Extraction score | 1.00 |
| Inference server | llama-server (llama.cpp, custom CUDA build) |
| Context size | 2048 |
| GPU layers | 99 (all offloaded) |

The generation bottleneck is **memory bandwidth**. Autoregressive decoding reads the full model weight matrix per token. At 68 GB/s and 3.2GB model, theoretical max is ~21 tok/s, but actual is 1.4 tok/s (~6% efficiency).

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr15`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files** for full context:
   - `tools/autoresearch/program.md` — this file (research instructions)
   - `tools/autoresearch/run_experiment.py` — the experiment runner you modify
   - `scripts/benchmark.py` — the evaluation harness (DO NOT MODIFY)
   - `scripts/experiments.yaml` — existing experiment results for reference
   - `apps/llama-server/deployment.yaml` — current server configuration
   - `apps/llama-server/build-job.yaml` — how llama-server was built
4. **Verify connectivity**: `curl -s http://192.168.150.41:30083/health` should return OK.
5. **Initialize results.tsv**: Create `tools/autoresearch/results.tsv` with the header row.
6. **Confirm and go**.

## Experimentation

Each experiment modifies inference configuration and runs the benchmark suite against the live Jetson endpoint.

**What you CAN do:**
- Modify `tools/autoresearch/run_experiment.py` — this is the file you edit
- Change llama-server launch arguments (requires pod restart via kubectl)
- Swap GGUF model files (if they exist on disk at `/var/lib/ollama/models`)
- Change client-side parameters: system prompt, max_tokens, temperature
- Try speculative decoding by adding a draft model
- Modify llama.cpp build flags and rebuild (via `apps/llama-server/build-job.yaml`)
- SSH to Jetson for system-level changes if needed

**What you CANNOT do:**
- Modify `scripts/benchmark.py` — it is the fixed evaluation harness
- Change the test cases in `scripts/test_cases.jsonl`
- Fake or skip the extraction quality check
- Use cloud/remote inference — everything must run on the Jetson

**The goal is simple: maximize gen_tok_per_sec while keeping extraction_score >= 0.80.**

## Research Directions (ranked by expected impact)

### Tier 1 — High potential, try first
1. **Speculative decoding**: Use a tiny draft model (Gemma 4 E1B or distilled) to predict N tokens, verify in batch. llama.cpp supports `--model-draft`. Both models must fit in 8GB.
2. **Diagnose the 6% efficiency gap**: Profile with `nsys` or instrument llama-server to find where 94% of theoretical bandwidth is lost. Suspects: 262K vocab output projection, CUDA sync overhead, KV cache pressure.
3. **Reduce context size**: Try `--ctx-size 1024` (our prompts + outputs total ~750 tokens). Less KV cache = more bandwidth for weights.
4. **Thread minimization**: Try `--threads 1` to reduce CPU-side memory traffic competing with GPU.

### Tier 2 — Medium potential
5. **Aggressive quantization**: Run the Q2_K experiment (defined but never executed). Watch extraction quality.
6. **Compact output schema**: Shorter JSON output = fewer gen tokens = faster wall-clock. Expand abbreviations client-side.
7. **Batch size tuning**: Try `--batch-size` and `--ubatch-size` flags for KV cache operations.
8. **Flash attention toggle**: `--flash-attn` was tested and HURT on unified memory (0.85 tok/s). But verify with latest llama.cpp.

### Tier 3 — Ambitious, try if Tier 1/2 plateau
9. **Tensor parallelism across Jetsons**: Split model across 3 nodes. Network latency vs bandwidth gain tradeoff.
10. **Custom GGUF with pruned vocab**: The 262K vocab output head is massive. Prune to medical-only tokens.
11. **Rebuild llama.cpp with Jetson-specific flags**: `-march=armv8.6-a`, NEON optimizations, link-time optimization.
12. **Alternative inference engine**: Try MLC-LLM or TensorRT-LLM instead of llama.cpp.

## Running an Experiment

```bash
# From repo root:
cd /Users/thinkstudio/gemma4

# Run the experiment script (it calls benchmark.py internally)
python tools/autoresearch/run_experiment.py \
  --endpoint http://192.168.150.41:30083 \
  --name "experiment-name" \
  > tools/autoresearch/run.log 2>&1

# Check results
grep "^gen_tok_s:\|^extraction_score:\|^prompt_tok_s:" tools/autoresearch/run.log
```

## Output Format

The experiment runner prints a summary:

```
---
gen_tok_s:          1.452
prompt_tok_s:       44.100
extraction_score:   1.00
success_rate:       1.00
total_seconds:      1650.3
model_file:         cliniq-gemma4-e2b-Q3_K_M.gguf
server_args:        --ctx-size 2048 --n-gpu-layers 99
description:        Q3_K_M + no-confidence prompt
```

## Logging Results

Log to `tools/autoresearch/results.tsv` (tab-separated):

```
commit	gen_tok_s	extraction_score	prompt_tok_s	status	description
```

- **commit**: git short hash (7 chars)
- **gen_tok_s**: average generation tokens/sec (0.000 for crashes)
- **extraction_score**: average quality score 0.0-1.0 (0.00 for crashes)
- **prompt_tok_s**: average prompt tokens/sec
- **status**: `keep`, `discard`, or `crash`
- **description**: what this experiment tried

Example:
```
commit	gen_tok_s	extraction_score	prompt_tok_s	status	description
a1b2c3d	1.331	1.00	43.4	keep	baseline Q4_K_M
b2c3d4e	1.453	1.00	44.0	keep	Q3_K_M + no-confidence prompt
c3d4e5f	1.289	1.00	40.1	discard	IQ4_XS (slower gen)
d4e5f6g	0.000	0.00	0.0	crash	speculative decoding OOM
```

## Logging to Obsidian

After each experiment, also push results to Obsidian via the MCP server:
- Append to note `ClinIQ/Research/Inference Optimization Log`
- Format as a timestamped entry with all metrics
- Tag with `#autoresearch #cliniq #inference`

## The Experiment Loop

LOOP FOREVER:

1. Look at current results.tsv — what's been tried, what worked, what's the current best
2. Pick a research direction from the list above (or invent a new one)
3. Modify `run_experiment.py` with the experimental config
4. git commit the change
5. Run: `python tools/autoresearch/run_experiment.py --endpoint http://192.168.150.41:30083 --name "description" > tools/autoresearch/run.log 2>&1`
6. Read results: `grep "^gen_tok_s:\|^extraction_score:" tools/autoresearch/run.log`
7. If grep is empty, it crashed. `tail -n 50 tools/autoresearch/run.log` to diagnose.
8. Record in results.tsv
9. Push summary to Obsidian
10. If gen_tok_s improved AND extraction_score >= 0.80: keep the commit
11. If gen_tok_s didn't improve OR extraction_score < 0.80: git reset back

**NEVER STOP.** Once the loop begins, do NOT pause to ask. Run indefinitely until manually interrupted. If stuck, think harder — re-read llama.cpp docs, check Jetson forums, try combining near-misses.

## Simplicity Criterion

All else being equal, simpler is better. A marginal speed gain that requires a fragile multi-process setup is not worth it. Removing complexity while maintaining speed is a win.
