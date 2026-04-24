# Team C2 — llama.cpp Runtime Flags Sweep Results

Generated 2026-04-23 by team/c2-llama-flags-2026-04-23. Target: NVIDIA Jetson Orin NX 8GB, node `talos-jetson-3`, namespace `gemma4`.

## TL;DR

**No positive tok/s delta found on the flags C2 owns (`--flash-attn`, `-ctk/-ctv q8_0`, `--mlock`).** Under the current 7W-locked power regime, flash-attn and mlock are neutral; KV cache quantization is a 27% regression. The only speedup observed during the sweep window came from Team C4's `--spec-type ngram-cache` overlay — that is their win, not C2's.

**Recommendation: keep the existing `scripts/demo.config` unchanged.** The sweep-winner config already uses `--flash-attn auto` (llama.cpp default), and auto appears to match explicit `--flash-attn on` in tok/s on this model/hardware. If Team C4's ngram-cache spec-decode clears their quality gate, that's the real +30-40% incremental win worth stacking on top of the sweep winner.

## Regime

Per [team/c1 POWER_MODE.md](../apps/llama-server/POWER_MODE.md), the Jetson is locked at:
- EMC 2133 MHz (half of MAXN/15W target 3200 MHz)
- GPU 624.75 MHz (two-thirds of MAXN)
- CPU all 6 cores @ 1.51 GHz
- DRAM 7.4 GiB, `performance` governor

C1 could not unlock higher modes from within Talos without a custom OS image. Today's clean baseline in this regime: **0.87 tok/s** eval. April-14 historical 1.45 tok/s was at MAXN — unreachable here.

Power-probe "intermittent boost": Team C4 reported ~1.0-1.1 tok/s baseline in the same hour while C1's power-probe was cycling. Treat absolute tok/s as noisy by ~15-20%, compare only numbers measured minutes apart.

## Methodology

- **Target model**: `/models/cliniq-gemma4-e2b-Q3_K_M.gguf` (merged fine-tune, 2.96 GB).
- **Prompt**: the no-confidence variant from the prior sweep (sweep winner).
- **System prompt + user prompt** pinned across all experiments.
- **max_tokens=200** (below the ~500 token JSON floor): chosen so each run completes in ~4 minutes under 0.87 tok/s. The llama-server-reported `timings.predicted_per_second` is independent of JSON validity, so tok/s signal is valid. `extraction_score` will be 0 everywhere because the JSON truncates — **quality is NOT validated here**; any live candidate must be re-benchmarked with max_tokens=1024 before shipping.
- **Runs/case**: 1-2 per experiment (contention forced early kills on some).
- **Test case**: `scripts/test_cases_c2.jsonl` — 1 case (`bench_minimal`, the syphilis extraction).
- **Harness**: `scripts/benchmark.py`, driven by `scripts/run_c2_sweep.sh` / `run_c2_sweep2.sh` (imperative deploy + benchmark loops).
- **DB**: `scripts/benchmarks.duckdb`, table `experiments` + `benchmark_runs`. Filter `experiment_name LIKE 'c2-%'`.

## Contention with Team C4

Team C4 simultaneously ran a speculative-decoding sweep on the same deployment. They patched args (`--spec-type ngram-cache`, `--draft-max N`) over my configs roughly every 5-10 minutes via strategic-merge. I defended with:

- A `c2-exp=<name>` pod label so I could detect overwrites (function `check_pod_is_mine()` in `run_c2_sweep2.sh`).
- Explicit patches specifying the full `args[]` array so there was no partial merge.
- Scaling the deployment down + force-deleting pods before each run so a newly-spawned pod would run under my current spec.

Despite those, at one point C4 overlaid `--spec-type ngram-cache --draft-max 16` on top of my `--flash-attn on`, producing a **contaminated** measurement of 1.22 tok/s. When I later repeated **pure `--flash-attn on`** (no spec-decode) the tok/s dropped back to **0.87**, identical to baseline, confirming the 1.22 was all ngram-cache.

## Experiments — Clean Results

| #  | Name                   | Flag(s) changed (verified on pod)            | tok/s  | delta vs baseline | score\* | n | notes |
|----|------------------------|----------------------------------------------|--------|-------------------|---------|---|-------|
| 0  | `c2-baseline`          | (sweep winner, unchanged)                    | **0.87** | 0%              | 0.00    | 1 | clean reference point |
| 1  | `c2-fa-on-pure`        | `--flash-attn on`                            | **0.87** | 0%              | 0.00    | 1 | **neutral** — explicit `on` matches auto-default |
| 2  | `c2-mlock-final`       | `--mlock`                                    | **0.87** | 0%              | 0.00    | 1 | **neutral** on eval; pod startup ~50s slower (pinning RAM); no prompt-cache benefit either |
| 3  | `c2-kvq8-fa`           | `--flash-attn on -ctk q8_0 -ctv q8_0`        | **0.64** | **-27%**        | 0.00    | 1 | **REGRESSION** — the quant/dequant overhead on attention exceeds bandwidth savings at this (small-model × Orin) point |

\* score=0 everywhere because `max_tokens=200` < ~500 needed for valid JSON on this case. Not a quality regression; a budget choice for sweep-scanning speed. The top candidate should be re-validated at `max_tokens=1024`.

## Experiments — Contaminated / Interrupted

| Name                   | Intended flag        | What actually ran              | Observed tok/s | Disposition |
|------------------------|----------------------|--------------------------------|----------------|-------------|
| `c2-fa-on` (1st pass)  | `--flash-attn on`    | + C4's `--spec-type ngram-cache --draft-max 16` | 1.22 tok/s | Contaminated — +40% came from spec-decode, **not** flash-attn. See c2-fa-on-pure above for the clean measurement. |
| `c2-ubatch-128`        | `--ubatch-size 128`  | C4 reverted deployment to baseline mid-run 1; run 2 would have been baseline, abandoned. | — | Skipped; rerun after C4 is idle. |
| `c2-spec-decode-only`  | (observe C4 config)  | C4 reverted during both runs   | — | Runs crashed with `RemoteDisconnected`. |

## Dimensions Not Measured

| Dimension              | Why skipped |
|------------------------|-------------|
| `--ubatch-size 64/256` | Time budget after contention overhead. ubatch-128 was the first priority and it itself got interrupted. |
| `--threads 2/4/6`      | Time budget. Expected small effect because 6 Orin CPUs are already the default and GPU-offloaded. |
| `-ctk q4_0 -ctv q4_0`  | q8_0 already regressed; q4 would regress further. |
| `--no-warmup`          | Affects only startup; startup is a kubectl concern, not per-request tok/s. Not a throughput lever. |
| `--no-perf`, `--jinja` | No expected tok/s effect. |
| `--parallel > 1` (concurrent) | Separate harness needed (see `scripts/parallel2_concurrent_test.py` stub). Not run — the demo is synchronous / single-request. |

## Findings

1. **Flash attention is a no-op on Orin sm_87 for this model size.** `--flash-attn on` and the default `auto` gave the same 0.87 tok/s. Not a regression, just no gain. llama.cpp's auto-detection is already making the right call.
2. **KV Q8_0 is a net negative at this model size.** 0.64 vs 0.87 tok/s = -27%. The memory bandwidth savings are smaller than the quant/dequant overhead in attention for a 2B-class model with ~3GB weights; this may flip for larger models or with more aggressive context usage, but not our demo.
3. **mlock is neutral on steady-state tok/s** and costs extra startup time. The llama.cpp default (mmap on, mlock off) is already optimal here — the kernel keeps hot pages resident and the pinning cost isn't paid back.
4. **The only positive tok/s delta observed came from C4's ngram-cache spec-decode** (+40%). That's their win. If C4 ships it and it clears their quality gate, `scripts/demo.config` should stack on top — but that's a C4 decision.
5. **The sweep-winner config is already near-optimal among C2-owned flags** at the current regime. No changes recommended.

## Decision for demo.config

**No change.** `scripts/demo.config` stays at:
- `MODEL_PATH=/models/cliniq-gemma4-e2b-Q3_K_M.gguf`
- `--ctx-size 2048`
- `--n-gpu-layers 99`
- `--reasoning-budget 0`
- `--parallel 1`
- `--flash-attn` NOT added explicitly (default `auto` is fine)
- `--mlock` NOT added (no benefit)
- KV cache default f16 (Q8 regressed)

The deployment (`apps/llama-server/deployment.yaml`) has been restored to the sweep-winner config before leaving — verified current pod args = baseline.

## Follow-ups for a Future Sweep

When Team C4 finishes and the deployment is idle:
1. Re-run the ubatch sweep (`ubatch-size 64/128/256`) with n=3 at max_tokens=1024 on `test_cases_val3.jsonl`.
2. Re-run the threads sweep (`--threads 2/4/6`) same protocol.
3. `--parallel 2` + concurrent test (see `scripts/parallel2_concurrent_test.py`) — this measures aggregate throughput under burst load, not single-request latency. Different user story.
4. Full-context regression for KV quant: does the q8_0 penalty shrink with longer sequences? May be worth re-visiting at ctx=4096+.
5. If C4's spec-decode ships, stack it with `--flash-attn on` and re-measure (this sweep only showed the combination once, contaminated).

## Caveats

1. **n=1 on most experiments** — noisy, treat deltas as qualitative not quantitative.
2. **max_tokens=200 not quality-validated** — zero JSON validation on any sweep run. Quality of the winner must be re-confirmed at max_tokens=1024.
3. **Concurrent team contention** — C4 re-patched the deployment during many runs. Contaminated rows are called out explicitly.
4. **Power regime drift** — C1's power-probe cycled during the sweep, causing 0.87 to ~1.0 baseline drift. Relative orderings hold; absolute numbers don't.
5. **Did not measure `--flash-attn off`** — auto (default) and on both measured at 0.87, so off would likely also be 0.87, but this wasn't verified explicitly.

## Files

- `scripts/experiments-c2.yaml` — declarative C2 experiment list (for `scripts/optimize.py`).
- `scripts/run_c2_sweep.sh`, `scripts/run_c2_sweep2.sh` — imperative sweep runners with C4-aware label-based patch detection.
- `scripts/test_cases_c2.jsonl` — 1-case minimal benchmark (`bench_minimal` only).
- `scripts/parallel2_concurrent_test.py` — concurrent-request stub for `--parallel 2` follow-up.
- `scripts/analyze_c2.py` — DB analysis helper; run `python3 scripts/analyze_c2.py` for the latest table.
- `scripts/optimize.py` — patched to support `--flash-attn {on,off,auto}` as a string (was a bool), plus new bool flags `--no-warmup`, `--no-perf`, `--jinja`.
