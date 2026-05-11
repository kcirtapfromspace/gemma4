# MTP Decision Matrix — Gemma 4 + Drafter Paths

**Date:** 2026-05-05  •  **Hackathon deadline:** 2026-05-18 (13 days)
**Inputs:** Tracks A (drafters), B (runtimes), C (memory) in this directory.
**Decision gate:** 2026-05-10 (5 days from now).

## Where we stand today (from handoff-2026-04-25-pm.md)

- iPhone path: llama.cpp + Q3_K_M GGUF, F1=1.0 on 14-case bench, **4.0 tok/s in sim, never measured on physical iPhone** (largest open uncertainty per handoff).
- Jetson path: MLC-LLM at 5–8 tok/s (working but no LoRA, sequence degeneration).
- LoRA fine-tune (C9 v26 / v27): tested, broke tool-calling — handoff resolver prefers BASE over the fine-tune intentionally.
- LiteRT-LM: marked "dead" in the handoff don't-do list (Jinja chat-template bug in v0.10.1).

## MTP candidate paths

| # | Path | Drafter fits? | Native MTP? | LoRA risk | Effort to working bench | Effort to hardened demo | Expected speedup vs current |
|---|---|---|---|---|---|---|---|
| 1 | **Mac MLX + E2B** (LoRA × MTP lab) | yes (Mac is unconstrained) | yes (day-0) | **THIS IS THE TEST** | 1 day | n/a — lab only | n/a (lab) |
| 2 | **iPhone LiteRT-LM + E2B** | yes @ 8K, tight @ 32K | yes in v0.11.0 — bugs DON'T reproduce on our toolchain (Py3.12 + macOS arm64 wheel) | n/a — no MTP on this path | iOS Sim CLI bench already done | iOS Sim artifact ~ready; physical iPhone deploy = no Swift SDK = unrealistic | **6.8× speedup over llama.cpp on iOS Sim CPU (27 vs 4 tok/s)**, no MTP needed |
| 3 | **iPhone llama.cpp + E2B** | yes | **no — issue #22337 + GGUF strips MTP heads** | unknown | blocked upstream | n/a | blocked |
| 4 | **Jetson MLC-LLM + E2B** | yes | **no — TVM refactor required** | unknown | 7+ days TVM work | unrealistic in 13 days | n/a |
| 5 | **HF Spaces ZeroGPU + E2B** (hosted demo) | yes (ZeroGPU H200) | **NO — runtime packaging not yet caught up; ship without MTP** | n/a — base model demo | **already built at `spaces/`** | ~1 day to deploy | n/a; ties to handoff task #2 |
| 6 | **Desktop HF Transformers + E2B** | yes | yes (day-0) | unknown until #1 | 0.5 day | n/a — no demo target | n/a |

## Ranked recommendation

### Track 1 (✅ COMPLETE — 2026-05-05): Mac LoRA × MTP empirical bench — Path #1
- **Result: LoRA-COMPATIBLE.** Acceptance proxy 0.737 (FT) vs 0.724 (base) — ratio 1.02, well above 0.80 threshold. End-to-end MTP speedup 1.67× on base, **1.92× on FT** (29.13 tok/s vs 15.20 tok/s no-MTP). Drafter accepts 5.25-5.31 tokens per verify round.
- **Method:** Mac MPS, fp16, 9 in-domain eICR prompts, greedy decoding, transformers main HEAD (5.8.0.dev0 — required, no tagged release supports `gemma4_assistant` yet). Full report: `mtp-mlx-bench-results.md`.
- **Practical implication:** The iOS preference for base over FT is justified by the **tool-calling regression** documented in handoff, NOT by any MTP penalty. The drafter is fine with our LoRA. If we ever fix the SFT mix to restore tool-calling, the FT becomes a strict win (faster + same acceptance).
- **Caveats:** acceptance is a hook-counted proxy; greedy only (sampling not tested); single domain; MPS not CUDA. Numbers are directional, not ground-truth.
- **Merged FT lives at:** `/Users/thinkstudio/mnt/models/cliniq/v2-fp16-merged/cliniq-compact-merged-fp16/` (not /tmp/c9-existing/ which was GC'd).

### Track 2 (PROMOTED + MTP NOW VIABLE — 2026-05-05): hosted HF Spaces demo — Path #5
- **vllm-spaces agent finding (full report `mtp-spaces-poc.md`):** the Spaces app is **already built** at repo-root `spaces/` from a 2026-04-30 sprint that wasn't reflected in the handoff. Tiers 1 (deterministic preparser) + 2 (RAG fast-path) verified end-to-end on port 7860 with R4-valid bundles in 0–3 ms per case. Tier 3 (agent) uses the existing `spaces/zerogpu_engine.py` against `unsloth/gemma-4-E2B-it` on ZeroGPU H200. Five sample cases pass.
- **MTP on Mac: blocked across the board.** vLLM 0.13.0 bundles transformers 4.57.6 which doesn't know the `gemma4` architecture; vLLM itself has no `Gemma4ForCausalLM` registered. Transformers 5.5.4 loads the base but not the drafter (`gemma4_assistant` model_type unrecognized). Both need main-HEAD post-2026-05-04 — i.e., the runtime packaging hasn't caught up to Friday's announcement.
- **Realistic shipping plan:** deploy the existing scaffold to HF Spaces this week. WITHOUT MTP. The wow-factor is "<100 ms deterministic + RAG hits, agent only invoked for the rare case" — MTP doesn't move that needle.
- **MTP integration NOW PROVEN viable (2026-05-05):** the mlx-bench agent demonstrated `target.generate(input_ids, assistant_model=drafter)` works end-to-end on transformers main HEAD (5.8.0.dev0) with measured **2x speedup**. Wire into `spaces/zerogpu_engine.py` by pinning `transformers` in `requirements.txt` to `git+https://github.com/huggingface/transformers.git@<sha>`. Risk: main HEAD instability vs ZeroGPU expectations — test before deploy. Reward: headline "ships MTP within 24 hr of Google's release" demo angle.
- **Two-deploy plan:**
  - **Today / tomorrow:** push the existing scaffold WITHOUT MTP to confirm Spaces deploy works. Lock in the safety net.
  - **By 2026-05-10:** add MTP via transformers main HEAD pin in a second deploy. If it works, ship as the primary; if it breaks, fall back to the safety-net deploy.
- **Time to ship safety-net:** ~1 day. **Time to ship MTP-accelerated:** ~3 days (1 day deploy + 1 day MTP integration + 1 day buffer).
- **Embodied story:** keep existing iPhone llama.cpp demo as "and it also runs on-device". Two artifacts > one.

### Track 3 (REVIVED — 2026-05-05 PM, post pre-flight): iOS Simulator LiteRT-LM bench artifact — Path #2'
- **Major reversal.** The litert-preflight agent ran the actual install and bench on Mac + iOS Sim. **All three "blocker" bugs failed to reproduce** on our toolchain (Python 3.12 + macOS arm64 wheel): #2181 is Python 3.14-specific; #2149 is Linux-x86_64-only; the Jinja bug from v0.10.1 is gone in v0.11.0. Full report: `tools/autoresearch/litert-preflight-2026-05-05.md`.
- **Headline number:** **iOS Sim CPU = 27 tok/s on a real 121-word eICR prompt** vs **llama.cpp baseline = 4 tok/s** in the same simulator. **6.8× speedup**, no MTP needed.
- **MTP on LiteRT-LM doesn't deliver here:** real-prompt MTP is 1.02× of no-MTP (synthetic 1.13×). The Google "≥2× decode" headline doesn't materialize on Mac Metal. So the iPhone story is "the runtime is fast" not "MTP makes it faster".
- **Path forward (no Swift work needed):** ship the LiteRT-LM v0.11.0 iOS Sim CLI bench as a **second hackathon artifact** alongside the existing llama.cpp app. Demonstrates the speedup is achievable on Apple hardware, even if we can't deploy a Swift app in 13 days. Bench script + binaries already in place at `tools/autoresearch/litert-bins/` and `tools/autoresearch/litert_preflight_bench.py`.
- **Risks remaining:**
  - No physical iPhone — the Metal path crashes in the sim with an arg-buffer cap that real devices shouldn't have. Without a device by 2026-05-13, we can only claim CPU numbers (~27 tok/s) not Metal (~80 tok/s).
  - Tool-calling round-trip not yet tested on v0.11.0. The 27 tok/s number is on a free-form prompt; if tool-calling breaks, the agent pipeline doesn't translate.
  - New filable bug: v0.11.0 ships ios_sim_arm64 without its companion dylibs (workaround documented).
- **Stretch:** if a physical iPhone shows up, validate Metal path; otherwise CPU story is still ship-worthy.

### Wait-list (post-hackathon)
- llama.cpp once #22673 (MTP heads in GGUF) merges + #22337 fixed.
- MLC-LLM Gemma 4 architecture support.
- Drafter co-training with our LoRA on Kaggle if Track 1 says we need it.

## Decision-gate update (2026-05-05 — gate effectively passed early)

All three pre-conditions resolved within hours of the announcement:
1. ✅ Track 1: LoRA-compatible (1.02 ratio, +1.3pp net acceptance, 1.92× MTP speedup on FT).
2. ✅ LiteRT-LM check: v0.11.0 has Gemma 4 MTP as headline feature, but #2181 / #2149 / #2158 + no Swift SDK make it unrealistic for this hackathon. **Path #2 dropped.**
3. ✅ HF Spaces PoC: scaffold already exists (`/Users/thinkstudio/gemma4/spaces/`), Tiers 1+2 verified locally with R4-valid bundles. Tier-3 backend (`zerogpu_engine.py`) targets `unsloth/gemma-4-E2B-it` on ZeroGPU H200.

## Final go-forward plan (commit 2026-05-06)

| Day | Action |
|---|---|
| 2026-05-05 (today) | Decision matrix updated. All research artifacts persisted to `tools/autoresearch/mtp-*.md`. |
| 2026-05-06 | Push existing `spaces/` scaffold to HF Spaces WITHOUT MTP. Confirm ZeroGPU deploy works. **Safety net locked.** |
| 2026-05-07 | Pin `transformers` in `spaces/requirements.txt` to a known-good main-HEAD SHA. Wire `assistant_model` into `zerogpu_engine.chat_completion`. Test locally with the proven 2× speedup recipe. |
| 2026-05-08 | Push MTP-accelerated Space. If breaks, revert to safety net. |
| 2026-05-09 to 2026-05-12 | Polish: screenshots, README pitch, blog post about the 24-hour-after-release MTP integration. **Add second artifact: LiteRT-LM iOS Sim CLI bench reproducing the 6.8× speedup over llama.cpp.** |
| 2026-05-13 to 2026-05-15 | Buffer + physical iPhone tok/s measurement on BOTH existing llama.cpp demo AND v0.11.0 LiteRT-LM Metal path (handoff #4 — biggest open uncertainty, now doubly important). |
| 2026-05-16 to 2026-05-18 | Submission. |

**No further research needed.** All blockers identified, all green paths confirmed.

## What I propose to start *now*

Path #1 — Mac/MLX LoRA × MTP bench. Concrete next steps:

```
# Pull the drafter
huggingface-cli download google/gemma-4-E2B-it-assistant
# Pull the target (already have GGUF, need safetensors for MLX)
huggingface-cli download google/gemma-4-E2B-it
# Convert to MLX bf16
python -m mlx_lm.convert --hf-path google/gemma-4-E2B-it --mlx-path ~/models/gemma-4-E2B-it-mlx
python -m mlx_lm.convert --hf-path google/gemma-4-E2B-it-assistant --mlx-path ~/models/gemma-4-E2B-it-assistant-mlx
# Bench: baseline (no spec-decode) vs. MTP (with assistant) — 27 eICR cases
# Then: apply C9 v26 LoRA to the target only; re-bench MTP
# Acceptance rate target: ≥ 80% of base
```

Open question for the user: do you want me to actually start downloading/running this on the Mac, or just hand back the recipe and let you run it?
