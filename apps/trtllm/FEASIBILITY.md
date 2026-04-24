# TensorRT-LLM on Jetson Orin NX 8GB for Gemma 4 E2B: Feasibility Study

**Team:** C3
**Branch:** `team/c3-trtllm-2026-04-23`
**Date:** 2026-04-23
**Phase completed:** Phase 1 (desk research only)
**Target:** Jetson Orin NX 8GB, L4T R36.4.7 (JetPack 6.2), sm_87

## TL;DR — Go / No-Go Decision: **NO-GO**

Do **not** port Gemma 4 E2B to TensorRT-LLM for this hackathon. Four independent blockers stack, any one of which alone would kill the 1-week porting budget:

1. **NVIDIA does not officially support TRT-LLM on Jetson.** The Jetson-specific fork (`v0.12.0-jetson`) is frozen at 0.12.0 (Nov 2024, JetPack 6.1). Mainline 1.2.1 (Apr 2026) does **not list sm_87** on the support matrix. NVIDIA staff have stated explicitly on the forums: *"We don't officially support TensorRT-LLM on Jetson."*
2. **Gemma 4 is not supported by mainline TRT-LLM.** Gemma 3 support is only 1B-text via a PyTorch-workflow PR (#3999, merged May 2025). Gemma 4 has a single open bug (#12764) on DGX Spark/GB10 (Blackwell/sm_121, **not** sm_87) where even NVFP4 exports fail to load because the runtime's bundled `transformers` library doesn't recognize the `gemma4` architecture. No release has shipped a working Gemma 4 builder.
3. **The dustynv jetson-containers image is stale.** `dustynv/tensorrt_llm:0.12-r36.4.0` was published >12 months ago for JetPack 6.0. No `0.13+` or `r36.4.7` tag exists. GitHub issue #4502 confirms current jetson-containers Dockerfiles fail on JetPack 6.2 ("TensorRT does not currently build wheels for Tegra systems").
4. **LoRA path is incompatible with our artifact.** TRT-LLM LoRA runtime requires HF safetensors → a custom NumPy format (`hf_lora_convert.py`). Our model is a merged Q8_0 GGUF; there is no GGUF → TRT-LLM engine path. Reconstructing from HF would require re-running our training artifacts through a full HF→ONNX→TRT engine rebuild on the Jetson itself — and the engine build is known to kernel-panic on 8 GB Orin Nano (forum #350532).

Projected effort to port Gemma 4 E2B (text only) on top of these blockers: 2–4 weeks of low-level work (patching the Jetson fork forward, adding `gemma4` to the PyTorch workflow, fighting tokenizer/transformers skew), with non-trivial probability of failure. We have 3.5 weeks until the hackathon deadline and a working llama.cpp baseline at 0.9–1.45 tok/s plus a working MLC-LLM build at 5–8 tok/s. **Stay with MLC-LLM as the optimization target; use llama.cpp as the fallback.**

---

## Question 1 — Build feasibility on Jetson L4T R36.4.7 / sm_87

**Answer: Marginal. No currently-published, supported artifact.**

| Artifact | Version | JetPack | sm_87? | Published | Status |
|---|---|---|---|---|---|
| `nvcr.io/nvidia/tensorrt-llm:*-jetson` | — | — | — | — | **Does not exist.** NGC has no Jetson-tagged TRT-LLM images. |
| `dustynv/tensorrt_llm:0.12-r36.4.0` | 0.12.0 | 6.0 (r36.4.0) | yes | ~Nov 2024 | Stale (>12 months old), targets JetPack 6.0 not 6.2 |
| TRT-LLM `v0.12.0-jetson` branch | 0.12.0 | 6.1 | yes (`--cuda_architectures 87`) | Nov 2024 | Source build; wheel must be built locally; ~6.8 GB VRAM after load |
| Mainline TRT-LLM 1.2.1 | 1.2.1 | — | **NOT listed** | Apr 20, 2026 | Support matrix: Blackwell / Hopper / Ada / Ampere (non-Jetson) / GB200 only |
| `jetson-containers` mainline Dockerfile | variable | 6.x | configurable | current | Fails on JetPack 6.2 per open issue #4502 |

**NVIDIA's own guidance for Jetson in 2026** (per *Getting Started with Edge AI on NVIDIA Jetson*, Jan 2026 and the Jetson AI Lab) is to use **vLLM**, **llama.cpp**, **Ollama**, or the new **TensorRT Edge-LLM** (a separate C++ runtime, not TRT-LLM). TRT-LLM is pointedly absent from the recommended framework list for Orin-class devices.

**Sources:**
- TRT-LLM support matrix: https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
- `v0.12.0-jetson` branch README: https://github.com/NVIDIA/TensorRT-LLM/blob/v0.12.0-jetson/README4Jetson.md
- NVIDIA forum "We don't officially support TRT-LLM on Jetson": https://forums.developer.nvidia.com/t/orin-nano-building-tensorrt-llm-from-source/350532
- Build-broken on JetPack 6.2: https://github.com/NVIDIA/TensorRT-LLM/issues/4502
- `dustynv/tensorrt_llm` Docker Hub (only `0.12-r36.4.0` tag): https://hub.docker.com/r/dustynv/tensorrt_llm/tags
- Jan 2026 NVIDIA Edge AI guide (recommends vLLM / llama.cpp): https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/

## Question 2 — Gemma 4 E2B architecture support

**Answer: No native support. Partial Gemma 3 support only. Gemma 4 is an open bug on non-Jetson hardware.**

| Model | TRT-LLM support | Release | Workflow | Notes |
|---|---|---|---|---|
| Gemma 1 (2B, 7B) | Yes | long-standing | TRT backend | `examples/models/core/gemma/` |
| Gemma 2 (9B, 27B) | Yes | long-standing | TRT backend | bf16 examples |
| Gemma 3 (1B-text) | Yes | 1.0 (May 2025, PR #3999) | PyTorch workflow | Sliding window support partial; VSWA KV-cache optimization "future MR" |
| Gemma 3 (4B, 12B, 27B) | **No** | — | — | Not in PR #3999, no follow-up |
| Gemma 3 (VLM/multimodal) | Partial (`Gemma3ForConditionalGeneration` listed) | 1.x | — | Text-only; vision tower not validated |
| Gemma 4 E2B / E4B | **No** | — | — | **Closed as "not planned"** on the only user request (#3143). Only Gemma 4 trace is AutoDeploy bug #12764 on DGX Spark where NVFP4 export can't be loaded (tokenizer + transformers version skew; `gemma4` arch unrecognized by bundled `transformers==4.57.3`) |

Gemma 4 E2B's architecture features (double-wide MLP, shared-KV 20 layers, per-layer sliding-vs-full attention, MatFormer nesting) are **not represented** in any TRT-LLM model definition file. They would have to be added from scratch in `tensorrt_llm/models/` (PyTorch workflow) — equivalent to what our Team MLC-LLM port has already done in `gemma4_model.py` for TVM, but with the additional TRT engine-compile step and no reference implementation to copy from.

**Sources:**
- Supported-models list: https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
- Gemma 3 1B PR: https://github.com/NVIDIA/TensorRT-LLM/pull/3999 (merged 2025-05-14)
- "When will Gemma 3 be supported?" issue closed as not planned: https://github.com/NVIDIA/TensorRT-LLM/issues/3143
- Gemma 4 NVFP4 runtime bug: https://github.com/NVIDIA/TensorRT-LLM/issues/12764
- AutoDeploy support matrix (no Gemma 4): https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/support_matrix.html

## Question 3 — LoRA support (merged GGUF in, TRT-LLM inference)

**Answer: Supported in principle, but not from our artifact. Path is HF only; no GGUF ingest.**

TRT-LLM LoRA workflow:

1. **Build time**: `trtllm-build --lora_dir <HF_LORA_DIR> --lora_plugin auto --max_lora_rank R --lora_target_modules ...`
2. **Format conversion**: `python examples/hf_lora_convert.py -i <hf_model> -o <output_dir> --storage-type float16` — HF safetensors → TRT-LLM NumPy tensor format.
3. **Inference**: Runtime accepts `LoraConfig(task_id, weights, config)` per request; multiple adapters swappable against a single base engine.

What this means for us:

- **GGUF is not accepted.** There is no path from `cliniq-gemma4-e2b.gguf` (Q8_0 merged) to a TRT-LLM engine. We would need the original HF safetensors of the base Gemma 4 E2B + the LoRA `adapter_model.safetensors`.
- **Merged LoRA path**: TRT-LLM's own docs say merged HF is the *simpler* path (skips step 1's `--lora_dir`). So if we had HF-merged weights, we'd just build a standard engine — but we don't, and the base isn't cached locally. And we'd still hit blockers 1 + 2 above.
- **int4_awq base requirement**: The officially-tested LoRA workflow requires the base quantized as INT4-AWQ first — another Jetson-side quantization step that has its own toolchain dependencies.

**Sources:**
- Official LoRA docs: https://nvidia.github.io/TensorRT-LLM/advanced/lora.html
- RTX-AI-Toolkit LoRA deployment guide: https://github.com/NVIDIA/RTX-AI-Toolkit/blob/main/llm-deployment/TensorRT-LLM-LoRA-deployment.md

## Question 4 — `trtllm-serve` OpenAI-compatible API

**Answer: Yes, the server exists; feasibility on Jetson is unvalidated.**

`trtllm-serve` ships with TRT-LLM ≥1.0 and exposes `/v1/completions` and `/v1/chat/completions` on :8000 with OpenAI-compatible schemas. Supports PyTorch, TRT, and AutoDeploy backends. For our `/v1/chat/completions` contract: yes, the endpoint is wire-compatible with llama-server and vLLM.

**However:** every `trtllm-serve` example on Jetson uses the stale 0.12-jetson fork (which shipped before `trtllm-serve` existed in its current form). The mainline 1.x server has not been demonstrated running on Orin; the GKE/DGX tutorials assume non-Jetson hardware. This is downstream of blocker 1 — if we can't get a working 1.x wheel on Jetson, we don't get the modern server.

**Sources:**
- `trtllm-serve` docs: https://nvidia.github.io/TensorRT-LLM/1.0.0rc2/commands/trtllm-serve.html
- Shahizat AGX Orin writeup (uses 0.12-jetson, Llama-2-7B, no serve-layer benchmarks, no Gemma): https://www.hackster.io/shahizat/running-llms-with-tensorrt-llm-on-nvidia-jetson-agx-orin-34372f

## Question 5 — POC and measurement

**Not performed.** Per the 4-hour constraint and Phase 1 gate, Questions 1–4 have already yielded enough red flags to abort. Spinning up the POC would burn the remaining budget building a known-stale container (0.12-r36.4.0) on the wrong JetPack, to run a non-Gemma-4 stand-in model (Llama-2 or Phi-2) whose tok/s number would not generalize to our workload.

---

## What the path would look like if we ignored this recommendation

For completeness, the minimum-viable engineering plan to make this work would be:

1. Fork `v0.12.0-jetson` and forward-port to L4T R36.4.7 / JetPack 6.2 (deal with libopenmpi, dill/datasets pin fights, NvInferVersion.h probing per issue #4502). Estimate: **2–5 days**.
2. Back-port the Gemma 3 PyTorch workflow (PR #3999) to the 0.12 branch — non-trivial, PyTorch workflow post-dates 0.12. Estimate: **3–5 days**.
3. Write a new `Gemma4ForCausalLM` PyTorch model with double-wide MLP, shared-KV, per-layer sliding, and proportional RoPE. Validate logit parity vs HF reference on x86 first. Estimate: **3–7 days**.
4. Unmerge our LoRA (or retrain-and-save HF-native), convert HF→int4-AWQ on the Jetson, build engine (risk: kernel-panic during build per forum reports). Estimate: **1–3 days**.
5. Stand up `trtllm-serve`, benchmark vs baseline. Estimate: **1 day**.

**Total: 10–21 working days.** We have ~3.5 weeks until deadline and one optimizer agent. This is not a responsible bet given MLC-LLM already delivers 5–8 tok/s (6× llama.cpp baseline) and the *inference* goal is effectively already achieved.

## Recommendation to the team

- **Hackathon deliverable**: Ship MLC-LLM at 5–8 tok/s. Do not touch TRT-LLM.
- **Post-hackathon**: If NVIDIA publishes a 1.x-family Jetson container with sm_87 and Gemma 3/4 support (watch `dustynv/tensorrt_llm` tags, NGC `jetson` tag on `tensorrt-llm`, and TRT-LLM releases mentioning "Orin"/"L4T"), re-open this investigation.
- **For C1 and C2**: continue on nvpmodel tuning and llama.cpp flag tuning respectively; TRT-LLM is not a near-term alternative path for them to hand off to.

---

## Citation summary

| Claim | Source |
|---|---|
| NVIDIA does not officially support TRT-LLM on Jetson | https://forums.developer.nvidia.com/t/orin-nano-building-tensorrt-llm-from-source/350532 |
| Mainline 1.2.1 support matrix excludes sm_87 | https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html |
| Jetson fork frozen at 0.12.0, targets JetPack 6.1 | https://github.com/NVIDIA/TensorRT-LLM/blob/v0.12.0-jetson/README4Jetson.md |
| Build broken on JetPack 6.2 | https://github.com/NVIDIA/TensorRT-LLM/issues/4502 |
| Gemma 3 support is 1B-text only | https://github.com/NVIDIA/TensorRT-LLM/pull/3999 |
| Gemma 3 request closed as not planned | https://github.com/NVIDIA/TensorRT-LLM/issues/3143 |
| Gemma 4 open runtime bug on DGX Spark (not Jetson) | https://github.com/NVIDIA/TensorRT-LLM/issues/12764 |
| AutoDeploy support matrix omits Gemma 4 | https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/support_matrix.html |
| LoRA path is HF safetensors only | https://nvidia.github.io/TensorRT-LLM/advanced/lora.html |
| `trtllm-serve` exposes OpenAI-compatible endpoints | https://nvidia.github.io/TensorRT-LLM/1.0.0rc2/commands/trtllm-serve.html |
| NVIDIA Jan 2026 edge guide recommends vLLM / llama.cpp, not TRT-LLM | https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/ |
| `dustynv/tensorrt_llm` has only the 0.12-r36.4.0 tag | https://hub.docker.com/r/dustynv/tensorrt_llm/tags |
| TRT-LLM v1.2.1 release (Apr 20, 2026) | https://github.com/NVIDIA/TensorRT-LLM/releases |
