# Mobile Feasibility — Gemma 4 E2B + compact LoRA for offline clinic demo

**Team C5** - 2026-04-23 - branch `team/c5-mobile-2026-04-23`

Scope: can we run Gemma 4 E2B + our 48 MB clinical LoRA *fully offline* on a consumer iPhone / Android handset in < 30-60 s per eICR case? Short answer: **yes, comfortably, on the GPU path of any modern flagship from the last two years.** Details below.

---

## Q1 — Support for **Gemma 4** (not Gemma 2 / 3)

| Runtime | Gemma 4 support? | Evidence |
|---|---|---|
| **LiteRT-LM** (Google AI Edge) | **Yes — first-class.** Gemma 4 E2B and E4B ship as official `.litertlm` bundles on HF `litert-community`. Shipped in LiteRT-LM v0.10.1 (Apr 2 2026); `v0.10.2` landed Apr 14 2026. | [LiteRT-LM GitHub](https://github.com/google-ai-edge/LiteRT-LM), [Gemma 4 edge launch blog](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/), [HF litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) |
| **MediaPipe LLM Inference API** | Gemma 2/3 only. **Android + iOS mobile implementations officially deprecated**; Google explicitly says "Migrate your mobile projects to LiteRT-LM." Web path is still supported. | [LLM Inference guide for iOS](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios), [mediapipe issue #5255](https://github.com/google-ai-edge/mediapipe/issues/5255) |
| **MLC-LLM mobile** | **No.** Nightly 2026-04-05 still errors `ValueError: Unknown model type: gemma4`. Architecture issues documented (nested `model.language_model.*` weight path, unhandled `audio_tower`/`vision_tower`, missing `query_pre_attn_scalar`/`head_dim` at root). Same class of patching our Jetson port is doing. | [mlc-llm issue #3477](https://github.com/mlc-ai/mlc-llm/issues/3477) |
| **llama.cpp iOS/Android** | **Yes — day-0 support on Apr 2 2026** for all Gemma 4 sizes; works with `llama-cli`, `llama-server`, `llama-mtmd-cli` (multimodal). Our existing GGUFs drop in. | [HF gemma4 blog post](https://huggingface.co/blog/gemma4) |
| **Apple MLX (iOS)** | **Yes.** `mlx-community/gemma-4` collection on HF has E2B/E4B ports in 4-bit and 8-bit. Demo'd running on iPhone via Locally AI. | [mlx-community Gemma 4 collection](https://huggingface.co/collections/mlx-community/gemma-4), [StartupHub Gemma-4-on-iPhone-MLX](https://www.startuphub.ai/ai-news/artificial-intelligence/2026/gemma-4-runs-on-iphone-using-mlx) |

**Bottom line:** Three real choices for Gemma 4 today — LiteRT-LM, llama.cpp, MLX. MediaPipe and MLC-LLM are out for our deadline.

---

## Q2 — Speed (tok/s) on flagship phones for Gemma 4 E2B

Official Google benchmarks (`ai.google.dev/edge/litert-lm/overview`, LiteRT-LM docs):

| Device | Backend | Prefill tk/s | **Decode tk/s** | TTFT | Peak mem |
|---|---|---|---|---|---|
| Samsung S26 Ultra | CPU | 557 | 47 | 1.8 s | 1733 MB |
| **Samsung S26 Ultra** | **GPU** | **3808** | **52** | **0.3 s** | **676 MB** |
| iPhone 17 Pro | CPU | 532 | 25 | 1.9 s | 607 MB |
| **iPhone 17 Pro** | **GPU** | **2878** | **56** | **0.3 s** | **1450 MB** |
| MacBook Pro M4 | GPU | 7835 | 160 | 0.1 s | 1623 MB |
| Raspberry Pi 5 | CPU | 133 | 8 | 7.8 s | 1546 MB |

[Source: ai.google.dev/edge/litert-lm/overview](https://ai.google.dev/edge/litert-lm/overview)

Community data points:

- iPhone 14 (A15) running Gemma 4 E2B via AI Edge Gallery: ~12 tok/s, 1-1.5 GB RAM live ([modelfit.io](https://modelfit.io/iphone/iphone-14/)).
- iPhone 15 Pro and newer via Google AI Edge Gallery: up to 30 tok/s ([XDA](https://www.xda-developers.com/google-gemma-4-finally-made-me-care-about-running-local-llms/)).
- No published Pixel 8/9 numbers specifically for Gemma 4 E2B yet — but Samsung S26 Ultra GPU ≈ iPhone 17 Pro GPU (both ~52-56 decode) suggests Pixel 9 Tensor G4 will be in the same ballpark.

**Our workload budget check:** ~700-token prefill + ~500-token decode.
- iPhone 17 Pro GPU: 700/2878 + 500/56 ≈ 0.24 s + 8.9 s = **~9 s**.
- Samsung S26 Ultra GPU: 700/3808 + 500/52 ≈ 0.18 s + 9.6 s = **~10 s**.
- iPhone 14 (worst case floor, A15): 700/~150 + 500/12 ≈ 4.7 s + 41.7 s = **~46 s** — still inside the 60 s budget.

All four priority devices meet the 30-60 s demo budget. We actually have headroom to ship to a 3-year-old iPhone.

---

## Q3 — LoRA adapter loading

- **LiteRT-LM: no documented runtime LoRA.** Must pre-merge into base weights, export as a single `.litertlm` bundle. Our compact LoRA is 48 MB; merged base at 4-bit stays ≈ 2.58 GB (the number Google quotes for `gemma-4-E2B-it-litert-lm`). Merge path: HF `transformers` + `PeftModel.merge_and_unload()` + Google's `ai-edge-torch` converter; tutorials exist ([Lushbinary QLoRA merge](https://lushbinary.com/blog/fine-tune-gemma-4-lora-qlora-complete-guide/)).
- **MediaPipe:** static LoRA at init time ([iOS guide](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios)) — but the whole mobile API is deprecated, so this is a dead end for us.
- **llama.cpp:** LoRA supported via `--lora` flag at load time; no merge required. Our 48 MB `cliniq-compact-lora.gguf` works as-is. **Cheapest LoRA path of any option.**
- **MLX / SwiftLM:** No LoRA loading surface documented in SwiftLM; MLX core supports LoRA training/merging (`mlx-lm fuse`), so pre-merge is the path. Comparable to LiteRT-LM.

**On-device file size if we pre-merge (LiteRT-LM path):** ~2.58 GB for E2B-int4 `.litertlm`. Fine for a modern phone.

---

## Q4 — Integration cost per option (to working demo app)

Assumes 1 mobile engineer, both iOS and Android targets.

| Option | Fork/base | Effort | Why |
|---|---|---|---|
| **LiteRT-LM + AI Edge Gallery fork** | `github.com/google-ai-edge/gallery` (Swift iOS + Kotlin Android, entirely open-source, Apache-2.0) | **2-3 days** | Official sample app already does: download-from-HF, load `.litertlm`, stream tokens, chat UI. We strip chat, add "paste eICR → extract JSON" flow + our merged model bundled in-app or side-loaded. Minimum iOS 17. |
| **llama.cpp (`llama.swiftui` fork)** | `llama.cpp/examples/llama.swiftui` + Android Termux or `llama.android` sample | **3-5 days** | Less packaged than AI Edge Gallery; Android story is weaker (no official UI app, Termux is hobbyist). Upside: our existing GGUFs + LoRA drop in unchanged; no merge step. Metal on iPhone is solid; OpenCL/Vulkan on Android is flakier per-device. |
| **MLX + SwiftBuddy fork** | `github.com/SharpAI/SwiftLM` (SwiftBuddy sample app, "running live on iPhone 13 Pro 6 GB") | **3-4 days iOS-only** | Great for iPhone/iPad; **zero Android story** (MLX is Apple Silicon only). If the user commits to iOS only, this is the cleanest. Would need LoRA pre-merge via `mlx-lm fuse`. |
| MLC-LLM mobile | existing `MLCChat` apps | **10+ days + blocked** | Needs same class of Gemma 4 architecture patches we're fighting on Jetson. Not recommended. |
| MediaPipe | deprecated | n/a | Don't. |

---

## Q5 — Multimodal / future-proof (vision for clinical images)

- **LiteRT-LM:** Gemma 4 E2B is multimodal-capable; Google's edge launch post explicitly targets "multimodal Gemma 4 E2B on the edge". Qualcomm QNN demos show vision on Snapdragon 8 Elite (Android). iOS vision support status is less clearly documented but the `litertlm` format includes the vision tower. **Future-proof — yes.**
- **llama.cpp:** `llama-mtmd-cli` supports Gemma 4 vision at launch. Swift wrappers usually lag a few weeks behind multimodal features.
- **MLX:** Text-only is mature; vision ports for Gemma 4 exist in `mlx-vlm` upstream but not all quantizations. Some gap.
- **MediaPipe:** deprecated on mobile, moot.

For a future clinical image extension (e.g., scanned lab report OCR → structured fields), LiteRT-LM is the safest bet.

---

## Key citations

- [LiteRT-LM overview + benchmark table](https://ai.google.dev/edge/litert-lm/overview)
- [LiteRT-LM GitHub releases](https://github.com/google-ai-edge/LiteRT-LM)
- [Gemma 4 edge launch blog (developers.googleblog)](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)
- [Google AI Edge Gallery release v1.0.11 (2026-04-02, Gemma 4 support)](https://github.com/google-ai-edge/gallery/releases)
- [Google AI Edge Gallery on iOS App Store](https://apps.apple.com/us/app/google-ai-edge-gallery/id6749645337)
- [Google AI Edge Gallery on Google Play](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery)
- [HF litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
- [HF mlx-community Gemma 4 collection](https://huggingface.co/collections/mlx-community/gemma-4)
- [HF gemma4 launch blog (llama.cpp + MLX day-0)](https://huggingface.co/blog/gemma4)
- [MLC-LLM Gemma 4 status issue #3477](https://github.com/mlc-ai/mlc-llm/issues/3477)
- [MediaPipe LoRA merge issue #5255](https://github.com/google-ai-edge/mediapipe/issues/5255)
- [MediaPipe deprecation notice (LLM Inference iOS guide)](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios)
