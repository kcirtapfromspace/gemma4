# Mobile Runtime Recommendation

**TL;DR — Fork Google AI Edge Gallery, bundle a LoRA-merged Gemma 4 E2B `.litertlm`, ship iOS + Android demo in ~3 days engineering.**

---

## Recommended path: LiteRT-LM via the AI Edge Gallery fork

**Why this one, not llama.cpp or MLX:**

1. **Only option with a first-party, open-source, published-to-both-stores sample app** (`github.com/google-ai-edge/gallery`, Swift iOS + Kotlin Android, iOS 17+). Gemma 4 E2B support shipped April 2 2026.
2. **Official benchmarks prove the target budget with 5-6x margin.** iPhone 17 Pro GPU: 56 tok/s decode, 0.3 s TTFT, 1.45 GB peak RAM. Samsung S26 Ultra GPU: 52 tok/s. Even iPhone 14 hits ~12 tok/s which still clears our 60 s/case ceiling.
3. **Actively maintained by Google with Gemma 4 as the flagship demo model** — we inherit future Gemma 4 mobile upstream work (vision, NPU, quantization improvements) for free.
4. **Our `.litertlm` pipeline is well-trodden** — HF has E2B-int4 pre-quantized at 2.58 GB ready to download; `ai-edge-torch` converts a LoRA-merged HF model to `.litertlm` in one command.
5. **NPU path opens up on Android.** Qualcomm QNN on Snapdragon 8 Elite hit 3700 prefill / 31 decode tk/s on Dragonwing IQ8 per the Gemma 4 edge blog. For future hardware acceleration this has a runway MLX and llama.cpp don't share.

## Runner-up: llama.cpp (pick this instead if Android LoRA-swappability matters more than polish)

- Only option that keeps LoRA as a separate runtime-loadable adapter (our existing `cliniq-compact-lora.gguf` just works, no merge step, no re-export when we retrain).
- Downside: no official Android app. `llama.swiftui` works on iOS but is a tech demo UX. More plumbing, less polish.
- Rough effort: 3-5 days.

## Hard no on MLC-LLM and MediaPipe

- MLC-LLM mobile: Gemma 4 architecture support is blocked on the same patches our Jetson port is still fighting. Wrong tool for this timeline.
- MediaPipe LLM Inference API: **officially deprecated on mobile** by Google in favor of LiteRT-LM.

---

## Reference app to fork

- **Repo:** `https://github.com/google-ai-edge/gallery` (Apache-2.0)
- **iOS (Swift):** builds with Xcode 15+, deploys iOS 17+
- **Android (Kotlin):** builds with Android Studio, targets Android 12+ / API 31+
- **Published binaries:** live on [App Store](https://apps.apple.com/us/app/google-ai-edge-gallery/id6749645337) and [Google Play](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery) — worth installing to validate device performance before committing engineering.
- **Model source:** `huggingface.co/litert-community/gemma-4-E2B-it-litert-lm` (2.58 GB int4)

## Engineering plan (3 days, 1 mobile eng)

**Day 0 (prep, ~4 h, doable now by any team member):**
- Merge LoRA: `PeftModel.from_pretrained(base, "models/cliniq-compact-lora") ; model.merge_and_unload() ; model.save_pretrained("models/cliniq-gemma4-e2b-merged/")`.
- Convert to `.litertlm`: `ai-edge-torch convert --model-path ... --output cliniq-gemma4-e2b.litertlm --quantize int4`.
- Host the `.litertlm` on a private HF repo or bundle in-app (~2.6 GB TestFlight payload — fine for internal distribution; use app-side download for production).

**Day 1:** Fork gallery, strip the generic chat UI, add a single-screen "Paste eICR → Extract" flow. Swap in our model. Wire up the clinical extraction system prompt.
**Day 2:** JSON output parsing, retry-on-malformed, basic error handling, loading UI. Test on physical iPhone 15 Pro + Pixel 8.
**Day 3:** Polish, packaging, TestFlight/Firebase App Distribution setup for clinic-worker handoff.

## What we'd lose going from Jetson cluster to phone

| Dimension | Jetson Orin NX 8GB (current) | Phone (proposed) | Impact |
|---|---|---|---|
| Decode throughput | 5-8 tok/s (MLC-LLM port) | **52-56 tok/s GPU on flagship, ~12 tok/s on A15** | **Phone is actually faster for flagship devices.** |
| Prefill | modest | 2878-3808 tk/s GPU | Phone wins, TTFT drops from seconds to ~0.3 s. |
| LoRA hot-swap | yes (llama.cpp) | no — must pre-merge and re-ship | Each retrain = new app build / model download. ~30 min pipeline overhead per iteration. |
| Concurrency | multi-tenant possible | single user per device | Irrelevant for single-clinic-worker use case. |
| Power / thermals | wall-powered | battery, throttles after sustained load | For a ~10 s "submit and wait" workflow this is fine; not fine for streaming-heavy use. |
| Multimodal vision | our port doesn't do vision today | LiteRT-LM Gemma 4 vision tower is on-device-capable | **We gain** a plausible future clinical-image path. |
| Offline operation | already offline | fully offline after first-time model download | Equivalent. Check the box. |
| Deploy overhead | talos/k8s/kubectl | App Store / Play / TestFlight / sideload | Phone is dramatically simpler for the target use case (remote clinic worker with a handset). |

**Net:** phone deployment is *better* than Jetson for this demo on every axis except runtime LoRA swappability. The LoRA merge-and-ship cost is a one-time ~30 min CI step.

---

## Go/no-go recommendation

**Go.** If leadership signs off on the LoRA-merge constraint, this is a 3-day engineering task with a known-good reference app, official Google-maintained runtime, documented benchmarks 5x above our budget, and an iOS+Android deploy path that actually matches the remote-clinic field scenario. See `SKETCH.md` for the minimal integration code.
