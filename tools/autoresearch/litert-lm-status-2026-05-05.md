# LiteRT-LM Status Report — 2026-05-05

## Current Version & Release Date

**v0.11.0** • Released 2026-05-05 (today)

## Jinja Chat-Template Bug Status

**Unknown / Not found in public issues.** The handoff document (2026-04-25) explicitly cites "Jinja chat-template bug in 0.10.1" as a blocker that made LiteRT-LM "dead" for the iPhone path. However:

- No open GitHub issue titled "jinja" or "chat template" exists in the LiteRT-LM repo.
- No commits since 2026-01-01 reference jinja or chat_template fixes.
- v0.10.1 release notes (2026-04-03) mention CLI migration and speculative decoding support, but no Jinja bug workaround or caveat.

**Verdict: Unknown whether this was ever filed or is fixed.** If you experienced it with v0.10.1, you'll need to test v0.11.0 directly. The absence of a public issue suggests either: (a) it was fixed silently and not documented, or (b) the root cause was external (downstream Gemma 4 template changes in HuggingFace model repos, not LiteRT-LM itself).

## Gemma 4 MTP Support

**YES, confirmed in v0.11.0.** Evidence:

- **Release headline:** "Gemma 4 Multi-token Prediction (MTP) Support — Single Position Multi Token Prediction, delivering >2x faster decode speeds on mobile GPUs with zero quality degradation"
- **Commit (2026-04-30):** `[litertlm] add enable_speculative_decoding to C API` (Rev 904066251)
- **Recent build config (2026-04-29):** CI updated to use `gemma-4-E4B-it-litert-lm` as the default test model
- **Blog link:** https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/

**Usage:** CLI flag `--enable-speculative-decoding=true` and C/Python API parameter `enable_speculative_decoding: bool`.

**Caveat:** Issue #2181 (opened 2026-05-05) reports that on macOS, `enable_speculative_decoding` fails with a type mismatch in the Python Engine constructor. Status: open, likely a binding regression.

## iOS Swift API Status

**In development / partially available.** Key findings:

- **Issue #2125** (open, 2026-04-30): "Swift SDK roadmap & integration concerns" — external developer asking when the Swift SDK will ship with feature parity (Engine/Conversation, function calling, multimodal, streaming).
- **Issue #2160** (open, 2026-05-04): "When will Session::Clone / Conversation::Clone be available for iOS?" — users need `Conversation::Clone()` for warm KV cache reuse; currently unimplemented in `SessionBasic` (the default iOS engine).
- **Closed issue #2085** (2026-04-28): "Build .dylib correctly for iOS" — fixed, but indicates recent toolchain struggles.

**Prebuilt binaries:**
- v0.11.0 release *does* include iOS simulator binaries (e.g., `litert_lm_main.ios_sim_arm64`), but not explicitly advertised.
- **Issue #2158** (open, 2026-05-03): App Store rejects bundled iOS apps because `libGemmaModelConstraintProvider.dylib` is linked with `minos 26.2` (iOS 26.2 SDK), but other companion dylibs use `minos 14.0`. Workaround: patch with `vtool -set-build-version ios 16.0 26.2`.

**Verdict:** C API is stable on iOS; Swift SDK is not yet public. iOS apps can integrate via C bindings or direct C API calls, but the Swift wrapper you'd want doesn't exist. Prebuilt iOS binaries exist but have deployment target issues (#2158).

## Top 3 Sharp Edges (April–May 2026)

1. **#2181 — Speculative decoding on macOS broken.** TypeError in Python Engine constructor when `enable_speculative_decoding=True` on macOS (opened 2026-05-05, still open). May affect iOS Metal path similarly. *Impact: Cannot use MTP feature on device until fixed.*

2. **#2149 — CPU decode crash on Gemma 4.** Segfault on E4B, silent hang on E2B post-prefill when using the C API `litert_lm_conversation_send_message`. Only decode phase affected; prefill succeeds. Affects Linux x86_64 docker, likely affects any non-GPU backend. *Impact: Tool-calling / multi-step inference on CPU-only devices fails.* Status: open, root cause unknown.

3. **#2158 — iOS App Store deployment target mismatch.** `libGemmaModelConstraintProvider.dylib` shipped with `minos 26.2` instead of `14.0`; App Store rejects bundles. Workaround exists (vtool patch), but blocks out-of-the-box iOS deployment. *Impact: Any iOS app shipping v0.11.0 prebuilt binaries must patch dylibs post-download or use a Flutter wrapper (flutter_gemma v0.14.3) that does this for you.*

## Verdict: GO / NO-GO for iPhone LiteRT-LM Path (13-Day Hackathon Window)

**CONDITIONAL GO, with risks.**

**Rationale:**
- ✅ Gemma 4 MTP support is live and documented (unblocks the original Google announcement claim).
- ✅ Prebuilt iOS binaries exist in v0.11.0 release.
- ❌ Jinja bug status unknown — no evidence it's fixed; test v0.11.0 directly on your use case to be sure.
- ❌ Speculative decoding (the headline MTP feature) broken on macOS (#2181, open); likely affects iOS Metal path.
- ❌ Swift SDK doesn't exist — you'll need C API bindings or raw C calls.
- ❌ iOS App Store deployment blocker (#2158) — dylib minos version must be patched.
- ⚠️ CPU decode crashes on Gemma 4 (#2149) — if your iPhone runs on CPU, tool-calling will hang/crash.

**Action items to unblock the iPhone path within 13 days:**
1. Test v0.11.0 with your Gemma 4 model on iOS simulator to confirm Jinja bug is resolved (or document workaround).
2. Verify Gemma 4 CPU decoding on A-series chip (physical iPhone 15 Pro if available). If it hangs/crashes, fallback to Metal/GPU backend only.
3. Patch `libGemmaModelConstraintProvider.dylib` deployment target before App Store submission (vtool or flutter_gemma v0.14.3 precedent).
4. Build C API wrapper for Conversation/Engine in Swift, or use raw C bindings (Swift C interop).
5. Monitor #2181 (speculative decoding) for a fix; if not fixed by 2026-05-10, disable `enable_speculative_decoding` on iOS.

**Do not rely on v0.11.0 as-is for iPhone production; pre-flight testing is mandatory.**

---

**Report Date:** 2026-05-05 (v0.11.0 release day)  
**Source:** GitHub API, v0.11.0 release notes, issues #2125, #2149, #2158, #2160, #2181
