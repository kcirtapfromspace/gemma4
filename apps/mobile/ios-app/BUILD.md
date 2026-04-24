# ClinIQ iOS — Build, Run, and Sideload

Team C10 — 2026-04-23 — initial SwiftUI scaffold + stub engine.
Team C12 — 2026-04-23 — real inference via the llama.cpp xcframework.

This doc captures the exact commands used to build, install, and launch
ClinIQ in the iPhone 17 Pro simulator on this Mac, plus notes on a physical
iPhone sideload with a free Apple ID.

## Environment

- Xcode 26.4.1 (17E202)
- iOS SDK: `iPhoneSimulator26.4.sdk`
- iPhone 17 Pro simulator: UDID `CADA1806-F64D-4B02-B983-B75F197D1EF3`, iOS 26.4
- Swift 5.0, deployment target iOS 17.0
- **Vendored dependency**: `llama.cpp` b8913 xcframework at
  `ClinIQ/Frameworks/llama.xcframework/` (iOS arm64 + iOS sim fat arm64/x86_64;
  other slices stripped for size — 153 MB → 14 MB after stripping dSYMs and
  non-iOS slices). Downloaded from
  https://github.com/ggml-org/llama.cpp/releases/tag/b8913
  (asset `llama-b8913-xcframework.zip`).

## Simulator build + run (copy-paste)

```bash
# 1. Build for the iPhone 17 Pro simulator
cd apps/mobile/ios-app/ClinIQ
xcodebuild \
  -project ClinIQ.xcodeproj \
  -scheme ClinIQ \
  -destination 'id=CADA1806-F64D-4B02-B983-B75F197D1EF3' \
  -configuration Debug \
  -derivedDataPath build \
  build

# 2. Boot the simulator (idempotent)
xcrun simctl boot CADA1806-F64D-4B02-B983-B75F197D1EF3 || true

# 3. Install the freshly-built app bundle
xcrun simctl install \
  CADA1806-F64D-4B02-B983-B75F197D1EF3 \
  build/Build/Products/Debug-iphonesimulator/ClinIQ.app

# 4. SEED THE MODEL into the simulator's Documents dir (first run only;
#    persists across re-installs).
CONTAINER=$(xcrun simctl get_app_container \
  CADA1806-F64D-4B02-B983-B75F197D1EF3 com.cliniq.ClinIQ data)
mkdir -p "$CONTAINER/Documents"
cp ../../../models/cliniq-gemma4-e2b-Q3_K_M.gguf \
   "$CONTAINER/Documents/"
# (optional base-model fallback, smaller + faster for first smoke test:)
# cp ../../../models/gemma-4-E2B-it-Q3_K_M.gguf "$CONTAINER/Documents/"

# 5. Launch (optionally with auto-extract for headless validation)
# The CLINIQ_CASE env var picks one of the 5 bundled test cases before
# auto-extract fires — use this to batch-run the full validation set.
# Case IDs: bench_minimal, bench_typical_covid, bench_complex_multi,
#            bench_meningitis, bench_negative_lab.
SIMCTL_CHILD_CLINIQ_AUTO_EXTRACT=1 \
SIMCTL_CHILD_CLINIQ_CASE=bench_typical_covid \
  xcrun simctl launch --terminate-running-process --console \
    CADA1806-F64D-4B02-B983-B75F197D1EF3 \
    com.cliniq.ClinIQ

# After the run completes, read the persisted JSON output for scoring:
CONTAINER=$(xcrun simctl get_app_container \
  CADA1806-F64D-4B02-B983-B75F197D1EF3 com.cliniq.ClinIQ data)
cat "$CONTAINER/Documents/extractions.log"

# 6. Capture a screenshot of the running app
#    (first extract can take 2-5 minutes cold on simulator CPU — see
#    "Performance" below)
xcrun simctl io CADA1806-F64D-4B02-B983-B75F197D1EF3 \
  screenshot screenshot-llamacpp.png
```

The auto-extract env var is handled in `ContentView.swift` `.onAppear`. When
unset (normal launch from the springboard), the user taps Extract manually.

## Run the headless validator (stub)

```bash
cd apps/mobile/ios-app
swift validate.swift
```

This still runs against the regex stub (dependency-free). The real model
is exercised in-app via the auto-extract env var and the per-case test
selector in the UI. See `VALIDATION.md` for per-case scores from both
paths.

## Inference backend

The project is factored behind `InferenceEngine` (see
`ClinIQ/Inference/InferenceEngine.swift`). Three concrete backends exist:

1. **`LlamaCppInferenceEngine`** (C12, default) — wraps the vendored
   `llama.xcframework`. Loads a GGUF on first `generate(...)` call, streams
   tokens through an `actor LlamaContext`. Preference order for the model
   file:
   1. `Bundle.main.url(forResource: "<name>", withExtension: "gguf")`
   2. `FileManager.default.urls(for: .documentDirectory, ...)` first dir
   3. `NSTemporaryDirectory()`

   Candidate names tried in order: `cliniq-gemma4-e2b-Q3_K_M`,
   `gemma-4-E2B-it-Q3_K_M`, `cliniq-gemma4-e2b-Q2_K`.

   Simulator always forces `n_gpu_layers = 0` (CPU only — the simulator's
   Metal implementation does not expose the tensor features ggml needs).
   Physical iPhone leaves the framework default, which uses Metal when
   available.

2. **`StubInferenceEngine`** — deterministic regex fallback. Kept as the
   CI / SwiftUI-Preview engine for when no GGUF is seeded. `ExtractionViewModel.
   makeDefaultEngine()` picks it automatically when
   `LlamaCppInferenceEngine.resolveModelPath() == nil`.

3. **(historical) `LiteRtLmEngine`** — Google's in-progress Swift binding.
   See the prior version of this doc in git history. Not currently wired;
   superseded by llama.cpp as of C12.

Swap in `ExtractionViewModel.makeDefaultEngine()` is a one-line change.

## Model distribution

The GGUF files are 2.5-3.2 GB each; **do not** commit them to git
(`.gitignore` covers `*.gguf`). Three distribution options:

1. **Seed into simulator sandbox** (dev only, fastest iteration):
   Use the `simctl get_app_container ... data` + `cp` pattern in § 4 above.
   Model persists until the app is uninstalled.

2. **Bundle into the .app** (simulator demo, larger build time):
   Drop the GGUF into `ClinIQ/ClinIQ/Resources/Models/` and add it to the
   app target's Copy Bundle Resources phase in Xcode. `Bundle.main.url(
   forResource:withExtension:)` picks it up automatically. Avoid for
   TestFlight — archives >4 GB hit App Store Connect limits.

3. **First-launch download** (production path, not wired):
   Download from HF or a dev endpoint into
   `FileManager.default.urls(for: .documentDirectory, ...).first!` on first
   launch. Sketch left in git history on the C10 branch. Needs a progress
   UI and resume support; out of scope for the 4-hour C12 budget.

## Performance (simulator CPU)

The iPhone 17 Pro simulator runs under x86_64 Rosetta on Apple Silicon, and
ggml's gemma4 graph is split into ~311 CPU segments (no SIMD fusion for
sliding-window + gated-delta kernels on the simulator backend). Expect:

- **Model load**: ~15-25 s (mmap + KV cache reservation)
- **Prompt prefill**: ~30-90 s for 300-500 tokens
- **Decode tok/s**: 1-1.5 tok/s (observed 1.3 tok/s on Q3_K_M fine-tune,
  gemma4 E2B, 4-thread CPU path, iPhone 17 Pro simulator)
- **Resident memory**: ~4.3 GB peak (close to simulator's 4-6 GB working set)

These numbers are **expected** on simulator CPU. Real iPhone with Metal is
projected at 10-20 tok/s decode on the 17 Pro A19 GPU per upstream
benchmarks; physical-device validation is left for team C13.

For scoring a single case during the demo, plan on **2-5 minutes per
extraction** on simulator. Running all 5 bundled test cases back-to-back
on simulator in one launch is feasible but slow (~15-25 min total); the
auto-extract env var only triggers the first case — subsequent cases need
a manual tap. See `VALIDATION.md` for per-case methodology.

## Physical iPhone sideload (free Apple ID)

NOT tested on a physical device by C12 either. Steps unchanged from C10:

1. Plug an iPhone in via USB and trust the Mac from the device.
2. In Xcode, open `apps/mobile/ios-app/ClinIQ/ClinIQ.xcodeproj`.
3. Select the **ClinIQ** target → **Signing & Capabilities**.
4. Toggle **Automatically manage signing**.
5. Under **Team**, click **Add an Account** and sign in with a free Apple
   ID. The account appears as "Personal Team".
6. Select that Personal Team. Xcode will provision a local, 7-day signing
   certificate. Bundle ID `com.cliniq.ClinIQ` is already unique to this
   project so the provisioning profile should auto-generate.
7. Copy the GGUF to the device via one of:
   - Files.app (Shared / This iPhone / ClinIQ) if you add
     `LSSupportsOpeningDocumentsInPlace`=YES and `UIFileSharingEnabled`=YES
     to `Info.plist` (future work).
   - Download at first launch from a known URL (add network permissions
     and the download code).
8. Change the destination to the connected device (top bar, next to Run).
9. Click Run (Cmd-R). First run asks the user to trust the developer
   profile on the iPhone under **Settings → General → VPN & Device
   Management**. After trusting, the app runs for 7 days before re-signing
   is required.

Caveats with a free Apple ID:
- App expires after 7 days; re-run from Xcode to re-sign.
- Limit of 3 free apps per phone.
- A **paid** Apple Developer account ($99/yr) is required for TestFlight,
  App Store distribution, and longer-lived local dev certificates.
- Personal-team provisioning profiles don't support some entitlements
  (push notifications, CloudKit, HealthKit, etc.). ClinIQ requires none of
  those.

## Project layout

```
apps/mobile/ios-app/
├── BUILD.md                          (this file)
├── VALIDATION.md                     (per-case extraction scores)
├── screenshot.png                    (C10 stub, bench_typical_covid)
├── screenshot-llamacpp.png           (C12 real inference, same case)
├── validate.swift                    (headless validator — stub path)
└── ClinIQ/
    ├── ClinIQ.xcodeproj/             (hand-authored pbxproj, scheme)
    ├── Frameworks/
    │   └── llama.xcframework/        (C12 — vendored llama.cpp b8913)
    └── ClinIQ/
        ├── ClinIQApp.swift           (@main entry point)
        ├── Views/
        │   ├── ContentView.swift
        │   └── ExtractionViewModel.swift
        ├── Models/
        │   └── TestCase.swift        (5 bundled test cases)
        ├── Inference/
        │   ├── PromptBuilder.swift   (unsloth gemma-4 turn delimiters)
        │   ├── InferenceEngine.swift (protocol + errors)
        │   ├── StubInferenceEngine.swift     (regex fallback)
        │   └── LlamaCppInferenceEngine.swift (C12 — real inference)
        └── Resources/
            └── Info.plist
```
