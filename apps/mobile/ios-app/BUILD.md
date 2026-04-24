# ClinIQ iOS — Build, Run, and Sideload

Team C10 — 2026-04-23 — branch `team/c10-ios-app-2026-04-23`.

This doc captures the exact commands used to build, install, and launch
ClinIQ in the iPhone 17 Pro simulator on this Mac, plus notes on a physical
iPhone sideload with a free Apple ID.

## Environment

- Xcode 26.4.1 (17E202)
- iOS SDK: `iPhoneSimulator26.4.sdk`
- iPhone 17 Pro simulator: UDID `CADA1806-F64D-4B02-B983-B75F197D1EF3`, iOS 26.4
- Swift 5.0, deployment target iOS 17.0
- No third-party dependencies (see "Inference backend" below — LiteRT-LM is
  not currently linked; simulator runs use the stub engine).

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

# 4. Launch (optionally with auto-extract for headless validation)
SIMCTL_CHILD_CLINIQ_AUTO_EXTRACT=1 \
  xcrun simctl launch --terminate-running-process \
    CADA1806-F64D-4B02-B983-B75F197D1EF3 \
    com.cliniq.ClinIQ

# 5. Capture a screenshot of the running app (after the extract completes)
sleep 8
xcrun simctl io CADA1806-F64D-4B02-B983-B75F197D1EF3 \
  screenshot screenshot.png
```

The auto-extract env var is handled in `ContentView.swift` `.onAppear`. When
unset (normal launch from the springboard), the user taps Extract manually.

## Run the headless validator (stub)

```bash
cd apps/mobile/ios-app
swift validate.swift
```

Outputs CSV to stdout with per-case scores and the final total. See
`VALIDATION.md` for interpretation.

## Inference backend

The project is factored behind `InferenceEngine` (see
`ClinIQ/Inference/InferenceEngine.swift`). Two concrete backends were
scoped:

1. **`StubInferenceEngine`** — deterministic, no-model, ships today. Used
   for all simulator runs. It emits a minified JSON derived from regex rules
   that mirror the SKILL.md examples. It is NOT a language model; its only
   purpose is to prove the app scaffolding is correctly wired.
2. **`LiteRtLmEngine`** (not yet written) — will wrap `litert-lm`'s Swift
   `Session` API once Google publishes the Swift bindings. As of 2026-04-23
   the LiteRT-LM README lists Swift as **"In Dev / Coming Soon"**
   ([repo root](https://github.com/google-ai-edge/LiteRT-LM)). There is no
   `Package.swift`, no xcframework, and no published static library
   artifact. The only iOS deliverable in the LiteRT-LM v0.9.0 release is
   `litert_lm_main.ios_sim_arm64` — a standalone CLI binary, not a linkable
   library. Vendoring the C++ core via Bazel + bridging header (as sketched
   in `apps/mobile/SKETCH.md`) is feasible but requires a ~1-day effort
   cross-compiling LiteRT-LM's Bazel build with the iOS toolchain; this sat
   outside the 4-hour budget. Current recommendation: ship the stub to
   prove the SwiftUI + prompt-formatting + streaming + scoring surfaces end
   to end; revisit LiteRT-LM integration when the Swift bindings drop or
   when we allocate a full day to the Bazel-iOS build.

When a working `LiteRtLmEngine` lands, the swap in
`ExtractionViewModel.makeDefaultEngine()` is a one-line change.

## Model distribution (stubbed)

The spec calls for a first-run HF download of the stock
`litert-community/gemma-4-E2B-it-litert-lm` bundle (~2.58 GB) cached into
`FileManager.default.urls(for: .documentDirectory, ...).first!`. Because
the current backend is the stub, no model file is actually needed and the
download path is not yet wired. The download code shape is:

```swift
// Pseudocode for LiteRtLmEngine init:
let docsURL = FileManager.default
  .urls(for: .documentDirectory, in: .userDomainMask).first!
let modelURL = docsURL.appending(path: "gemma-4-E2B-it.litertlm")
if !FileManager.default.fileExists(atPath: modelURL.path) {
    try await downloadFromHF(
      repo: "litert-community/gemma-4-E2B-it-litert-lm",
      file: "gemma-4-E2B-it.litertlm",
      into: modelURL,
      progress: progressBlock)
}
```

This path is commented in `InferenceEngine.swift`'s TODO; the stub ignores
the model path entirely.

## Physical iPhone sideload (free Apple ID)

NOT tested on a physical device — the spec requires simulator only for C10
but documents the sideload steps for later.

1. Plug an iPhone in via USB and trust the Mac from the device.
2. In Xcode, open `apps/mobile/ios-app/ClinIQ/ClinIQ.xcodeproj`.
3. Select the **ClinIQ** target → **Signing & Capabilities**.
4. Toggle **Automatically manage signing**.
5. Under **Team**, click **Add an Account** and sign in with a free Apple
   ID. The account appears as "Personal Team".
6. Select that Personal Team. Xcode will provision a local, 7-day signing
   certificate. Bundle ID `com.cliniq.ClinIQ` is already unique to this
   project so the provisioning profile should auto-generate.
7. Change the destination to the connected device (top bar, next to Run).
8. Click Run (Cmd-R). First run asks the user to trust the developer
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
├── screenshot.png                    (running app on simulator)
├── validate.swift                    (headless validator harness)
└── ClinIQ/
    ├── ClinIQ.xcodeproj/             (hand-authored pbxproj, scheme)
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
        │   └── StubInferenceEngine.swift (regex stub)
        └── Resources/
            └── Info.plist
```
