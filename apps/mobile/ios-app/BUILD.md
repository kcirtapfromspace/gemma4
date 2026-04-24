# ClinIQ iOS — Build, Run, and Sideload

Team C10 — 2026-04-23 — initial SwiftUI scaffold + stub engine.
Team C12 — 2026-04-23 — real inference via the llama.cpp xcframework.
Team C13 — 2026-04-23 — clinician field case-reporting PoC (this version).

This doc captures the exact commands used to build, install, and launch
ClinIQ in the iPhone 17 Pro simulator on this Mac, plus notes on a physical
iPhone sideload with a free Apple ID.

## What changed in C13

- **New product shape**: tab-based clinician app (Cases / Outbox / History
  / Settings). The C10/C12 developer JSON dumper now lives as a legacy
  testbench (`ContentView`); the app root is `RootView.swift`. See
  `LEGACY.md`.
- **Persistence**: SwiftData store under Application Support/ with file
  protection `completeUntilFirstUserAuthentication`. Models: `ClinicalCase`,
  `Patient`, `ExtractedCondition`, `ExtractedLab`, `ExtractedMedication`,
  `Vitals`, `SyncRecord`.
- **Sync**: `SyncService` actor that drains pending cases to a (mock)
  public-health endpoint. NWPathMonitor-based offline banner, auto-sync
  on network return, per-attempt audit history.
- **UX**: Every user-facing view shows conditions / labs / meds / vitals as
  clean rows with the coded identifier in muted subtitle type. No raw JSON
  in the UI.
- **Inference**: unchanged — reuses `LlamaCppInferenceEngine` with
  `<|turn>` prompt wrapping.

## Environment

- Xcode 26.4.1 (17E202)
- iOS SDK: `iPhoneSimulator26.4.sdk`
- iPhone 17 Pro simulator: UDID `CADA1806-F64D-4B02-B983-B75F197D1EF3`, iOS 26.4
- Swift 5.0, deployment target iOS 17.0
- **Vendored dependency**: `llama.cpp` b8913 xcframework at
  `ClinIQ/Frameworks/llama.xcframework/` (iOS arm64 + iOS sim fat
  arm64/x86_64).

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

# 5. Launch — the app seeds 4 realistic demo cases on first launch.
xcrun simctl launch --terminate-running-process \
  CADA1806-F64D-4B02-B983-B75F197D1EF3 \
  com.cliniq.ClinIQ
```

## Demo-mode environment variables

The app honours several env vars so the screenshot harness can jump
directly to a UI state without chasing tap coordinates.

| Var                           | Effect                                                |
| ----------------------------- | ----------------------------------------------------- |
| `CLINIQ_SIMULATE_OFFLINE=1`   | Forces the NetworkMonitor to report offline, shows the banner, disables Sync button. |
| `CLINIQ_TAB=outbox\|history\|settings` | Selects a tab on launch. |
| `CLINIQ_OPEN_NEW_CASE=1`      | Presents the NewCaseView sheet over the Cases tab. |
| `CLINIQ_PREFILL_NEW_CASE=1`   | Pre-populates the new case with the COVID template. |
| `CLINIQ_OPEN_REVIEW=1`        | Opens the AI Review sheet for the first populated case. |
| `CLINIQ_OPEN_CASE_DETAIL=1`   | Pushes the Case Detail view for the first populated case. |

Pass any of these via `SIMCTL_CHILD_…` to propagate into the simulator:

```bash
SIMCTL_CHILD_CLINIQ_SIMULATE_OFFLINE=1 \
SIMCTL_CHILD_CLINIQ_TAB=outbox \
  xcrun simctl launch --terminate-running-process \
    CADA1806-F64D-4B02-B983-B75F197D1EF3 com.cliniq.ClinIQ
```

## Reproducing the six PoC screenshots

All six PNGs live at the `apps/mobile/ios-app/` root.

```bash
# 01 — case list with offline banner + mixed statuses
SIMCTL_CHILD_CLINIQ_SIMULATE_OFFLINE=1 \
  xcrun simctl launch --terminate-running-process ...
sleep 3 && xcrun simctl io ... screenshot poc-01-case-list.png

# 02 — new case intake (prefilled COVID narrative)
SIMCTL_CHILD_CLINIQ_SIMULATE_OFFLINE=1 \
SIMCTL_CHILD_CLINIQ_OPEN_NEW_CASE=1 \
SIMCTL_CHILD_CLINIQ_PREFILL_NEW_CASE=1 \
  xcrun simctl launch --terminate-running-process ...
sleep 3 && xcrun simctl io ... screenshot poc-02-new-case-intake.png

# 03 — AI review screen (entities as rows, tok/s counter)
SIMCTL_CHILD_CLINIQ_SIMULATE_OFFLINE=1 \
SIMCTL_CHILD_CLINIQ_OPEN_REVIEW=1 \
  xcrun simctl launch --terminate-running-process ...
sleep 3 && xcrun simctl io ... screenshot poc-03-ai-review.png

# 04 — Outbox with 1 queued report
SIMCTL_CHILD_CLINIQ_SIMULATE_OFFLINE=1 \
SIMCTL_CHILD_CLINIQ_TAB=outbox \
  xcrun simctl launch --terminate-running-process ...
sleep 3 && xcrun simctl io ... screenshot poc-04-outbox.png

# 05 — History tab, filtered by status
SIMCTL_CHILD_CLINIQ_SIMULATE_OFFLINE=1 \
SIMCTL_CHILD_CLINIQ_TAB=history \
  xcrun simctl launch --terminate-running-process ...
sleep 3 && xcrun simctl io ... screenshot poc-05-history.png

# 06 — Case Detail with offline banner persisting through nav
SIMCTL_CHILD_CLINIQ_SIMULATE_OFFLINE=1 \
SIMCTL_CHILD_CLINIQ_OPEN_CASE_DETAIL=1 \
  xcrun simctl launch --terminate-running-process ...
sleep 3 && xcrun simctl io ... screenshot poc-06-offline-banner.png
```

## Seed data

On first launch `DemoSeed` inserts four cases so the demo has visible
content without waiting for model inference:

| Patient          | Condition              | Status     | Notes                       |
| ---------------- | ---------------------- | ---------- | --------------------------- |
| Maria Garcia     | COVID-19               | Submitted  | Recently posted (−6 h)      |
| Daniel Johnson   | Meningococcal disease  | Queued     | Waiting in outbox           |
| Michael Martinez | HIV infection          | Submitted  | −4 days; shows in History   |
| Jennifer Brown   | (pending review)       | Draft      | Just started — no entities  |

The seed runs only when the store is empty; subsequent launches preserve
whatever state the clinician left behind.

## Inference backend

The project is factored behind `InferenceEngine` (see
`ClinIQ/Inference/InferenceEngine.swift`). Three concrete backends exist:

1. **`LlamaCppInferenceEngine`** (default) — wraps the vendored
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
   CI / SwiftUI-Preview engine. The C13 `ExtractionService` automatically
   picks it when `LlamaCppInferenceEngine.resolveModelPath() == nil`, so
   the review flow runs end-to-end even without a bundled GGUF.

## Model distribution

The GGUF files are 2.5-3.2 GB each; **do not** commit them to git
(`.gitignore` covers `*.gguf`). See the C12 notes (unchanged): seed into
the simulator sandbox, bundle into the .app, or download on first launch.

## Performance (simulator CPU)

Unchanged from C12: 1-1.5 tok/s decode on Q3_K_M fine-tune, ~2-5 min per
extraction. Physical iPhone with Metal is projected 10-20 tok/s — never
validated on device by C12 or C13.

## Physical iPhone sideload (free Apple ID)

NOT tested on a physical device by any of C10/C12/C13. Steps unchanged
from C10 — open the project in Xcode, select a Personal Team, connect a
device, hit Run. A free signing cert is good for 7 days; 3 free apps per
phone. See the C10 BUILD.md in git history for the full flow.

## Project layout

```
apps/mobile/ios-app/
├── BUILD.md                          (this file)
├── DEMO_SCRIPT.md                    (60-second demo narration)
├── LEGACY.md                         (C10/C12 JSON-dumper notes)
├── VALIDATION.md                     (per-case extraction scores)
├── poc-01-…-poc-06-….png             (C13 PoC screenshots, committed)
├── screenshot*.png                   (C10/C12 dev-path screenshots)
├── validate.swift                    (headless validator — stub path)
└── ClinIQ/
    ├── ClinIQ.xcodeproj/
    ├── Frameworks/llama.xcframework/
    └── ClinIQ/
        ├── ClinIQApp.swift           (@main entry point, wires container + services)
        ├── Views/
        │   ├── RootView.swift        (C13 — tab shell)
        │   ├── ContentView.swift     (C10/C12 legacy testbench)
        │   ├── ExtractionViewModel.swift (legacy testbench VM)
        │   ├── Components/           (Theme, StatusBadge, OfflineBanner, EntityRow)
        │   ├── Cases/                (CasesTab, NewCaseView, CaseDetailView)
        │   ├── Review/               (ReviewFlowView)
        │   ├── Outbox/               (OutboxTab)
        │   ├── History/              (HistoryTab)
        │   └── Settings/             (SettingsTab)
        ├── Persistence/
        │   ├── Models.swift          (SwiftData @Model types)
        │   ├── PersistenceController.swift (container factory + file protection)
        │   └── DemoSeed.swift        (4 first-launch cases)
        ├── Sync/
        │   ├── NetworkMonitor.swift  (NWPathMonitor wrapper)
        │   ├── SyncConfig.swift      (endpoint + toggles)
        │   └── SyncService.swift     (drain loop + audit history)
        ├── Extraction/
        │   ├── ExtractionParser.swift (JSON tolerant parser)
        │   └── ExtractionService.swift (streams model → ParsedExtraction)
        ├── Models/
        │   └── TestCase.swift        (legacy C10/C12 test cases — still referenced by ContentView testbench)
        ├── Inference/                (InferenceEngine protocol + 3 impls, unchanged from C12)
        └── Resources/Info.plist
```
