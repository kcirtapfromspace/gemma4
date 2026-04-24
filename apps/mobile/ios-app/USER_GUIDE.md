# ClinIQ — Complete User Guide

*Offline clinician case-reporting on iPhone, powered by on-device Gemma 4.*

Target reader: demo presenters, clinician end-users, reviewers evaluating the PoC. Assumes the app is already built and installed per `BUILD.md`.

For the short 60-second narration while holding a phone, see `DEMO_SCRIPT.md`. This document is the long-form reference.

---

## 1. What the app is

**ClinIQ** is an iOS app that lets a front-line clinician in a remote clinic with **no cellular or Wi-Fi coverage** during shift:

1. **Open a case** from either the field or a prior shift
2. **Paste or type a clinical narrative** (or import an eICR fragment)
3. **Tap "Review with AI"** — a quantized Gemma 4 E2B model + a clinical-extraction LoRA runs entirely **on the device**, no network, and proposes structured entities: conditions (SNOMED CT), labs (LOINC), medications (RxNorm), patient demographics, and vitals
4. **Curate** the extraction — accept, edit, or reject each entity individually
5. **Queue** the case to a local outbox — encrypted at rest, survives app kills and reboots
6. **Auto-sync** when the phone reconnects to a network — the Outbox drains to a configurable public-health endpoint, each submission gets a stamped receipt, and the case moves to History

No raw JSON appears in the user-facing UI. Codes appear as audit subtitles under human-readable names.

---

## 2. Who it's for

- **Field clinicians / CHWs** in remote districts who see notifiable conditions (meningococcal, measles, TB, HIV, STI, enteric outbreaks, Dengue, Lyme, hepatitis) and must report to state / district surveillance systems.
- **Public-health-informatics engineers** evaluating whether an offline-first mobile AI extraction tool is viable on a phone the clinician already carries.
- **Demo reviewers** who want to see on-device LLM inference doing real work inside a product-shaped UI rather than a developer testbench.

---

## 3. Architecture at a glance

```
 ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐
 │  Case intake UI  │  │   AI Review UI  │  │   Outbox UI    │
 │  (SwiftUI)       │  │   (SwiftUI)     │  │   (SwiftUI)    │
 └────────┬─────────┘  └────────┬────────┘  └───────┬────────┘
          │                     │                   │
          ▼                     ▼                   ▼
 ┌──────────────────────────────────────────────────────────┐
 │                  ExtractionService (actor)               │
 │  wraps InferenceEngine protocol                          │
 │   • LlamaCppInferenceEngine (real: llama.cpp xcframework)│
 │   • StubInferenceEngine (fallback: rule-based)           │
 │  streams AsyncThrowingStream<InferenceChunk> with        │
 │  token-by-token output + tok/s counter                   │
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────────────┐
 │              ExtractionParser (tolerant JSON)            │
 │   turns streamed model output into typed SwiftData rows  │
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────────────┐
 │                       SwiftData store                    │
 │   Case · Patient · Condition · Lab · Medication · Vitals │
 │   SyncRecord · encrypted at rest                         │
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────────────┐
 │                     SyncService (actor)                  │
 │   listens to NWPathMonitor (Network framework)           │
 │   drains pending cases to SyncConfig.endpoint on network │
 │   stamp receipt + status per attempt                     │
 └──────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **On-device inference only.** The model file (`cliniq-gemma4-e2b-Q3_K_M.gguf`, ~3.0 GB) is copied into `Documents/` at first run or sideloaded. No inference call ever leaves the device.
- **Zero raw JSON in UI.** JSON is a transport detail between the model and the parser. Entities are rendered as structured rows.
- **SwiftData for persistence.** iOS 17+ native ORM, encrypted at rest with `completeUntilFirstUserAuthentication` data-protection class.
- **`NWPathMonitor` for reachability.** Not polling; framework-native path-status changes drive offline banner + auto-sync trigger.
- **Mock sync endpoint toggleable in Settings.** Real interop (mTLS, jurisdiction routing, FHIR transaction bundle) is documented in `SyncConfig.swift` but not wired — explicitly out of PoC scope.

---

## 4. The five tabs

### 4.1 Cases

**What the user sees:** a list of all cases on the device — draft, queued, syncing, submitted, or failed — sorted by most recent. Each row: patient name, primary condition display name, status chip, time since last edit. Top-of-screen **offline banner** (yellow) or **online chip** (green). A **`+`** button in the navigation bar opens the New Case sheet.

**What happens under the hood:**
- `CasesView` is a `@Query` from SwiftData filtered by status != `.archived`
- Each row is a `NavigationLink` into `CaseDetailView`
- Row background color mirrors status (subtle)
- `NetworkMonitor` drives the banner via `@ObservedObject`

**User actions:**
- Tap `+` → open New Case intake sheet
- Tap any row → open Case Detail
- Swipe-left on a row → archive (soft delete)
- Pull-to-refresh → no-op in PoC (future: force a sync attempt)

### 4.2 New Case (sheet from Cases tab)

**What the user sees:** a form with four sections:
1. **Patient** — first name, last name, gender (M/F/U picker), DOB, optional MRN
2. **Encounter** — date, facility name, location (city / ZIP), reason-for-visit short text
3. **Clinical narrative** — large multi-line `TextEditor` for the free-text note. A **Sample ▾** menu in the section header pre-fills with one of the 10 `test_cases.jsonl` examples.
4. **Actions** — `Cancel` (dismiss, discard) and **`Review with AI`** (primary CTA)

**What happens under the hood:**
- A new `ClinicalCase` with `status = .draft` is created in SwiftData on sheet open
- Form bindings write through to the SwiftData object live — draft persists if the user backgrounds the app
- Tapping Review with AI:
  1. Serializes `system prompt + user narrative` via `PromptBuilder.build(...)` wrapped in `<|turn>` delimiters
  2. Opens the AI Review sheet while extraction runs in the background via the `ExtractionService` actor
  3. Stream of `InferenceChunk` is displayed token-by-token; a live `tok/s` counter is visible in the title bar

### 4.3 AI Review (sheet from New Case)

**What the user sees:**
- Header: patient name, "AI Review" title, `tok/s` counter (reads from `InferenceChunk.tokensPerSecond`), **`✕`** cancel
- Four sections, one per entity type: **Conditions**, **Labs**, **Medications**, **Vitals**
- Each entity row:
  - Large text: human-readable display name
  - Small muted text underneath: `SNOMED 840539006` / `LOINC 94500-6` / `RxNorm 2599543`
  - A status chip: **Proposed** (default from AI), **Accepted** (user tapped check), **Edited** (user changed something), **Rejected** (user tapped minus)
  - Right-side buttons: green **check** (accept), orange **pencil** (edit), red **minus** (reject)
- Footer: **`Queue to Outbox`** (primary, only enabled when at least one entity is accepted)

**What happens under the hood:**
- While the stream is still running, rows appear progressively as the parser identifies complete entities from the partial JSON
- The **parser is tolerant** — unbalanced braces, trailing commas, half-emitted strings are handled; entities appear only when they validate
- Tapping **edit** on a condition opens a sheet with an autocomplete search over a bundled SNOMED slim terminology subset (100 most common notifiable conditions). LOINC and RxNorm get the same shape.
- Each accept/edit/reject writes through to the SwiftData `ExtractedCondition`/`ExtractedLab`/`ExtractedMedication` row's `reviewState` column so the audit trail is preserved

### 4.4 Outbox

**What the user sees:** a list of cases with status `.queued` or `.syncing` or `.failed`. Each row: patient name, condition, queued-at time, status chip. Top-right button: **`Sync now`** (disabled when offline). The tab-bar badge on the Outbox tab shows the count of pending cases.

**What happens under the hood:**
- `OutboxView` is a `@Query` where `status in [.queued, .syncing, .failed]`
- The `Sync now` button invokes `SyncService.drain()`; while offline the button is disabled (button state bound to `NetworkMonitor.isReachable`)
- **Auto-drain**: when `NWPathMonitor` flips to `.satisfied`, `SyncService.drain()` fires automatically in the background — Outbox updates live via SwiftData's `@Query` reactivity
- Each sync attempt creates a `SyncRecord` row: timestamp, endpoint, response body, outcome (`.submitted` / `.failed` + error). Survives app kills.
- Failed sends stay in the Outbox with a `Failed` chip + the error text; the user can tap the row to inspect and manually retry

### 4.5 History

**What the user sees:** a list of submitted cases (and archived drafts), with filters at top: **Status** chip (`All / Submitted / Archived`) and **Condition** picker (derived from the unique SNOMED displays in the store). Tapping a row opens a read-only Case Detail with the audit trail.

**What happens under the hood:**
- `HistoryView` is another `@Query` keyed on `status == .submitted || status == .archived`
- The Case Detail in History mode is the same `CaseDetailView` with `editable = false`
- Each row's submission timestamp and receipt reference are visible

### 4.6 Settings (gear icon in nav bar)

**What the user sees:**
- **Sync endpoint** URL (display only; configured in `SyncConfig.swift`)
- **Simulate offline for demo** toggle — overrides `NWPathMonitor` result to false
- **Mock sync success/fail** toggle — controls whether the stub endpoint 200s or 500s
- **Force CPU inference** toggle — sets `n_gpu_layers = 0` on the engine; useful on iPhones where Metal may be flaky
- **Model info** — shows the loaded GGUF filename and size
- **About** — version, build hash, Apache 2.0 credit to llama.cpp and upstream Gemma 4

**What happens under the hood:**
- Each toggle is an `@AppStorage`-backed `Bool`
- `NetworkMonitor` reads the "Simulate offline" override before returning `.isReachable`
- `ExtractionService.makeDefaultEngine()` reads `ForceCPU` to pass `n_gpu_layers: 0` to `LlamaCppInferenceEngine.init(...)`

---

## 5. Demo runbook — what to show, in what order

### 5.1 The 60-second pitch

See `DEMO_SCRIPT.md`. Verbatim narration, pre-seeded cases, controlled screen sequence.

### 5.2 The 3-minute guided walkthrough

1. **(0:00-0:15) Set the scene.** Open the app (fresh launch or `xcrun simctl launch`). Cases tab appears. Point to the **yellow offline banner** — "offline because the simulator has no network, same as a clinic dead zone."
2. **(0:15-0:30) Case list.** 3 seeded cases with different statuses (Draft / Queued / Submitted). Explain the color chips.
3. **(0:30-0:45) New case.** Tap `+`. Fill in demographics OR tap Sample ▾ → COVID. Show the narrative field. Say: "this is literally the text the clinician types or pastes."
4. **(0:45-1:30) AI review.** Tap **Review with AI**. As the tok/s counter runs, entities appear row-by-row. Point to the SNOMED/LOINC/RxNorm codes in the muted subtitles — "these are the standardized codes for FHIR interop; the clinician doesn't have to know them."
5. **(1:30-1:45) Curate.** Tap check on each entity. Show editing one condition — brings up a SNOMED autocomplete.
6. **(1:45-2:00) Queue to outbox.** Tap **Queue to Outbox**. Switch to **Outbox** tab. Show the queued case. Sync button disabled (still offline).
7. **(2:00-2:30) Sync.** Either toggle off "Simulate offline" in Settings, or (on a real device) watch the banner turn green when Wi-Fi returns. Outbox auto-drains. Point at the live status change: `queued → syncing → submitted`.
8. **(2:30-3:00) History.** Switch to History. Submitted case is there with receipt reference. Filter by condition.

### 5.3 The 5-minute technical walkthrough

Add before each product step:
- Open the Xcode console or `xcrun simctl spawn … log stream --predicate 'processImagePath endswith "ClinIQ"'` and show inference log lines with `tok/s` during Review
- Open the simulator `Documents/extractions.log` (`xcrun simctl get_app_container …`) between steps to show the raw audit records
- Open Settings and flip "Force CPU" — point out this exists for iPhones where Metal is unreliable on Gemma 4's sliding-window kernels

---

## 6. How-to — specific tasks

### 6.1 Record a new case from a typed narrative

1. Cases tab → tap `+`
2. Fill Patient section → gender, DOB, optional MRN
3. Fill Encounter section → date, facility, location
4. In Clinical Narrative, paste or type the encounter text (anything: eICR fragment, SOAP note, CDA summary, plain notes)
5. Tap **Review with AI**
6. Watch entities populate
7. For each entity: tap check to accept, pencil to edit, minus to reject
8. Tap **Queue to Outbox**

### 6.2 Work through a backlog of drafts offline

- Cases tab lists all drafts mixed with other statuses
- Sort is by last-edited-desc; drafts bubble to top
- Drafts persist across app launches — tap any to resume

### 6.3 Force a sync attempt

- Go to **Outbox** tab
- If the Sync button is disabled, you're offline. Option A: wait for real network to return. Option B: toggle off "Simulate offline" in Settings.
- Sync button enables → tap it → each case's chip cycles `Queued → Syncing → Submitted` (or `Failed` with an error).

### 6.4 See what was already submitted

- **History** tab
- Filter by Condition or Status at the top
- Tap any row → read-only Case Detail with timestamp + receipt reference

### 6.5 Drive the UI from the command line (for recording)

Per C13's `BUILD.md`, env vars jump the UI straight into the screenshot-ready states:

```bash
SIMCTL_CHILD_CLINIQ_TAB=outbox \
SIMCTL_CHILD_CLINIQ_OPEN_CASE_DETAIL=daniel \
SIMCTL_CHILD_CLINIQ_SIMULATE_OFFLINE=1 \
  xcrun simctl launch --terminate-running-process \
    CADA1806-F64D-4B02-B983-B75F197D1EF3 com.cliniq.ClinIQ
```

Supported: `CLINIQ_TAB`, `CLINIQ_OPEN_NEW_CASE`, `CLINIQ_OPEN_REVIEW`, `CLINIQ_OPEN_CASE_DETAIL`, `CLINIQ_PREFILL_NEW_CASE`, `CLINIQ_SIMULATE_OFFLINE`.

---

## 7. What's real vs stubbed vs mocked

| Feature | Real | Stubbed | Mocked |
|---|---|---|---|
| On-device LLM inference | ✅ llama.cpp + Gemma 4 E2B GGUF | fallback to rule-based `StubInferenceEngine` when no GGUF present | — |
| Tokenization, decode streaming, tok/s counter | ✅ | — | — |
| SwiftData persistence (encrypted at rest) | ✅ | — | — |
| Offline detection via NWPathMonitor | ✅ | overridable via Settings toggle | — |
| Sync to public health endpoint | — | — | ✅ HTTP POST to `SyncConfig.endpoint` (default `http://localhost:8080/reports`); success/fail toggleable |
| FHIR R4 transaction bundle formatting | — | ✅ sketched in `SyncConfig.swift` | — |
| mTLS / jurisdiction routing | — | ✅ documented | — |
| SNOMED autocomplete | ✅ ships w/ 100-condition slim subset | — | — |
| LOINC / RxNorm autocomplete | — | ✅ edit sheet allows free text | — |
| Receipt reference | — | ✅ timestamp-based id | — |

---

## 8. Performance reality check

| Inference path | tok/s | 200-token extraction |
|---|---|---|
| Simulator CPU (measured by C12, C13 independently) | **1.0–4.4** (median 4.0 warm, cold 1.3) | 45s–200s |
| iPhone 17 Pro CPU (projected, llama.cpp Metal off) | 5–8 | 25–40s |
| iPhone 17 Pro Metal (projected, llama.cpp Metal on) | **10–20** | **10–20s** |
| iPhone 17 Pro Metal via LiteRT-LM Swift (C11 package, not wired yet) | **52–56** | **~4s** |

For a live demo on a physical iPhone, expect 10-20s per extraction. For the simulator during recording, lean on the seeded pre-extracted cases — the `Review with AI` flow is shown with the stub fallback so the screen renders promptly.

---

## 9. Known limitations

1. **Simulator inference is slow.** 1-4 tok/s CPU. Seeded cases ship pre-extracted for that reason. Real iPhone is 10x-50x faster.
2. **Metal on device is unvalidated** for Gemma 4's sliding-window attention kernels. If the graph fails to compile, the Review flow hangs at "Extracting...". Mitigation: Settings → Force CPU.
3. **SNOMED autocomplete is a slim 100-condition subset.** Real deployment needs full SNOMED CT US Edition behind a licensed terminology service.
4. **LoRA v1 quality gaps** documented by Team C8: `bench_minimal` (syphilis) scored 0/3 and `bench_typical_covid` scored 1/3 on the LiteRT-LM validator. The llama.cpp path (C12) recovers some of these but not all. LoRA v2 retrain (Kaggle, in flight) targets the two specific failure modes.
5. **No real public-health-system integration.** Sync is a mock HTTP POST. Production would need FHIR R4 MessageHeader + Bundle, mTLS with jurisdiction-specific CAs, and resilient retry with exponential backoff — all documented in `SyncConfig.swift` but not wired.
6. **No authentication.** The clinician opens the app; no login. Production would add either FaceID gate or an OAuth flow to the state surveillance system.

---

## 10. Next steps mapped to effort

| Task | Effort | Unblocks |
|---|---|---|
| Wait for Kaggle v2 LoRA to finish; re-merge → re-validate | ~2 hrs human wait + 30 min rebuild | Higher per-case extraction scores |
| Sideload to a physical iPhone with free Apple ID | ~30 min per `BUILD.md` § Sideload | Real Metal tok/s measurement |
| Add FaceID gate on app launch | ~2 hrs | Clinical-grade auth story |
| Swap from llama.cpp to C11's LiteRT-LM Swift package | ~1-2 days | 5x faster inference (~56 tok/s) |
| Wire real FHIR R4 transaction bundle to a jurisdiction endpoint | 3-5 days + jurisdictional sign-off | Production-grade submission |
| Build full SNOMED CT US Edition autocomplete | 1-2 days (need license or MIT-licensed mirror) | Clinical-grade terminology coverage |
| Add an eICR XML upload path (paste CDA/XML, extract) | 1 day | Handles the "forwarded from an EHR" case |

---

## 11. Files to read next

- `DEMO_SCRIPT.md` — 60-second verbatim narration
- `BUILD.md` — build / install / launch / sideload commands
- `VALIDATION.md` — per-case extraction scores and comparisons
- `LEGACY.md` — notes on the prior chat-style UI (deprecated, preserved for git blame)
- `ClinIQ/ClinIQ/Persistence/` — SwiftData model definitions
- `ClinIQ/ClinIQ/Extraction/ExtractionService.swift` — inference orchestration + parser
- `ClinIQ/ClinIQ/Sync/SyncService.swift` — outbox drainer
- `ClinIQ/ClinIQ/Sync/SyncConfig.swift` — real-endpoint hook points for production integration
- `ClinIQ/Frameworks/llama.xcframework/` — the vendored llama.cpp binary distribution (iOS arm64 + sim fat)
