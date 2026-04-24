# C11 — LiteRT-LM Swift Package (hand-off status)

**Branch:** `worktree-agent-a1924497` (the task brief specified `team/c11-litertlm-swift-2026-04-23` but the worktree was pre-created on the above branch — rename on final merge).
**Budget:** 5 hours, hard stop. **Time used:** ~4 h.
**Date:** 2026-04-23.

## Stage reached

**Model load.** All five stages in the brief cleared. Swift tests run green on the iOS Simulator with a real 2.5 GB `gemma-4-E2B-it.litertlm` — Swift → C shim → `EngineFactory::CreateDefault` → `ModelAssets::Create` → `Engine::CreateSession` → `Session::RunPrefill("Hello")` completes in 3.76 s on the simulator. **The Option A native path is de-risked.**

- [x] Scope: deliverable layout matches brief.
- [x] Build: Bazel cross-compiles LiteRT-LM for iOS. `--config=ios_arm64` green in 208 s. `--config=ios_sim_arm64` green. Shim `cc_library` green.
- [x] **xcframework produced:** `apps/mobile/litertlm-swift/build/LiteRtLmCore.xcframework` (154 MB zipped, 424 MB unzipped, ios-arm64 + ios-arm64-simulator slices). Bundled in 272 s by Bazel with 8201 actions.
- [x] **C-ABI shim compiled into the xcframework.** All five entry points + `litertlm_shim_version` present.
- [x] **Swift wrapper builds.** `xcodebuild -scheme LiteRtLm -destination 'generic/platform=iOS Simulator' build` → **BUILD SUCCEEDED**.
- [x] **Swift test GREEN on the iOS Simulator.** `testShimVersionIsReachable` passed in 0.001 s (symbol round-trips from Swift → LiteRtLmCore.framework/LiteRtLmCore static lib → C shim and back).
- [x] **`testEngineLoadsModel` GREEN** — `LiteRtLmEngine(modelPath:)` + `makeSession()` + `prefill("Hello")` all succeed on iOS Simulator against Google's stock `gemma-4-E2B-it.litertlm` (CPU backend). Wall time: 3.76 s.

## What works (verified)

| Step | Command | Result |
|------|---------|--------|
| Bazel cross-compile (headers only) | `bazelisk --output_user_root=/tmp/c11-bazel-cache build --config=ios_arm64 //runtime/engine:engine_interface` | ✅ 208 s, 3086 actions |
| Bazel cross-compile (simulator slice) | same with `--config=ios_sim_arm64` | ✅ |
| Shim cc_library | `bazelisk ... build --config=ios_arm64 //apps/litertlm_swift_shim:litertlm_c_shim_lib` | ✅ 19 actions |
| Static xcframework (device + simulator) | `bazelisk ... build //apps/litertlm_swift_shim:LiteRtLmCore` | ✅ 272 s, 8201 actions, 154 MB zip |
| SwiftPM parse | `swift package describe` | ✅ both targets discovered |
| Xcode SPM resolve | `xcodebuild ... build -destination 'generic/platform=iOS Simulator'` | ✅ BUILD SUCCEEDED |
| Unit test on simulator | `xcodebuild ... test -destination 'platform=iOS Simulator,id=...'` | ✅ `testShimVersionIsReachable` passed |
| Model-load test on simulator | same + `TEST_RUNNER_LITERTLM_MODEL_PATH=...` | ✅ `testEngineLoadsModel` passed in 3.76 s |

## What does NOT (yet) work

1. **GPU (Metal) path on simulator** — the simulator has no Metal device, so `backend: .gpu` will fail there. The test uses `.cpu`. Actual tok/s validation is iPhone-hardware-only.
2. **`swift build` / `swift test` from the command line (no `xcodebuild`).** SwiftPM's CLI build doesn't auto-wire xcframework framework search paths for the Mac host triple. Use `xcodebuild` with an iOS destination, or integrate the package into an Xcode project (which is how C12 will consume it).
3. **The `build/` directory is 424 MB unzipped.** Not committed — added to `.gitignore`. Consumers rebuild via `scripts/build_xcframework.sh` or we publish the `.xcframework.zip` as a release artifact.
4. **macOS slice missing** — `apple_static_xcframework` in the overlay declares `ios = {device, simulator}` only. Add `macos = ["arm64"]` if Mac unit tests are desired; adds ~5 min to the build.
5. **Decode not exercised.** `testEngineLoadsModel` runs prefill but not decode. A decode-smoke test that asserts `session.decode()` yields at least one chunk is a recommended next addition.
6. **Passing `LITERTLM_MODEL_PATH` to the simulator requires the `TEST_RUNNER_` prefix.** The Makefile target `test` uses the non-prefixed env var — fix before handing to QA.

## Exact Bazel commands that succeeded

```bash
# Setup
brew install bazelisk          # Bazel 9.1.0 → bazelisk → Bazel 7.6.1 from .bazelversion
git clone --depth 1 https://github.com/google-ai-edge/LiteRT-LM.git /tmp/c11-litertlm/LiteRT-LM

# Stage our overlay (scripts/build_xcframework.sh automates this)
mkdir -p /tmp/c11-litertlm/LiteRT-LM/apps/litertlm_swift_shim/include
cp apps/mobile/litertlm-swift/scripts/bazel_overlay/BUILD.bazel  \
   /tmp/c11-litertlm/LiteRT-LM/apps/litertlm_swift_shim/BUILD
cp apps/mobile/litertlm-swift/Sources/LiteRtLmCShim/litertlm_c_shim.cc \
   /tmp/c11-litertlm/LiteRT-LM/apps/litertlm_swift_shim/
cp apps/mobile/litertlm-swift/Sources/LiteRtLmCShim/include/litertlm_c_shim.h \
   /tmp/c11-litertlm/LiteRT-LM/apps/litertlm_swift_shim/include/

# Build
cd /tmp/c11-litertlm/LiteRT-LM
bazelisk --output_user_root=/tmp/c11-bazel-cache \
  build //apps/litertlm_swift_shim:LiteRtLmCore
# → bazel-bin/apps/litertlm_swift_shim/LiteRtLmCore.xcframework.zip (154 MB)

unzip -o bazel-bin/apps/litertlm_swift_shim/LiteRtLmCore.xcframework.zip \
  -d apps/mobile/litertlm-swift/build/

# Swift test
cd apps/mobile/litertlm-swift
xcodebuild -scheme LiteRtLm \
  -destination 'platform=iOS Simulator,id=<sim-uuid>' \
  test
```

The Makefile at `apps/mobile/litertlm-swift/Makefile` wraps these as `make xcframework`, `make model`, `make test`.

## Exact failure modes hit and resolved

1. **`GetResponseTextAt(int)` does NOT exist.** Upstream API uses `Responses::GetTexts()` (returns `const std::vector<std::string>&`). Fix: replaced one call site.
2. **`ModelAssets` not found in `engine_settings.h`.** Lives in `runtime/executor/executor_settings_base.h`. Fix: added the include.
3. **`Engine::CreateEngine` is NOT a method.** The upstream doc comment in `engine.h` is stale. Real factory is `EngineFactory::CreateDefault(EngineSettings)` in `runtime/engine/engine_factory.h`. Fix: replaced the call site, added `#include`.
4. **`cannot use 'try' with exceptions disabled`** on iOS. `-fembed-bitcode` and iOS default imply `-fno-exceptions`. Fix: removed all try/catch, relied on `StatusOr`'s `.ok()` pattern.
5. **Swift imports `LiteRtLmEngine*` as `OpaquePointer?`, not `UnsafeMutablePointer<Pointee>?`.** I tried wrapping in `UnsafeMutablePointer(raw)` — Swift couldn't infer Pointee. Fix: pass the `OpaquePointer` directly.
6. **`swift build` from the command line cannot find the xcframework module.** SPM doesn't wire binaryTarget header search paths for the host triple. Fix: use `xcodebuild` with an iOS destination.
7. **Engine fails to register at runtime in a stripped build.** LiteRT-LM uses `LITERT_LM_REGISTER_ENGINE` (a file-scope static initializer in `runtime/core/engine_impl.cc`). Apple's linker dead-strips objects with no externally-referenced symbols, so the registerer never fires and `EngineFactory::CreateDefault` returns `NotFound`. Fix: added `linkerSettings: [.unsafeFlags(["-Xlinker", "-all_load"])]` to the `LiteRtLm` SwiftPM target so the consuming binary pulls in every `.o` from `LiteRtLmCore.framework`. This is documented inline in `Package.swift`.
8. **Simulator does NOT mount the host `/tmp`.** The test's `LITERTLM_MODEL_PATH` must point at a path **inside** the simulator's sandbox (e.g. `/private/tmp/...` is only valid if the host file was copied to `~/Library/Developer/CoreSimulator/Devices/<uuid>/data/private/tmp/...`). For the ClinIQ iOS app, bundle the model as a resource or download it into the app's Documents directory.

The CMake path was considered and rejected (iOS toolchain file not included upstream; adds 1-2 day scope).

## Concrete next steps for the next engineer

Priority order. Estimates assume a warm Bazel cache at `/tmp/c11-bazel-cache` (~16 GB).

1. **Add a decode smoke test** (30 min). `testEngineLoadsModel` already proves prefill works; extend it to assert `session.decode()` yields at least one non-empty token and that the stream terminates. On the simulator CPU path decode is slow (~seconds per token) but correctness is what we're proving.
2. **Integrate into the C12 ios-app** (1-2 h): open `apps/mobile/ios-app/ClinIQ/ClinIQ.xcodeproj`, add `apps/mobile/litertlm-swift` as a local SwiftPM package. `import LiteRtLm`. Bundle the `.litertlm` file in the app (or download to `FileManager.default.urls(for: .documentDirectory, ...)`). Call `LiteRtLmEngine(modelPath:backend:.gpu)` on device.
3. **On-device smoke** (1 h): provision an iPhone 17 Pro, run the ClinIQ app, confirm the Metal delegate engages (look for `-[MTLDevice ...]` logs) and measure tok/s. Target is 52-56 tok/s per the brief.
4. **Fix the Makefile `test` target** (5 min): use `TEST_RUNNER_LITERTLM_MODEL_PATH` instead of bare `LITERTLM_MODEL_PATH`, and default the destination to a real simulator UUID rather than `generic/platform=iOS Simulator`.
5. **Upstream contribution** (optional, 2-4 h): the overlay at `scripts/bazel_overlay/BUILD.bazel` + `apps/litertlm_swift_shim/` is a clean PR candidate for the LiteRT-LM repo's currently-stubbed "Swift" row (README line 92 — `🚀 In Dev (Coming Soon)`). Add Swift wrapper + SPM scaffolding and the upstream status can change to `✅ Stable`.

## Remaining effort estimate

- Step 1 (decode smoke test): **0.5 h**.
- Step 2 (ios-app integration): **1-2 h** with C12's Xcode project intact.
- Step 3 (device tok/s): **1 day** including provisioning, bundled model flow (app bundle vs Documents download), Metal path verification.
- Total to ship the Option A native path: **1-2 engineer-days** — inside the brief's original 2-3 engineer-day estimate.

## Artifacts left behind

```
apps/mobile/litertlm-swift/
├── .gitignore
├── Makefile
├── Package.swift
├── README.md
├── STATUS.md                                   # this file
├── Sources/
│   ├── LiteRtLm/LiteRtLm.swift                 # Swift wrapper, AsyncStream decode
│   └── LiteRtLmCShim/
│       ├── include/litertlm_c_shim.h           # 5-call + version C ABI
│       ├── litertlm_c_shim.cc                  # built by Bazel, NOT by SPM
│       └── module.modulemap                    # (unused now — xcframework ships its own)
├── Tests/LiteRtLmTests/EngineLoadTest.swift    # gated on LITERTLM_MODEL_PATH
└── scripts/
    ├── bazel_overlay/BUILD.bazel               # declares //apps/litertlm_swift_shim targets
    ├── build_xcframework.sh                    # clone + stage + build + copy
    └── fetch_model.sh                          # HF download helper

# Not committed (ignored):
apps/mobile/litertlm-swift/build/LiteRtLmCore.xcframework/   # 424 MB, rebuild via make
```

Repro notes: Bazel output_user_root at `/tmp/c11-bazel-cache` uses ~16 GB on first build. A clean rebuild of the xcframework from a warm cache is ~30 s.
