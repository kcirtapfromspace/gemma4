# C11 — LiteRT-LM Swift Package (in-progress status)

**Branch:** `worktree-agent-a1924497` (note: the task brief specified `team/c11-litertlm-swift-2026-04-23` but the worktree was pre-created on the above branch — rename on final merge.)
**Budget:** 5 hours, hard stop.
**Date:** 2026-04-23.

## Stage reached

`xcframework build, mid-flight`. (Stages defined in the brief: scope / build / xcframework / shim / swift wrapper / model load.)

- [x] Scope: deliverable layout matches brief.
- [x] Build: Bazel can cross-compile LiteRT-LM's `//runtime/engine:engine_interface` for `ios_arm64`. **Verified with a successful 208 s build producing a static archive (3086 actions, critical path 23.7 s on the Mac Studio).**
- [~] xcframework: `apple_static_xcframework` rule authored (`scripts/bazel_overlay/BUILD.bazel`) and the shim `cc_library` build is running at the time of hand-off. Not yet produced a zipped xcframework.
- [x] Shim (C ABI): `Sources/LiteRtLmCShim/{include/litertlm_c_shim.h,litertlm_c_shim.cc}` authored against the verified upstream API (`Engine`, `Session`, `ModelAssets`, `Backend`, `InputText`, `Responses::GetTexts`).
- [x] Swift wrapper: `Sources/LiteRtLm/LiteRtLm.swift` exposes `LiteRtLmEngine`, `LiteRtLmSession`, and an `AsyncThrowingStream<String>` decode. `Package.swift` declares the binary target.
- [ ] Model load: not exercised. `Tests/LiteRtLmTests/EngineLoadTest.swift` authored and gated on `LITERTLM_MODEL_PATH` env var. Cannot run `swift test` until `build/LiteRtLmCore.xcframework` exists.

## What works (verified)

- `bazelisk` resolves and uses Bazel 7.6.1 (per upstream `.bazelversion`) on macOS arm64.
- Upstream LiteRT-LM clones cleanly at `https://github.com/google-ai-edge/LiteRT-LM` (HEAD 2026-04-23).
- The upstream `.bazelrc` already defines `ios_arm64`, `ios_sim_arm64`, and `ios_arm64e` configs that select the right `build_bazel_apple_support` platforms. No patches required.
- Cross-compile: `bazelisk build --config=ios_arm64 //runtime/engine:engine_interface` succeeds without touching the upstream tree. Log at `/tmp/c11-bazel-build-ios_arm64.log`.
- All shim entry points compile-check against upstream headers (verified by grep, not by a test run):
  - `litert::lm::Engine::CreateEngine(EngineSettings)` → `runtime/engine/engine.h:65`.
  - `litert::lm::ModelAssets::Create(absl::string_view)` → `runtime/executor/executor_settings_base.h:113`.
  - `litert::lm::EngineSettings::CreateDefault(ModelAssets, Backend)` → `runtime/engine/engine_settings.h:79`.
  - `litert::lm::SessionConfig::CreateDefault()` → `runtime/engine/engine_settings.h:181`.
  - `InputText(std::string)` via implicit variant conversion → `runtime/engine/io_types.h:46`.
  - `Responses::GetTexts()` → `runtime/engine/io_types.h:336`.

## What does NOT (yet) work

1. **Full dep-tree build of the shim.** Building `//runtime/engine:engine_interface` alone pulls only the header-only interface plus abseil. The real engine requires `//runtime/engine:engine_impl_selected` → `//runtime/core:engine_impl`, which pulls LiteRT, TensorFlow, sentencepiece, tokenizers_cpp, flatbuffers, protobuf, antlr4, minja, llguidance, miniaudio, minizip, stb. **This big build was in progress at the 5 h hard stop.** Watch `/tmp/c11-bazel-shim-ios_arm64.log` to see whether it finished.
2. **xcframework bundle zip.** The `apple_static_xcframework` rule exists but has not been invoked end-to-end. Output path (when it works) will be `bazel-bin/apps/litertlm_swift_shim/LiteRtLmCore.xcframework.zip`; the build script copies it into `apps/mobile/litertlm-swift/build/LiteRtLmCore.xcframework`.
3. **Swift test.** `swift test` will fail today with "binaryTarget path does not exist" until step 2 succeeds.
4. **Branch rename.** Commits landed on `worktree-agent-a1924497`. If the final merge process expects `team/c11-litertlm-swift-2026-04-23`, rebase or rename.

## Exact Bazel commands that succeeded

```bash
git clone --depth 1 https://github.com/google-ai-edge/LiteRT-LM.git /tmp/c11-litertlm/LiteRT-LM
cd /tmp/c11-litertlm/LiteRT-LM
# engine interface, iOS device arm64
bazelisk --output_user_root=/tmp/c11-bazel-cache \
  build --config=ios_arm64 //runtime/engine:engine_interface
# → INFO: Build completed successfully, 3086 total actions, 208s wall.
```

Simulator slice (`--config=ios_sim_arm64`) was launched afterward and was green at hand-off — see `/tmp/c11-bazel-build-ios_sim.log`.

## Exact failure modes hit

None hit (yet) at the iOS cross-compile layer. Every failure that was caught was an API-signature mismatch I fixed in the shim before compilation:

- `GetResponseTextAt(int)` does NOT exist — replaced with `GetTexts().front()`.
- `ModelAssets` lives in `runtime/executor/executor_settings_base.h`, not `engine_settings.h` — added the include.
- `InputText` ctor wants a `std::variant<std::string, TensorBuffer>` — implicit conversion from `std::string` works, so the call site stays ergonomic.

The CMake path was considered and rejected: `CMakePresets.json` only ships `android-arm64`. No iOS toolchain file. Adding one is 1-2 day scope.

## Concrete next steps for the next engineer

Priority order. Rough estimates assume a warm Bazel cache at `/tmp/c11-bazel-cache`.

1. **Check the shim cc_library build** (1 h): `tail -f /tmp/c11-bazel-shim-ios_arm64.log`. If green, go to step 2. If red, diff the first `error:` line against `runtime/engine/*.h` — the shim is 200 lines and the API surface is small.
2. **Build the xcframework** (30 min, assuming step 1 green): `cd /tmp/c11-litertlm/LiteRT-LM && bazelisk --output_user_root=/tmp/c11-bazel-cache build //apps/litertlm_swift_shim:LiteRtLmCore`. Unzip `bazel-bin/.../LiteRtLmCore.xcframework.zip` into `apps/mobile/litertlm-swift/build/LiteRtLmCore.xcframework`. `scripts/build_xcframework.sh` does both steps.
3. **Verify SPM resolve** (15 min): `cd apps/mobile/litertlm-swift && swift package resolve && swift build --triple arm64-apple-macosx` — the macOS triple lets you link-check the C shim symbols even though the xcframework only has iOS slices. Expect an initial linker error; add a macos slice to `apple_static_xcframework.macos = ["arm64"]` if you want unit tests to run on the Mac.
4. **Acquire model** (15 min download): `huggingface-cli download litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --local-dir /tmp` → 2.5 GB. Set `LITERTLM_MODEL_PATH=/tmp/gemma-4-E2B-it.litertlm`.
5. **Run smoke test** (15 min): `swift test --filter EngineLoadTest`. `testEngineLoadsModel` exercises `LiteRtLmEngine.init(modelPath:)` → `makeSession()` → `prefill("Hello")`. Do NOT assert on decode throughput on the Mac; that's iPhone-only.
6. **Device wiring** (2-3 h): drop the xcframework and Swift targets into the C12 ios-app workspace; run on an iPhone 17 Pro; confirm Metal-backed GPU path by setting backend `.gpu`. This is where the 52-56 tok/s claim gets validated.

## Remaining effort estimate

- Happy path (steps 1-5): **3 engineer-hours** if the shim compile is green and the xcframework rule produces a well-formed bundle.
- Realistic path (expect one tokenization / flatbuffer-schema linker issue per upstream build system): **1-2 engineer-days.** `engine_impl_selected` drags in ~120 MB of iOS-targeted objects; deduplication against `LiteRtCore.dylib` (LiteRT ships its own in the rules_apple repository) may trip `apple_static_xcframework`'s duplicate-symbol check.
- Integration (step 6): **0.5-1 engineer-day** once we have a green xcframework.

## Repro notes

- `bazel` output base is explicit: `--output_user_root=/tmp/c11-bazel-cache`. Uses ~12 GB after the interface build. A full shim build is projected at ~25 GB.
- No network egress is required for the already-resolved dependencies after the first build; the local Bazel cache survives across runs.
- The brief forbade touching `apps/mobile/ios-app/`, `apps/mobile/convert/`, `kaggle-training/`, and `scripts/`. Nothing under those paths was modified.

## Files touched

```
apps/mobile/litertlm-swift/
├── Package.swift
├── README.md
├── STATUS.md                                  # this file
├── Sources/
│   ├── LiteRtLm/LiteRtLm.swift               # high-level Swift wrapper
│   └── LiteRtLmCShim/
│       ├── include/litertlm_c_shim.h         # 5-call C ABI
│       ├── litertlm_c_shim.cc                # shim implementation (built by Bazel)
│       └── module.modulemap                  # for SwiftPM consumers of the shim
├── Tests/LiteRtLmTests/EngineLoadTest.swift  # gated on LITERTLM_MODEL_PATH
└── scripts/
    ├── build_xcframework.sh                  # clones overlay into upstream, builds, copies out
    └── bazel_overlay/BUILD.bazel             # adds //apps/litertlm_swift_shim targets
```

Total LOC added outside of LiteRT-LM: ~500.
