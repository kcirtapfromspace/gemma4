# LiteRtLm (Swift Package)

Swift wrapper around Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) C++ runtime for on-device LLM inference. Target: iPhone 17 Pro GPU inference of `gemma-4-E2B-it.litertlm` at 52-56 tok/s.

## Status

In progress. See `STATUS.md` for exact state, what works, and the next-engineer hand-off checklist.

## Layout

- `Package.swift` — SPM manifest; binaryTarget at `build/LiteRtLmCore.xcframework`.
- `Sources/LiteRtLmCShim/` — C-ABI shim header (`include/litertlm_c_shim.h`) + implementation (`litertlm_c_shim.cc`). The `.cc` is compiled by Bazel, NOT by SPM — SPM consumes the prebuilt xcframework.
- `Sources/LiteRtLm/` — high-level Swift API: `LiteRtLmEngine` / `LiteRtLmSession` + `AsyncThrowingStream<String>` decode.
- `Tests/LiteRtLmTests/` — XCTest smoke tests gated on `LITERTLM_MODEL_PATH`.
- `scripts/build_xcframework.sh` — invokes Bazel in an upstream LiteRT-LM checkout and copies the xcframework out.
- `scripts/bazel_overlay/BUILD.bazel` — the overlay that declares `cc_library(litertlm_c_shim_lib)` and `apple_static_xcframework(LiteRtLmCore)` against `//runtime/engine:*` targets.

## Build

```bash
# 1. Clone LiteRT-LM (~1 GB)
git clone https://github.com/google-ai-edge/LiteRT-LM.git /tmp/c11-litertlm/LiteRT-LM

# 2. Produce the xcframework (~30-60 min cold, ~5 min warm)
LITERTLM_UPSTREAM=/tmp/c11-litertlm/LiteRT-LM ./scripts/build_xcframework.sh

# 3. Run Swift tests (xcframework is now at build/LiteRtLmCore.xcframework)
swift test
```

To exercise the model load path, also set `LITERTLM_MODEL_PATH` to a local copy of `gemma-4-E2B-it.litertlm`.

## API sketch

```swift
let engine = try LiteRtLmEngine(
    modelPath: URL(fileURLWithPath: "/path/to/gemma-4-E2B-it.litertlm"),
    backend: .gpu
)
let session = try engine.makeSession()
try session.prefill("Summarize this eICR: …")
for try await chunk in session.decode() {
    print(chunk, terminator: "")
}
```
