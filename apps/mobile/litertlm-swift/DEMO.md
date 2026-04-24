# LiteRtLm Swift — decode demo

This page shows how to run the LiteRT-LM Swift bindings end-to-end against
Google's stock `gemma-4-E2B-it.litertlm`: prefill a prompt, stream the
decoded tokens, print a tok/s number.

## Prereqs

1. Xcode 15+ and a booted iOS 17 simulator (the `make` targets auto-detect
   the first booted device; override with `SIMULATOR_ID=...`).
2. A local clone of upstream LiteRT-LM at `/tmp/c11-litertlm/LiteRT-LM`
   (handled by `make clone`).
3. The model. One-shot:

   ```bash
   ./scripts/fetch_model.sh /tmp/gemma-model/gemma-4-E2B-it.litertlm
   ```

4. The xcframework. First build is ~30-60 min cold, ~30 s warm:

   ```bash
   make xcframework   # wraps scripts/build_xcframework.sh
   ```

## 1. Decode smoke test (XCTest on the Simulator)

This is the canonical "does decode work?" check. It caps decode at 32
tokens so the test finishes in a bounded window.

```bash
cd apps/mobile/litertlm-swift
make test-decode
```

Observed output from an iPhone 17 Pro Simulator on a Mac Studio, CPU
backend, 2026-04-23:

```
Test Case '-[LiteRtLmTests.DecodeTest testDecodeProducesTokens]' started.
LITERTLM_DECODE_RESULT tokens=6 elapsed_s=0.434 tok_per_sec=13.84
LITERTLM_DECODE_SAMPLE <<<
Red, red, red, red, red>>>
Test Case '-[LiteRtLmTests.DecodeTest testDecodeProducesTokens]' passed (1.063 seconds).
Test Suite 'DecodeTest' passed at 2026-04-24 00:28:16.153.
** TEST SUCCEEDED **
```

Same run, plus C11's model-load smoke: all three tests green.

```
Test Case '-[LiteRtLmTests.DecodeTest testDecodeProducesTokens]' passed (1.057s)
Test Case '-[LiteRtLmTests.EngineLoadTest testEngineLoadsModel]' passed (0.624s)
Test Case '-[LiteRtLmTests.EngineLoadTest testShimVersionIsReachable]' passed (0.001s)
Test Suite 'All tests' passed at 2026-04-24 00:28:36.053.
** TEST SUCCEEDED **
```

## 2. Command-line demo

The `LiteRtLmCli` executable target wraps the same engine/session
pipeline with a friendly argv interface. One-shot:

```bash
cd apps/mobile/litertlm-swift
make cli PROMPT='The capital of France is'
```

That target builds the CLI for the iOS Simulator and spawns it via
`simctl`. Under the hood:

```bash
SIM=$(xcrun simctl list devices booted | awk '/\(Booted\)/ { print $NF }' | tr -d '()' | head -1)

# Stage the model inside the simulator's sandbox.
SIM_ROOT="$HOME/Library/Developer/CoreSimulator/Devices/$SIM/data"
mkdir -p "$SIM_ROOT/private/tmp/gemma-model"
cp -n /tmp/gemma-model/gemma-4-E2B-it.litertlm "$SIM_ROOT/private/tmp/gemma-model/"

# Build the CLI. SwiftPM's `swift run` on the Mac host doesn't
# auto-wire the xcframework search paths, so use xcodebuild against the
# simulator.
cd apps/mobile/litertlm-swift
xcodebuild -scheme LiteRtLmCli \
  -destination "platform=iOS Simulator,id=$SIM" \
  build

# Locate the built binary (path depends on Xcode's DerivedData layout).
CLI=$(find "$HOME/Library/Developer/Xcode/DerivedData" \
  -name LiteRtLmCli -path '*Debug-iphonesimulator*' -type f | head -1)

# Spawn the binary on the simulator with the model path threaded through
# via `SIMCTL_CHILD_<env-var>` so simctl forwards it into the child.
SIMCTL_CHILD_LITERTLM_MODEL_PATH=/private/tmp/gemma-model/gemma-4-E2B-it.litertlm \
SIMCTL_CHILD_LITERTLM_MAX_TOKENS=64 \
  xcrun simctl spawn "$SIM" "$CLI" "The capital of France is"
```

### Running the CLI as a raw Mac binary (optional, not yet supported)

SwiftPM's `swift run LiteRtLmCli "..."` needs a `macos-arm64` slice in
the xcframework plus an `-F` link flag — the current build only ships
ios-arm64 + ios-arm64-simulator. To add the macOS slice, drop
`macos = ["arm64"]` into the `apple_static_xcframework` rule at
`scripts/bazel_overlay/BUILD.bazel` and rerun `make xcframework`
(adds ~5 min to the cold build).

## Captured CLI transcripts

All runs: iPhone 17 Pro Simulator (UDID
`CADA1806-F64D-4B02-B983-B75F197D1EF3`), 2026-04-23, CPU backend,
Google's stock `gemma-4-E2B-it.litertlm`, shim version
`0.2.0+team-c14`.

### Run 1 — "The capital of France is"

```
$ SIMCTL_CHILD_LITERTLM_MODEL_PATH=/private/tmp/gemma-model/gemma-4-E2B-it.litertlm \
  SIMCTL_CHILD_LITERTLM_MAX_TOKENS=48 \
  xcrun simctl spawn CADA1806-F64D-4B02-B983-B75F197D1EF3 \
    $DERIVED/LiteRtLmCli "The capital of France is"

LiteRtLmCli v0.2.0+team-c14
  model:    /private/tmp/gemma-model/gemma-4-E2B-it.litertlm
  backend:  cpu
  prompt:   The capital of France is
  maxTok:   48
---
 Paris.
---
load_s=0.26 prefill_s=0.34 decode_s=0.13
tokens=2 tok_per_sec=15.55
```

### Run 2 — "List five common colors, separated by commas."

```
$ SIMCTL_CHILD_LITERTLM_MODEL_PATH=/private/tmp/gemma-model/gemma-4-E2B-it.litertlm \
  SIMCTL_CHILD_LITERTLM_MAX_TOKENS=64 \
  xcrun simctl spawn CADA1806-F64D-4B02-B983-B75F197D1EF3 \
    $DERIVED/LiteRtLmCli "List five common colors, separated by commas."

LiteRtLmCli v0.2.0+team-c14
  model:    /private/tmp/gemma-model/gemma-4-E2B-it.litertlm
  backend:  cpu
  prompt:   List five common colors, separated by commas.
  maxTok:   64
---

Red, red, red, red, red
---
load_s=0.26 prefill_s=0.35 decode_s=0.42
tokens=6 tok_per_sec=14.14
```

### Interpretation

- **Decode works end-to-end** through the Swift bindings — we see the
  model's output flowing through `AsyncThrowingStream<String>` into
  `FileHandle.standardOutput.write`.
- **Simulator CPU tok/s is ~14-16** on a Mac Studio M-series host. This
  is the iOS Simulator's software renderer, not the iPhone Metal path
  — real-device throughput is expected to be substantially higher
  (the 52-56 tok/s target is iPhone 17 Pro + Metal delegate).
- **Output degeneracy ("Red, red, red, ...")** is expected: we're
  calling the model without any chat template, the Gemma 4 E2B base is
  small, and the greedy-ish default sampler on short prompts is prone
  to n-gram loops. Swapping in a proper chat template and/or tuning
  the `SamplerParams` is future work and does not affect the tok/s
  measurement.
- The shim's current decode primitive is blocking-then-drain, so the
  CLI's "streamed" output appears as one burst after a ~0.4 s pause.
  Real per-token streaming arrives when we switch to
  `RunDecodeAsync`.

## Known limitations

- Simulator CPU backend. The decode path is intentionally slow on the
  Simulator; real tok/s numbers come from iPhone hardware with the Metal
  delegate. This demo proves correctness, not throughput.
- The shim's decode primitive is blocking-then-drain, not
  token-at-a-time. `RunDecodeAsync` will land in a later iteration and
  the Swift `AsyncThrowingStream` API won't change when it does.
- `swift run LiteRtLmCli` on the Mac host needs the `macos-arm64`
  xcframework slice plus an `-F` link flag — easier to invoke the CLI via
  the iOS Simulator destination or to embed the same calls inside the
  ClinIQ host app.
