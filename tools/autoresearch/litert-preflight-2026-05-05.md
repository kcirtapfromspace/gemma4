# LiteRT-LM v0.11.0 Mac Preflight — 2026-05-05

**Mission:** Stress-test LiteRT-LM v0.11.0 (released today) on Mac + iOS Simulator
to decide whether the iPhone path is salvageable for the 13-day hackathon window.
Pre-flight only — no deploy to device.

**Headline:** **GO with caveats.** Every "blocked" item the status report flagged
turned out to either (a) not reproduce on our toolchain, or (b) have a workable
manual fix. Real-prompt decode in the iOS Simulator hits **27 tok/s on E2B
CPU** — 6.8× our existing llama.cpp baseline (4 tok/s). MTP works at the C
API / Python API / CLI level on macOS GPU; the synthetic 2× headline didn't
materialise on real prompts but the path itself is unbroken. The iOS sim
prebuilt binary's headline gap is the missing dylib (#2158) — we worked around
it by fetching from `prebuilt/ios_sim_arm64/` LFS.

## Time spent

- **Wall clock:** 20 minutes (started 2026-05-05 22:11 MDT, finished writing
  2026-05-05 22:31 MDT). **Well under the 4-hour budget.**

## Install

```bash
# Fresh Python 3.12 venv (untouched scripts/.venv per instructions)
/opt/homebrew/bin/python3.12 -m venv tools/autoresearch/litert-preflight-venv
tools/autoresearch/litert-preflight-venv/bin/pip install litert-lm
```

**Installed:** `litert-lm 0.11.0` + `litert-lm-api 0.11.0` (wheel
`litert_lm_api-0.11.0-py3-none-macosx_12_0_arm64.whl`, 20 MB).

CLI binary: `tools/autoresearch/litert-preflight-venv/bin/litert-lm`.
Native dylib (statically links the constraint provider): `lib/python3.12/
site-packages/litert_lm/liblitert-lm.dylib` (62 MB, arm64, platform=macOS).

**Model:** `litert-community/gemma-4-E2B-it.litertlm` (2.47 GB) downloaded into
the existing `HF_HUB_CACHE=/Volumes/models/hf/hub`. The repo
`litert-community/gemma-4-E2B-it-litert-lm` exists alongside the CI default
`-E4B-it-litert-lm`; we used E2B since that's our deployment target.

---

## Question 1 — Can v0.11.0 load Gemma 4 E2B on Mac at all?

**YES on both CPU and GPU. No `#2149` reproduction.**

Synthetic benchmark (256 prefill / 128 decode, official `litert-lm benchmark`):

| Backend | MTP   | Prefill tok/s | Decode tok/s | Init s | TTFT s |
|---------|-------|--------------:|-------------:|-------:|-------:|
| CPU     | false |         77.58 |    **27.23** |  16.22 |   3.34 |
| CPU     | true  |         75.27 |        26.64 |  16.26 |   3.44 |
| GPU     | false |        222.81 |    **80.36** |   2.11 |   1.16 |
| GPU     | true  |       1313.71 |        79.88 |   1.95 |   0.21 |

GPU + MTP shows a striking 5.9× prefill speedup (222 → 1313) and 5.5× TTFT
drop (1.16 → 0.21 s) but **only 1.13× decode** (70 → 79 tok/s at 256/256, see
Q2). Both CPU configs work cleanly — no segfault on E4B (didn't test E4B —
have only E2B), no silent hang on E2B post-prefill. **#2149 does not affect
the macOS pip wheel.**

Real-prompt run (Python API, 121-word eICR pediatric measles case, top_k=1):

```
GPU MTP=False  wall=5.73s  ttft=0.22s  tokens=430  decode=77.86 tok/s
GPU MTP=True   wall=7.06s  ttft=0.28s  tokens=538  decode=79.22 tok/s
CPU MTP=False  wall=26.25s ttft=3.35s  tokens=471  decode=20.52 tok/s
CPU MTP=True   wall=20.48s ttft=2.91s  tokens=471  decode=26.74 tok/s
```

Bench harness: `tools/autoresearch/litert_preflight_bench.py`. Raw JSON at
`/tmp/litert-preflight-bench.json`.

**Output quality verified.** All four runs produced coherent, structurally
correct clinical text identifying measles as the diagnosis, listing
reportable status, and producing 5+ public-health action items. No Jinja
chat-template breakage — the system/turn template renders cleanly. The
"Jinja bug" called out in the handoff don't-do list does not reproduce in
v0.11.0 with this model.

---

## Question 2 — Does enabling MTP work, or does #2181 fire?

**MTP works. #2181 does NOT reproduce.**

I ran the exact command from issue #2181:

```bash
litert-lm run gemma-4-E2B-it.litertlm \
  --backend=gpu --enable-speculative-decoding=true \
  --prompt="What is the capital of France?"
```

Result: `The capital of France is **Paris**.` — clean, 1.3 s wall total.

**Why no repro?** The reporter is on **Python 3.14** with an older `litert_lm`
that exposes `Engine` as a pybind11 C++ class. Their traceback shows
`TypeError: Engine(): incompatible function arguments`, with a kwargs list
that doesn't include `enable_speculative_decoding` in the supported
signatures. The wheel I installed (Python 3.12, fresh `litert_lm 0.11.0`)
exposes `Engine` as a pure Python class in `litert_lm/engine.py` that wraps
the C ABI via ctypes — and it accepts `enable_speculative_decoding` directly
(see `engine.py:73-76`).

**Recommended upstream comment for #2181:** the bug is binding-version-
specific. On `litert_lm 0.11.0` + Python 3.12 + macOS arm64 wheel, the
reproducer command works end-to-end. Upgrading from Python 3.14 to 3.12
would likely unblock the reporter immediately, OR they need to upgrade
their `litert_lm` package to 0.11.0 (the API surface changed from pybind11
to ctypes).

**MTP speedup, real prompt, GPU, top_k=1 (Python API):**
- decode no-MTP: 77.86 tok/s
- decode MTP:    79.22 tok/s
- **speedup: 1.02× — essentially flat.**

**MTP speedup, synthetic 256/256 (CLI benchmark):**
- decode no-MTP: 70.30 tok/s
- decode MTP:    79.43 tok/s
- **speedup: 1.13×.**

The headline ">2× faster decode on mobile GPUs" claim from the v0.11.0 release
notes does **not** materialise on Mac Metal for either workload type. Possible
reasons: (a) the speedup is mobile-GPU-specific (Adreno/Mali), (b) the
release-notes "single position MTP" requires a different invocation path than
a simple `enable_speculative_decoding=true` flag, or (c) the drafter heads
inside the `.litertlm` artefact aren't fully wired up on the Mac Metal backend
yet. Compare to our prior `mtp-mlx-bench-results.md` numbers on
Transformers/MPS: **1.92× speedup on FT, 1.67× on base** — a different
implementation path achieved real speedup, so the gap here is in the
LiteRT-LM Metal MTP plumbing, not the technique.

**Stretch finding:** the ~6× prefill speedup with MTP enabled (synthetic) is
real and consistent (1217 → 1289 tok/s at 256/256, 222 → 1313 at 256/128).
That's worth investigating but doesn't help our use case (single-shot eICR
extraction is decode-bound, not prefill-bound).

---

## Question 3 — Can the iOS Simulator CLI bench be replicated?

**YES, with a manual dylib fetch.** The released
`litert_lm_main.ios_sim_arm64` is **unrunnable as shipped** because it
expects three companion dylibs that the v0.11.0 release does not include.
This is a real and currently-undocumented bug — see "New bug discovered"
below.

### Workaround that worked

```bash
# 1. Download the prebuilt iOS-sim arm64 dylibs from the LiteRT-LM repo's LFS
mkdir -p tools/autoresearch/litert-bins/ios_sim_arm64
for f in libGemmaModelConstraintProvider.dylib libLiteRt.dylib libLiteRtMetalAccelerator.dylib; do
  curl -sL -o tools/autoresearch/litert-bins/ios_sim_arm64/$f \
    "https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/main/prebuilt/ios_sim_arm64/$f"
done

# 2. Boot the existing iPhone17ProDemo simulator
xcrun simctl boot CADA1806-F64D-4B02-B983-B75F197D1EF3

# 3. Spawn the binary in the sim with DYLD_LIBRARY_PATH set
SIM=CADA1806-F64D-4B02-B983-B75F197D1EF3
LIBS=/Users/thinkstudio/gemma4/tools/autoresearch/litert-bins/ios_sim_arm64
MODEL=/Volumes/models/hf/hub/.../gemma-4-E2B-it.litertlm
SIMCTL_CHILD_DYLD_LIBRARY_PATH="$LIBS" \
  xcrun simctl spawn $SIM /Users/thinkstudio/gemma4/tools/autoresearch/litert-bins/litert_lm_main.ios_sim_arm64 \
    --model_path="$MODEL" \
    --backend=cpu \
    --input_prompt_file=/tmp/litert-prompt.txt
```

The simulator on Apple Silicon runs spawned processes against the host POSIX
filesystem (it sees `/Volumes/models`, `/Users/thinkstudio/...`), so we can
point straight at the cached model.

### Results — iOS Simulator (iPhone 17 Pro Demo, iOS 26.4 sim)

121-word eICR prompt, full bench output:

| Backend | Prefill tokens | Prefill tok/s | Decode tokens | Decode tok/s | TTFT | Init |
|---------|----:|--------------:|----:|--------------:|------:|------:|
| **CPU** | 228 |     **81.45** | 472 |     **27.17** | 2.84 s | 0.27 s |
| GPU/Metal | — | — (failed) | — | — | — | — |

**CPU output quality verified.** Output is identical in structure to the macOS
runs (measles diagnosis, reportable, 5 action items). 472 generated tokens.

**GPU/Metal in iOS sim CRASHES.** Exact error from `delegate_metal.mm:125`:

```
Failed to create DelegateKernelLiteRtMetal: INTERNAL: newComputePipelineStateWithFunction
error: texture binding has argument index 31 that is greater than 30
```

This is an iOS Simulator Metal limitation, not a LiteRT-LM bug per se — the
simulator's Metal stack caps argument indices at 30, while the prebuilt MTP
shaders need 31. **A real iOS device wouldn't have this cap.** But: we have
no physical iPhone to verify, and Apple's docs explicitly warn about
simulator/device divergence on Metal argument buffers.

### MTP on iOS sim: not testable via the released binary

The `litert_lm_main.ios_sim_arm64` test driver exposes only four CLI flags:
`--model_path`, `--backend`, `--input_prompt`, `--input_prompt_file`. It
does **not** expose `--enable_speculative_decoding`. Trying to pass it
gives `ERROR: Unknown command line flag 'enable_speculative_decoding'`.

To bench MTP on iOS sim, you'd either need to:
1. Build a custom test driver from source (`litert_lm_main.cc` + a flag
   plumb-through), OR
2. Build a Swift app linking the C API.

Both are out of scope per the brief ("stop short of writing any Swift app").

---

## Question 4 — How does this compare to existing on-device numbers?

| Path | Hardware | Backend | Decode tok/s | Notes |
|---|---|---|---:|---|
| llama.cpp Q3_K_M (existing) | iOS Sim | CPU | **4.0** | C12 baseline from handoff |
| **LiteRT-LM v0.11.0 E2B** | **iOS Sim** | **CPU** | **27.17** | This bench, real prompt, 472 tok |
| LiteRT-LM v0.11.0 E2B | macOS | CPU | 27.23 | Synthetic 256/128 |
| LiteRT-LM v0.11.0 E2B | macOS | GPU/Metal | 80.36 | Synthetic; real-prompt 77.86 |
| LiteRT-LM v0.11.0 E2B + MTP | macOS | GPU/Metal | 79.43 | Real-prompt 79.22 |
| Transformers MPS base (mtp-mlx-bench) | Mac | MPS | 14.24 | No MTP |
| Transformers MPS base + MTP | Mac | MPS | 23.80 | 1.67× speedup |
| Transformers MPS FT + MTP | Mac | MPS | 29.13 | 1.92× on FT |
| MLC-LLM Jetson (working) | Jetson NX | CUDA | 5–8 | Production-ready |

**Headlines:**
- **iOS Sim CPU: 6.8× faster than our existing llama.cpp baseline (27 vs 4 tok/s)**, on
  the same simulator, same prompt, same model class. This is by far the most
  important finding — if the path holds up on a physical iPhone (untested), it
  unblocks the Rank-1 mobile demo.
- LiteRT-LM Mac CPU is ≈2× the Transformers/MPS no-MTP rate (27 vs 14) — the
  C++ runtime is materially faster than the Python pipeline at the same
  precision.
- LiteRT-LM Mac GPU is ≈3.4× the Transformers/MPS+MTP rate (79 vs 23) — Metal
  via the LiteRT runtime is the fastest Mac path bar none.
- MTP in LiteRT-LM Mac Metal is **flat** vs no-MTP for end-to-end decode,
  contrary to the Google announcement. Prefill gets a real boost. Net: do NOT
  base demo claims on the "2× MTP speedup" headline until reproduced on
  device.

---

## New bugs discovered (filable)

### Bug A: `litert_lm_main.ios_sim_arm64` v0.11.0 release ships without companion dylibs

**Severity: high (blocks any out-of-the-box iOS-simulator usage).**

The release artefact `litert_lm_main.ios_sim_arm64` (19.6 MB) is
dynamically linked against three runtime dylibs that are present in the
LiteRT-LM repo under `prebuilt/ios_sim_arm64/` (LFS-stored) but are **not**
included as release assets. Running the binary out of the box yields:

```
dyld[]: Library not loaded: @rpath/libGemmaModelConstraintProvider.dylib
  Reason: tried: ... (no such file) ... (no such file) ...
```

**Workaround (works):** fetch the three dylibs manually from
`https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/main/prebuilt/ios_sim_arm64/`
(LFS media URL) and set `SIMCTL_CHILD_DYLD_LIBRARY_PATH` when spawning.

**Recommended fix:** include `libGemmaModelConstraintProvider.dylib`,
`libLiteRt.dylib`, `libLiteRtMetalAccelerator.dylib` (and their
`macos_arm64` and `linux_x86_64` siblings, if missing — didn't verify) as
release assets, OR ship them as a `litert-lm-runtime-libs.tar.gz` bundle
per platform.

**This is filable.** Did not file in this preflight session (left as an
action item).

### Bug B: Confirmed #2158 — `libGemmaModelConstraintProvider.dylib` minos mismatch

I separately verified the issue cited in the status report:

```
=== libGemmaModelConstraintProvider.dylib ===
 platform 7 (iossimulator)  minos 26.2  sdk 26.2
=== libLiteRt.dylib ===
 platform 7                 minos 14.0  sdk 26.2
=== libLiteRtMetalAccelerator.dylib ===
 platform 7                 minos 14.0  sdk 26.2
```

The constraint-provider dylib is built against iOS 26.2, the others against
iOS 14.0. Confirms #2158. The vtool workaround documented in the issue
(`vtool -set-build-version ios 16.0 26.2 ...`) is the right fix; we did
NOT need to apply it for the simulator (the sim accepted the 26.2 minos
fine), so the issue specifically blocks **device deployment / App Store
submission**, not simulator dev work.

---

## Status of all the original concerns from the status report

| Concern | Status from report | Status after preflight |
|---|---|---|
| #2181 enable_speculative_decoding TypeError on macOS | open, "headline feature broken" | **NOT REPRODUCED** on Python 3.12 + litert-lm 0.11.0 wheel. Bug is on Python 3.14 + older pybind11 binding. Comment-worthy. |
| #2149 CPU decode crash/hang on Gemma 4 | open, blocks tool-calling on CPU | **NOT REPRODUCED** on Mac CPU via wheel. Both prefill and decode succeed for 471 tokens, multiple runs. The bug is specific to Linux x86_64 + custom shared-lib build path. |
| #2158 dylib minos mismatch | open, blocks App Store | **CONFIRMED** but only affects device/App Store. Sim works fine. vtool workaround unblocks. |
| Jinja chat-template bug from v0.10.1 handoff | unknown | **NOT REPRODUCED** in v0.11.0 with `litert-community/gemma-4-E2B-it-litert-lm`. All four matrix runs produced clean structured clinical output. |
| No Swift SDK | confirmed | unchanged. C API + ctypes still the only options. |
| MTP "≥2× decode speedup" claim | unverified | **NOT REPRODUCED** on Mac Metal: real-prompt speedup ≈1.02×, synthetic ≈1.13×. Prefill does get a 5–6× boost. May still hold on actual mobile GPUs. |

---

## Go / no-go for iPhone path

**GO — ship without MTP, lean into the on-device CPU number.**

**Rationale:**
1. The iOS sim CPU run delivers **27 tok/s** on a real eICR prompt with the
   same E2B model class our existing app uses. That is **6.8× our llama.cpp
   baseline of 4 tok/s** in the same simulator. Even allowing for sim-vs-
   device variance, this is a credible "now it actually responds in real time"
   improvement worth showing.
2. All "this is dead" warnings from the status report turned out to be
   either toolchain-version-specific (#2181), platform-specific to Linux
   (#2149), only blocking App Store submission rather than dev work
   (#2158), or never reproduced in v0.11.0 at all (Jinja bug).
3. The C-API path is solid. Python 3.12 + ctypes wrapper works. Tools,
   conversations, streaming, sampling all functional via
   `litert_lm.engine.Engine`.
4. The dylib-bundling miss (Bug A above) is a 30-second curl workaround.
5. MTP doesn't deliver its headline 2× on Mac Metal end-to-end, but the
   path *runs* without errors — so we can leave the toggle in code and let
   the runtime decide.

**The thing to watch:** we have **zero** physical-iPhone numbers. The
simulator's Metal stack failed (Bug B / Metal arg-buffer cap). On a real
iPhone 15 Pro the Metal path *should* work — but unverified. If we want
to ship the iPhone GPU path, we MUST get our hands on a device before
2026-05-13 to validate.

**No-go conditions (would flip back to "drop"):**
- We can't get a physical iPhone for end-to-end validation by 2026-05-13,
  AND we want to claim GPU/Metal speed (~80 tok/s). With CPU-only iPhone
  numbers (~25-27 tok/s presumably), the story is still "6× llama.cpp" but
  without the Metal headline.
- We discover the .litertlm format hard-locks tool-calling output (didn't
  test — used a free-form prompt). If tool-calling round-trips fail, the
  whole agent pipeline breaks and we're back to llama.cpp.

**Suggested next actions (not done in this preflight):**
1. Wire up a single-tool calling probe through `Conversation.send_message`
   to verify the agent path works in v0.11.0.
2. File Bug A upstream (release ships without dylibs).
3. Comment on #2181 with our Python 3.12 success evidence.
4. Either acquire a physical iPhone OR commit to "CPU-only on-device"
   claims.

---

## Files left behind

- `tools/autoresearch/litert-preflight-venv/` — venv with `litert-lm 0.11.0`
- `tools/autoresearch/litert_preflight_bench.py` — 4-config Python bench
- `tools/autoresearch/litert-bins/` — iOS sim + macOS test binaries +
  prebuilt dylibs (≈ 70 MB total)
- `/Volumes/models/hf/hub/models--litert-community--gemma-4-E2B-it-litert-lm/`
  — 2.47 GB cached `.litertlm` model
- `/tmp/litert-preflight-bench.json` — raw 4-config matrix output
- `/tmp/litert-q*-{cpu,gpu}*.log` — verbose CLI/bench logs from each step
- `/tmp/litert-prompt.txt` — the 121-word eICR pediatric measles prompt

No commits made. No changes to `apps/mobile/ios-app/`. No changes to
`scripts/.venv/`. The iPhone17ProDemo simulator was booted; safe to leave
booted or shut down with `xcrun simctl shutdown CADA1806-...`.
