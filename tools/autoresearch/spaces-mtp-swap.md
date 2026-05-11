# Spaces MTP Engine — Swap Procedure

**Date:** 2026-05-04
**Audience:** whoever ships the Gemma 4 hackathon Space after the safety-net
deploy lands.
**TL;DR:** the MTP-enabled engine is a parallel-track copy of
`spaces/zerogpu_engine.py` that adds Hugging Face assisted decoding via the
official 78 M `google/gemma-4-E2B-it-assistant` drafter. To swap it in, change
two file references in the deploy bundle. Reverting to the safety-net is the
same change in reverse.

## What's new vs the safety-net

- `spaces/zerogpu_engine_mtp.py` — engine module with assisted-decoding wired
  in. Public API (`chat_completion`, `chat_http_shim`, `model_banner`) is
  bit-identical to `zerogpu_engine.py` so `app.py` only needs an import swap.
- `spaces/requirements-mtp.txt` — same deps as `requirements.txt` but pins
  `transformers` to commit `41c3a5ac425e81aa1c9b3e6288eebaccf0c89835` (HEAD
  of `huggingface/transformers` main as of 2026-05-04). That commit is the
  first to ship the `gemma4_assistant` model_type. Tagged transformers
  releases (≤ 5.5.4) raise `ValueError: gemma4_assistant is not a recognized
  model_type` on drafter load.
- `MTP_ENABLED` env var (default `true`) — set to `false` / `0` / `no` /
  `off` to bypass the drafter at runtime without redeploying. The target
  model still loads; only the assisted-generation path is skipped.

The MTP engine never modifies `spaces/zerogpu_engine.py`,
`spaces/requirements.txt`, or `spaces/app.py`. The two files coexist in the
repo; deploy chooses one.

## How to swap the deploy from safety-net → MTP

The Spaces deploy is a flat directory built by `spaces/build.sh` (copies
`spaces/app.py`, `spaces/zerogpu_engine.py`, `spaces/requirements.txt`, and
the pipeline modules into `out/space/`). Three options:

### Option A — patch `build.sh` (preferred, single source of truth)

Edit `spaces/build.sh` so it copies the MTP variants:

```diff
-cp "${REPO_ROOT}/spaces/zerogpu_engine.py"  "${OUT_DIR}/zerogpu_engine.py"
-cp "${REPO_ROOT}/spaces/requirements.txt"   "${OUT_DIR}/requirements.txt"
+cp "${REPO_ROOT}/spaces/zerogpu_engine_mtp.py"  "${OUT_DIR}/zerogpu_engine.py"
+cp "${REPO_ROOT}/spaces/requirements-mtp.txt"   "${OUT_DIR}/requirements.txt"
```

We keep the destination filenames stable (`zerogpu_engine.py`,
`requirements.txt`) so `app.py` (which imports `from zerogpu_engine import
…`) needs no change. Re-run `bash spaces/build.sh` and push the bundle.

### Option B — modify the deploy bundle directly

After `bash spaces/build.sh` has run:

```bash
cp spaces/zerogpu_engine_mtp.py  out/space/zerogpu_engine.py
cp spaces/requirements-mtp.txt   out/space/requirements.txt
```

(Same effect as Option A, but the change lives only in `out/space/` and gets
clobbered by the next build.)

### Option C — leave the engine filename intact, edit `app.py`

If for some reason both engines need to coexist in a single deploy, copy
`spaces/zerogpu_engine_mtp.py` to `out/space/` alongside the original and
patch the import in `out/space/app.py`:

```diff
-from zerogpu_engine import (  # noqa: E402
+from zerogpu_engine_mtp import (  # noqa: E402
     chat_http_shim as _chat_http_shim,
     model_banner as _model_banner,
 )
```

Plus copy `spaces/requirements-mtp.txt` over `out/space/requirements.txt`.
This option exists for completeness; Option A is cleaner.

## Reverting MTP → safety-net

If the MTP deploy misbehaves on Spaces, kill it from the runtime control
panel and either:

1. **Fast revert (no rebuild):** in the Space's Settings, set the env var
   `MTP_ENABLED=false`. This keeps the MTP-pinned transformers but disables
   the drafter — generation falls back to plain `model.generate(...)` (no
   speculative decoding). The Space stays up, behavior is identical to the
   safety-net plus the wider transformers pin.
2. **Full revert:** undo the build.sh diff above, re-run `bash
   spaces/build.sh`, push the bundle. This restores the tagged-transformers
   pin too.

Prefer option 1 first — it's a config flip with zero rebuild risk and lets
us A/B the wrapper overhead vs MTP without touching the deploy.

## Bench results — wrapper vs bare-generate

See `tools/autoresearch/spaces-mtp-bench.json` for raw per-prompt numbers.
Reference numbers from the bare-generate bench (mtp-mlx-bench, MPS fp16, 9
eICR prompts, base E2B-it):

- baseline (no MTP):         14.24 tok/s
- with drafter (bare):       23.80 tok/s  (1.67× speedup)
- with drafter + cliniq FT:  29.13 tok/s  (1.92× speedup)

Wrapper bench was run on **CPU** in the smoke venv (no GPU available on the
dev host); the absolute numbers are an order of magnitude slower than the
MPS reference, but the *MTP-on / MTP-off* ratio is what matters for ZeroGPU
extrapolation. See `spaces-mtp-bench.json` for the per-prompt detail; the
wrapper-overhead delta vs bare `model.generate` should be sub-millisecond
per call (one tokenizer apply_chat_template + one decode + the regex tool
parser — all sub-ms compared to seconds of generation).

## Caveats discovered during integration

1. **`apply_chat_template(..., return_tensors="pt")` shape regression.**
   Transformers main HEAD (5.8.0.dev0, the SHA we pin) returns a
   `BatchEncoding` (dict-like) here, not a bare `torch.Tensor`. The MTP
   engine handles both shapes; `model.generate(...)` is called with
   `**gen_inputs` instead of positional `prompt_ids`. **The safety-net
   `zerogpu_engine.py` will hit the same bug if it's ever run against a
   transformers version that returns BatchEncoding** — the existing code
   does `prompt_ids.to(...)` and `prompt_ids.shape[1]`, both of which fail
   on BatchEncoding. Tagged transformers ≤ 5.5.4 returns Tensor and is
   unaffected. If we ever bump the safety-net's transformers pin past the
   chat-template-API change, port the same shape-normalize helper.
2. **`torch_dtype=` kwarg deprecated.** Both `from_pretrained` calls log a
   deprecation warning telling us to use `dtype=` on transformers main.
   Functionally still works on 5.8.0.dev0; safe to ignore for now. Fix in a
   follow-up by renaming `torch_dtype=_DTYPE` → `dtype=_DTYPE` once the
   tagged release we eventually target also supports it.
3. **MTP_ENABLED is read at module import time.** Flipping the env var
   without restarting the Space's Python process has no effect — the
   drafter is loaded (or not) based on the value seen at first import. To
   toggle live, restart the Space worker.
4. **Drafter-load failure is non-fatal.** If `from_pretrained(DRAFTER_ID)`
   raises (e.g. wrong revision in cache, network blip), the engine logs the
   error, sets `_drafter = None`, and continues serving requests with
   plain `model.generate`. `model_banner()` reflects the degraded state and
   `timings.mtp.drafter_load_error` carries the error string in every
   response — easy to spot from a single chat-completion call.
5. **Drafter weights are 78 MB only** (78 M params @ fp16/bf16 ≈ 156 MB on
   disk; bf16 in memory ≈ 156 MB). Cold-start delta over the safety-net is
   ~one HF download for the drafter and a few tenths of a second of extra
   load. No new HF auth needed — `google/gemma-4-E2B-it-assistant` is
   public.
6. **CPU smoke only.** We could not exercise the engine on real H200
   ZeroGPU from the dev host. The first ZeroGPU deploy IS the first
   GPU-path test for this code. Watch the Space build log for any
   ZeroGPU-specific complaints (e.g. `assistant_model` interaction with
   `@spaces.GPU(duration=120)` — both models live on the same `cuda`
   device per `device_map=_DEVICE`, which is the documented assisted-decode
   pattern).

## Environment variables (full list, MTP engine)

| Var | Default | Effect |
|---|---|---|
| `CLINIQ_GEMMA_MODEL_ID` | `unsloth/gemma-4-E2B-it` | Target model HF id |
| `CLINIQ_GEMMA_DRAFTER_ID` | `google/gemma-4-E2B-it-assistant` | Drafter HF id |
| `MTP_ENABLED` | `true` | `0`/`false`/`no`/`off` skips drafter load |
| `SPACES_ZERO_GPU` | (set by Spaces) | Forces CUDA + bf16 path |
| `CLINIQ_DISABLE_AGENT` | unset | Inherited from safety-net; skips engine import in `app.py` |

## Files

- `spaces/zerogpu_engine_mtp.py` — the engine
- `spaces/requirements-mtp.txt` — the pinned-SHA requirements
- `tools/autoresearch/spaces_mtp_bench.py` — bench harness through the
  engine wrapper
- `tools/autoresearch/spaces-mtp-bench.json` — merged per-prompt results
- `tools/autoresearch/mtp-mlx-bench-results.md` — reference bench from the
  bare `model.generate(...)` path (MPS, fp16, real numbers)
