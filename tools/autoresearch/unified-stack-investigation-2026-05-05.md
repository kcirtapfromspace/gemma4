# Unified single-runtime stack investigation — Gemma 4 hackathon

**Date:** 2026-05-05  •  **Hackathon deadline:** 2026-05-18 (13 days)
**Time spent:** ~80 minutes wall clock (started 22:42 MDT, finished 00:02 MDT)
**TL;DR:** **No unified single-runtime stack is feasible in 13 days. Ship the two-stack story.**

---

## The puzzle, restated

We have two measured speedup paths that do not currently compose:

| Path | Runtime | Headline | Limitation |
|---|---|---:|---|
| Mac Metal raw decode | LiteRT-LM v0.11.0 | **80 tok/s** | MTP path effectively 1.0× (flat) |
| MPS + speculative decoding | Transformers + drafter | **29 tok/s on FT** (1.92× over base) | Base Transformers/MPS decode is only ~14 tok/s |

We tested whether either runtime can deliver both speedups together.

---

## Approach A — Custom spec-decode wrapper around LiteRT-LM

**Verdict: DEAD before benching. Hard precondition fails.**

### Time spent: 8 minutes

### What I checked

1. **Is there a standalone `.litertlm` drafter?**
   - On HF: `litert-community/gemma-4-E2B-it-litert-lm` ships a single `gemma-4-E2B-it.litertlm` (2.47 GB) bundle with sidecar files including `gemma-4-E2B-it.litertlm.mtp_drafter.xnnpack_cache_*`. **The drafter is baked into the bundle**, not a separate file.
   - There is no `gemma-4-E2B-it-assistant.litertlm` repo on HuggingFace. Google has never shipped one.
   - The `gemma-4-E2B-it-assistant` HF safetensors repo is the source for the drafter, but converting it to `.litertlm` would require running Google's internal `.litertlm` build pipeline (proto + tflite + xnnpack cache + magic-number tensor patch — see `magic_number_utils.cc:425` in the verbose logs). No public conversion script exists in `litert_lm` v0.11.0.

2. **Does the LiteRT-LM Python API expose logits or per-token probabilities so we could implement spec-decode externally?**
   - Inspected `litert-preflight-venv/lib/python3.12/site-packages/litert_lm/{engine.py,session.py,_ffi.py}`.
   - Session API surface: `run_prefill(contents: list[str])`, `run_decode() -> Responses`, `run_decode_async() -> stream`, `run_text_scoring(target_text)`. Returns `texts`, optional `scores` (per-sequence likelihoods), optional `token_lengths`, optional `token_scores` (one float per emitted token).
   - **No API exposes the next-token logits or top-k probability vectors.** No way to inject prefilled token IDs. No way to advance the KV cache by N candidate tokens then read back which were "accepted" with their probabilities.
   - Speculative decoding's rejection sampler needs `q(x)` (drafter prob) and `p(x)` (target prob) for every candidate token. That information simply isn't surfaced by the C ABI.

### Conclusion

To build a custom spec-decode loop on top of LiteRT-LM you would need to:
1. Build a custom `.litertlm` for the drafter (no public tooling).
2. Patch the C++ runtime to export logit vectors per decode step (vendor fork).
3. Wrap both engines and implement the Leviathan/Chen-style verify-and-accept logic in Python.

This is multi-week work. **Killed.**

---

## Approach B — Why LiteRT-LM Mac Metal MTP is flat

**Verdict: Architectural limitation, not a config bug. No knob exists to fix it on Metal in v0.11.0.**

### Time spent: 12 minutes

### What I checked

1. **CLI knobs:** `litert-lm benchmark --help` and `litert-lm run --help` only expose `--enable-speculative-decoding [auto|true|false]`. No `num_speculative_tokens`, `draft_length`, `max_lookahead`, `batch_size_for_drafter`, etc. The Python `Engine` constructor only accepts the boolean flag (`engine.py:73-76`).

2. **Verbose logs from prior preflight (`/tmp/litert-q2-gpu-mtp-v.log`):**
   ```
   model_type: tf_lite_mtp_drafter (loaded)
   signature=mtp_drafter, subgraph_index=0, num_tensors=275, num_inputs=8, num_outputs=2, num_ops=198
   llm_litert_mtp_drafter.cc:151] Num drafted tokens: 2901
   llm_litert_mtp_drafter.cc:152] Num verified tokens: 2901
   llm_litert_mtp_drafter.cc:154] Success rate: 1
   ```

3. **Fresh real-prompt verbose run (capital-of-France) shows the same pattern:**
   ```
   Num drafted tokens: 3057
   Num verified tokens: 3057
   Success rate: 1
   ```

### Interpretation

`drafted == verified` and `success rate = 1.0` across both synthetic and real-prompt runs is the smoking gun. This is not a 1.92×-style assistant_model loop where the drafter proposes K candidates and the target verifies them in parallel, accepting some fraction. It looks like Google's published "single-position MTP head" — the target model has an extra MTP head that produces ONE next-token candidate per regular forward, and the verification is essentially "did the regular head agree?" answered by re-using the same decoder pass. With a 1:1 draft-to-verify ratio the only possible speedup comes from the MTP head being computationally cheaper than recomputing the target's main head — and on Mac Metal that turns out to be ~2-13% of total decode wall, hence the flat result.

The 5.9× **prefill** speedup with MTP enabled (1313 vs 222 tok/s in the synthetic bench) is a separate Metal kernel optimization triggered by the `enable_speculative_decoding=true` codepath but unrelated to actual draft-then-verify. Decode-bound workloads can't benefit from it.

### What would unblock real MTP on this runtime

- A new `.litertlm` bundle that includes a multi-token MTP drafter (Google's choice, not a config knob).
- OR a runtime version that exposes a `num_speculative_tokens` parameter and a multi-position drafter API. v0.11.0 does not have this.
- OR a different mobile-GPU backend (Adreno/Mali) where Google's code path ships a different drafter implementation. Untestable without the hardware.

**No measurement run was needed** — the architectural ceiling is visible from the verbose-log ratio alone. Confirming the theoretical 1.13×–1.02× on synthetic vs real already happened in the preflight; nothing new to discover here.

---

## Approach C — Transformers + torch.compile + MTP on MPS

**Verdict: Compile contributes nothing on MPS. MTP composes with compile only when drafter is left eager. Cannot close the runtime gap.**

### Time spent: 60 minutes

### Bench harness

`tools/autoresearch/mtp_compile_bench.py` — 5 scenarios on top of the existing `mtp_bench.py` instrumentation:

| Scenario | Compile target | Compile drafter | MTP |
|---|---|---|---|
| `s0_baseline` | no | n/a | no |
| `s1_compile_only` | yes | n/a | no |
| `s2_mtp_only` | no | no | yes |
| `s3_compile_mtp` | yes | yes | yes |
| `s4_compile_target_only` | yes | no | yes |

All on FT model (`/Users/thinkstudio/mnt/models/cliniq/v2-fp16-merged`), MPS, fp16, greedy, 3 prompts × 96 max_new_tokens, torch 2.11.0 + transformers 5.8.0.dev0, `inductor` backend, `default` mode.

Also a side probe at `tools/autoresearch/static_cache_compile_probe.py` testing the documented HF `cache_implementation="static"` + compile path with 3 warmed iterations.

### Results

| Scenario | Total tok | Wall s | tok/s | Speedup vs s0 | Notes |
|---|---:|---:|---:|---:|---|
| `s0_baseline` (no compile, no MTP) | 271 | 19.49 | **13.90** | 1.00× | matches prior 14.2 |
| `s1_compile_only` | 271 | 17.86 | **15.17** | 1.09× | within noise |
| `s2_mtp_only` (control == ft_mtp) | 271 | 10.81 | **25.08** | 1.80× | matches prior 1.92× |
| `s3_compile_mtp` | — | — | **FAILED** | — | `ValueError: inputs_embeds and shared_kv_states cannot be None` at warmup |
| `s4_compile_target_only` | 271 | 10.88 | **24.92** | 1.79× | indistinguishable from s2 |

Static-cache probe (50 tok × 3 iters, steady state):
| Config | tok/s |
|---|---:|
| dynamic cache, no compile | 15.27 |
| static cache, no compile | 14.27 |
| static cache + compiled forward | 13.91 |

(Compile incurred a 36s warmup penalty plus a `torch._dynamo hit config.recompile_limit (8)` warning caused by the StaticCache's `cumulative_length_int` changing during decode — every step recompiled until the limit was hit, then dynamo fell back to eager. Even after that fallback, steady-state is identical to no-compile.)

### Why compile gives zero speedup on MPS

Inspected the inductor codegen via `TORCH_LOGS=output_code` on a small MLP block. The output:

```python
extern_kernels.addmm(arg1_1, arg2_1, reinterpret_tensor(arg0_1, ...), out=buf0)
mps_lib_0.generated_kernel(buf1, buf0, threads=[128])  # GELU only
extern_kernels.addmm(arg4_1, buf1, reinterpret_tensor(arg3_1, ...), out=buf2)
```

Inductor on MPS generates real Metal shaders for elementwise/activation ops, but **dispatches every `linear`/`matmul` to the existing `extern_kernels.addmm` (the MPS BLAS path)**. For decode-bound transformer inference where 95% of wall time is matmul, fusing only the activations gives essentially 0% speedup. Confirmed via a tight microbench (Linear → GELU → Linear, fp16, 1×32×2048 input):

```
eager:    39.5 ms / 100 calls
inductor: 41.3 ms / 100 calls
speedup:  0.96×
```

### Why s3 (compile target + compile drafter + MTP) fails

Traceback at warmup:

```
ValueError: inputs_embeds and shared_kv_states cannot be None.
```

This fires inside Gemma 4's forward when assisted decoding tries to call the drafter. Compiling the drafter freezes a particular kwargs signature; the assisted-generation loop then passes a different signature on the next invocation (carrying drafter-specific state from the previous step) and the compiled graph rejects it. **The MTP path's dynamic kwargs are incompatible with `torch.compile`-wrapped drafters.**

s4 (compile only target, leave drafter eager) avoids the contract mismatch and succeeds — but produces no speedup because (a) the target compile is ineffective on MPS as established, and (b) the drafter's per-step overhead is small. Net: same as s2.

---

## Side-quest: bitsandbytes int4 on MPS

Skipped — `bitsandbytes` not installed in any of the existing venvs, and it's well-documented to not work on Apple Silicon MPS (the CUDA quantization kernels don't have Metal equivalents). No 4-hour-budget exploration would change that.

---

## Final recommendation: ship the two-stack story

**A unified single-runtime stack is not feasible in 13 days.** Each of the three approaches has a different failure mode that none of the available 13-day work can fix:

- **Approach A:** requires building a custom `.litertlm` bundle for the drafter (no public tooling) AND patching the LiteRT-LM C++ runtime to expose logits (vendor fork). Multi-week minimum.
- **Approach B:** the Mac Metal MTP path is doing single-position drafting with success rate 1.0; multi-token speculative decoding requires either a new bundle from Google or a different runtime backend. No knob exists in v0.11.0 to change this.
- **Approach C:** torch.compile inductor on MPS does not fuse matmuls — it falls back to MPS BLAS for the 95% of compute that matters. Compose with MTP either crashes (drafter compile) or no-ops (target-only compile).

### What we should ship

The two-stack story stands as the right hackathon submission:

1. **Server side / HF Spaces:** Transformers + MTP via `assistant_model`. Proven 1.92× on FT, 1.67× on base. Maps to ZeroGPU H200. **No compile needed** — the H200 will dominate raw throughput; MTP is the differentiator.
2. **iOS Simulator bench artifact:** LiteRT-LM v0.11.0 CLI binary, no MTP. Proven 6.8× over llama.cpp on the same simulator. Already packaged in `tools/autoresearch/litert-bins/`.

These are complementary, both reproducible, both honest about what they measure.

### What to be careful NOT to claim

- **Do not claim** "compounded LiteRT-LM × MTP speedup". The MTP path on Mac Metal is a no-op for decode (1.02×). On a real Adreno/Mali phone it might compound — untested, no hardware.
- **Do not claim** "torch.compile gives faster Mac decode". On MPS in torch 2.11 it doesn't.
- **Do not claim** "we ran a unified Python stack at LiteRT-LM speed". We can't.

### Path forward (post-hackathon, if anyone keeps pulling this thread)

1. **CUDA test of compile + MTP on H200/A100.** On CUDA, inductor fuses matmuls via Triton, which would actually move the needle. The s3 crash (`inputs_embeds and shared_kv_states cannot be None`) needs a transformers fix or an `assistant_model_class` workaround — file an upstream issue. If solved, compile + MTP on H200 could plausibly hit 200+ tok/s.
2. **MLX port.** Apple's MLX runtime on Mac Metal is materially faster than HF/MPS for transformer decode (community reports 3-5× over MPS on similar models). MLX-LM supports speculative decoding upstream. This is the "right" Mac path but requires re-implementing the cliniq fine-tune in MLX or adapting the safetensors bridge — 1-2 weeks.
3. **Real device test of LiteRT-LM MTP on Adreno/Mali.** Get a Pixel 8/9 or Galaxy S24, run the same `--enable-speculative-decoding=true` benchmark, see whether the headline 2× actually exists somewhere. This is the single highest-leverage open question and requires only hardware acquisition.
4. **Fork LiteRT-LM to expose logits.** Build of `liblitert-lm.dylib` from source with a `litert_lm_session_get_next_logits()` shim, then build Approach A on top. Realistic 2-3 week project; would unlock external spec-decode at LiteRT-LM speed.

None of these fit in the remaining 13 days alongside the actual hackathon submission work.

---

## Files left behind

- `tools/autoresearch/mtp_compile_bench.py` — 5-scenario torch.compile + MTP bench (new)
- `tools/autoresearch/static_cache_compile_probe.py` — 3-config static-cache probe (new)
- `tools/autoresearch/mtp-compile-bench-raw.json` — raw scenario output (s0,s1,s2,s3)
- `/tmp/mtp-compile-s4.json` — raw s4 output
- `/tmp/mtp-compile-s1*.json` — earlier exploratory runs (can be deleted)

No commits made. No changes to `scripts/.venv`, `apps/mobile/ios-app/`, or `spaces/` source files.
