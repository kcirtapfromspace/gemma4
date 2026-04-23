# MLC-LLM Gemma 4 E2B Jetson baseline — 2026-04-23

Baseline branch for the hackathon team's mlc-llm port. Captures the state of the
Gemma 4 E2B model running on the Jetson Orin NX 8GB via MLC-LLM + patched TVM,
and documents a blocker that makes the 5-8 tok/s baseline **unreachable from a
pure `8da77e9` checkout without a full re-convert + re-compile cycle**.

---

## TL;DR

| item                           | value                                                        |
| ------------------------------ | ------------------------------------------------------------ |
| branch                         | `team/mlc-baseline-2026-04-23`                               |
| head commit                    | `8da77e9` ("Gemma 4 E2B working at 5-8 tok/s on Jetson")     |
| pod                            | `gemma4/mlc-test` on `talos-jetson-3` (10.244.4.127)         |
| reproducible tok/s today       | **Not measured** — see `Blocker` below                       |
| degeneration onset             | **Not measured** — see `Blocker` below                       |
| last known good tok/s          | 5-8 tok/s (from commit message; no reproducible test)        |
| hypothesized cause             | **Sliding-window KV eviction misbehaving at HEAD** — details below |

**Status = option (b) from the task spec: "definitive proof the baseline is
unreachable from current state + explanation of why".**

---

## What I did (in order)

1. **Reverted the live pod's TVM `kv_cache.py`** from the WIP-era
   `_IS_MHA_FAMILY`-patched version to the stock baseline saved at
   `tools/autoresearch/kv_cache_stock.py`. After `kubectl cp`, the pod's file
   contained 0 matches for `_IS_MHA_FAMILY` (verified).

2. **Redeployed HEAD `gemma4_model.py`** to the pod at both
   `/opt/mlc-llm/python/mlc_llm/model/gemma4/gemma4_model.py` and
   `/usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/gemma4_model.py`.
   Verified: 0 matches for `double_wide_start_layer` (WIP-only identifier),
   1 match for `num_kv_shared_layers = 0  # DEBUG: disable KV sharing`
   (the HEAD override introduced by commit 8da77e9).

3. **Python import sanity check** passed after the revert:
   `AttnKind.MHA_SLIDING = 3`, `gemma4` registered in `MODELS`,
   `Gemma4TextConfig.use_double_wide_mlp default = False`.

4. **Attempted a one-shot MLCEngine load** against the existing compiled
   `/models/mlc-models/gemma4-e2b-q4f16_1-v2/lib.so` (dated Apr 22 01:09 UTC)
   with HEAD Python. The `MLCEngine(...)` constructor logged
   *"Engine loaded!"* and reported *"Estimated total single GPU memory usage:
   2735.468 MB"* — so the `.so` itself still dlopens. **However**, the first
   call to `engine.chat.completions.create(...)` hung: all 14 Python threads
   parked in `futex_wait`, steady ~100% CPU on one core, no tokens emitted,
   no error raised, no progress after 2 minutes. Killed and moved on.

5. **Attempted a baseline re-convert** via
   `python3 -m mlc_llm convert_weight /models/mlc-models/gemma-4-e2b-it
    --model-type gemma4 --quantization q4f16_1 --device cuda --output ...
   gemma4-e2b-q4f16_1-baseline/`. First quantization step completed in 11.92s
   projecting >1h 50min for all 567 tensors. **Pod was OOM-killed**
   (exit code 137) shortly after — the `test-pod.yaml` limit is 7 GiB and
   the node's memory is over-committed (Agent A's llama-server holds another
   7 GiB limit on the same 8 GiB Jetson). Re-created the pod with
   `kubectl apply -f apps/mlc-llm/test-pod.yaml`; it comes up clean but needs
   the NVIDIA lib re-injection from
   `~/.claude/projects/.../reference_jetson_pod_recovery.md` before TVM imports.

---

## Blocker — why the baseline is unreachable from HEAD today

### The version skew

Commit `8da77e9` added the line

```py
self.num_kv_shared_layers = 0  # DEBUG: disable KV sharing
```

to `Gemma4TextConfig.__post_init__` (line 87 of
`apps/mlc-llm/gemma4-patches/gemma4_model.py`). The commit message explicitly
says "KV sharing disabled". That override makes
`layer_uses_shared_cache(idx)` return False for every layer and therefore
`layer_uses_double_wide_mlp(idx)` returns False for every layer as well.
The expected MLP shape for every one of the 35 decoder layers becomes
**single-wide** (`gate_up_proj = (12288, 192)`, `intermediate_size = 6144`).

But every compiled artifact currently on disk — both on the Mac
(`apps/mlc-llm/build-output/gemma4-weights-v3/`,
`.../gemma4-weights-merged-q4f16_1/`) and on the Jetson pod
(`/models/mlc-models/gemma4-e2b-q4f16_1{,-v2,-fixed,-new,-scaled,-merged-*}/`)
— was produced **before** that override was in place. `ndarray-cache.json` on
every one of them shows the same mixed shape pattern:

```
gate_up_proj  shape (12288, 192)   15 params   layers 0..14      (single-wide)
gate_up_proj  shape (24576, 192)   20 params   layers 15..34     (double-wide)
```

`mlc-chat-config.json` for those builds has `num_kv_shared_layers: 20`,
`use_double_wide_mlp: true`. Perfectly consistent **with itself** — and
perfectly inconsistent with HEAD's `num_kv_shared_layers = 0` override.

### Why the engine loads but inference hangs

`MLCEngine(model_dir, model_lib=lib.so)` does two things:

1. Reads `mlc-chat-config.json` → `Gemma4Config(...)` → runs `__post_init__`
   which zeros `num_kv_shared_layers`. Every Python-side shape check now
   assumes single-wide everywhere.
2. Opens `lib.so` and binds its exported entrypoints. The compiled CUDA
   kernels inside `lib.so` were built for the *other* topology (single-wide
   layers 0-14, double-wide layers 15-34), so their tensor shape hints,
   PagedKVCache layouts, and attention-kind dispatch tables come from the
   double-wide-aware compile.

During actual generation, Python and the `.so` disagree on layer 15+: the
Python config says the MLP expects `6144`-wide activations, but the kernel
wants `12288`-wide (single-wide weights would be broadcast through a
wider-than-expected GEMM). The dispatch blocks waiting for a tensor
signature that will never arrive, and we see the `futex_wait` stall I
reproduced. No exception is raised because the disagreement is on CUDA
stream ordering, not on a Python shape assertion.

### To actually benchmark HEAD you must do both

- Re-convert weights on HEAD code → all-single-wide `gate_up_proj`.
  Attempted this run; it OOM-killed the pod after ~1 minute at step 1 of
  567. Would take ~2 hours if the OOM were solved (either raise pod memory
  limit beyond the 7 Gi cap, or run conversion on the Mac in Docker with
  more headroom). **Weight conversion on the Mac in Docker is the
  committed path** (`apps/mlc-llm/convert-weights.sh` expects HF
  safetensors at `/tmp/gemma4-hf/` — not currently present on the Mac).
- Re-compile on-device → new `lib.so` whose graph matches the
  all-single-wide weights. Takes ~15 minutes per the commit message.

Until both happen, every serve attempt will either hang (current state) or
garble output (WIP-era Python that re-enabled double-wide via
`double_wide_start_layer`).

---

## Where that leaves the ~20-token degeneration

The task asked me to characterize a ~20-token degeneration. I never reached
that measurement today because no tokens were emitted. But since the
baseline commit `8da77e9`'s message calls out **"long-sequence
degeneration"** as a known issue, and the pod's `mlc-chat-config.json`
records `sliding_window_size = 512` / sliding attention on 28 of 35 layers,
the hypothesis I would test first once the baseline serves is:

### Hypothesis: the sliding-window attention sink is evicting position 0

Gemma 4 E2B alternates sliding-window (w=512) and full-attention layers
(every 5th is full). In the stock
`tools/autoresearch/kv_cache_stock.py`, `_get_kv_chunk_len` and
`_get_seq_offset` assume an **attention sink** of a fixed size (`sink_size`)
that stays pinned at the beginning of the cache. For the MLC-LLM Gemma 4
port the sink size appears to default to 0 — no special start tokens are
kept when the window rolls. The Gemma architecture paper specifies an
attention sink of the first few BOS/turn tokens; without it, once the
sliding window rolls past the chat template's `<|turn>` preamble (~20-30
tokens from the start of the user prompt), the model loses its anchoring
and output collapses into repetition or language-switch.

Concretely: the first detectable degeneration would likely appear at the
token where `seq_len - sliding_window_size == prompt_prefix_length`, i.e.
when the *last* header token is about to be evicted. For a short user
prompt that's ~20-40 tokens in; for a long user prompt it's 500+ tokens.
This matches the "~20-token degeneration" hint in the task description
(if the user tested with minimal prompts).

Supporting evidence:
- The commit just before baseline, `7413a4e`, is literally titled
  *"norm weight fix + KV sharing disabled"*; the two subsequent attempts
  (`6e826c7`, the embed-scale fix, still garbled output) also chased
  prompt/context-window issues, all consistent with a sink/window defect
  rather than a weight/RoPE defect.
- The WIP branch's `_IS_MHA_FAMILY` helper sits in exactly the code path
  that routes sliding-window layers to the correct `mha_sliding` kernel.
  A future fix probably needs to pass an explicit sink size into
  `PagedKVCache.create_generic(...)` and/or teach the TVM kernel to never
  evict the first N tokens of a sliding-window layer.

Alternative hypotheses, ranked lower:
- **RoPE phase drift on the full-attention layers** (they use
  `partial_rotary_factor=0.25` + `freq_dim_base`; the convert-weights
  patch already handles this but it's subtle).
- **The double-wide MLP actually matters for output quality** (current
  HEAD forces it off; if the training distribution assumed double-wide,
  zeroing it is the degeneration source — but this would wreck output
  from token 1, not at token 20).
- **Small KV capacity** (512 in interactive mode per pod logs) — this
  would cause degradation *exactly at token 512*, which is ruled out by
  the "~20 token" hint.

The sink-eviction hypothesis is the one I would test first by adding a
`sink_size=4` parameter to the PagedKVCache constructor and rebuilding
`lib.so`.

---

## Reproducing (after the blocker is cleared)

### Step 0 — prerequisites

Pod must be up and have NVIDIA host libs injected (see
`~/.claude/projects/-Users-thinkstudio-gemma4/memory/reference_jetson_pod_recovery.md`).
Verify:

```bash
kubectl -n gemma4 get pod mlc-test              # Running
kubectl -n gemma4 exec mlc-test -- bash -c '
  export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm:/usr/local/cuda/lib64
  python3 -c "import tvm; print(tvm.__version__)"'
```

If the pod was just recreated, you also need to redeploy patched TVM .so
files and the gemma4 model package — see
`apps/mlc-llm/deploy-to-jetson.sh`.

### Step 1 — deploy HEAD Python patches to pod

```bash
kubectl cp tools/autoresearch/kv_cache_stock.py \
  gemma4/mlc-test:/usr/local/lib/python3.10/dist-packages/tvm/relax/frontend/nn/llm/kv_cache.py
kubectl cp apps/mlc-llm/gemma4-patches/gemma4_model.py \
  gemma4/mlc-test:/opt/mlc-llm/python/mlc_llm/model/gemma4/gemma4_model.py
kubectl cp apps/mlc-llm/gemma4-patches/gemma4_model.py \
  gemma4/mlc-test:/usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/gemma4_model.py

# Verify
kubectl -n gemma4 exec mlc-test -- grep -c '_IS_MHA_FAMILY' \
  /usr/local/lib/python3.10/dist-packages/tvm/relax/frontend/nn/llm/kv_cache.py   # → 0
kubectl -n gemma4 exec mlc-test -- grep -c 'double_wide_start_layer' \
  /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/gemma4_model.py     # → 0
kubectl -n gemma4 exec mlc-test -- grep -c 'num_kv_shared_layers = 0' \
  /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/gemma4_model.py     # → 1
```

### Step 2 — re-convert weights under HEAD (THIS CURRENTLY OOMs)

Option A — raise pod memory limit (recommended):

```yaml
# apps/mlc-llm/test-pod.yaml
          limits:
            memory: "12Gi"     # was 7Gi
```

Then `kubectl delete pod mlc-test && kubectl apply -f test-pod.yaml` and
re-inject NVIDIA libs per recovery recipe.

Option B — convert on Mac in Docker (`apps/mlc-llm/convert-weights.sh`
already exists, but needs HF safetensors staged at `/tmp/gemma4-hf/`;
you can pull them off the Jetson with
`kubectl cp gemma4/mlc-test:/models/mlc-models/gemma-4-e2b-it/ /tmp/gemma4-hf/`
first — ~10.2 GiB transfer).

Either option runs the conversion:

```bash
python3 -m mlc_llm convert_weight \
    /models/mlc-models/gemma-4-e2b-it \
    --model-type gemma4 --quantization q4f16_1 --device cuda \
    --output /models/mlc-models/gemma4-e2b-q4f16_1-baseline/
```

**Verification target**: `ndarray-cache.json` should report *all 35 layers*
of `gate_up_proj` at shape `(12288, 192)`. If you see any `(24576, 192)`,
HEAD's `num_kv_shared_layers = 0` override was not in effect during conversion
and the weights are stale.

### Step 3 — compile `lib.so` on-device

Recipe in `apps/mlc-llm/compile-model-inner.sh` (run via
`apps/mlc-llm/compile-model.sh`; ~15 minutes per the baseline commit).

### Step 4 — serve + benchmark

Copy `apps/mlc-llm/inference-test.py` into the pod, then:

```bash
kubectl cp apps/mlc-llm/inference-test.py gemma4/mlc-test:/tmp/inference-test.py
kubectl -n gemma4 exec mlc-test -- bash -c '
  export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm:/usr/local/cuda/lib64
  python3 -u /tmp/inference-test.py'
```

The script emits `[stats]` lines with tokens/s/TTFT and
`[output-marked-by-20]` blocks that let you pinpoint the exact token where
degeneration begins.

---

## File pointers

- HEAD code: `apps/mlc-llm/gemma4-patches/gemma4_model.py`
  (lines 87 `num_kv_shared_layers = 0`; 237-238 `layer_uses_double_wide_mlp`)
- TVM kv_cache stock (team baseline, with MHA_SLIDING + per-layer attn_kind
  patches, without WIP `_IS_MHA_FAMILY`): `tools/autoresearch/kv_cache_stock.py`
- Inference test script: `apps/mlc-llm/inference-test.py`
- Pod spec: `apps/mlc-llm/test-pod.yaml`
  (note: bump `resources.limits.memory` to 12Gi to un-block weight conversion)
- Pod recovery recipe: `~/.claude/projects/-Users-thinkstudio-gemma4/memory/reference_jetson_pod_recovery.md`
- WIP branch (do NOT cherry-pick from): `wip/mlc-gemma4-doublewide-2026-04-23`
  commit `2ced904` — introduces `double_wide_start_layer` which decouples
  double-wide from shared-KV layers and regresses output quality.

---

## Open questions for Agent A / the next session

1. Can the pod's memory limit safely be raised to 12Gi given Agent A's
   llama-server also runs on this node with a 7Gi limit? Node has 8 GiB.
   We probably need llama-server to downscale during a conversion pass.
2. The committed `apps/mlc-llm/convert-weights.sh` expects HF safetensors
   at `/tmp/gemma4-hf/` on the *Mac* — but the pod already has them at
   `/models/mlc-models/gemma-4-e2b-it/`. Future runs should just use the
   pod's copy via an on-device conversion (already scripted in the
   reproduction steps above).
3. Is the `self.num_kv_shared_layers = 0  # DEBUG: disable KV sharing`
   override a temporary bisect or the intended final state? If temporary,
   the degeneration may already have been mis-diagnosed and the fix is to
   revert that line and re-enable the double-wide path. The commit message
   hints at the former ("DEBUG").
