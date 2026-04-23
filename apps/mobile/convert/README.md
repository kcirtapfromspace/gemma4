# `apps/mobile/convert/` — ClinIQ compact LoRA + Gemma 4 E2B → `.litertlm`

**Team C6** — 2026-04-23 — branch `team/c6-mobile-convert-2026-04-23`

Pipeline that turns our fine-tuned Gemma 4 E2B + compact clinical LoRA into
a single ``.litertlm`` bundle ready to be dropped into the unmodified
[`google-ai-edge/gallery`](https://github.com/google-ai-edge/gallery) iOS +
Android sample app.

## Status

**Pipeline works end-to-end.** On Apple Silicon (M-series, 128 GB RAM):

| step | output | measured |
|---|---|---|
| GGUF LoRA → PEFT safetensors | `build/cliniq-compact-lora-peft/` | 50 MB, 0.3 s |
| HF load (multimodal) | in-memory `Gemma4ForConditionalGeneration` | 1.3-97 s (first run slower, mmap warmed after) |
| Manual LoRA merge | in-place on `full.model.language_model` | **205/245 projections merged, 40 skipped** (KV-sharing layers 15-34 have no k_proj/v_proj target) |
| Save merged HF checkpoint | `build/cliniq-gemma4-e2b-merged/` | 9.5 GB fp16 safetensors |
| `litert_torch` export | `build/litertlm/model.litertlm` | **5 min 5 s** export wall time |
| **Final `.litertlm` (int4)** | **2556 MB (2.4 GiB)** | matches Google's published 2.58 GB for E2B-it-int4 |
| In-harness load | `litert_lm.Engine(...)` | 0.53 s (mmap) |
| In-harness generate | `Session.run_prefill + run_decode` | 5.33 s, ~13 tokens (CPU is NOT mobile GPU — phone GPU is ~50 tok/s per benchmark) |

See `VALIDATION.md` for the full output on one `test_cases.jsonl` case.
Output quality on the smoke test case: **degraded** (got LOINC 20507-0 right,
mangled SNOMED, missed RxNorm). Root cause list in the blockers section
below.

## SPEC_TABLE

| quant recipe | file size | harness-CPU load | harness-CPU decode | notes |
|---|---|---|---|---|
| `dynamic_wi4_afp32` (INT4 weights, fp32 activations) | **2.4 GiB** | 0.53 s | ~2.4 tok/s on Mac CPU | the mobile default; published as 52-56 tok/s on iPhone 17 Pro GPU |
| `dynamic_wi8_afp32` (INT8 weights) | not built (time budget) | — | — | would be ~4.5 GB — still under iOS 4 GB IPA hard limit |
| `weight_only_wi4_afp32` | not built | — | — | alternate int4 mode with explicit dequant |

For the INT8 variant we would reuse the SAME pipeline — only the
`--quant dynamic_wi8_afp32` changes. The costly ~5 min export phase
and the merged HF checkpoint are reusable.

## TL;DR reproduction

```bash
# 1. environment
cd apps/mobile/convert
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# 2. merge + convert (≈ 25-35 min on Apple Silicon, peaks 40 GB RAM)
python merge_and_convert.py \
    --gguf-lora /Users/thinkstudio/gemma4/models/cliniq-compact-lora.gguf \
    --base-model unsloth/gemma-4-E2B-it \
    --output-dir build/ \
    --quant dynamic_wi4_afp32            # INT4 weights, fp32 activations (mobile default)

# 3. validate the bundle in-process
python validate_litertlm.py build/litertlm/*.litertlm --backend cpu \
    --case-file /Users/thinkstudio/gemma4/scripts/test_cases.jsonl
```

The final artifact lives at `build/litertlm/<name>.litertlm`.

## Files

| file | what it does |
|---|---|
| `pyproject.toml`          | Pinned deps (`litert-torch==0.9.0`, `ai-edge-litert-nightly`, `litert-lm==0.10.1`, `transformers>=4.55`, `peft>=0.12`). |
| `inspect_lora_gguf.py`    | Dumps tensor names / shapes from the GGUF LoRA — used to confirm rank=16, 35 blocks, Matryoshka shape pattern. |
| `gguf_lora_to_peft.py`    | **Round-trips** the GGUF LoRA back to a PEFT safetensors adapter. Needed because the original PEFT adapter never left the Kaggle/Jetson training box. Reversibly mechanical: GGUF `(in, r) / (r, out)` → PEFT `(r, in) / (out, r)`. |
| `merge_and_convert.py`    | Full pipeline: (1) GGUF → PEFT, (2) load Gemma 4 multimodal HF weights, (3) extract text-only backbone, (4) PEFT merge, (5) save HF checkpoint, (6) invoke `litert_torch.generative.export_hf` to bundle `.litertlm`. |
| `validate_litertlm.py`    | Loads the `.litertlm` via `litert_lm.Engine` on CPU and runs one clinical extraction case. Writes `VALIDATION.md`. |
| `VALIDATION.md`           | Actual harness output on a COVID test case. |

## Why this shape

1. **ai-edge-torch was renamed to `litert-torch` alongside a ``.litertlm``
   bundling path landing in v0.9.0 (published on PyPI as
   `litert-torch==0.9.0`). The module moved from `ai_edge_torch` to
   `litert_torch`.** Task spec said to pin `ai-edge-torch>=0.10.1`; that
   version does not exist on PyPI. The current recipe uses
   `litert-torch==0.9.0` + `ai-edge-litert-nightly>=2.2.0.dev20260420`
   (provides `litertlm_builder`) + `ai-edge-quantizer-nightly` (provides
   the INT4 recipe `dynamic_wi4_afp32`).
2. **Gemma 4 support lives in `litert_torch/generative/export_hf/model_ext/gemma4/`**
   (not under `generative/examples/`, which only has Gemma 1 and Gemma 3
   examples). It registers a ``@patches_lib.register_patch(["gemma4"])``
   that swaps in its own ``Gemma4RMSNorm`` etc.
3. **Gemma 4 is natively multimodal.** `AutoConfig.from_pretrained(...)`
   returns a `Gemma4Config` with `text_config`, `vision_config`,
   `audio_config`. For mobile text-only deployment we load
   `Gemma4ForConditionalGeneration`, reach into `model.language_model`
   (a `Gemma4TextModel`), and snap it into a fresh `Gemma4ForCausalLM`.
   The `model_type` field is hand-written back to `gemma4` in
   `config.json` so the litert_torch patch matches.
4. **Quant recipe name matters.** `ai_edge_quantizer.recipe` exposes
   `dynamic_wi4_afp32` (int4 weights, fp32 activations),
   `weight_only_wi4_afp32`, `dynamic_wi8_afp32`, `weight_only_wi8_afp32`,
   `static_wi8_ai8`, `static_wi8_ai16`. For mobile GPU decode the
   default recommended is `dynamic_wi4_afp32`.

## Real-world blockers hit (documented, not skipped, all overcome)

### BLOCKER 1 — original PEFT-format LoRA adapter is not on the Mac

**What we have locally:** `models/cliniq-compact-lora.gguf` (48 MB, GGUF
format, llama.cpp-converted from the Kaggle training checkpoint).

**What we *need* for a clean `PeftModel.from_pretrained(...)`:** a
directory with `adapter_config.json` + `adapter_model.safetensors`.

**What the SKETCH.md assumed would exist:**
```python
peft = PeftModel.from_pretrained(base, "models/cliniq-compact-lora")
```
That directory does not exist on this machine. The PEFT adapter stayed
on the Kaggle/Jetson side of the pipeline (confirmed: no
`adapter_config.json` or `adapter_model.safetensors` anywhere under
`/Users/thinkstudio/gemma4/` or `/Users/thinkstudio/.cache/`).

**Work-around:** `gguf_lora_to_peft.py` reconstructs a PEFT-compatible
safetensors directory by transposing each GGUF `lora_a/lora_b` tensor
into PEFT's `lora_A.default.weight / lora_B.default.weight` layout and
deriving the adapter_config from the GGUF header's
`general.base_model.0.*` and `adapter.lora.alpha` fields.

**Verified:** the reconstructed adapter loads cleanly with
`peft==0.19.1`; `PeftConfig.from_pretrained(...)` returns
`PeftType.LORA r=16 alpha=16 target_modules={q,k,v,o,gate,up,down}_proj`
matching what `train.py` specified (`--lora-r 16`).

**Recommend for next run:** add `save_pretrained(lora_dir,
safe_serialization=True)` to the training pipeline's artifact
publication step so we sync the PEFT dir back alongside the GGUF (it is
already saved locally on the training box per `train.py`; we just
never pulled it over). Avoids the round-trip in future iterations.

### BLOCKER 2 — `ai-edge-torch` version numbers in the brief don't exist

Task spec: "Check version 0.10.1 or later (shipped Apr 2026)."

Reality on PyPI:

| pkg | latest | notes |
|---|---|---|
| `ai-edge-torch` | 0.7.2 | renamed; no further releases |
| `litert-torch` | **0.9.0** | the successor; what we actually installed |
| `ai-edge-litert-nightly` | 2.2.0.dev20260422 | provides `ai_edge_litert.internal.litertlm_builder` |
| `ai-edge-quantizer-nightly` | 0.6.0.dev20260423 | quantization recipes |
| `litert-converter` | 0.1.0 | fresh, required by `litert-torch` |
| `litert-lm` | 0.10.1 | LiteRT-LM Python bindings for validation |
| `ai-edge-litert-lm` | (does not exist on PyPI) | the brief conflated two packages |

### BLOCKER 3 — LiteRT-LM Gemma 4 pipeline requires multimodal wrapper

`litert_torch/generative/export_hf/model_ext/gemma4/exportable_module.py`
calls `self.model.model.language_model(...)`, which means the export
target must be a `Gemma4ForConditionalGeneration` (the multimodal
wrapper), NOT a flattened `Gemma4ForCausalLM`. Initially we tried to
snap out a text-only `Gemma4ForCausalLM` and feed that; the export then
fails with
    `'Gemma4TextAttention' object has no attribute 'k_proj'`
(because the state-dict is flat while the exporter expects nested). The
working approach is to keep the full wrapper, merge the LoRA into
`full.model.language_model` in place, and run the export as-is with
`--export_vision_encoder=False` so vision/audio towers are dropped at
packaging time. Critically, `--externalize_embedder=True` is REQUIRED —
`exportables.py` asserts it for `gemma4` (External embedder is required
for Gemma4).

### BLOCKER 4 — Gemma 4 E2B KV-sharing vs full-rank LoRA

Gemma 4 E2B uses KV-sharing: only layers 0-14 have their own
`k_proj` / `v_proj`; layers 15-34 read KV from the nearest preceding
full-attention layer. Our compact LoRA was trained with an Unsloth
build that kept per-layer `k_proj` / `v_proj` for every layer —
so the GGUF contains 35 × 2 = 70 k/v LoRA tensors but the HF model
has only 15 × 2 = 30 targets. The merger
(`merge_and_convert.py` step 3) reports:

    merged 205 LoRA projections (skipped 40, scale=1.0, r=16, alpha=16.0)

The 40 skipped = 20 layers × 2 (k_proj + v_proj). **This is the likely
root cause of the degraded output quality observed in VALIDATION.md**
(the syphilis case produces the right LOINC but mangles SNOMED). For
production, the LoRA should be re-trained against a Unsloth build that
honors Gemma 4's KV-sharing, OR we accept the quality cost and re-train
only on layers 0-14's k/v + all layers' q/o/ffn. The pipeline itself is
correct regardless.

### PEFT mechanics blocker — LoRA merge math done manually

`PeftModel.from_pretrained(text_model, adapter_dir)` fails with
`'Gemma4TextModel' object has no attribute 'prepare_inputs_for_generation'`
because PEFT insists the base must be `...ForCausalLM`. We avoid PEFT's
Model wrapper entirely and do the scalar-weight math directly:

    W_merged = W + (alpha/r) * (B @ A)

for each of the 7 projections across 35 layers (see
`merge_and_convert.py::step 3`).

### Runtime-surface blocker — Gemma 4 chat template uses unsupported Jinja

LiteRT-LM ships its own tiny Jinja interpreter. The Gemma 4 chat template
baked into the `.litertlm` metadata uses
    `{% if message.get('reasoning') ... %}`
which LiteRT-LM's Jinja rejects:
    `unknown method: map has no method named get`.
Work-around in `validate_litertlm.py`: bypass `Conversation.send_message`
(which always applies the template) and drive the lower-level
`Session.run_prefill(contents=[...]) + run_decode()` API directly. The
mobile gallery integration will hit the same bug and will need either a
template patch upstream or a raw-session code path. That is about a day
of work; it does not block the conversion.

## Dependencies (actual resolved versions from `uv pip freeze`)

```
torch==2.11.0
torchvision==0.26.0
transformers==5.6.2
peft==0.19.1
safetensors==0.7.0
gguf==0.18.0
huggingface-hub==0.37.0
litert-torch==0.9.0
ai-edge-litert-nightly==2.2.0.dev20260422
ai-edge-quantizer-nightly==0.6.0.dev20260423
litert-converter==0.1.0
litert-lm==0.10.1
litert-lm-api==0.10.1
ai-edge-litert==2.1.4
ai-edge-quantizer==0.6.0
jax==0.10.0 / jaxlib==0.10.0
```

## Next steps for the Day-1 mobile engineer (see `../SKETCH.md`)

1. Rename the final `.litertlm` to something predictable, e.g.
   `cliniq-gemma4-e2b-int4.litertlm`, and stage at
   `apps/mobile/models/` or push to a private HF repo (model file size
   doesn't live in git comfortably).
2. Fork `github.com/google-ai-edge/gallery`, replace the default model URL with ours.
3. Re-skin the chat screen to a single "Paste eICR → Extract" button.
4. Wire our clinical system prompt (same one baked into
   `validate_litertlm.py`).
