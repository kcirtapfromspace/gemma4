#!/usr/bin/env python3
"""End-to-end pipeline: ClinIQ compact LoRA (GGUF) + Gemma 4 E2B → `.litertlm`.

Pipeline:
    1. Reverse-engineer the GGUF LoRA into a PEFT safetensors adapter
       (see gguf_lora_to_peft.py; this is *only* needed because the
       original PEFT adapter never left the Jetson/Kaggle training box).
    2. Load Gemma 4 E2B text-only backbone from the local HF cache.
    3. ``PeftModel.from_pretrained(base, peft_dir).merge_and_unload()``.
    4. Save merged model as a text-only HF checkpoint.
    5. Invoke ``litert_torch.generative.export_hf.export_main`` to bundle
       a ``.litertlm`` file with int4 quantization and the Gemma 4 patch.

Run:
    python merge_and_convert.py \\
        --gguf-lora /Users/thinkstudio/gemma4/models/cliniq-compact-lora.gguf \\
        --base-model unsloth/gemma-4-E2B-it \\
        --output-dir build/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def step(name: str):
    bar = "=" * 8
    print(f"\n{bar} {name} {bar}\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gguf-lora",
        default="/Users/thinkstudio/gemma4/models/cliniq-compact-lora.gguf",
    )
    ap.add_argument("--base-model", default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--output-dir", default="build/")
    ap.add_argument(
        "--quant",
        default="dynamic_wi4_afp32",
        help=(
            "Quantization recipe name from ai_edge_quantizer.recipe. "
            "INT4 weights with fp32 activations is the mobile default. "
            "Alternatives: dynamic_wi8_afp32, weight_only_wi4_afp32, "
            "weight_only_wi8_afp32, static_wi8_ai16, static_wi8_ai8."
        ),
    )
    ap.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip LoRA merge, convert vanilla base model only (sanity check)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. GGUF LoRA -> PEFT safetensors ----
    step("1 · GGUF LoRA -> PEFT safetensors")
    peft_dir = out / "cliniq-compact-lora-peft"
    if not peft_dir.exists():
        from gguf_lora_to_peft import convert as gguf_to_peft

        gguf_to_peft(
            Path(args.gguf_lora), peft_dir, base_model_override=args.base_model
        )
    else:
        print(f"[skip] {peft_dir} already exists")
    if args.dry_run:
        return

    merged_dir = out / "cliniq-gemma4-e2b-merged"
    # ---- 2. load base + 3. merge LoRA ----
    if args.skip_merge or (merged_dir / "model.safetensors").exists():
        print(f"[skip] merged dir already populated at {merged_dir}")
    else:
        step("2 · load Gemma 4 multimodal backbone")
        import torch
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

        t0 = time.time()
        full = Gemma4ForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print(f"loaded full multimodal in {time.time()-t0:.1f}s")

        step("3 · merge LoRA into language_model (manual math)")
        # PEFT requires a `...ForCausalLM` root to wrap, but the gemma4 LiteRT
        # exporter needs the full `Gemma4ForConditionalGeneration` wrapper.
        # Rather than surgery into PEFT internals, we do the scalar-weight
        # math directly:  W_merged = W + (alpha/r) * B @ A
        # for each target projection in each decoder layer. That is exactly
        # what `PeftModel.merge_and_unload()` does under the hood.
        import json as _json
        from safetensors.torch import load_file

        t0 = time.time()
        adapter_cfg = _json.loads((peft_dir / "adapter_config.json").read_text())
        alpha = float(adapter_cfg["lora_alpha"])
        r = int(adapter_cfg["r"])
        scale = alpha / r
        # PEFT/HF attr suffixes inside each decoder layer
        PROJ_SUFFIX = {
            "q_proj": ("self_attn", "q_proj"),
            "k_proj": ("self_attn", "k_proj"),
            "v_proj": ("self_attn", "v_proj"),
            "o_proj": ("self_attn", "o_proj"),
            "gate_proj": ("mlp", "gate_proj"),
            "up_proj": ("mlp", "up_proj"),
            "down_proj": ("mlp", "down_proj"),
        }
        lora_sd = load_file(str(peft_dir / "adapter_model.safetensors"))
        # Keys look like:
        #   base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight
        #   base_model.model.model.layers.N.self_attn.q_proj.lora_B.default.weight
        import re as _re

        pat = _re.compile(
            r"^base_model\.model\.model\.layers\.(\d+)\.(\w+)\.(\w+)\.lora_([AB])\.default\.weight$"
        )
        pairs: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        for k, v in lora_sd.items():
            m = pat.match(k)
            if not m:
                continue
            layer_idx = int(m.group(1))
            parent, proj = m.group(2), m.group(3)
            ab = m.group(4)
            pairs.setdefault((layer_idx, f"{parent}.{proj}"), {})[ab] = v

        # Walk decoder layers and do the merge.
        # NOTE: Gemma 4 E2B KV-sharing kicks in at layer 15 — layers 15-34
        # have NO k_proj / v_proj (they read KV from a preceding full-attn
        # layer). Our GGUF LoRA was trained on an Unsloth build that kept
        # per-layer k_proj / v_proj for every layer. The deltas for the
        # shared-KV layers are discarded here with a count — they have no
        # architectural target to land on.
        layers = full.model.language_model.layers
        merged_count = 0
        skipped_count = 0
        for (layer_idx, path), ab_tensors in pairs.items():
            parent_name, proj_name = path.split(".")
            parent_mod = getattr(layers[layer_idx], parent_name)
            if not hasattr(parent_mod, proj_name):
                skipped_count += 1
                continue
            A = ab_tensors["A"].to(torch.float32)  # (r, in)
            B = ab_tensors["B"].to(torch.float32)  # (out, r)
            delta = scale * (B @ A)  # (out, in)
            proj_mod = getattr(parent_mod, proj_name)
            # Some gemma4 linear projs may have mismatched output dims on
            # KV-sharing / Matryoshka boundaries — skip those too.
            if tuple(proj_mod.weight.shape) != tuple(delta.shape):
                print(
                    f"[skip] layer {layer_idx} {path}: base shape "
                    f"{tuple(proj_mod.weight.shape)} != delta shape "
                    f"{tuple(delta.shape)}"
                )
                skipped_count += 1
                continue
            with torch.no_grad():
                proj_mod.weight.add_(delta.to(proj_mod.weight.dtype))
            merged_count += 1
        print(
            f"merged {merged_count} LoRA projections "
            f"(skipped {skipped_count}, scale={scale}, r={r}, alpha={alpha}) "
            f"in {time.time()-t0:.1f}s"
        )

        step("4 · save merged multimodal HF checkpoint")
        merged_dir.mkdir(parents=True, exist_ok=True)
        full.save_pretrained(merged_dir, safe_serialization=True)
        tok = AutoTokenizer.from_pretrained(args.base_model)
        tok.save_pretrained(merged_dir)
        del full
        import gc

        gc.collect()
        # config.json keeps model_type='gemma4' (multimodal wrapper) by default
        print(f"merged HF checkpoint -> {merged_dir}")

    # ---- 5. invoke litert_torch exporter ----
    step("5 · litert_torch export -> .litertlm")
    litertlm_out = out / "litertlm"
    litertlm_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "litert_torch.generative.export_hf",
        f"--model={merged_dir}",
        f"--output_dir={litertlm_out}",
        "--task=text_generation",
        "--bundle_litert_lm=True",
        f"--quantization_recipe={args.quant}",
        # Gemma 4 requires externalized embedder (asserted in the model_ext
        # exportables dispatch): the per-layer embedder and main token
        # embedder run as separate TFLite subgraphs so the mobile runtime can
        # stream tokens through them without re-entering the transformer.
        "--externalize_embedder=True",
        # Also skip re-exporting the vision encoder — we are text-only for
        # now. Can be flipped to True later once we add clinical OCR.
        "--export_vision_encoder=False",
        "--keep_temporary_files=False",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    final = next(litertlm_out.rglob("*.litertlm"), None)
    if final:
        size_mb = final.stat().st_size / 1e6
        print(f"\nDONE — {final} ({size_mb:.1f} MB)")
    else:
        print("\nWARN — no .litertlm file found in output_dir; inspect logs above.")


if __name__ == "__main__":
    main()
