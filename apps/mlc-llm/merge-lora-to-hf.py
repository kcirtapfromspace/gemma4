#!/usr/bin/env python3
"""Merge a GGUF LoRA adapter into HuggingFace safetensors base model.

Produces a merged HF model directory suitable for MLC-LLM convert_weight.

Usage:
    python merge-lora-to-hf.py \
        --lora models/cliniq-compact-lora.gguf \
        --base /tmp/gemma4-hf \
        --output /tmp/gemma4-hf-merged
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import torch
from gguf import GGUFReader
from safetensors.torch import load_file, save_file


# GGUF tensor name component -> HuggingFace weight path component
GGUF_TO_HF_MODULE = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

# GGUF tensor type -> numpy dtype
GGUF_TYPE_TO_DTYPE = {
    0: np.float32,   # GGML_TYPE_F32
    1: np.float16,   # GGML_TYPE_F16
}


def read_gguf_lora(lora_path: str):
    """Read GGUF LoRA adapter, return (alpha, rank, lora_pairs dict)."""
    reader = GGUFReader(lora_path)

    # Extract alpha
    alpha = 16.0  # default
    if "adapter.lora.alpha" in reader.fields:
        alpha = float(reader.fields["adapter.lora.alpha"].parts[-1][0])

    # Parse tensors into pairs: {base_name: (lora_a_data, lora_b_data)}
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t

    lora_pairs = {}
    rank = None
    pattern = re.compile(r"blk\.(\d+)\.(\w+)\.weight\.lora_([ab])")

    for name, tensor in tensors.items():
        m = pattern.match(name)
        if not m:
            continue

        layer_idx = int(m.group(1))
        gguf_module = m.group(2)
        ab = m.group(3)

        if gguf_module not in GGUF_TO_HF_MODULE:
            print(f"  WARNING: Unknown GGUF module {gguf_module}, skipping")
            continue

        hf_module = GGUF_TO_HF_MODULE[gguf_module]
        hf_key = f"model.language_model.layers.{layer_idx}.{hf_module}.weight"

        if hf_key not in lora_pairs:
            lora_pairs[hf_key] = {}

        # Read tensor data as float32 numpy array
        dtype = GGUF_TYPE_TO_DTYPE.get(tensor.tensor_type, np.float32)
        data = tensor.data.copy()
        if dtype != np.float32:
            data = data.astype(np.float32)
        data = data.reshape(tensor.shape[::-1])  # GGUF stores reversed dims

        lora_pairs[hf_key][ab] = data

        if ab == "a" and rank is None:
            rank = data.shape[0]  # rank is the first dim of lora_a after reshape

    # Verify all pairs are complete
    for key, pair in list(lora_pairs.items()):
        if "a" not in pair or "b" not in pair:
            print(f"  WARNING: Incomplete LoRA pair for {key}, skipping")
            del lora_pairs[key]

    return alpha, rank, lora_pairs


def main():
    parser = argparse.ArgumentParser(description="Merge GGUF LoRA into HF safetensors")
    parser.add_argument("--lora", required=True, help="Path to GGUF LoRA adapter")
    parser.add_argument("--base", required=True, help="Path to HF base model directory")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--dry-run", action="store_true", help="Verify shapes without merging")
    args = parser.parse_args()

    base_safetensors = os.path.join(args.base, "model.safetensors")
    if not os.path.exists(base_safetensors):
        print(f"ERROR: {base_safetensors} not found")
        sys.exit(1)

    print(f"=== Merge GGUF LoRA into HuggingFace model ===")
    print(f"  LoRA:   {args.lora}")
    print(f"  Base:   {args.base}")
    print(f"  Output: {args.output}")
    print()

    # Step 1: Read LoRA
    print("Step 1: Reading GGUF LoRA adapter...")
    alpha, rank, lora_pairs = read_gguf_lora(args.lora)
    scaling = alpha / rank
    print(f"  alpha={alpha}, rank={rank}, scaling={scaling}")
    print(f"  {len(lora_pairs)} weight matrices to merge")
    print()

    # Step 2: Load base weights
    print("Step 2: Loading base model weights...")
    base_weights = load_file(base_safetensors)
    print(f"  {len(base_weights)} tensors loaded")
    print()

    # Step 3: Verify shapes
    print("Step 3: Verifying shape compatibility...")
    mismatches = 0
    for hf_key, pair in sorted(lora_pairs.items()):
        if hf_key not in base_weights:
            print(f"  MISSING: {hf_key} not in base model!")
            mismatches += 1
            continue

        base_shape = tuple(base_weights[hf_key].shape)
        a_shape = pair["a"].shape  # [rank, in_features] after reshape
        b_shape = pair["b"].shape  # [out_features, rank] after reshape
        delta_shape = (b_shape[0], a_shape[1])

        if delta_shape != base_shape:
            print(f"  MISMATCH: {hf_key}: delta {delta_shape} != base {base_shape}")
            mismatches += 1
        else:
            # Only print first and last for brevity
            layer_match = re.search(r"layers\.(\d+)\.", hf_key)
            layer = int(layer_match.group(1)) if layer_match else -1
            if layer <= 0 or layer == 34:
                print(f"  OK: {hf_key} {base_shape}")

    if mismatches > 0:
        print(f"\n  {mismatches} shape mismatches! Aborting.")
        sys.exit(1)
    print(f"  All {len(lora_pairs)} shapes verified OK")
    print()

    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    # Step 4: Apply LoRA deltas
    print("Step 4: Merging LoRA weights (W' = W + scaling * B @ A)...")
    merged_weights = dict(base_weights)  # shallow copy of tensor dict

    for i, (hf_key, pair) in enumerate(sorted(lora_pairs.items())):
        lora_a = torch.from_numpy(pair["a"])  # [rank, in_features]
        lora_b = torch.from_numpy(pair["b"])  # [out_features, rank]
        base_w = merged_weights[hf_key].float()

        delta = scaling * (lora_b @ lora_a)  # [out_features, in_features]
        merged_weights[hf_key] = (base_w + delta).to(torch.bfloat16)

        if (i + 1) % 35 == 0 or i == 0:
            print(f"  Merged {i + 1}/{len(lora_pairs)} weights")

    print(f"  All {len(lora_pairs)} weights merged")
    print()

    # Step 5: Save merged model
    print(f"Step 5: Saving merged model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)

    # Save merged safetensors
    output_path = os.path.join(args.output, "model.safetensors")
    save_file(merged_weights, output_path)
    size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"  Saved: {output_path} ({size_gb:.2f} GB)")

    # Copy config files (config.json, tokenizer, etc.)
    for fname in os.listdir(args.base):
        if fname == "model.safetensors":
            continue
        src = os.path.join(args.base, fname)
        dst = os.path.join(args.output, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")

    print()
    print("=== Merge complete! ===")
    print(f"Merged model: {args.output}")
    print()
    print("Next: Run MLC-LLM weight conversion:")
    print(f"  python -m mlc_llm convert_weight {args.output} \\")
    print(f"    --model-type gemma4 --quantization q4f16_1 \\")
    print(f"    --output <output-dir>")


if __name__ == "__main__":
    main()
