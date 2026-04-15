#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter with its base model and export to GGUF.

Requires: transformers, peft, torch, accelerate

Usage:
    python merge_lora_to_gguf.py \
        --lora-path /tmp/kaggle-output/outputs/checkpoint-500 \
        --base-model unsloth/gemma-4-E2B-it \
        --output-dir models/ \
        --quant q3_k_m
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA + base model → GGUF")
    parser.add_argument("--lora-path", required=True, help="Path to PEFT LoRA adapter dir")
    parser.add_argument("--base-model", default="unsloth/gemma-4-E2B-it",
                        help="HuggingFace base model ID")
    parser.add_argument("--output-dir", default="models/", help="Output directory")
    parser.add_argument("--quant", default="q3_k_m",
                        help="GGUF quantization method (q3_k_m, q4_k_m, q8_0, f16)")
    parser.add_argument("--merged-dir", default="/tmp/cliniq-merged",
                        help="Temp dir for merged model")
    args = parser.parse_args()

    print(f"Step 1: Loading base model {args.base_model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU merge to avoid GPU memory issues
    )

    print(f"Step 2: Loading LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Step 3: Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Step 4: Saving merged model to {args.merged_dir}...")
    os.makedirs(args.merged_dir, exist_ok=True)
    model.save_pretrained(args.merged_dir)
    tokenizer.save_pretrained(args.merged_dir)

    print("Step 5: Converting to GGUF...")
    convert_script = "/tmp/llama-cpp-tools/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print(f"ERROR: {convert_script} not found. Clone llama.cpp first.")
        sys.exit(1)

    gguf_f16 = os.path.join(args.output_dir, "cliniq-compact-merged-f16.gguf")
    subprocess.run([
        sys.executable, convert_script,
        args.merged_dir,
        "--outfile", gguf_f16,
        "--outtype", "f16",
    ], check=True)

    print(f"Step 6: Quantizing to {args.quant}...")
    final_gguf = os.path.join(args.output_dir, f"cliniq-compact-{args.quant.replace('_', '-')}.gguf")
    subprocess.run([
        "llama-quantize", gguf_f16, final_gguf, args.quant.upper(),
    ], check=True)

    print(f"\nDone! Final model: {final_gguf}")
    print(f"Size: {os.path.getsize(final_gguf) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
