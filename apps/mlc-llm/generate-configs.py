#!/usr/bin/env python3
"""Generate mlc-chat-config.json variants for different quantization + context combinations.

Usage:
    python generate-configs.py          # generates all variants in model-config/
    python generate-configs.py --list   # list all variants
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "model-config")

# Base text config (shared across all variants)
TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_k_eq_v": False,
    "bos_token_id": 2,
    "dtype": "bfloat16",
    "enable_moe_block": False,
    "eos_token_id": 1,
    "expert_intermediate_size": None,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 1536,
    "hidden_size_per_layer_input": 256,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 131072,
    "layer_types": (
        ["sliding_attention"] * 4 + ["full_attention"]
    ) * 7,
    "model_type": "gemma4_text",
    "num_attention_heads": 8,
    "num_hidden_layers": 35,
    "num_key_value_heads": 1,
    "num_kv_shared_layers": 20,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
    },
    "sliding_window": 512,
    "tie_word_embeddings": True,
    "use_bidirectional_attention": None,
    "use_cache": True,
    "use_double_wide_mlp": True,
    "vocab_size": 262144,
    "vocab_size_per_layer_input": 262144,
}

# Variants to generate: (quant, context_size, prefill_chunk, notes)
VARIANTS = [
    ("q4f16_1", 2048, 512, "current baseline"),
    ("q4f16_1", 1536, 512, "reduced context for memory headroom"),
    ("q4f16_0", 2048, 512, "different quantization for sequence quality"),
    ("q4f16_0", 1536, 512, "q4f16_0 + smaller context"),
    ("q3f16_1", 2048, 512, "more aggressive quant, saves ~200MB"),
    ("q3f16_1", 1536, 512, "q3f16_1 + smaller context"),
]


def make_config(quant: str, ctx_size: int, prefill_chunk: int) -> dict:
    return {
        "model_type": "gemma4",
        "quantization": quant,
        "context_window_size": ctx_size,
        "prefill_chunk_size": prefill_chunk,
        "sliding_window_size": 512,
        "tensor_parallel_shards": 1,
        "max_batch_size": 1,
        "text_config": TEXT_CONFIG,
        "vocab_size": 262144,
    }


def main():
    if "--list" in sys.argv:
        print("Available variants:")
        for quant, ctx, pfx, notes in VARIANTS:
            name = f"mlc-chat-config-{quant}-ctx{ctx}.json"
            print(f"  {name:50s} — {notes}")
        return

    os.makedirs(CONFIG_DIR, exist_ok=True)

    for quant, ctx, pfx, notes in VARIANTS:
        config = make_config(quant, ctx, pfx)
        name = f"mlc-chat-config-{quant}-ctx{ctx}.json"
        path = os.path.join(CONFIG_DIR, name)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"  Generated: {name} — {notes}")

    # Also update the default
    default = make_config("q4f16_1", 2048, 512)
    default_path = os.path.join(CONFIG_DIR, "mlc-chat-config.json")
    with open(default_path, "w") as f:
        json.dump(default, f, indent=2)
        f.write("\n")
    print(f"  Updated:   mlc-chat-config.json (default)")

    print(f"\nGenerated {len(VARIANTS) + 1} configs in {CONFIG_DIR}/")


if __name__ == "__main__":
    main()
