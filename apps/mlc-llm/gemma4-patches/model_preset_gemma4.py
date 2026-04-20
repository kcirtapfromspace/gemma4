"""Gemma4 E2B preset extracted from MakotoUwu's PR #3485.

Add this entry to the MODEL_PRESETS dict in model_preset.py.
"""

GEMMA4_E2B_IT_PRESET = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "text_config": {
        "model_type": "gemma4_text",
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_hidden_layers": 35,
        "vocab_size": 262144,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "rms_norm_eps": 1e-06,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 131072,
        "sliding_window": 512,
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
        ],
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        },
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "global_head_dim": 512,
        "num_kv_shared_layers": 20,
        "use_double_wide_mlp": True,
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    },
    "vocab_size": 262144,
}
