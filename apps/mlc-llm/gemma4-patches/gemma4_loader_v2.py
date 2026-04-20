"""HuggingFace parameter mapping for Gemma 4 text model (MLC 0.20.0 API).

Key insight: Gemma4ForCausalLM wraps the text model under language_model.*
but export_tvm() flattens params WITHOUT the language_model prefix.
The quant map however retains the prefix. The loader maps between these worlds.
"""
import functools
import numpy as np
from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization
from .gemma4_model import Gemma4Config, Gemma4ForCausalLM

def _mlc_to_hf(mlc_name):
    """Transform MLC param name to HF weight name.
    MLC: model.layers.0.X -> HF: model.language_model.layers.0.X
    """
    if mlc_name.startswith("model."):
        return "model.language_" + mlc_name
    return mlc_name

def huggingface(model_config: Gemma4Config, quantization: Quantization) -> ExternMapping:
    model = Gemma4ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    named_parameters = dict(_named_params)
    mapping = ExternMapping()
    _dtype = quantization.model_dtype if quantization else "float16"
    n_layers = model_config.text_config.num_hidden_layers

    for i in range(n_layers):
        # Gate + up projection fusion
        mlc_name = f"model.layers.{i}.mlp.gate_up_proj.weight"
        if mlc_name in named_parameters:
            hf_mlp = f"model.language_model.layers.{i}.mlp"
            mapping.add_mapping(mlc_name,
                [f"{hf_mlp}.gate_proj.weight", f"{hf_mlp}.up_proj.weight"],
                functools.partial(lambda g, u, d: np.concatenate([g, u], axis=0).astype(d),
                                  d=named_parameters[mlc_name].dtype))

        # RMS norms: Gemma adds 1 to weights
        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight",
                       "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                       "self_attn.k_norm.weight", "self_attn.q_norm.weight"]:
            mlc_name = f"model.layers.{i}.{suffix}"
            if mlc_name in named_parameters:
                mapping.add_mapping(mlc_name, [_mlc_to_hf(mlc_name)],
                    functools.partial(lambda x, d: (x + 1).astype(d),
                                      d=named_parameters[mlc_name].dtype))

        # Layer scalar: zero-pad to 2 elements
        mlc_name = f"model.layers.{i}.layer_scalar"
        if mlc_name in named_parameters:
            mapping.add_mapping(mlc_name, [_mlc_to_hf(mlc_name)],
                functools.partial(lambda w, dt=_dtype: np.concatenate(
                    [w.astype(dt), np.zeros((1,), dtype=dt)])))

    # Final norm
    mlc_name = "model.norm.weight"
    if mlc_name in named_parameters:
        mapping.add_mapping(mlc_name, [_mlc_to_hf(mlc_name)],
            functools.partial(lambda x, d: (x + 1).astype(d),
                              d=named_parameters[mlc_name].dtype))

    # Main embed_tokens: pre-multiply by sqrt(hidden_size) to fold the scale
    # into quantized weights. Gemma multiplies embed output by this factor;
    # folding it here avoids a separate multiply at runtime.
    embed_scale = model_config.text_config.hidden_size ** 0.5
    mlc_name = "model.embed_tokens.weight"
    if mlc_name in named_parameters:
        mapping.add_mapping(mlc_name, [_mlc_to_hf(mlc_name)],
            functools.partial(lambda x, sc=embed_scale, d=_dtype:
                (x.astype("float32") * sc).astype(d)))

    # Per-layer embeddings: split single HF tensor into shards
    # Map using unquantized names — pipeline handles quantization
    shard_names = [k for k in sorted(named_parameters.keys())
                   if "embed_tokens_per_layer.shards" in k]
    if shard_names:
        shard_dims = [named_parameters[k].shape[1] for k in shard_names]
        offsets = [0]
        for d in shard_dims:
            offsets.append(offsets[-1] + d)
        hf_source = "model.language_model.embed_tokens_per_layer.weight"
        per_layer_scale = model_config.text_config.hidden_size_per_layer_input ** 0.5
        for idx, mlc_name in enumerate(shard_names):
            start, end = offsets[idx], offsets[idx + 1]
            mapping.add_mapping(mlc_name, [hf_source],
                functools.partial(lambda w, s=start, e=end, sc=per_layer_scale, d=_dtype:
                    (w[:, s:e].astype("float32") * sc).astype(d)))

    # Per-layer projection norm: +1
    mlc_name = "model.per_layer_projection_norm.weight"
    if mlc_name in named_parameters:
        mapping.add_mapping(mlc_name, [_mlc_to_hf(mlc_name)],
            functools.partial(lambda x, d: (x + 1).astype(d),
                              d=named_parameters[mlc_name].dtype))

    # All remaining: direct mapping
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(mlc_name, [_mlc_to_hf(mlc_name)],
                functools.partial(lambda x, d: x.astype(d), d=mlc_param.dtype))

    return mapping
