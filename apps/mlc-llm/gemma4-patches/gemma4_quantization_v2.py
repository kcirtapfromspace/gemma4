"""Gemma 4 quantization with language_model prefix rewriting.

The quantization map uses language_model.* prefix (from the model's internal
structure) but export_tvm() strips this prefix. This module rewrites the
quant map to match the exported parameter names.
"""
from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import GroupQuantize, NoQuantize

from .gemma4_model import Gemma4Config, Gemma4ForCausalLM

_PREFIX = "language_model."


def _strip_prefix(name: str) -> str:
    return name[len(_PREFIX):] if name.startswith(_PREFIX) else name


def _rewrite_quant_map(qmap: QuantizeMapping) -> QuantizeMapping:
    """Strip language_model. prefix from quant map keys to match exported params."""
    if not qmap.param_map:
        return qmap
    new_param_map = {
        _strip_prefix(k): [_strip_prefix(v) for v in vs]
        for k, vs in qmap.param_map.items()
    }
    new_map_func = {
        _strip_prefix(k): v
        for k, v in qmap.map_func.items()
    }
    return QuantizeMapping(new_param_map, new_map_func)


def group_quant(
    model_config: Gemma4Config,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Gemma 4 model using group quantization."""
    model = Gemma4ForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(model, quant_map, "")
    quant_map = _rewrite_quant_map(quant_map)
    return model, quant_map


def no_quant(
    model_config: Gemma4Config,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Gemma 4 model without quantization."""
    model = Gemma4ForCausalLM(model_config)
    model.to(quantization.model_dtype)
    return model, QuantizeMapping({}, {})
