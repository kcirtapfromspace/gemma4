"""Apply all MLC-LLM + TVM patches for Gemma 4 support.
Run inside the dustynv/mlc:0.20.0-r36.4.0 container.
"""
import os
import shutil

SITE = "/usr/local/lib/python3.10/dist-packages"
TVM_LLM = f"{SITE}/tvm/relax/frontend/nn/llm"
MLC_MODEL = f"{SITE}/mlc_llm/model"
MLC_PASS = f"{SITE}/mlc_llm/compiler_pass"
MLC_NN = f"{SITE}/mlc_llm/nn"

# === 1. TVM kv_cache.py ===
path = f"{TVM_LLM}/kv_cache.py"
with open(path) as f:
    c = f.read()
c = c.replace("    MHA = 0\n    MLA = 1", "    MHA = 0\n    MLA = 1\n    MHA_SLIDING = 3")
old = "[int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]"
new = "[int(getattr(AttnKind, k.upper())) for k in attn_kind] if isinstance(attn_kind, list) else [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]"
c = c.replace(old, new)

# CRITICAL: When attn_kind is a list, all the `attn_kind == "mha"` checks in the
# FlashInfer/TIR constructors will be False, causing MLA params (=0) to be used.
# Fix: normalize attn_kind to a single string for kernel-selection decisions.
# Insert normalization right after class AttnKind definition.
attn_kind_normalize = '''

def _normalize_attn_kind(attn_kind):
    """When attn_kind is a per-layer list, extract single kind for kernel decisions."""
    if isinstance(attn_kind, list):
        # If any element is MLA, use MLA kernels; otherwise MHA.
        return "mla" if "mla" in attn_kind else "mha"
    return attn_kind

'''
# Insert after the MHA_SLIDING line
c = c.replace(
    "    MHA_SLIDING = 3\n",
    "    MHA_SLIDING = 3\n" + attn_kind_normalize
)

# Now add normalization at the start of FlashInferPagedKVCache.__init__ and TIRPagedKVCache.__init__
# The constructors both have the same pattern: they use `attn_kind` for == checks.
# We need to add: attn_kind_single = _normalize_attn_kind(attn_kind)
# Then replace all `attn_kind == "mha"` with `attn_kind_single == "mha"` etc.
# But that's too many replacements. Simpler: shadow attn_kind with normalized value
# AFTER the ShapeExpr creation (which needs the list).

# For FlashInferPagedKVCache: after the ShapeExpr line, add normalization
fi_marker = "            rx.PrimValue(enable_disaggregation),\n            rx.PrimValue(rope_mode),"
fi_insert = "        attn_kind = _normalize_attn_kind(attn_kind)  # normalize for kernel selection\n"
# Find first occurrence (FlashInfer) — insert BEFORE the constructor body starts using attn_kind for == checks
# Actually better: insert right after the ShapeExpr block in both constructors

# Simpler approach: insert normalization right at the start of __init__ body,
# but save the original list for ShapeExpr. Replace the ShapeExpr to use a local var.
#
# Cleanest fix: replace all `attn_kind == "mha"` and `attn_kind == "mla"` with
# `_normalize_attn_kind(attn_kind) == "mha"` etc.
c = c.replace('attn_kind == "mha"', '_normalize_attn_kind(attn_kind) == "mha"')
c = c.replace('attn_kind == "mla"', '_normalize_attn_kind(attn_kind) == "mla"')
with open(path, "w") as f:
    f.write(c)
print("  Patched: kv_cache.py")

# === 2. TVM position_embedding.py ===
path = f"{TVM_LLM}/position_embedding.py"
with open(path) as f:
    c = f.read()
c = c.replace(
    'def rope_freq_gptj(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):\n'
    '    """Compute the inverse frequency of RoPE for gptj RoPE scaling."""\n'
    '    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(d_range, "float32"))',
    'def rope_freq_gptj(\n'
    '    s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str,\n'
    '    freq_dim_base: int = 0,\n'
    '):\n'
    '    """Compute the inverse frequency of RoPE for gptj RoPE scaling."""\n'
    '    denom = freq_dim_base if freq_dim_base > 0 else d_range\n'
    '    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(denom, "float32"))'
)
old_sw = '    if rope_scaling["rope_type"] == "gptj":\n        return rope_freq_gptj'
new_sw = ('    if rope_scaling["rope_type"] == "gptj":\n'
           '        freq_dim_base = rope_scaling.get("freq_dim_base", 0)\n'
           '        if freq_dim_base > 0:\n'
           '            from functools import partial\n'
           '            return partial(rope_freq_gptj, freq_dim_base=freq_dim_base)\n'
           '        return rope_freq_gptj')
if old_sw in c:
    c = c.replace(old_sw, new_sw)
c = c.replace("    @T.prim_func\n    def fused_rope(", "    @T.prim_func(private=True)\n    def fused_rope(")
c = c.replace("    @T.prim_func\n    def fused_rope_longrope_scaling(", "    @T.prim_func(private=True)\n    def fused_rope_longrope_scaling(")
c = c.replace("        apply_rope: T.int32,", "        apply_rope: T.int64,")
with open(path, "w") as f:
    f.write(c)
print("  Patched: position_embedding.py")

# === 3. MLC dispatch_kv_cache_creation.py ===
path = f"{MLC_PASS}/dispatch_kv_cache_creation.py"
with open(path) as f:
    c = f.read()
c = c.replace(
    '    assert isinstance(args[0], relax.StringImm)\n    assert args[0].value in ["mha", "mla"]',
    '    assert isinstance(args[0], relax.StringImm)\n    assert args[0].value in ["mha", "mla"] or args[0].value.startswith("[")'
)
c = c.replace(
    '    return {\n        "attn_kind": args[0].value,',
    '    _raw = args[0].value\n    if _raw.startswith("["):\n        import json as _json\n        _attn_kind = _json.loads(_raw)\n    else:\n        _attn_kind = _raw\n    return {\n        "attn_kind": _attn_kind,'
)
with open(path, "w") as f:
    f.write(c)
print("  Patched: dispatch_kv_cache_creation.py")

# === 4. MLC nn/kv_cache.py ===
path = f"{MLC_NN}/kv_cache.py"
with open(path) as f:
    c = f.read()
c = c.replace(
    '    @staticmethod\n    def create_generic(  # pylint: disable=too-many-locals\n        attn_kind: Literal["mha", "mla"],',
    '    @staticmethod\n    def create_generic(  # pylint: disable=too-many-locals\n        attn_kind,  # str or List[str]'
)
c = c.replace(
    "                rx.StringImm(attn_kind),",
    '                rx.StringImm(__import__("json").dumps(attn_kind) if isinstance(attn_kind, list) else attn_kind),'
)
with open(path, "w") as f:
    f.write(c)
print("  Patched: mlc_llm/nn/kv_cache.py")

# === 5. Install gemma4 model ===
gemma4_dir = f"{MLC_MODEL}/gemma4"
os.makedirs(gemma4_dir, exist_ok=True)
shutil.copy("/patches/gemma4-patches/__init__.py", gemma4_dir)
shutil.copy("/patches/gemma4-patches/gemma4_model.py", gemma4_dir)
shutil.copy("/patches/gemma4-patches/gemma4_loader_v2.py", f"{gemma4_dir}/gemma4_loader.py")
shutil.copy("/patches/gemma4-patches/gemma4_quantization_v2.py", f"{gemma4_dir}/gemma4_quantization.py")
print("  Installed: gemma4 model files")

# === 6. Register in model.py (Gemma4LanguageModel for correct param names) ===
path = f"{MLC_MODEL}/model.py"
with open(path) as f:
    c = f.read()
if "gemma4" not in c:
    c = c.replace(
        "from .gemma3 import gemma3_loader, gemma3_model, gemma3_quantization",
        "from .gemma3 import gemma3_loader, gemma3_model, gemma3_quantization\n"
        "from .gemma4 import gemma4_loader, gemma4_model, gemma4_quantization"
    )
    entry = '''    "gemma4": Model(
        name="gemma4",
        model=gemma4_model.Gemma4LanguageModel,
        config=gemma4_model.Gemma4Config,
        source={
            "huggingface-torch": gemma4_loader.huggingface,
            "huggingface-safetensor": gemma4_loader.huggingface,
        },
        quantize={
            "no-quant": gemma4_quantization.no_quant,
            "group-quant": gemma4_quantization.group_quant,
        },
    ),
'''
    c = c.replace('    "gpt2": Model(', entry + '    "gpt2": Model(')
    with open(path, "w") as f:
        f.write(c)
    print("  Registered: gemma4 (Gemma4LanguageModel)")
else:
    # Ensure it uses Gemma4LanguageModel
    if "Gemma4ForCausalLM" in c:
        c = c.replace("Gemma4ForCausalLM", "Gemma4LanguageModel")
        with open(path, "w") as f:
            f.write(c)
        print("  Fixed: Gemma4ForCausalLM -> Gemma4LanguageModel")
    else:
        print("  gemma4 already registered correctly")

print("\nAll patches applied.")
