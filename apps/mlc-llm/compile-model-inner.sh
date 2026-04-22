#!/bin/bash
# Inner script — runs inside Docker for model compilation
set -e

echo "=== Phase 0: Install NVIDIA BSP libs (fix stub libnvrm_gpu.so) ==="
echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" > /etc/apt/sources.list.d/nvidia-l4t.list
echo "deb https://repo.download.nvidia.com/jetson/t234 r36.4 main" >> /etc/apt/sources.list.d/nvidia-l4t.list
apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc 2>/dev/null
DEBIAN_FRONTEND=noninteractive apt-get update -qq 2>/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq -o Dpkg::Options::="--force-confnew" nvidia-l4t-core nvidia-l4t-cuda 2>&1 | tail -3
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm:/usr/local/cuda/lib64
echo "  Done"

echo ""
echo "=== Phase 1: Apply patches ==="
# Apply Python patches only (skip TVM C++ build — we already have built .so files in /output)
cd /opt/mlc-llm/3rdparty/tvm

python3 -c "
path = 'python/tvm/relax/frontend/nn/llm/kv_cache.py'
with open(path) as f:
    content = f.read()
content = content.replace('    MHA = 0\n    MLA = 1', '    MHA = 0\n    MLA = 1\n    MHA_SLIDING = 3')
# FlashInfer per-layer
old_fi = '''            rx.ShapeExpr(
                [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]
            ),
            rx.PrimValue(enable_disaggregation),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype), \"kv_cache_transpose_append\"),
            bb.add_func(_kv_cache_transpose_append_mla(qk_head_dim, dtype), \"kv_cache_transpose_append_mla\"),'''
new_fi = old_fi.replace(
    '[int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]',
    '[int(getattr(AttnKind, k.upper())) for k in attn_kind]\n                if isinstance(attn_kind, list)\n                else [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]'
)
content = content.replace(old_fi, new_fi)
# TIR per-layer (has pylint comment)
old_tir = old_fi.replace('bb.add_func(_kv_cache_transpose_append', '# pylint: disable=line-too-long\n            # fmt: off\n            bb.add_func(_kv_cache_transpose_append')
if old_tir in content:
    new_tir = old_tir.replace(
        '[int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]',
        '[int(getattr(AttnKind, k.upper())) for k in attn_kind]\n                if isinstance(attn_kind, list)\n                else [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]'
    )
    content = content.replace(old_tir, new_tir)
with open(path, 'w') as f:
    f.write(content)
print('  Patched: kv_cache.py')
"

python3 -c "
path = 'python/tvm/relax/frontend/nn/llm/position_embedding.py'
with open(path) as f:
    content = f.read()
content = content.replace(
    '''def rope_freq_gptj(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
    \\\"\\\"\\\"Compute the inverse frequency of RoPE for gptj RoPE scaling.\\\"\\\"\\\"
    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(d_range, \\\"float32\\\"))''',
    '''def rope_freq_gptj(
    s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str,
    freq_dim_base: int = 0,
):
    \\\"\\\"\\\"Compute the inverse frequency of RoPE for gptj RoPE scaling.\\\"\\\"\\\"
    denom = freq_dim_base if freq_dim_base > 0 else d_range
    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(denom, \\\"float32\\\"))'''
)
# switch_rope_freq_func
old_sw = '    if rope_scaling[\"rope_type\"] == \"gptj\":\n        return rope_freq_gptj'
new_sw = '    if rope_scaling[\"rope_type\"] == \"gptj\":\n        freq_dim_base = rope_scaling.get(\"freq_dim_base\", 0)\n        if freq_dim_base > 0:\n            from functools import partial\n            return partial(rope_freq_gptj, freq_dim_base=freq_dim_base)\n        return rope_freq_gptj'
if old_sw in content:
    content = content.replace(old_sw, new_sw)
# fused_rope private
content = content.replace('    @T.prim_func\n    def fused_rope(', '    @T.prim_func(private=True)\n    def fused_rope(')
content = content.replace('    @T.prim_func\n    def fused_rope_longrope_scaling(', '    @T.prim_func(private=True)\n    def fused_rope_longrope_scaling(')
content = content.replace('        apply_rope: T.int32,', '        apply_rope: T.int64,')
with open(path, 'w') as f:
    f.write(content)
print('  Patched: position_embedding.py')
"

# Copy patched TVM Python to site-packages
cp python/tvm/relax/frontend/nn/llm/kv_cache.py /usr/local/lib/python3.10/dist-packages/tvm/relax/frontend/nn/llm/kv_cache.py
cp python/tvm/relax/frontend/nn/llm/position_embedding.py /usr/local/lib/python3.10/dist-packages/tvm/relax/frontend/nn/llm/position_embedding.py

# Patch MLC-LLM
cd /opt/mlc-llm
python3 -c "
path = 'python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py'
with open(path) as f:
    content = f.read()
content = content.replace(
    '    assert isinstance(args[0], relax.StringImm)\n    assert args[0].value in [\"mha\", \"mla\"]',
    '    assert isinstance(args[0], relax.StringImm)\n    assert args[0].value in [\"mha\", \"mla\"] or args[0].value.startswith(\"[\")'
)
content = content.replace(
    '    return {\n        \"attn_kind\": args[0].value,',
    '    _raw = args[0].value\n    if _raw.startswith(\"[\"):\n        import json as _json\n        _attn_kind = _json.loads(_raw)\n    else:\n        _attn_kind = _raw\n    return {\n        \"attn_kind\": _attn_kind,'
)
with open(path, 'w') as f:
    f.write(content)
print('  Patched: dispatch_kv_cache_creation.py')
"

python3 -c "
import json
path = 'python/mlc_llm/nn/kv_cache.py'
with open(path) as f:
    content = f.read()
content = content.replace(
    '    @staticmethod\n    def create_generic(  # pylint: disable=too-many-locals\n        attn_kind: Literal[\"mha\", \"mla\"],',
    '    @staticmethod\n    def create_generic(  # pylint: disable=too-many-locals\n        attn_kind,  # str or List[str]'
)
content = content.replace(
    '                rx.StringImm(attn_kind),',
    '                rx.StringImm(json.dumps(attn_kind) if isinstance(attn_kind, list) else attn_kind),'
)
with open(path, 'w') as f:
    f.write(content)
print('  Patched: mlc_llm/nn/kv_cache.py')
"

# Copy to site-packages
cp python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py /usr/local/lib/python3.10/dist-packages/mlc_llm/compiler_pass/
cp python/mlc_llm/nn/kv_cache.py /usr/local/lib/python3.10/dist-packages/mlc_llm/nn/

# Install gemma4 model
mkdir -p /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4
cp /patches/gemma4-patches/__init__.py /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/
cp /patches/gemma4-patches/gemma4_model.py /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/
cp /patches/gemma4-patches/gemma4_loader_v2.py /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/gemma4_loader.py
cp /patches/gemma4-patches/gemma4_quantization_v2.py /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/gemma4_quantization.py
echo "  Installed gemma4 model files"

# Register in site-packages model.py
python3 -c "
path = '/usr/local/lib/python3.10/dist-packages/mlc_llm/model/model.py'
with open(path) as f:
    content = f.read()
if 'gemma4' not in content:
    content = content.replace(
        'from .gemma3 import gemma3_loader, gemma3_model, gemma3_quantization',
        'from .gemma3 import gemma3_loader, gemma3_model, gemma3_quantization\nfrom .gemma4 import gemma4_loader, gemma4_model, gemma4_quantization'
    )
    entry = '''    \\\"gemma4\\\": Model(
        name=\\\"gemma4\\\",
        model=gemma4_model.Gemma4LanguageModel,
        config=gemma4_model.Gemma4Config,
        source={
            \\\"huggingface-torch\\\": gemma4_loader.huggingface,
            \\\"huggingface-safetensor\\\": gemma4_loader.huggingface,
        },
        quantize={
            \\\"no-quant\\\": gemma4_quantization.no_quant,
            \\\"group-quant\\\": gemma4_quantization.group_quant,
        },
    ),
'''
    content = content.replace('    \"gpt2\": Model(', entry + '    \"gpt2\": Model(')
    with open(path, 'w') as f:
        f.write(content)
    print('  Registered gemma4 (Gemma4LanguageModel) in model.py')
"

# CRITICAL FIX: Use Gemma4LanguageModel (not Gemma4ForCausalLM) for correct param names
echo "=== Applying param name fix (Gemma4LanguageModel) ==="
python3 << "EOF"
path = "/usr/local/lib/python3.10/dist-packages/mlc_llm/model/model.py"
with open(path) as f:
    content = f.read()
if "Gemma4ForCausalLM" in content:
    content = content.replace("Gemma4ForCausalLM", "Gemma4LanguageModel")
    with open(path, "w") as f:
        f.write(content)
    print("  Fixed: model.py -> Gemma4LanguageModel")
EOF

# Also fix the fused_rope private issue
python3 << "EOF"
import os
for path in [
    "/opt/mlc-llm/3rdparty/tvm/python/tvm/relax/frontend/nn/llm/position_embedding.py",
    "/usr/local/lib/python3.10/dist-packages/tvm/relax/frontend/nn/llm/position_embedding.py",
]:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        content = f.read()
    changed = False
    if "    @T.prim_func\n    def fused_rope(" in content:
        content = content.replace("    @T.prim_func\n    def fused_rope(", "    @T.prim_func(private=True)\n    def fused_rope(")
        changed = True
    if "    @T.prim_func\n    def fused_rope_longrope_scaling(" in content:
        content = content.replace("    @T.prim_func\n    def fused_rope_longrope_scaling(", "    @T.prim_func(private=True)\n    def fused_rope_longrope_scaling(")
        changed = True
    if "        apply_rope: T.int32," in content:
        content = content.replace("        apply_rope: T.int32,", "        apply_rope: T.int64,")
        changed = True
    if changed:
        with open(path, "w") as f:
            f.write(content)
        print("  Fixed: " + path + " (private rope + int64)")
EOF

echo ""
echo "=== Phase 2: Setup model config ==="
mkdir -p /tmp/model-compile
cp /patches/model-config/config.json /tmp/model-compile/
cp /patches/model-config/mlc-chat-config.json /tmp/model-compile/

echo ""
echo "=== Phase 3: Compile model ==="
# Install the patched TVM .so files we built previously
cp /output/libtvm.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm.so
cp /output/libtvm_runtime.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm_runtime.so

# Verify imports work
python3 -c "
from tvm.relax.frontend.nn.llm.kv_cache import AttnKind
assert AttnKind.MHA_SLIDING == 3
from mlc_llm.model import MODELS
assert 'gemma4' in MODELS
print('  Imports OK: AttnKind.MHA_SLIDING=3, gemma4 registered')
"

# Compile — target cuda SM 8.7 for Jetson Orin
python3 -m mlc_llm compile \
    /tmp/model-compile \
    --model-type gemma4 \
    --quantization q4f16_1 \
    --device "cuda -arch=sm_87" \
    --host "aarch64-unknown-linux-gnu" \
    --output /output/lib.so \
    --overrides "context_window_size=2048;prefill_chunk_size=512;max_batch_size=1" \
    2>&1

echo ""
echo "=== Done ==="
ls -lh /output/lib.so
