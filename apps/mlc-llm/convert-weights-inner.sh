#!/bin/bash
# Inner script — runs inside Docker container for weight conversion
# Mounts:
#   /patches     -> repo dir (read-only)
#   /output      -> build-output dir (read-write)
#   /hf-model    -> HF safetensors (read-only)
set -e

echo "=== Phase 0: Create CUDA stubs (no real GPU needed for weight conversion) ==="
# Extract undefined CUDA symbols from the pre-built TVM and create stub .so
NEEDED=$(nm -D /output/libtvm.so 2>/dev/null | grep " U " | grep "^.*cu[A-Z]" | awk "{print \$2}" | sort -u)
echo "// cuda stub for weight conversion" > /tmp/s.c
echo "$NEEDED" | while read sym; do [ -n "$sym" ] && echo "int $sym() { return 0; }" >> /tmp/s.c; done
mkdir -p /usr/lib/aarch64-linux-gnu/nvidia
gcc -shared -o /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 /tmp/s.c
ln -sf libcuda.so.1 /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so
# Replace any broken/tiny stub libs with a valid empty .so
gcc -shared -o /tmp/libstub.so -x c /dev/null
for lib in /usr/lib/aarch64-linux-gnu/nvidia/lib*.so*; do
    [[ "$lib" == *libcuda* ]] && continue
    size=$(stat -c%s "$lib" 2>/dev/null || echo "999999")
    [ "$size" -lt 1000 ] && [ -f "$lib" ] && cp /tmp/libstub.so "$lib"
done
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm:/usr/local/cuda/lib64
echo "  Created CUDA stubs ($(echo "$NEEDED" | wc -l | tr -d ' ') symbols)"

echo ""
echo "=== Phase 1: Install patched TVM .so ==="
cp /output/libtvm.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm.so
cp /output/libtvm_runtime.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm_runtime.so
echo "  Done"

echo ""
echo "=== Phase 2: Apply all Python patches ==="

# --- TVM kv_cache.py ---
cd /opt/mlc-llm/3rdparty/tvm
python3 -c "
path = 'python/tvm/relax/frontend/nn/llm/kv_cache.py'
with open(path) as f:
    content = f.read()
content = content.replace('    MHA = 0\n    MLA = 1', '    MHA = 0\n    MLA = 1\n    MHA_SLIDING = 3')
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

# --- TVM position_embedding.py ---
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
old_sw = '    if rope_scaling[\"rope_type\"] == \"gptj\":\n        return rope_freq_gptj'
new_sw = '    if rope_scaling[\"rope_type\"] == \"gptj\":\n        freq_dim_base = rope_scaling.get(\"freq_dim_base\", 0)\n        if freq_dim_base > 0:\n            from functools import partial\n            return partial(rope_freq_gptj, freq_dim_base=freq_dim_base)\n        return rope_freq_gptj'
if old_sw in content:
    content = content.replace(old_sw, new_sw)
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

# --- MLC-LLM patches ---
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

# --- Install gemma4 model ---
mkdir -p /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4
cp /patches/gemma4-patches/__init__.py /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/
cp /patches/gemma4-patches/gemma4_model.py /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/
cp /patches/gemma4-patches/gemma4_loader_v2.py /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/gemma4_loader.py
cp /patches/gemma4-patches/gemma4_quantization_v2.py /usr/local/lib/python3.10/dist-packages/mlc_llm/model/gemma4/gemma4_quantization.py
echo "  Installed gemma4 model files"

# --- Register gemma4 in model.py ---
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
    print('  Registered gemma4 in model.py')
elif 'Gemma4ForCausalLM' in content:
    content = content.replace('Gemma4ForCausalLM', 'Gemma4LanguageModel')
    with open(path, 'w') as f:
        f.write(content)
    print('  Fixed: Gemma4ForCausalLM -> Gemma4LanguageModel')
else:
    print('  gemma4 already registered correctly')
"

echo ""
echo "=== Phase 3: Verify ==="
python3 -c "
from mlc_llm.model import MODELS
assert 'gemma4' in MODELS, 'gemma4 not in MODELS'
print('  gemma4 model registered OK')

# Quick test: load config and instantiate model
from mlc_llm.model.gemma4.gemma4_model import Gemma4Config
import json
with open('/hf-model/config.json') as f:
    cfg = json.load(f)
config = Gemma4Config.from_dict(cfg)
print(f'  Config loaded: {config.text_config.num_hidden_layers} layers, hidden_size={config.text_config.hidden_size}')
"

echo ""
echo "=== Phase 4: Convert weights ==="
echo "Input: /hf-model/ ($(ls -lh /hf-model/model.safetensors | awk '{print $5}'))"
echo "Output: /output/gemma4-weights-v3/"
echo ""

python3 -m mlc_llm convert_weight \
    /hf-model \
    --model-type gemma4 \
    --quantization q4f16_1 \
    --output /output/gemma4-weights-v3/ \
    2>&1

echo ""
echo "=== Phase 5: Results ==="
echo "Output files:"
ls -lh /output/gemma4-weights-v3/
echo ""
echo "Total size:"
du -sh /output/gemma4-weights-v3/

echo ""
echo "=== Phase 6: Param inventory ==="
python3 << 'PYEOF'
import json
import os

output_dir = "/output/gemma4-weights-v3"

# Read ndarray-cache.json to get param names and shapes
cache_path = os.path.join(output_dir, "ndarray-cache.json")
if os.path.exists(cache_path):
    with open(cache_path) as f:
        cache = json.load(f)

    records = cache.get("records", [])
    print(f"Total converted params: {len(records)}")
    print()

    # Categorize params
    categories = {}
    for rec in records:
        name = rec[0]  # param name
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            cat = ".".join(parts[:idx+1]) + ".X." + ".".join(parts[idx+2:])
        else:
            cat = name
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, rec[1]))

    print("Param patterns (first instance):")
    for cat in sorted(categories.keys()):
        instances = categories[cat]
        name, info = instances[0]
        shape = info[0] if isinstance(info, list) else info.get("shape", "?")
        dtype = info[1] if isinstance(info, list) else info.get("dtype", "?")
        print(f"  {name}  shape={shape} dtype={dtype}  (x{len(instances)})")
else:
    print("WARNING: ndarray-cache.json not found!")
    # Try listing .bin files
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")
PYEOF

echo ""
echo "=== Weight conversion complete ==="
