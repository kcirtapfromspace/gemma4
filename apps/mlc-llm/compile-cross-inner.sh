#!/bin/bash
# Cross-compile model lib inside Docker — targets Jetson sm_87
# Runs in dustynv/mlc:0.20.0-r36.4.0 on Mac Studio
set -e

QUANT="${1:-q4f16_1}"
MODEL_DIR="/output/gemma4-weights-merged-${QUANT}"

echo "=== Phase 0: CUDA stubs ==="
NEEDED=$(nm -D /output/libtvm.so 2>/dev/null | grep " U " | grep "^.*cu[A-Z]" | awk "{print \$2}" | sort -u)
echo "// cuda stub" > /tmp/s.c
echo "$NEEDED" | while read sym; do [ -n "$sym" ] && echo "int $sym() { return 0; }" >> /tmp/s.c; done
mkdir -p /usr/lib/aarch64-linux-gnu/nvidia
gcc -shared -o /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 /tmp/s.c
ln -sf libcuda.so.1 /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so
gcc -shared -o /tmp/libstub.so -x c /dev/null
for lib in /usr/lib/aarch64-linux-gnu/nvidia/lib*.so*; do
    [[ "$lib" == *libcuda* ]] && continue
    size=$(stat -c%s "$lib" 2>/dev/null || echo "999999")
    [ "$size" -lt 1000 ] && [ -f "$lib" ] && cp /tmp/libstub.so "$lib"
done
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm:/usr/local/cuda/lib64

echo "=== Phase 1: Install patched TVM ==="
cp /output/libtvm.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm.so
cp /output/libtvm_runtime.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm_runtime.so

echo "=== Phase 2: Apply all patches ==="
SITE=/usr/local/lib/python3.10/dist-packages
MLC=/opt/mlc-llm

# TVM kv_cache.py patch
cd $MLC/3rdparty/tvm
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

# TVM position_embedding.py patch
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

cp python/tvm/relax/frontend/nn/llm/kv_cache.py $SITE/tvm/relax/frontend/nn/llm/kv_cache.py
cp python/tvm/relax/frontend/nn/llm/position_embedding.py $SITE/tvm/relax/frontend/nn/llm/position_embedding.py

# MLC-LLM patches
cd $MLC
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
# FlashInfer normalization
old = '                cache = kv_cache.FlashInferPagedKVCache(target=self.target, **kwargs)'
new = '                fi_kwargs = dict(kwargs)\n                ak = fi_kwargs[\"attn_kind\"]\n                if isinstance(ak, list):\n                    fi_kwargs[\"attn_kind\"] = \"mla\" if \"mla\" in set(ak) else \"mha\"\n                cache = kv_cache.FlashInferPagedKVCache(target=self.target, **fi_kwargs)'
if old in content:
    content = content.replace(old, new)
# TIR normalization
old2 = '            cache = kv_cache.TIRPagedKVCache(target=self.target, **kwargs)'
new2 = '            tir_k = dict(kwargs)\n            ak = tir_k[\"attn_kind\"]\n            if isinstance(ak, list): tir_k[\"attn_kind\"] = \"mla\" if \"mla\" in set(ak) else \"mha\"\n            cache = kv_cache.TIRPagedKVCache(target=self.target, **tir_k)'
if old2 in content:
    content = content.replace(old2, new2)
with open(path, 'w') as f:
    f.write(content)
print('  Patched: dispatch_kv_cache_creation.py (JSON + FlashInfer + TIR)')
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

# Copy patches to site-packages
cp python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py $SITE/mlc_llm/compiler_pass/
cp python/mlc_llm/nn/kv_cache.py $SITE/mlc_llm/nn/

# Install gemma4 model (to BOTH paths)
for BASE in $SITE/mlc_llm/model/gemma4 $MLC/python/mlc_llm/model/gemma4; do
    mkdir -p $BASE
    cp /patches/gemma4-patches/__init__.py $BASE/
    cp /patches/gemma4-patches/gemma4_model.py $BASE/gemma4_model.py
    cp /patches/gemma4-patches/gemma4_loader_v2.py $BASE/gemma4_loader.py
    cp /patches/gemma4-patches/gemma4_quantization_v2.py $BASE/gemma4_quantization.py
done

# Register gemma4
python3 -c "
path = '$SITE/mlc_llm/model/model.py'
with open(path) as f:
    content = f.read()
if 'gemma4' not in content:
    content = content.replace(
        'from .gemma3 import gemma3_loader, gemma3_model, gemma3_quantization',
        'from .gemma3 import gemma3_loader, gemma3_model, gemma3_quantization\nfrom .gemma4 import gemma4_loader, gemma4_model, gemma4_quantization'
    )
    entry = '    \"gemma4\": Model(\n        name=\"gemma4\",\n        model=gemma4_model.Gemma4LanguageModel,\n        config=gemma4_model.Gemma4Config,\n        source={\n            \"huggingface-torch\": gemma4_loader.huggingface,\n            \"huggingface-safetensor\": gemma4_loader.huggingface,\n        },\n        quantize={\n            \"no-quant\": gemma4_quantization.no_quant,\n            \"group-quant\": gemma4_quantization.group_quant,\n        },\n    ),\n'
    content = content.replace('    \"gpt2\": Model(', entry + '    \"gpt2\": Model(')
    with open(path, 'w') as f:
        f.write(content)
    print('  Registered gemma4')
else:
    print('  gemma4 already registered')
"

echo ""
echo "=== Phase 3: Verify ==="
python3 -c "
from mlc_llm.model import MODELS
assert 'gemma4' in MODELS
from mlc_llm.model.gemma4.gemma4_model import Gemma4Config
import json
with open('$MODEL_DIR/mlc-chat-config.json') as f:
    cfg = json.load(f)
mc = cfg.get('model_config', cfg)
config = Gemma4Config.from_dict(mc)
tc = config.text_config
print(f'  double_wide_start_layer={tc.double_wide_start_layer}')
print(f'  layer 15 double_wide={tc.layer_uses_double_wide_mlp(15)}')
"

echo ""
echo "=== Phase 4: Compile for sm_87 (cross-compile, no GPU needed) ==="
python3 -c "
import json
from pathlib import Path
import tvm
from tvm import relax
from mlc_llm.interface.compile import compile as mlc_compile, ModelConfigOverride, OptimizationFlags
from mlc_llm.model import MODELS
from mlc_llm.quantization import QUANTIZATION

model_dir = '$MODEL_DIR'
output = '/output/gemma4-merged-cuda.so'
quant = '$QUANT'

with open(model_dir + '/mlc-chat-config.json') as f:
    config = json.load(f)

target = tvm.target.Target(
    {'kind': 'cuda', 'arch': 'sm_87', 'max_threads_per_block': 1024,
     'max_num_threads': 1024, 'max_shared_memory_per_block': 49152,
     'thread_warp_size': 32, 'libs': ['thrust']},
    host={'kind': 'llvm', 'mtriple': 'aarch64-unknown-linux-gnu', 'mcpu': 'generic'},
)

def build_func(mod, args, pipeline=None):
    relax.build(mod, target=args.target, relax_pipeline=pipeline,
                system_lib=False).export_library(str(args.output))

# Debug: trace the config through the same path as compile()
import copy
cfg_copy = copy.deepcopy(config)
if 'model_config' in cfg_copy:
    mc = cfg_copy.pop('model_config')
    mc.update(cfg_copy)
    debug_config = MODELS['gemma4'].config.from_dict(mc)
else:
    debug_config = MODELS['gemma4'].config.from_dict(cfg_copy)
tc = debug_config.text_config
print('DEBUG compile config:')
print('  double_wide_start_layer:', tc.double_wide_start_layer)
print('  num_kv_shared_layers:', tc.num_kv_shared_layers)
print('  use_double_wide_mlp:', tc.use_double_wide_mlp)
print('  layer 15 double_wide:', tc.layer_uses_double_wide_mlp(15))

# Monkey-patch _compile to inspect model shapes
import mlc_llm.interface.compile as _comp_mod
_orig_compile = _comp_mod._compile
def _debug_compile(args, model_config):
    tc = model_config.text_config
    print('INSIDE _compile:')
    print('  double_wide_start_layer:', getattr(tc, 'double_wide_start_layer', 'MISSING'))
    print('  num_kv_shared_layers:', tc.num_kv_shared_layers)
    print('  use_double_wide_mlp:', tc.use_double_wide_mlp)
    # Check model param shapes
    model = args.model.model(model_config)
    _, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    params = dict(params)
    for key in ['model.layers.0.mlp.gate_up_proj.weight', 'model.layers.15.mlp.gate_up_proj.weight']:
        if key in params: print('  %s: %s' % (key, params[key].shape))
    # Now run original
    return _orig_compile(args, model_config)
_comp_mod._compile = _debug_compile

mlc_compile(
    config=config,
    quantization=QUANTIZATION[quant],
    model_type=MODELS['gemma4'],
    target=target,
    opt=OptimizationFlags.from_str('O2'),
    build_func=build_func,
    system_lib_prefix='',
    output=Path(output),
    overrides=ModelConfigOverride.from_str('context_window_size=1536;prefill_chunk_size=512'),
)
print('Generated:', output)
"

echo ""
echo "=== Result ==="
ls -lh /output/gemma4-merged-cuda.so
echo "=== Cross-compilation complete ==="
