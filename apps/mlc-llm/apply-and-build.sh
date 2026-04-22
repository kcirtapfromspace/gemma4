#!/bin/bash
# Internal script — runs INSIDE the Docker container
# Called by build-gemma4.sh
set -e

echo "=== Phase 1: Patch C++ (TVM) ==="
cd /opt/mlc-llm/3rdparty/tvm

python3 << "EOF"
# attn_utils.h — add kMHASliding enum + GetKVCacheShape
path = "src/runtime/relax_vm/attn_utils.h"
with open(path) as f:
    content = f.read()

content = content.replace(
    "enum class AttnKind : int {\n  kMHA = 0,\n  kMLA = 1,\n  kLinearAttn = 2,\n};",
    "enum class AttnKind : int {\n  kMHA = 0,\n  kMLA = 1,\n  kLinearAttn = 2,\n  kMHASliding = 3,\n};"
)
content = content.replace(
    "  if (attn_kind == AttnKind::kMHA) {\n    // Ignore v_head_dim",
    "  if (attn_kind == AttnKind::kMHA || attn_kind == AttnKind::kMHASliding) {\n    // Ignore v_head_dim"
)
with open(path, "w") as f:
    f.write(content)
print("  Patched: attn_utils.h")
EOF

python3 << "EOF"
# paged_kv_cache.cc — 4 fixes
path = "src/runtime/relax_vm/paged_kv_cache.cc"
with open(path) as f:
    content = f.read()

# Fix 1: Hoist ReserveAppendLengthInSeq unconditionally
old1 = "    if (append_before_attn_) {\n      // Right now we use different kernels when depth is 1 or not 1.\n      // For the case where maximum depth is 1, we create the auxiliary\n      // data structure with regard to the page table after appending.\n      for (int i = 0; i < cur_batch_size_; ++i) {\n        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);\n      }\n    }"
new1 = "    // Reserve pages unconditionally (Gemma 4 cross-attention fix).\n    for (int i = 0; i < cur_batch_size_; ++i) {\n      ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);\n    }"
assert old1 in content, "FAIL: first reserve block"
content = content.replace(old1, new1)

# Fix 2: Remove second conditional reserve
old2 = "    if (!append_before_attn_) {\n      // Right now we use different kernels when depth is 1 or not 1.\n      // For the case where maximum depth is not 1, we create the auxiliary\n      // data structure with regard to the page table before appending.\n      for (int i = 0; i < cur_batch_size_; ++i) {\n        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);\n      }\n    }"
assert old2 in content, "FAIL: second reserve block"
content = content.replace(old2, "    // (Reserve now done unconditionally above)")

# Fix 3: AttentionWithFusedQKV CHECK assertion
old_fused = "    CHECK(attn_kinds_[layer_id] == AttnKind::kMHA);"
new_fused = "    CHECK(attn_kinds_[layer_id] == AttnKind::kMHA || attn_kinds_[layer_id] == AttnKind::kMHASliding);"
assert old_fused in content, "FAIL: AttentionWithFusedQKV CHECK"
content = content.replace(old_fused, new_fused)

# Fix 4: Self-attention dispatch
old3 = "    if (attn_kind == AttnKind::kMHA) {\n      MHASelfAttnInternal(q_data, k_data, v_data, o_data, lse_data, sm_scale);\n    } else {\n      MLASelfAttnInternal(q_data, k_data, v_data, o_data, lse_data, sm_scale);\n    }"
new3 = "    if (attn_kind == AttnKind::kMHA || attn_kind == AttnKind::kMHASliding) {\n      MHASelfAttnInternal(q_data, k_data, v_data, o_data, lse_data, sm_scale);\n    } else {\n      MLASelfAttnInternal(q_data, k_data, v_data, o_data, lse_data, sm_scale);\n    }"
assert old3 in content, "FAIL: self-attn dispatch"
content = content.replace(old3, new3)

# Fix 5: Cross-attention dispatch
old4 = "    if (attn_kind == AttnKind::kMHA) {\n      MHACrossAttnInternal(local_layer_id, q_data, o_data, lse_data, sm_scale,\n                           /*is_first_kernel=*/true);\n    } else {\n      MLACrossAttnInternal(local_layer_id, q_data, o_data, lse_data, sm_scale);\n    }"
new4 = "    if (attn_kind == AttnKind::kMHA || attn_kind == AttnKind::kMHASliding) {\n      MHACrossAttnInternal(local_layer_id, q_data, o_data, lse_data, sm_scale,\n                           /*is_first_kernel=*/true);\n    } else {\n      MLACrossAttnInternal(local_layer_id, q_data, o_data, lse_data, sm_scale);\n    }"
assert old4 in content, "FAIL: cross-attn dispatch"
content = content.replace(old4, new4)

with open(path, "w") as f:
    f.write(content)
print("  Patched: paged_kv_cache.cc (4 fixes)")
EOF

echo ""
echo "=== Phase 2: Patch Python (TVM) ==="

python3 << "EOF"
# kv_cache.py — AttnKind enum + per-layer support
path = "python/tvm/relax/frontend/nn/llm/kv_cache.py"
with open(path) as f:
    content = f.read()

# Add MHA_SLIDING to enum
content = content.replace(
    "    MHA = 0\n    MLA = 1",
    "    MHA = 0\n    MLA = 1\n    MHA_SLIDING = 3"
)

# Per-layer support (FlashInfer)
old_fi = """            rx.ShapeExpr(
                [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]
            ),
            rx.PrimValue(enable_disaggregation),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_kv_cache_transpose_append_mla(qk_head_dim, dtype), "kv_cache_transpose_append_mla"),"""

new_fi = """            rx.ShapeExpr(
                [int(getattr(AttnKind, k.upper())) for k in attn_kind]
                if isinstance(attn_kind, list)
                else [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]
            ),
            rx.PrimValue(enable_disaggregation),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_kv_cache_transpose_append_mla(qk_head_dim, dtype), "kv_cache_transpose_append_mla"),"""

assert old_fi in content, "FAIL: FlashInfer block"
content = content.replace(old_fi, new_fi)

# Per-layer support (TIR)
old_tir = """            rx.ShapeExpr(
                [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]
            ),
            rx.PrimValue(enable_disaggregation),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            # pylint: disable=line-too-long
            # fmt: off
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_kv_cache_transpose_append_mla(qk_head_dim, dtype), "kv_cache_transpose_append_mla"),"""

new_tir = """            rx.ShapeExpr(
                [int(getattr(AttnKind, k.upper())) for k in attn_kind]
                if isinstance(attn_kind, list)
                else [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]
            ),
            rx.PrimValue(enable_disaggregation),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            # pylint: disable=line-too-long
            # fmt: off
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_kv_cache_transpose_append_mla(qk_head_dim, dtype), "kv_cache_transpose_append_mla"),"""

assert old_tir in content, "FAIL: TIR block"
content = content.replace(old_tir, new_tir)

with open(path, "w") as f:
    f.write(content)
print("  Patched: kv_cache.py")
EOF

python3 << "EOF"
# position_embedding.py — freq_dim_base for partial rotary
path = "python/tvm/relax/frontend/nn/llm/position_embedding.py"
with open(path) as f:
    content = f.read()

old = """def rope_freq_gptj(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
    \"\"\"Compute the inverse frequency of RoPE for gptj RoPE scaling.\"\"\"
    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(d_range, "float32"))"""

new = """def rope_freq_gptj(
    s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str,
    freq_dim_base: int = 0,
):
    \"\"\"Compute the inverse frequency of RoPE for gptj RoPE scaling.
    freq_dim_base: if > 0, use as denominator for partial rotary (Gemma 4).
    \"\"\"
    denom = freq_dim_base if freq_dim_base > 0 else d_range
    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(denom, "float32"))"""

assert old in content, "FAIL: rope_freq_gptj"
content = content.replace(old, new)

old_sw = """    if rope_scaling["rope_type"] == "gptj":
        return rope_freq_gptj"""
new_sw = """    if rope_scaling["rope_type"] == "gptj":
        freq_dim_base = rope_scaling.get("freq_dim_base", 0)
        if freq_dim_base > 0:
            from functools import partial
            return partial(rope_freq_gptj, freq_dim_base=freq_dim_base)
        return rope_freq_gptj"""

if old_sw in content:
    content = content.replace(old_sw, new_sw)
with open(path, "w") as f:
    f.write(content)
print("  Patched: position_embedding.py")
EOF

echo ""
echo "=== Phase 3: Patch MLC-LLM ==="
cd /opt/mlc-llm

python3 << "EOF"
# dispatch_kv_cache_creation.py — accept per-layer lists
path = "python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py"
with open(path) as f:
    content = f.read()

content = content.replace(
    '    assert isinstance(args[0], relax.StringImm)\n    assert args[0].value in ["mha", "mla"]',
    '    assert isinstance(args[0], relax.StringImm)\n    assert args[0].value in ["mha", "mla"] or args[0].value.startswith("[")'
)
content = content.replace(
    '    return {\n        "attn_kind": args[0].value,',
    '    _raw = args[0].value\n    if _raw.startswith("["):\n        import json as _json\n        _attn_kind = _json.loads(_raw)\n    else:\n        _attn_kind = _raw\n    return {\n        "attn_kind": _attn_kind,'
)
with open(path, "w") as f:
    f.write(content)
print("  Patched: dispatch_kv_cache_creation.py")
EOF

python3 << "EOF"
# mlc_llm/nn/kv_cache.py — list support
path = "python/mlc_llm/nn/kv_cache.py"
with open(path) as f:
    content = f.read()

content = content.replace(
    '    @staticmethod\n    def create_generic(  # pylint: disable=too-many-locals\n        attn_kind: Literal["mha", "mla"],',
    '    @staticmethod\n    def create_generic(  # pylint: disable=too-many-locals\n        attn_kind,  # str or List[str]'
)
content = content.replace(
    "                rx.StringImm(attn_kind),",
    '                rx.StringImm(json.dumps(attn_kind) if isinstance(attn_kind, list) else attn_kind),'
)
with open(path, "w") as f:
    f.write(content)
print("  Patched: mlc_llm/nn/kv_cache.py")
EOF

# Install gemma4 model
mkdir -p python/mlc_llm/model/gemma4
cp /patches/gemma4-patches/__init__.py python/mlc_llm/model/gemma4/
cp /patches/gemma4-patches/gemma4_model.py python/mlc_llm/model/gemma4/
cp /patches/gemma4-patches/gemma4_loader_v2.py python/mlc_llm/model/gemma4/gemma4_loader.py
cp /patches/gemma4-patches/gemma4_quantization_v2.py python/mlc_llm/model/gemma4/gemma4_quantization.py

# Register in model.py
python3 << "EOF"
path = "python/mlc_llm/model/model.py"
with open(path) as f:
    content = f.read()

if "gemma4" not in content:
    # Add import
    content = content.replace(
        "from .gemma3 import gemma3_loader, gemma3_model, gemma3_quantization",
        "from .gemma3 import gemma3_loader, gemma3_model, gemma3_quantization\n"
        "from .gemma4 import gemma4_loader, gemma4_model, gemma4_quantization"
    )
    # Add MODELS entry before gpt2
    entry = '''    "gemma4": Model(
        name="gemma4",
        model=gemma4_model.Gemma4ForCausalLM,
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
    content = content.replace('    "gpt2": Model(', entry + '    "gpt2": Model(')
    with open(path, "w") as f:
        f.write(content)
    print("  Registered gemma4 in model.py")
else:
    print("  gemma4 already registered")
EOF

echo "  Installed gemma4 model files"

# Fix: Use Gemma4LanguageModel (params = model.X, matching converted weights)
python3 << "EOF"
path = "python/mlc_llm/model/model.py"
with open(path) as f:
    content = f.read()
if "Gemma4ForCausalLM" in content:
    content = content.replace("Gemma4ForCausalLM", "Gemma4LanguageModel")
    with open(path, "w") as f:
        f.write(content)
    print("  Fixed: model.py -> Gemma4LanguageModel (param name compat)")
EOF

# Fix: fused_rope must be private to avoid duplicate global symbol
python3 << "EOF"
import os
for path in [
    "3rdparty/tvm/python/tvm/relax/frontend/nn/llm/position_embedding.py",
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
        print("  Fixed: position_embedding.py (private rope + int64)")
EOF

echo ""
echo "=== Phase 4: Build TVM ==="
cd /opt/mlc-llm/3rdparty/tvm
mkdir -p build && cd build

cat > config.cmake << "CMAKE"
set(USE_CUDA ON)
set(USE_CUDNN ON)
set(USE_LLVM ON)
set(USE_THRUST ON)
set(USE_CUTLASS ON)
set(USE_FLASH_ATTN OFF)
set(USE_FP_ATTN_GEMM ON)
set(USE_GRAPH_EXECUTOR OFF)
set(USE_PROFILER OFF)
set(USE_MICRO OFF)
set(HIDE_PRIVATE_SYMBOLS ON)
CMAKE

echo "  Configuring CMake (SM 8.7 for Jetson Orin)..."
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    2>&1 | tail -5

echo ""
NCPU=$(nproc)
echo "  Building TVM with $NCPU jobs (this takes 20-40 min)..."
time ninja -j$NCPU 2>&1 | tail -30

echo ""
echo "=== Phase 5: Package output ==="

# Copy .so files
cp libtvm.so libtvm_runtime.so /output/ 2>/dev/null || true
find . -name "libfpA_intB_gemm.so" -exec cp {} /output/ \; 2>/dev/null || true

# Package all patched Python files
cd /opt/mlc-llm
tar czf /output/mlc-python-patched.tar.gz \
    3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py \
    3rdparty/tvm/python/tvm/relax/frontend/nn/llm/position_embedding.py \
    python/mlc_llm/nn/kv_cache.py \
    python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py \
    python/mlc_llm/model/gemma4/ \
    python/mlc_llm/model/model.py \
    python/mlc_llm/model/__init__.py

echo ""
echo "=== OUTPUT ==="
ls -lh /output/
echo ""
echo "Deploy to Jetson:"
echo "  1. scp build-output/libtvm*.so to pod at:"
echo "     /usr/local/lib/python3.10/dist-packages/tvm/"
echo "  2. Extract mlc-python-patched.tar.gz:"
echo "     tar xzf mlc-python-patched.tar.gz -C /opt/mlc-llm/"
echo "  3. Re-run model compile (mlc_llm compile)"
