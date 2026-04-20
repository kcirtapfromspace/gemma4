#!/bin/bash
# Build TVM from source in Docker on Mac Studio, then copy .so files to Jetson
# This is ~10x faster than building on the Jetson at 7W
set -e

echo "============================================"
echo "  TVM Build — Mac Studio (Docker)"
echo "============================================"
echo "Cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu)"
echo ""

cd /opt/mlc-llm/3rdparty/tvm

# Apply patches
echo "=== Applying patches ==="

# 1. paged_kv_cache.cc — hoist ReserveAppendLengthInSeq
python3 -c "
path = 'src/runtime/relax_vm/paged_kv_cache.cc'
with open(path, 'r') as f:
    content = f.read()

old = '''    if (append_before_attn_) {
      // Right now we use different kernels when depth is 1 or not 1.
      // For the case where maximum depth is 1, we create the auxiliary
      // data structure with regard to the page table after appending.
      for (int i = 0; i < cur_batch_size_; ++i) {
        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);
      }
    }'''

new = '''    // Reserve pages unconditionally before aux-data loop (Gemma 4 fix).
    for (int i = 0; i < cur_batch_size_; ++i) {
      ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);
    }'''

content = content.replace(old, new)

old2 = '''    if (!append_before_attn_) {
      // Right now we use different kernels when depth is 1 or not 1.
      // For the case where maximum depth is not 1, we create the auxiliary
      // data structure with regard to the page table before appending.
      for (int i = 0; i < cur_batch_size_; ++i) {
        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);
      }
    }'''
content = content.replace(old2, '    // Removed: now done unconditionally above')

with open(path, 'w') as f:
    f.write(content)
print('Patched: paged_kv_cache.cc')
"

# 2. position_embedding.py — add freq_dim_base for partial rotary
python3 -c "
path = 'python/tvm/relax/frontend/nn/llm/position_embedding.py'
with open(path, 'r') as f:
    content = f.read()

old = '''def rope_freq_gptj(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
    \"\"\"Compute the inverse frequency of RoPE for gptj RoPE scaling.\"\"\"
    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(d_range, \"float32\"))'''

new = '''def rope_freq_gptj(
    s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str,
    freq_dim_base: int = 0,
):
    \"\"\"Compute the inverse frequency of RoPE for gptj RoPE scaling.
    freq_dim_base: if > 0, use as denominator for partial rotary (Gemma 4).
    \"\"\"
    denom = freq_dim_base if freq_dim_base > 0 else d_range
    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(denom, \"float32\"))'''

content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('Patched: position_embedding.py')
"

# Build
echo ""
echo "=== Building TVM ==="
mkdir -p build && cd build

cat > config.cmake << EOF
set(USE_CUDA ON)
set(USE_CUDNN ON)
set(USE_LLVM ON)
set(USE_THRUST ON)
set(USE_CUTLASS ON)
set(USE_FLASH_ATTN ON)
set(USE_FP_ATTN_GEMM ON)
set(USE_GRAPH_EXECUTOR OFF)
set(USE_PROFILER OFF)
set(USE_MICRO OFF)
set(HIDE_PRIVATE_SYMBOLS ON)
EOF

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    2>&1 | tail -5

echo ""
echo "Starting build with $(nproc) jobs..."
time ninja -j$(nproc)
echo ""
echo "=== Build complete ==="
ls -lh libtvm*.so libfpA_intB_gemm.so 2>/dev/null

# Copy results to output (find all .so files from the build tree)
mkdir -p /output
cp libtvm.so libtvm_runtime.so /output/ || true
find /opt/mlc-llm/3rdparty/tvm/build -name "libfpA_intB_gemm.so" -exec cp {} /output/ \; 2>/dev/null || true
find /opt/mlc-llm/3rdparty/tvm/build -name "libflash_attn.so" -exec cp {} /output/ \; 2>/dev/null || true
# Also copy the Python extensions
find /opt/mlc-llm/3rdparty/tvm/python -name "*.so" -exec cp {} /output/ \; 2>/dev/null || true
echo "Libraries copied to /output/"
ls -lh /output/
