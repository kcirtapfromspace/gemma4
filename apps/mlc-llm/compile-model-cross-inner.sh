#!/bin/bash
# Inner script for cross-compiling model lib — runs inside Docker
# Mounts: /patches (repo, ro), /output (build-output, rw)
set -e

QUANT="${QUANT:-q4f16_1}"
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

echo "=== Phase 2: Apply Python patches ==="
# Run the same patch logic as convert-weights-inner.sh (phases 2-3)
# Source it from the patches dir
bash /patches/convert-weights-inner.sh 2>&1 | grep -E '(Patched|Installed|registered|Config)' || true

echo ""
echo "=== Phase 3: Compile model library ==="
echo "Model: $MODEL_DIR"
echo "Quant: $QUANT"

# Ensure mlc-chat-config.json is present
if [ ! -f "$MODEL_DIR/mlc-chat-config.json" ]; then
    cp /patches/model-config/mlc-chat-config.json "$MODEL_DIR/"
fi

python3 -m mlc_llm compile \
    "$MODEL_DIR" \
    --quantization "$QUANT" \
    --device cuda \
    -o /output/gemma4-merged-cuda.so \
    2>&1

echo ""
echo "=== Result ==="
ls -lh /output/gemma4-merged-cuda.so
echo "=== Cross-compilation complete ==="
