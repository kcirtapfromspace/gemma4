#!/bin/bash
# Cross-compile Gemma 4 model library on Mac Studio Docker targeting Jetson (sm_87)
# Produces a .so that can be deployed directly to the Jetson
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/build-output"
MODEL_DIR="$OUTPUT_DIR/gemma4-weights-merged-q4f16_1"
QUANT="${1:-q4f16_1}"

echo "============================================"
echo "  Cross-compile Gemma 4 model lib (sm_87)"
echo "============================================"
echo "  Model: $MODEL_DIR"
echo "  Quant: $QUANT"
echo "  Output: $OUTPUT_DIR/gemma4-merged-cuda.so"
echo ""

if [ ! -f "$MODEL_DIR/ndarray-cache.json" ]; then
    echo "ERROR: Weights not found. Run convert-weights.sh --merged first."
    exit 1
fi

docker run --rm \
    -v "$SCRIPT_DIR:/patches:ro" \
    -v "$OUTPUT_DIR:/output" \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash -c '
set -e

# Phase 0: CUDA stubs
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

# Phase 1: Install patched TVM
cp /output/libtvm.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm.so
cp /output/libtvm_runtime.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm_runtime.so

# Phase 2: Apply all Python patches (same as convert-weights-inner.sh)
cd /opt/mlc-llm/3rdparty/tvm
'"$(cat "$SCRIPT_DIR/convert-weights-inner.sh" | sed -n '/Phase 2: Apply all Python patches/,/Phase 3: Verify/{ /Phase 3/d; p }')"'

# Phase 3: Compile model
echo ""
echo "=== Compiling model library (cross-compile for sm_87) ==="
MODEL_DIR="/output/gemma4-weights-merged-'"$QUANT"'"

# Need mlc-chat-config.json in the model dir
cp /patches/model-config/mlc-chat-config.json "$MODEL_DIR/" 2>/dev/null || true

python3 -m mlc_llm compile \
    "$MODEL_DIR" \
    --quantization '"$QUANT"' \
    --device cuda \
    --host aarch64-unknown-linux-gnu \
    -o /output/gemma4-merged-cuda.so \
    2>&1

echo ""
ls -lh /output/gemma4-merged-cuda.so
echo "=== Cross-compilation complete ==="
'
