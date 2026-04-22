#!/bin/bash
# Deploy patched TVM + MLC-LLM to the mlc-test pod on Jetson
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-output"
POD="mlc-test"
NS="gemma4"

TVM_LIB="/usr/local/lib/python3.10/dist-packages/tvm"
MLC_ROOT="/opt/mlc-llm"

echo "============================================"
echo "  Deploy Gemma 4 patches to Jetson"
echo "============================================"
echo "Pod: $POD (namespace: $NS)"
echo ""

# Verify build output exists
if [ ! -f "$BUILD_DIR/libtvm.so" ]; then
    echo "ERROR: Build output not found. Run build-gemma4.sh first."
    exit 1
fi

# Step 1: Backup existing .so files
echo "=== Step 1: Backup existing libraries ==="
kubectl exec -n $NS $POD -- bash -c "
    mkdir -p /tmp/tvm-backup
    cp $TVM_LIB/libtvm.so /tmp/tvm-backup/ 2>/dev/null || true
    cp $TVM_LIB/libtvm_runtime.so /tmp/tvm-backup/ 2>/dev/null || true
    echo '  Backed up to /tmp/tvm-backup/'
    ls -lh /tmp/tvm-backup/
"

# Step 2: Copy .so files
echo ""
echo "=== Step 2: Copy libraries (253 MB total) ==="
echo "  Copying libtvm.so (127 MB)..."
kubectl cp "$BUILD_DIR/libtvm.so" "$NS/$POD:$TVM_LIB/libtvm.so"
echo "  Copying libtvm_runtime.so (82 MB)..."
kubectl cp "$BUILD_DIR/libtvm_runtime.so" "$NS/$POD:$TVM_LIB/libtvm_runtime.so"
echo "  Copying libfpA_intB_gemm.so (43 MB)..."
kubectl cp "$BUILD_DIR/libfpA_intB_gemm.so" "$NS/$POD:$TVM_LIB/libfpA_intB_gemm.so"

# Step 3: Copy and extract Python patches
echo ""
echo "=== Step 3: Deploy Python patches ==="
kubectl cp "$BUILD_DIR/mlc-python-patched.tar.gz" "$NS/$POD:/tmp/mlc-python-patched.tar.gz"
kubectl exec -n $NS $POD -- bash -c "
    cd $MLC_ROOT
    tar xzf /tmp/mlc-python-patched.tar.gz
    rm /tmp/mlc-python-patched.tar.gz
    echo '  Extracted Python patches to $MLC_ROOT'
"

# Also deploy to the installed site-packages (MLC uses both paths)
kubectl exec -n $NS $POD -- bash -c "
    # Copy patched TVM Python files to site-packages
    SITE='/usr/local/lib/python3.10/dist-packages'
    cp $MLC_ROOT/3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py \$SITE/tvm/relax/frontend/nn/llm/kv_cache.py 2>/dev/null || true
    cp $MLC_ROOT/3rdparty/tvm/python/tvm/relax/frontend/nn/llm/position_embedding.py \$SITE/tvm/relax/frontend/nn/llm/position_embedding.py 2>/dev/null || true

    # Copy patched MLC-LLM Python files to site-packages
    cp $MLC_ROOT/python/mlc_llm/nn/kv_cache.py \$SITE/mlc_llm/nn/kv_cache.py 2>/dev/null || true
    cp $MLC_ROOT/python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py \$SITE/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py 2>/dev/null || true
    cp -r $MLC_ROOT/python/mlc_llm/model/gemma4 \$SITE/mlc_llm/model/gemma4 2>/dev/null || true
    cp $MLC_ROOT/python/mlc_llm/model/model.py \$SITE/mlc_llm/model/model.py 2>/dev/null || true
    cp $MLC_ROOT/python/mlc_llm/model/__init__.py \$SITE/mlc_llm/model/__init__.py 2>/dev/null || true
    echo '  Synced to site-packages'
"

# Step 4: Verify
echo ""
echo "=== Step 4: Verify deployment ==="
kubectl exec -n $NS $POD -- bash -c "
    export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm:/usr/local/cuda/lib64
    echo '  Library check:'
    ls -lh $TVM_LIB/libtvm.so $TVM_LIB/libtvm_runtime.so $TVM_LIB/libfpA_intB_gemm.so
    echo ''
    echo '  Python import check:'
    python3 -c '
import tvm
print(f\"  TVM version: {tvm.__version__}\")
from tvm.relax.frontend.nn.llm.kv_cache import AttnKind
assert hasattr(AttnKind, \"MHA_SLIDING\"), \"MHA_SLIDING missing!\"
print(f\"  AttnKind.MHA_SLIDING = {AttnKind.MHA_SLIDING} ✓\")

from mlc_llm.model import MODELS
assert \"gemma4\" in MODELS, \"gemma4 not in MODELS!\"
print(f\"  gemma4 registered in MODELS ✓\")
print(\"  All checks passed!\")
'
"

echo ""
echo "============================================"
echo "  Deployment complete!"
echo "============================================"
echo ""
echo "Next: Re-compile the model on the Jetson:"
echo "  kubectl exec -n $NS $POD -- bash -c '"
echo "    cd /models && python3 -m mlc_llm compile \\"
echo "      /models/gemma-4-e2b-it \\"
echo "      --quantization q4f16_1 \\"
echo "      --device cuda \\"
echo "      -o /models/mlc-models/gemma4-e2b-q4f16_1/'"
