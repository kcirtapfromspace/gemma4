#!/bin/bash
# Standalone compile script — runs in Docker, applies patches, compiles model
set -e

# === CUDA stubs ===
NEEDED=$(nm -D /output/libtvm.so 2>/dev/null | grep " U " | grep "^.*cu[A-Z]" | awk "{print \$2}" | sort -u)
echo "// stub" > /tmp/s.c
echo "$NEEDED" | while read sym; do echo "int $sym() { return 0; }" >> /tmp/s.c; done
gcc -shared -o /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 /tmp/s.c
ln -sf libcuda.so.1 /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so
gcc -shared -o /tmp/libstub.so -x c /dev/null
for lib in /usr/lib/aarch64-linux-gnu/nvidia/lib*.so*; do
    [[ "$lib" == *libcuda* ]] && continue
    size=$(stat -c%s "$lib" 2>/dev/null || echo "999999")
    [ "$size" -lt 1000 ] && [ -f "$lib" ] && cp /tmp/libstub.so "$lib"
done
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm:/usr/local/cuda/lib64

# === Install patched TVM .so ===
cp /output/libtvm.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm.so
cp /output/libtvm_runtime.so /usr/local/lib/python3.10/dist-packages/tvm/libtvm_runtime.so

# === Apply all Python patches ===
python3 /patches/apply-patches.py

# === Verify ===
python3 -c "
from tvm.relax.frontend.nn.llm.kv_cache import AttnKind
assert AttnKind.MHA_SLIDING == 3
from mlc_llm.model import MODELS
assert 'gemma4' in MODELS
print('Patches verified OK')
"

# === Compile ===
echo ""
echo "=== Compiling Gemma 4 lib.so ==="
python3 -m mlc_llm compile \
    /patches/model-config \
    --model-type gemma4 \
    --quantization q4f16_1 \
    --device "cuda -arch=sm_87 -max_shared_memory_per_block=49152 -max_num_threads=1024 -thread_warp_size=32" \
    --host "aarch64-unknown-linux-gnu" \
    --output /output/lib.so \
    --opt "flashinfer=0;cudagraph=0" \
    --overrides "context_window_size=2048;prefill_chunk_size=512;max_batch_size=1"

echo ""
echo "=== Done ==="
ls -lh /output/lib.so
