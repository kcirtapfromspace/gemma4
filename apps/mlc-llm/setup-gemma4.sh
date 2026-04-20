#!/bin/bash
# MLC-LLM Gemma 4 Setup Script
# Run inside the dustynv/mlc container on Jetson
set -e

echo "============================================"
echo "  MLC-LLM Gemma 4 Port — Setup Script"
echo "============================================"

# Step 0: Verify environment
echo ""
echo "=== Step 0: Environment Check ==="
python3 -c "import tvm; print(f'TVM version: {tvm.__version__}')" 2>/dev/null || echo "TVM not found"
python3 -c "import mlc_llm; print(f'MLC-LLM available')" 2>/dev/null || echo "MLC-LLM not found"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available (expected on Jetson)"
echo "GPU check:"
python3 -c "import tvm; print(tvm.cuda(0).exist)" 2>/dev/null || echo "CUDA check failed"

# Check TVM source and build dirs
echo ""
echo "=== TVM/MLC Source Locations ==="
ls -d /opt/mlc-llm 2>/dev/null && echo "MLC-LLM source: /opt/mlc-llm"
ls -d /opt/mlc-llm/3rdparty/tvm 2>/dev/null && echo "TVM source: /opt/mlc-llm/3rdparty/tvm"
ls -d /opt/mlc-llm/3rdparty/tvm/build 2>/dev/null && echo "TVM build: /opt/mlc-llm/3rdparty/tvm/build"
echo "Python site-packages TVM:"
python3 -c "import tvm; print(tvm.__file__)" 2>/dev/null

# Check what gemma models are already registered
echo ""
echo "=== Registered Gemma Models ==="
python3 -c "
from mlc_llm.model import MODELS
for k in sorted(MODELS.keys()):
    if 'gemma' in k.lower():
        print(f'  {k}')
" 2>/dev/null || echo "Could not list models"

echo ""
echo "=== Disk Space ==="
df -h /models /opt 2>/dev/null

echo ""
echo "Setup complete. Ready for next steps."
