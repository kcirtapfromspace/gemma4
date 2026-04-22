#!/bin/bash
# Convert Gemma 4 E2B weights using the updated model code (v3)
# Runs in Docker on Mac Studio, produces quantized weights in build-output/gemma4-weights-v3/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/build-output"
HF_MODEL_DIR="/tmp/gemma4-hf"

echo "============================================"
echo "  Convert Gemma 4 weights (v3) — Mac Studio"
echo "============================================"

# Validate HF model exists
if [ ! -f "$HF_MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: $HF_MODEL_DIR/model.safetensors not found."
    echo "Download it first with talosctl read."
    exit 1
fi

ACTUAL_SIZE=$(stat -f%z "$HF_MODEL_DIR/model.safetensors" 2>/dev/null || stat -c%s "$HF_MODEL_DIR/model.safetensors" 2>/dev/null)
if [ "$ACTUAL_SIZE" -lt 5000000000 ]; then
    echo "ERROR: model.safetensors is only $(( ACTUAL_SIZE / 1048576 )) MB — expected ~9.6 GB."
    echo "Download may still be in progress."
    exit 1
fi

echo "HF model: $HF_MODEL_DIR ($(du -sh "$HF_MODEL_DIR" | cut -f1))"
echo "Output: $OUTPUT_DIR/gemma4-weights-v3/"
echo ""

mkdir -p "$OUTPUT_DIR/gemma4-weights-v3"

docker run --rm \
    -v "$SCRIPT_DIR:/patches:ro" \
    -v "$OUTPUT_DIR:/output" \
    -v "$HF_MODEL_DIR:/hf-model:ro" \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash /patches/convert-weights-inner.sh
