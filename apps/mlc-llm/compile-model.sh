#!/bin/bash
# Compile Gemma 4 model lib.so on Mac Studio Docker
# Uses the patched TVM we already built + the model code
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/build-output"

echo "============================================"
echo "  Compile Gemma 4 model (Mac Studio Docker)"
echo "============================================"

# We need the HF config.json and mlc-chat-config.json
# Copy from existing model if not available locally
if [ ! -f "$SCRIPT_DIR/model-config/config.json" ]; then
    echo "Fetching model configs from Jetson..."
    mkdir -p "$SCRIPT_DIR/model-config"
    kubectl cp gemma4/mlc-test:/models/mlc-models/gemma-4-e2b-it/config.json "$SCRIPT_DIR/model-config/config.json" 2>/dev/null || true
    kubectl cp gemma4/mlc-test:/models/mlc-models/gemma4-e2b-q4f16_1-fixed/mlc-chat-config.json "$SCRIPT_DIR/model-config/mlc-chat-config.json" 2>/dev/null || true
fi

if [ ! -f "$SCRIPT_DIR/model-config/config.json" ]; then
    echo "ERROR: Need model-config/config.json. Copy from Jetson manually."
    exit 1
fi

docker run --rm \
    -v "$SCRIPT_DIR:/patches:ro" \
    -v "$OUTPUT_DIR:/output" \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash /patches/compile-model-inner.sh

echo ""
echo "Output: $OUTPUT_DIR/lib.so"
ls -lh "$OUTPUT_DIR/lib.so" 2>/dev/null
