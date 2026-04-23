#!/bin/bash
# Cross-compile Gemma 4 model lib on Mac Studio Docker for Jetson (sm_87)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/build-output"
QUANT="${1:-q4f16_1}"

echo "============================================"
echo "  Cross-compile model lib for Jetson sm_87"
echo "============================================"

docker run --rm \
    -v "$SCRIPT_DIR:/patches:ro" \
    -v "$OUTPUT_DIR:/output" \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash /patches/compile-cross-inner.sh "$QUANT"

echo ""
ls -lh "$OUTPUT_DIR/gemma4-merged-cuda.so"
echo "Done. Deploy with: kubectl cp $OUTPUT_DIR/gemma4-merged-cuda.so gemma4/mlc-test:/models/mlc-models/gemma4-e2b-merged-q4f16_1/gemma4-e2b-merged-q4f16_1-cuda.so"
