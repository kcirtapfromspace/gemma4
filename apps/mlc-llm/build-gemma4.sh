#!/bin/bash
# Build patched TVM + MLC-LLM for Gemma 4 support on Jetson Orin NX
# Run on Mac Studio: ./build-gemma4.sh
# Output: apps/mlc-llm/build-output/ with .so files and Python packages
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/build-output"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Gemma 4 MLC-LLM Build (Mac Studio Docker)"
echo "============================================"
echo "Output: $OUTPUT_DIR"
echo ""

docker run --rm \
    -v "$SCRIPT_DIR:/patches:ro" \
    -v "$OUTPUT_DIR:/output" \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash /patches/apply-and-build.sh

echo ""
echo "============================================"
echo "  Build complete! Output in: $OUTPUT_DIR"
echo "============================================"
ls -lh "$OUTPUT_DIR/" 2>/dev/null
