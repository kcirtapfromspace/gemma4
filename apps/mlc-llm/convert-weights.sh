#!/bin/bash
# Convert Gemma 4 E2B weights using the updated model code (v3)
# Runs in Docker on Mac Studio, produces quantized weights in build-output/
#
# Usage:
#   ./convert-weights.sh                              # base model, q4f16_1
#   ./convert-weights.sh --merged                     # merged (LoRA) model, q4f16_1
#   ./convert-weights.sh --merged --quant q4f16_0     # merged model, q4f16_0
#   ./convert-weights.sh --merged --quant q3f16_1     # merged model, q3f16_1
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/build-output"
HF_MODEL_DIR="/tmp/gemma4-hf"
QUANT="q4f16_1"
MODEL_TAG="base"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --merged) HF_MODEL_DIR="/tmp/gemma4-hf-merged"; MODEL_TAG="merged"; shift ;;
        --model-dir) HF_MODEL_DIR="$2"; MODEL_TAG="custom"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

WEIGHT_DIR="gemma4-weights-${MODEL_TAG}-${QUANT}"

echo "============================================"
echo "  Convert Gemma 4 weights — Mac Studio"
echo "============================================"
echo "  Model: $HF_MODEL_DIR ($MODEL_TAG)"
echo "  Quant: $QUANT"
echo "  Output: $OUTPUT_DIR/$WEIGHT_DIR/"
echo ""

# Validate HF model exists
if [ ! -f "$HF_MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: $HF_MODEL_DIR/model.safetensors not found."
    if [ "$MODEL_TAG" = "merged" ]; then
        echo "Run merge-lora-to-hf.py first to create the merged model."
    else
        echo "Download it first with talosctl read."
    fi
    exit 1
fi

ACTUAL_SIZE=$(stat -f%z "$HF_MODEL_DIR/model.safetensors" 2>/dev/null || stat -c%s "$HF_MODEL_DIR/model.safetensors" 2>/dev/null)
if [ "$ACTUAL_SIZE" -lt 5000000000 ]; then
    echo "ERROR: model.safetensors is only $(( ACTUAL_SIZE / 1048576 )) MB — expected ~9.6 GB."
    exit 1
fi

echo "HF model: $HF_MODEL_DIR ($(du -sh "$HF_MODEL_DIR" | cut -f1))"
echo ""

mkdir -p "$OUTPUT_DIR/$WEIGHT_DIR"

docker run --rm \
    -v "$SCRIPT_DIR:/patches:ro" \
    -v "$OUTPUT_DIR:/output" \
    -v "$HF_MODEL_DIR:/hf-model:ro" \
    -e QUANT="$QUANT" \
    -e WEIGHT_DIR="$WEIGHT_DIR" \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash /patches/convert-weights-inner.sh
