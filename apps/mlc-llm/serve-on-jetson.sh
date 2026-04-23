#!/bin/bash
# Serve Gemma 4 E2B via MLC-LLM's OpenAI-compatible API on Jetson
#
# Prerequisites:
#   1. Patches deployed (deploy-to-jetson.sh)
#   2. Model compiled on Jetson (compile-model.sh)
#   3. Weights converted and deployed
#
# Usage (run inside mlc-test pod):
#   bash /models/serve-on-jetson.sh                          # default: q4f16_1, ctx 2048
#   bash /models/serve-on-jetson.sh --quant q4f16_0          # different quantization
#   bash /models/serve-on-jetson.sh --ctx 1536               # reduced context
#   bash /models/serve-on-jetson.sh --port 8080              # custom port
set -e

# Defaults
QUANT="q4f16_1"
CTX=2048
PORT=8000
MODEL_TAG="merged"
MODEL_BASE="/models/mlc-models"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant) QUANT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --model-tag) MODEL_TAG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODEL_DIR="$MODEL_BASE/gemma4-e2b-${MODEL_TAG}-${QUANT}"
CONFIG_NAME="mlc-chat-config-${QUANT}-ctx${CTX}.json"

echo "============================================"
echo "  Gemma 4 E2B MLC-LLM Server"
echo "============================================"
echo "  Model:  $MODEL_DIR"
echo "  Quant:  $QUANT"
echo "  Context: $CTX"
echo "  Port:   $PORT"
echo ""

# Verify model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Available models:"
    ls "$MODEL_BASE/" 2>/dev/null || echo "  (none)"
    exit 1
fi

# Deploy the right config
CONFIG_SRC="/models/model-configs/${CONFIG_NAME}"
if [ -f "$CONFIG_SRC" ]; then
    cp "$CONFIG_SRC" "$MODEL_DIR/mlc-chat-config.json"
    echo "  Config: $CONFIG_NAME"
else
    echo "  Config: using existing mlc-chat-config.json in model dir"
fi

# Ensure tokenizer files are present
if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "  Copying tokenizer files..."
    for f in tokenizer.json tokenizer_config.json chat_template.jinja; do
        [ -f "/models/gemma-4-e2b-it/$f" ] && cp "/models/gemma-4-e2b-it/$f" "$MODEL_DIR/"
    done
fi

export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm

echo ""
echo "  Starting MLC-LLM serve on port $PORT..."
echo "  API: http://localhost:$PORT/v1/chat/completions"
echo ""

exec python3 -m mlc_llm serve \
    "$MODEL_DIR" \
    --device cuda \
    --host 0.0.0.0 \
    --port "$PORT" \
    --overrides "max_num_sequence=1"
