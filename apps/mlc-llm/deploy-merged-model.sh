#!/bin/bash
# Deploy merged (LoRA) model to Jetson for MLC-LLM serving
#
# Deploys:
#   1. Merged quantized weights (from convert-weights.sh --merged)
#   2. Model configs for all quantization variants
#   3. Tokenizer files
#   4. Serve and test scripts
#
# Usage:
#   ./deploy-merged-model.sh                     # deploy q4f16_1 merged weights
#   ./deploy-merged-model.sh --quant q4f16_0     # deploy q4f16_0 merged weights
#   ./deploy-merged-model.sh --all-quants        # deploy all available quantizations
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-output"
POD="mlc-test"
NS="gemma4"

QUANT="q4f16_1"
ALL_QUANTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant) QUANT="$2"; shift 2 ;;
        --all-quants) ALL_QUANTS=true; shift ;;
        --pod) POD="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  Deploy Merged Model to Jetson"
echo "============================================"
echo "  Pod: $POD (namespace: $NS)"
echo ""

# Step 1: Deploy model configs (all variants)
echo "=== Step 1: Deploy model configs ==="
kubectl exec -n $NS $POD -- mkdir -p /models/model-configs
for cfg in "$SCRIPT_DIR/model-config"/mlc-chat-config-*.json; do
    name=$(basename "$cfg")
    kubectl cp "$cfg" "$NS/$POD:/models/model-configs/$name"
    echo "  $name"
done
echo ""

# Step 2: Deploy tokenizer files
echo "=== Step 2: Deploy tokenizer ==="
HF_DIR="/tmp/gemma4-hf-merged"
if [ ! -d "$HF_DIR" ]; then
    HF_DIR="/tmp/gemma4-hf"
fi
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do
    if [ -f "$HF_DIR/$f" ]; then
        kubectl exec -n $NS $POD -- mkdir -p /models/gemma-4-e2b-it
        kubectl cp "$HF_DIR/$f" "$NS/$POD:/models/gemma-4-e2b-it/$f"
        echo "  $f"
    fi
done
echo ""

# Step 3: Deploy serve and test scripts
echo "=== Step 3: Deploy scripts ==="
kubectl cp "$SCRIPT_DIR/serve-on-jetson.sh" "$NS/$POD:/models/serve-on-jetson.sh"
kubectl cp "$SCRIPT_DIR/test-serve.sh" "$NS/$POD:/models/test-serve.sh"
echo "  serve-on-jetson.sh"
echo "  test-serve.sh"
echo ""

# Step 4: Deploy quantized weights
deploy_weights() {
    local q="$1"
    local weight_dir="gemma4-weights-merged-${q}"
    local src="$BUILD_DIR/$weight_dir"
    local dest="/models/mlc-models/gemma4-e2b-merged-${q}"

    if [ ! -d "$src" ]; then
        echo "  SKIP: $src not found (run: ./convert-weights.sh --merged --quant $q)"
        return 1
    fi

    echo "  Deploying $weight_dir → $dest"
    kubectl exec -n $NS $POD -- mkdir -p "$dest"

    # Copy weight files
    for f in "$src"/*; do
        name=$(basename "$f")
        size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
        echo "    $name ($(( size / 1048576 )) MB)"
        kubectl cp "$f" "$NS/$POD:$dest/$name"
    done

    # Copy config
    local cfg_name="mlc-chat-config-${q}-ctx2048.json"
    if [ -f "$SCRIPT_DIR/model-config/$cfg_name" ]; then
        kubectl cp "$SCRIPT_DIR/model-config/$cfg_name" "$NS/$POD:$dest/mlc-chat-config.json"
        echo "    mlc-chat-config.json (from $cfg_name)"
    fi

    # Copy config.json (model architecture)
    kubectl cp "$SCRIPT_DIR/model-config/config.json" "$NS/$POD:$dest/config.json"
    echo "    config.json"

    # Copy tokenizer
    for f in tokenizer.json tokenizer_config.json chat_template.jinja; do
        [ -f "$HF_DIR/$f" ] && kubectl cp "$HF_DIR/$f" "$NS/$POD:$dest/$f"
    done
    echo "    tokenizer files"

    echo "  Done: $dest"
}

echo "=== Step 4: Deploy weights ==="
if [ "$ALL_QUANTS" = true ]; then
    for q in q4f16_1 q4f16_0 q3f16_1; do
        deploy_weights "$q" || true
    done
else
    deploy_weights "$QUANT"
fi
echo ""

echo "============================================"
echo "  Deployment complete!"
echo "============================================"
echo ""
echo "Next steps on Jetson:"
echo "  1. Compile model (if not already):"
echo "     kubectl exec -n $NS $POD -- bash -c '"
echo "       export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm"
echo "       python3 -m mlc_llm compile /models/mlc-models/gemma4-e2b-merged-$QUANT \\"
echo "         --quantization $QUANT --device cuda \\"
echo "         -o /models/mlc-models/gemma4-e2b-merged-$QUANT/gemma4-e2b-merged-$QUANT-cuda.so'"
echo ""
echo "  2. Start server:"
echo "     kubectl exec -n $NS $POD -- bash /models/serve-on-jetson.sh --quant $QUANT"
echo ""
echo "  3. Test:"
echo "     kubectl exec -n $NS $POD -- bash /models/test-serve.sh"
