# MLC-LLM Gemma 4 Production Pipeline

Complete pipeline to deploy the fine-tuned (LoRA-merged) Gemma 4 E2B model
via MLC-LLM on Jetson Orin NX 8GB.

## Prerequisites

- Base HF model at `/tmp/gemma4-hf/` (9.5 GB safetensors)
- GGUF LoRA adapter at `models/cliniq-compact-lora.gguf` (48 MB)
- Patched TVM `.so` files built via `build-gemma4.sh` in `build-output/`
- `mlc-test` pod running on Jetson (talos-jetson-3)

## Step 1: Merge LoRA into Base Weights (Mac Studio, ~2 min)

```bash
cd apps/mlc-llm
python3 merge-lora-to-hf.py \
    --lora ../../models/cliniq-compact-lora.gguf \
    --base /tmp/gemma4-hf \
    --output /tmp/gemma4-hf-merged
```

Output: `/tmp/gemma4-hf-merged/model.safetensors` (9.54 GB)

## Step 2: Convert Weights (Mac Studio Docker, ~5 min)

Default quantization (q4f16_1):
```bash
./convert-weights.sh --merged
```

Try alternative quantizations for better long-sequence quality:
```bash
./convert-weights.sh --merged --quant q4f16_0
./convert-weights.sh --merged --quant q3f16_1
```

Output: `build-output/gemma4-weights-merged-<quant>/`

## Step 3: Deploy TVM Patches + Python (Mac Studio → Jetson, ~3 min)

Only needed once per TVM rebuild:
```bash
./deploy-to-jetson.sh
```

## Step 4: Deploy Merged Model (Mac Studio → Jetson, ~5 min)

```bash
./deploy-merged-model.sh --quant q4f16_1
# or for all quantizations:
./deploy-merged-model.sh --all-quants
```

## Step 5: Compile Model Library (ON Jetson, ~15 min)

```bash
kubectl exec -n gemma4 mlc-test -- bash -c '
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm
    python3 -m mlc_llm compile \
        /models/mlc-models/gemma4-e2b-merged-q4f16_1 \
        --quantization q4f16_1 \
        --device cuda \
        -o /models/mlc-models/gemma4-e2b-merged-q4f16_1/
'
```

## Step 6: Serve (ON Jetson)

```bash
kubectl exec -n gemma4 mlc-test -- bash /models/serve-on-jetson.sh --quant q4f16_1
```

With reduced context (less memory):
```bash
kubectl exec -n gemma4 mlc-test -- bash /models/serve-on-jetson.sh --quant q4f16_1 --ctx 1536
```

## Step 7: Test

```bash
kubectl exec -n gemma4 mlc-test -- bash /models/test-serve.sh
```

## Experiment Matrix

| Quant | Context | Est. Memory | Notes |
|-------|---------|-------------|-------|
| q4f16_1 | 2048 | ~2840 MB | Current baseline, degeneration at ~20 tokens |
| q4f16_1 | 1536 | ~2600 MB | Less memory pressure |
| q4f16_0 | 2048 | ~2840 MB | Different quant scheme, may fix degeneration |
| q4f16_0 | 1536 | ~2600 MB | Best combo for memory + quality? |
| q3f16_1 | 2048 | ~2400 MB | Smallest model, most headroom |
| q3f16_1 | 1536 | ~2200 MB | Maximum headroom |

## Key Files

| File | Purpose |
|------|---------|
| `merge-lora-to-hf.py` | Merge GGUF LoRA into HF safetensors |
| `convert-weights.sh` | Weight conversion (accepts `--merged --quant`) |
| `generate-configs.py` | Generate all config variants |
| `deploy-merged-model.sh` | Full deployment to Jetson |
| `serve-on-jetson.sh` | Start MLC-LLM serve API |
| `test-serve.sh` | Test suite for the serve API |
| `model-config/` | All config variants |
