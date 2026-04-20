#!/bin/bash
# Benchmark Gemma 3 on MLC-LLM to validate speedup before porting Gemma 4
set -e

MODEL_DIR="/models/mlc-models"
mkdir -p "$MODEL_DIR"

echo "============================================"
echo "  MLC-LLM Gemma 3 Baseline Benchmark"
echo "============================================"

# Step 1: Download Gemma 3 1B (smallest, fastest to test)
echo ""
echo "=== Step 1: Download Gemma 3 1B ==="
if [ ! -d "$MODEL_DIR/gemma-3-1b-it" ]; then
    echo "Downloading google/gemma-3-1b-it..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-3-1b-it', local_dir='$MODEL_DIR/gemma-3-1b-it',
                  ignore_patterns=['*.bin', '*.ot'])
print('Download complete')
" 2>&1 || {
    echo "HF download failed. Trying with git..."
    cd "$MODEL_DIR"
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google/gemma-3-1b-it
    cd gemma-3-1b-it
    git lfs pull --include="*.safetensors"
}
else
    echo "Model already downloaded"
fi

# Step 2: Convert weights to MLC format
echo ""
echo "=== Step 2: Convert Weights ==="
if [ ! -d "$MODEL_DIR/gemma-3-1b-it-q4f16_1" ]; then
    echo "Converting weights (q4f16_1 quantization)..."
    mlc_llm convert_weight \
        "$MODEL_DIR/gemma-3-1b-it" \
        --quantization q4f16_1 \
        -o "$MODEL_DIR/gemma-3-1b-it-q4f16_1"
else
    echo "Weights already converted"
fi

# Step 3: Compile model for CUDA
echo ""
echo "=== Step 3: Compile Model ==="
if [ ! -f "$MODEL_DIR/gemma-3-1b-it-q4f16_1/lib.so" ]; then
    echo "Compiling for CUDA..."
    mlc_llm compile \
        "$MODEL_DIR/gemma-3-1b-it" \
        --quantization q4f16_1 \
        --device cuda \
        -o "$MODEL_DIR/gemma-3-1b-it-q4f16_1/lib.so"
else
    echo "Model already compiled"
fi

# Step 4: Benchmark
echo ""
echo "=== Step 4: Benchmark ==="
echo "Running inference benchmark..."
python3 -c "
from mlc_llm import MLCEngine
import time

engine = MLCEngine('$MODEL_DIR/gemma-3-1b-it-q4f16_1')

# Warmup
print('Warming up...')
for r in engine.chat.completions.create(
    messages=[{'role': 'user', 'content': 'Hi'}],
    model='gemma-3-1b-it',
    max_tokens=10,
    stream=True
):
    pass

# Benchmark: short prompt, measure decode speed
prompt = 'Extract clinical entities from this eICR summary: Patient John Doe, 45M, presented with fever 101.2F, cough, diagnosed with COVID-19 (SNOMED 840539006). Labs: WBC 12.5, CRP 45. Prescribed Paxlovid. Output JSON.'

print(f'Prompt: {prompt[:60]}...')
print('Running benchmark (3 runs)...')

for run in range(3):
    tokens = []
    t0 = time.time()
    first_token_time = None
    for chunk in engine.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}],
        model='gemma-3-1b-it',
        max_tokens=256,
        stream=True
    ):
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            tokens.append(chunk.choices[0].delta.content)

    t1 = time.time()
    total_time = t1 - t0
    ttft = first_token_time - t0 if first_token_time else 0
    n_tokens = len(tokens)
    decode_time = t1 - first_token_time if first_token_time else total_time
    decode_tps = n_tokens / decode_time if decode_time > 0 else 0

    print(f'  Run {run+1}: {n_tokens} tokens, {decode_tps:.1f} tok/s decode, TTFT={ttft:.3f}s, total={total_time:.2f}s')

print()
print('Benchmark complete.')
engine.terminate()
" 2>&1

echo ""
echo "Done."
