#!/bin/bash
# Test MLC-LLM serve API with various prompts
# Run from within the Jetson pod or from any machine with access to the API
#
# Usage:
#   bash test-serve.sh                          # test localhost:8000
#   bash test-serve.sh http://10.0.0.5:8000     # test remote endpoint
set -e

API="${1:-http://localhost:8000}"

echo "============================================"
echo "  MLC-LLM Serve API Tests"
echo "  Endpoint: $API"
echo "============================================"
echo ""

# Helper function
test_prompt() {
    local desc="$1"
    local payload="$2"
    echo "--- Test: $desc ---"
    local resp
    resp=$(curl -s -X POST "$API/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>&1)

    local content
    content=$(echo "$resp" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    c = r['choices'][0]
    text = c['message']['content']
    tokens = r.get('usage', {})
    print(f'  Response: {text[:200]}')
    if tokens:
        print(f'  Tokens: prompt={tokens.get(\"prompt_tokens\",\"?\")}, completion={tokens.get(\"completion_tokens\",\"?\")}, total={tokens.get(\"total_tokens\",\"?\")}')
    finish = c.get('finish_reason', '?')
    print(f'  Finish reason: {finish}')
    if not text.strip():
        print('  WARNING: Empty response!')
except Exception as e:
    print(f'  ERROR: {e}')
    print(f'  Raw: {sys.stdin.read()[:200] if hasattr(sys.stdin, \"read\") else \"(consumed)\"}')
" 2>&1)
    echo "$content"
    echo ""
}

# Test 1: Simple factual (baseline)
test_prompt "Simple factual — capital of France" '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    "max_tokens": 32,
    "temperature": 0.1,
    "stop": ["<turn|>"]
}'

# Test 2: Math (baseline)
test_prompt "Simple math — 2+2" '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    "max_tokens": 16,
    "temperature": 0.0,
    "stop": ["<turn|>"]
}'

# Test 3: Stop token behavior — without explicit stop
test_prompt "No explicit stop tokens" '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "What color is the sky?"}],
    "max_tokens": 64,
    "temperature": 0.1
}'

# Test 4: Stop token behavior — with stop string
test_prompt "With stop string <turn|>" '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "What color is the sky?"}],
    "max_tokens": 64,
    "temperature": 0.1,
    "stop": ["<turn|>"]
}'

# Test 5: System prompt + clinical extraction (long output test)
test_prompt "Clinical extraction (long output, ~100 tokens)" '{
    "model": "gemma4",
    "messages": [
        {"role": "system", "content": "Extract clinical entities from this eICR. Return minified JSON: {\"patient\":{...},\"conditions\":[...],\"labs\":[...],\"meds\":[...],\"vitals\":[...]}. All sections are arrays. Include SNOMED for conditions, LOINC for labs, RxNorm for meds. No summary. No markdown. JSON only."},
        {"role": "user", "content": "Patient: Jane Smith\nGender: F\nDOB: 1985-03-15\nDx: COVID-19 (SNOMED 840539006)\nLab: SARS-CoV-2 RNA (LOINC 94500-6) - Detected\nMeds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543)"}
    ],
    "max_tokens": 512,
    "temperature": 0.1,
    "stop": ["<turn|>"]
}'

# Test 6: Long sequence quality test (~200 tokens)
test_prompt "Long sequence test — explain photosynthesis" '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Explain how photosynthesis works in 3-4 sentences."}],
    "max_tokens": 256,
    "temperature": 0.1,
    "stop": ["<turn|>"]
}'

echo "============================================"
echo "  Tests complete"
echo "============================================"
