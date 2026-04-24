#!/usr/bin/env bash
# Team C2 sweep runner. Under 7W regime we get ~0.9 tok/s so run budget matters.
# 1 case x 2 runs x max_tokens=200 → ~4 min per exp run, ~10 min per exp incl restart.

set -uo pipefail

SCRIPTS=/Users/thinkstudio/gemma4/scripts
ENDPOINT=http://192.168.150.41:30083
DB="$SCRIPTS/benchmarks.duckdb"
CASES="$SCRIPTS/test_cases_c2.jsonl"
RUNS=2
WARMUP=0
MAX_TOKENS=200
PROMPT="Extract clinical entities from this eICR summary. Output JSON with: patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), medications (RxNorm), vitals, and a case summary. Output valid JSON only."
MODEL=/models/cliniq-gemma4-e2b-Q3_K_M.gguf
LOG_DIR=/tmp/c2-logs
mkdir -p "$LOG_DIR"

say() { echo ""; echo "=== $(date -u +%H:%M:%S) $* ==="; }

deploy_args() {
  local patch
  patch=$(python3 -c "
import json,sys
args=sys.argv[1:]
p={'spec':{'template':{'spec':{'containers':[{'name':'llama-server','args':args}]}}}}
print(json.dumps(p))" "$@")
  kubectl -n gemma4 patch deployment llama-server --type=strategic -p "$patch" > /dev/null
  kubectl -n gemma4 scale deployment/llama-server --replicas=0 > /dev/null
  sleep 4
  kubectl -n gemma4 delete pods -l app=llama-server --force --grace-period=0 2>/dev/null || true
  kubectl -n gemma4 scale deployment/llama-server --replicas=1 > /dev/null
  local deadline=$((SECONDS + 240))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -s -m 3 "$ENDPOINT/health" 2>/dev/null | grep -q '"status":"ok"'; then
      echo "  health ok at $SECONDS"
      sleep 3
      return 0
    fi
    sleep 4
  done
  echo "  HEALTH TIMEOUT"
  return 1
}

run_exp() {
  local name=$1 config_json=$2
  local log="$LOG_DIR/$name.log"
  say "Running $name (max_tokens=$MAX_TOKENS, runs=$RUNS)"
  stdbuf -oL -eL python3 "$SCRIPTS/benchmark.py" \
    --experiment-name "$name" \
    --endpoint "$ENDPOINT" \
    --runs $RUNS --warmup $WARMUP \
    --test-cases "$CASES" \
    --system-prompt "$PROMPT" \
    --max-tokens $MAX_TOKENS \
    --no-stream \
    --output-db "$DB" \
    --config-json "$config_json" \
    --notes "C2 sweep — $name — 7W power, 1 case x $RUNS runs x max $MAX_TOKENS" 2>&1 | tee "$log"
}

# ------------------------------------------------------------
# EXPERIMENTS
# ------------------------------------------------------------

say "EXPERIMENT: c2-baseline"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1
run_exp "c2-baseline" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0}' || true

say "EXPERIMENT: c2-fa-on"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --flash-attn on
run_exp "c2-fa-on" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"flash_attn":true}' || true

say "EXPERIMENT: c2-kvq8-fa"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
run_exp "c2-kvq8-fa" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"flash_attn":true,"cache_type_k":"q8_0","cache_type_v":"q8_0"}' || true

say "EXPERIMENT: c2-kvq4-fa"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0
run_exp "c2-kvq4-fa" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"flash_attn":true,"cache_type_k":"q4_0","cache_type_v":"q4_0"}' || true

say "EXPERIMENT: c2-ubatch-256"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --ubatch-size 256
run_exp "c2-ubatch-256" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"ubatch_size":256}' || true

say "EXPERIMENT: c2-ubatch-128"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --ubatch-size 128
run_exp "c2-ubatch-128" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"ubatch_size":128}' || true

say "EXPERIMENT: c2-threads-2"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --threads 2
run_exp "c2-threads-2" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"threads":2}' || true

say "EXPERIMENT: c2-threads-4"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --threads 4
run_exp "c2-threads-4" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"threads":4}' || true

say "EXPERIMENT: c2-mlock"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --mlock
run_exp "c2-mlock" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"mlock":true}' || true

say "EXPERIMENT: c2-combo-fa-kvq8"
deploy_args -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --mlock
run_exp "c2-combo-fa-kvq8" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"flash_attn":true,"cache_type_k":"q8_0","cache_type_v":"q8_0","mlock":true}' || true

say "SWEEP COMPLETE"
