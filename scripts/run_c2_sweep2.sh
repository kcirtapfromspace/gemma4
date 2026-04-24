#!/usr/bin/env bash
# Team C2 sweep runner v2. Coordinate with C4 by checking deployment before each exp.

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

patch_and_relabel() {
  # Tag the pod with c2-exp=<name> so we can detect if someone overrides.
  local name=$1
  shift
  local patch
  patch=$(python3 -c "
import json,sys
args=sys.argv[1:]
p={'spec':{'template':{'metadata':{'labels':{'app':'llama-server','c2-exp':'$name'}},'spec':{'containers':[{'name':'llama-server','args':args}]}}}}
print(json.dumps(p))" "$@")
  kubectl -n gemma4 patch deployment llama-server --type=strategic -p "$patch" > /dev/null
}

restart_wait() {
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

check_pod_is_mine() {
  local name=$1
  local label
  label=$(kubectl -n gemma4 get pod -l app=llama-server -o jsonpath='{.items[0].metadata.labels.c2-exp}' 2>/dev/null)
  if [ "$label" = "$name" ]; then
    return 0
  else
    echo "  NOT MY POD: label=$label expected=$name — another team patched over me. Skipping."
    return 1
  fi
}

run_exp() {
  local name=$1 config_json=$2
  local log="$LOG_DIR/$name.log"
  say "Running $name (max=$MAX_TOKENS, runs=$RUNS)"
  if ! check_pod_is_mine "$name"; then
    return 1
  fi
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
    --notes "C2 sweep v2 — $name — 7W power" 2>&1 | tee "$log"
}

do_exp() {
  local name=$1 config_json=$2
  shift 2
  say "EXPERIMENT: $name"
  patch_and_relabel "$name" "$@"
  if restart_wait; then
    run_exp "$name" "$config_json" || true
  fi
}

# ------------------------------------------------------------

# fa-on
do_exp "c2-fa-on" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"flash_attn":true}' \
  -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --flash-attn on

# kvq8+fa
do_exp "c2-kvq8-fa" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"flash_attn":true,"cache_type_k":"q8_0","cache_type_v":"q8_0"}' \
  -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0

# ubatch-128
do_exp "c2-ubatch-128" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"ubatch_size":128}' \
  -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --ubatch-size 128

# ubatch-256
do_exp "c2-ubatch-256" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"ubatch_size":256}' \
  -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --ubatch-size 256

# threads-4
do_exp "c2-threads-4" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"threads":4}' \
  -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --threads 4

# mlock
do_exp "c2-mlock" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"mlock":true}' \
  -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --mlock

# combo (fa + kvq8 + mlock)
do_exp "c2-combo-fa-kvq8-mlock" '{"model_file":"'$MODEL'","ctx_size":2048,"n_gpu_layers":99,"parallel":1,"reasoning_budget":0,"flash_attn":true,"cache_type_k":"q8_0","cache_type_v":"q8_0","mlock":true}' \
  -m "$MODEL" --port 8080 --host 0.0.0.0 --ctx-size 2048 --n-gpu-layers 99 --reasoning-budget 0 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --mlock

say "SWEEP COMPLETE"
