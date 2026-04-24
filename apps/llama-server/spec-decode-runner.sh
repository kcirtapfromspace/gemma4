#!/usr/bin/env bash
# Team C4 spec-decode experiment runner.
#
# Usage:
#   ./spec-decode-runner.sh <exp-id> <label> [patch-args-json]
#
# Flow:
#   1. Apply a kubectl strategic-merge patch overriding the llama-server args.
#   2. Wait for the pod to become Ready.
#   3. Run scripts/benchmark.py against the endpoint (n=3 runs x 3 cases).
#   4. Print a one-line summary for appending to SPEC_DECODE_LOG.md.
#
# The caller is responsible for appending the summary to the log.

set -euo pipefail

EXP_ID="${1:?exp id required, e.g. ec3}"
LABEL="${2:?label required, e.g. ngram-N8-pmin60}"
# The third arg is the JSON array body for spec.template.spec.containers[0].args
ARGS_JSON="${3:?args-json required}"
RUNS="${RUNS:-3}"
TEST_CASES="${TEST_CASES:-test_cases_val3.jsonl}"

NAMESPACE=gemma4
DEPLOY=llama-server
ENDPOINT="http://192.168.150.41:30083"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DB="${REPO_ROOT}/apps/llama-server/spec_decode.duckdb"

echo "==[ ${EXP_ID} ${LABEL} ]=="

# Build strategic-merge patch (override container args + add a label so pods roll).
PATCH=$(python3 - "$ARGS_JSON" "$EXP_ID" <<'PY'
import json, sys, os
args = json.loads(sys.argv[1])
exp_id = sys.argv[2]
patch = {
    "spec": {
        "template": {
            "metadata": {
                "labels": {"c4-exp": exp_id}
            },
            "spec": {
                "containers": [{
                    "name": "llama-server",
                    "args": args,
                }]
            }
        }
    }
}
print(json.dumps(patch))
PY
)

echo "--- patch ---"
echo "$PATCH" | python3 -m json.tool

kubectl -n "$NAMESPACE" patch deployment "$DEPLOY" --type=strategic --patch "$PATCH"

echo "--- waiting for rollout ---"
kubectl -n "$NAMESPACE" rollout status deployment/$DEPLOY --timeout=180s

# Extra grace for weights to fully load onto GPU.
for i in {1..60}; do
  if curl -sSf "$ENDPOINT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# Verify which model is loaded.
curl -sS "$ENDPOINT/props" | python3 -c "import json,sys;d=json.load(sys.stdin);print('loaded:',d.get('model_path','?'))" || true

echo "--- benchmark ---"
cd "$REPO_ROOT/scripts"
python3 benchmark.py \
  --experiment-name "c4-${EXP_ID}-${LABEL}" \
  --endpoint "$ENDPOINT" \
  --runs "$RUNS" --warmup 0 \
  --test-cases "$TEST_CASES" \
  --system-prompt "Extract clinical entities from this eICR summary. Output JSON with: patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), medications (RxNorm), vitals, and a case summary. Output valid JSON only." \
  --max-tokens 1024 \
  --no-stream \
  --output-db "$DB" \
  --notes "C4 spec-decode ${EXP_ID} ${LABEL}" \
  --config-json '{"model_file":"see-args","ctx_size":2048,"n_gpu_layers":99}'

echo "--- done ${EXP_ID} ---"
