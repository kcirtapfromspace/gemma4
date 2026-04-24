#!/usr/bin/env bash
# Team C4 — run the entire spec-decode sweep.
# Invocations are gated by $ONLY to allow piecemeal runs.
#
# Usage:
#   ./run-sweep.sh                    # runs all configured experiments
#   ONLY="ec0 ec3" ./run-sweep.sh     # run just a subset

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/apps/llama-server"

CONFIG=spec-configs.json
RUNNER=./spec-decode-runner.sh
APPEND=./append-log-row.py

EXPERIMENTS=(ec0_baseline_noSpec ec6_kvQuant_q8 ec3_ngramCache_N8 ec4_ngramCache_N4 ec5_ngramCache_N16 ec1_self_Q2Kdraft_N8 ec2_base_draft_N8 ec7_self_Q3KSdraft_N8)
ONLY_LIST="${ONLY:-}"
DEFAULT_RUNS="${DEFAULT_RUNS:-2}"
DEFAULT_TEST_CASES="${DEFAULT_TEST_CASES:-test_cases_val3.jsonl}"

for key in "${EXPERIMENTS[@]}"; do
  short_id="${key%%_*}"
  if [[ -n "$ONLY_LIST" ]] && ! [[ " $ONLY_LIST " == *" $short_id "* ]]; then
    continue
  fi
  label="${key#*_}"
  label="${label//_/-}"
  args_json=$(python3 -c "import json;c=json.load(open('$CONFIG'));print(json.dumps(c['$key']['args']))")
  echo ""
  echo "#############################"
  echo "# running $short_id ($label)"
  echo "#############################"
  if ! RUNS="$DEFAULT_RUNS" TEST_CASES="$DEFAULT_TEST_CASES" "$RUNNER" "$short_id" "$label" "$args_json"; then
    echo "!! experiment $short_id failed — continuing"
  fi
  # best-effort log append (captures id from duckdb)
  python3 "$APPEND" "$short_id" "$label" "?" "?" "?" "?" || true
done

echo ""
echo "=== Sweep complete ==="
