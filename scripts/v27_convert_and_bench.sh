#!/usr/bin/env bash
# Convert + bench Kaggle kernel v27 (cliniq-compact-lora) against base
# Gemma 4 E2B. Fires the entire keep/discard pipeline in one shot.
#
# Steps:
#   1. Pull kernel artifacts to /tmp/c9-v27/
#   2. Convert merged HF → GGUF f16 via /tmp/llama-cpp-tools/convert_hf_to_gguf.py
#   3. Quantize Q3_K_M to match deployment
#   4. Stop current llama-server (base Gemma 4)
#   5. Start llama-server pointed at the v27 GGUF on the same port
#   6. Run agent_pipeline.py on combined-27 + adv4 = 35 cases
#   7. Run regex_preparser-only bench (no LLM; sanity check that base
#      tooling still works)
#   8. Restart base llama-server (deployment default)
#   9. Print keep/discard verdict per the handoff decision rule
#
# Decision rule:
#   keep    — v27 F1 ≥ base AND 0 false positives on combined-27
#   discard — v27 F1 < base OR FP > 0 on combined-27 (broke tool-calling
#             OR introduced precision regression)
#
# Idempotent re-runs: if /tmp/c9-v27/cliniq-gemma4-e2b-v2-Q3_K_M.gguf
# already exists, conversion is skipped and we go straight to the bench.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE_TOKEN="${KAGGLE_API_TOKEN:-KGAT_815bad4b042568001ff75ed86e46852b}"
WORK_DIR="/tmp/c9-v27"
LLAMA_TOOLS="/tmp/llama-cpp-tools"
BASE_GGUF="/Users/thinkstudio/gemma4/models/gemma-4-E2B-it-Q3_K_M.gguf"
V27_F16="${WORK_DIR}/cliniq-gemma4-e2b-v2.f16.gguf"
V27_Q3KM="${WORK_DIR}/cliniq-gemma4-e2b-v2-Q3_K_M.gguf"
PORT=8090
HOST=127.0.0.1
KAGGLE_KERNEL="patrickdeutsch/cliniq-compact-lora-training"
PY="${REPO_ROOT}/scripts/.venv/bin/python"

# Bench artifacts go alongside the others.
BUILD_DIR="${REPO_ROOT}/apps/mobile/convert/build"
mkdir -p "${BUILD_DIR}" "${WORK_DIR}"

log() { printf "[v27] %s\n" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

# ---------- 1. Pull artifacts ----------

if [ ! -f "${V27_Q3KM}" ]; then
  log "Polling kernel status…"
  status=$(KAGGLE_API_TOKEN="${KAGGLE_TOKEN}" kaggle kernels status "${KAGGLE_KERNEL}" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z]+' || true)
  log "  status = ${status}"
  if echo "${status}" | grep -qE 'RUNNING|QUEUED'; then
    die "Kernel still ${status}. Wait for COMPLETE before running this script."
  fi
  if ! echo "${status}" | grep -qE 'COMPLETE'; then
    die "Kernel ${status} (not COMPLETE). Investigate before bench."
  fi

  log "Pulling artifacts to ${WORK_DIR}…"
  KAGGLE_API_TOKEN="${KAGGLE_TOKEN}" kaggle kernels output "${KAGGLE_KERNEL}" -p "${WORK_DIR}"
  ls "${WORK_DIR}/" | head -10

  # ---------- 2. Convert merged HF → GGUF f16 ----------
  if [ ! -d "${WORK_DIR}/cliniq-compact-merged" ]; then
    die "Expected merged HF model at ${WORK_DIR}/cliniq-compact-merged — not found."
  fi
  log "Converting HF → GGUF f16…"
  "${PY}" "${LLAMA_TOOLS}/convert_hf_to_gguf.py" \
    "${WORK_DIR}/cliniq-compact-merged" \
    --outfile "${V27_F16}" \
    --outtype f16

  # ---------- 3. Quantize Q3_K_M ----------
  log "Quantizing Q3_K_M…"
  llama-quantize "${V27_F16}" "${V27_Q3KM}" Q3_K_M
  log "  v27 GGUF: ${V27_Q3KM} ($(du -h "${V27_Q3KM}" | cut -f1))"
else
  log "Reusing existing ${V27_Q3KM}"
fi

# ---------- 4–5. Restart llama-server with v27 GGUF ----------

stop_servers() {
  pkill -f "llama-server.*--port ${PORT}" 2>/dev/null || true
  # Wait until port frees
  while lsof -ti :${PORT} >/dev/null 2>&1; do sleep 1; done
}

start_server() {
  local model="$1"
  local logfile="$2"
  log "Starting llama-server with $(basename "${model}")…"
  llama-server \
    --model "${model}" \
    --port ${PORT} --host ${HOST} \
    --jinja --ctx-size 8192 --n-gpu-layers 99 --threads 8 \
    > "${logfile}" 2>&1 &
  # Wait for server ready
  until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do sleep 1; done
  log "  server up"
}

stop_servers
start_server "${V27_Q3KM}" "/tmp/llama-server-v27.log"

# ---------- 6. Run combined-27 + adv4 bench under v27 ----------

log "Running combined-27 + adv4 (35 cases) under v27…"
"${PY}" "${REPO_ROOT}/apps/mobile/convert/agent_pipeline.py" \
  --cases \
    "${REPO_ROOT}/scripts/test_cases.jsonl" \
    "${REPO_ROOT}/scripts/test_cases_adversarial.jsonl" \
    "${REPO_ROOT}/scripts/test_cases_adversarial2.jsonl" \
    "${REPO_ROOT}/scripts/test_cases_adversarial3.jsonl" \
    "${REPO_ROOT}/scripts/test_cases_adversarial4.jsonl" \
  --out-json "${BUILD_DIR}/v27_combined35_bench.json" \
  --endpoint "http://${HOST}:${PORT}" \
  | tee "${BUILD_DIR}/v27_combined35_bench.log"

# ---------- 7. Restart base, sanity-check ----------

stop_servers
start_server "${BASE_GGUF}" "/tmp/llama-server-cliniq.log"

# ---------- 8. Compute verdict ----------

log "Computing keep/discard verdict…"
"${PY}" - <<PY
import json, pathlib
v27 = json.loads(pathlib.Path("${BUILD_DIR}/v27_combined35_bench.json").read_text())
ok_rows = [r for r in v27 if "matched" in r]
total_m = sum(r["matched"] for r in ok_rows)
total_e = sum(r["expected"] for r in ok_rows)
total_fp = sum(r["false_positives"] for r in ok_rows)
perfect = sum(1 for r in ok_rows if r["expected"] and r["matched"] == r["expected"] and r["false_positives"] == 0)
recall = total_m / total_e if total_e else 0
prec = total_m / (total_m + total_fp) if (total_m + total_fp) else 0
f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
print(f"v27 result: F1={f1:.3f}  prec={prec:.3f}  recall={recall:.3f}  perfect={perfect}/{len(ok_rows)}  FP={total_fp}")
print(f"baseline:   F1=0.972  prec=1.000  recall=0.946  perfect=30/35  FP=0  (combined-27+adv4 with c19 fast-path)")
print()
if f1 >= 0.972 and total_fp == 0:
    print("VERDICT: KEEP — v27 holds F1 and precision; consider promoting in iOS Frameworks/")
elif total_fp > 0:
    print("VERDICT: DISCARD — v27 introduced false positives. Tool-calling likely broken.")
else:
    print(f"VERDICT: DISCARD — v27 F1 {f1:.3f} < baseline 0.972")
PY

log "Done."
