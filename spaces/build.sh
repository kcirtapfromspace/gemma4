#!/usr/bin/env bash
# Bundle a flat Hugging Face Spaces deploy directory.
#
# HF Spaces expects everything reachable from the Space root. This script
# copies the spaces/ files plus the pipeline modules from
# apps/mobile/convert/ into the destination directory so the deploy can be
# pushed as-is to a Space repo.
#
# Usage:
#   bash spaces/build.sh [out_dir]
#
# Default out_dir: ./out/space
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/out/space}"

echo "Building Spaces deploy bundle → ${OUT_DIR}"
mkdir -p "${OUT_DIR}/convert"

# Backend variant — defaults to "zerogpu" (in-process PyTorch on ZeroGPU hardware,
# the headline-feature path). Override with CLINIQ_SPACE_BACKEND=remote to deploy
# the HTTP-proxy variant that calls the Kaggle inference-server kernel (saves
# ZeroGPU minutes but adds tunnel-failure risk for live demos).
BACKEND="${CLINIQ_SPACE_BACKEND:-zerogpu}"
case "${BACKEND}" in
  zerogpu)
    ENGINE_SRC="${REPO_ROOT}/spaces/zerogpu_engine.py"
    REQ_SRC="${REPO_ROOT}/spaces/requirements.txt"
    ;;
  mtp)
    ENGINE_SRC="${REPO_ROOT}/spaces/zerogpu_engine_mtp.py"
    REQ_SRC="${REPO_ROOT}/spaces/requirements-mtp.txt"
    ;;
  remote)
    ENGINE_SRC="${REPO_ROOT}/spaces/zerogpu_engine_remote.py"
    REQ_SRC="${REPO_ROOT}/spaces/requirements-remote.txt"
    ;;
  *)
    echo "ERROR: unknown CLINIQ_SPACE_BACKEND=${BACKEND}; expected zerogpu|mtp|remote" >&2
    exit 1
    ;;
esac
echo "Engine: ${BACKEND} (${ENGINE_SRC##*/})"

cp "${REPO_ROOT}/spaces/app.py"  "${OUT_DIR}/app.py"
cp "${ENGINE_SRC}"               "${OUT_DIR}/zerogpu_engine.py"
cp "${REQ_SRC}"                  "${OUT_DIR}/requirements.txt"
cp "${REPO_ROOT}/spaces/README.md"          "${OUT_DIR}/README.md"

# Copy the pipeline modules — only the ones imported by app.py + their deps.
PIPELINE=(
  agent_pipeline.py
  case_diff.py
  fhir_bundle.py
  rag_search.py
  regex_preparser.py
  lookup_table.json
  reportable_conditions.json
)
for f in "${PIPELINE[@]}"; do
  cp "${REPO_ROOT}/apps/mobile/convert/${f}" "${OUT_DIR}/convert/${f}"
done

echo "Done. Push with:"
echo "  cd ${OUT_DIR}"
echo "  git init && git add . && git commit -m 'Initial commit'"
echo "  git remote add origin https://huggingface.co/spaces/<you>/<space-name>"
echo "  git push origin main"
