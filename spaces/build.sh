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

cp "${REPO_ROOT}/spaces/app.py"           "${OUT_DIR}/app.py"
cp "${REPO_ROOT}/spaces/requirements.txt" "${OUT_DIR}/requirements.txt"
cp "${REPO_ROOT}/spaces/README.md"        "${OUT_DIR}/README.md"

# Copy the pipeline modules — only the ones imported by app.py + their deps.
PIPELINE=(
  agent_pipeline.py
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
