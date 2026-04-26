#!/usr/bin/env bash
# Deploy the ClinIQ Gradio Space to Hugging Face.
#
# Wraps spaces/build.sh + the standard HF Spaces git push flow into a
# single idempotent script. Re-runnable to push updates — it reuses the
# existing out/ bundle's git remote when present.
#
# Usage:
#   bash spaces/deploy.sh                                    # default repo
#   bash spaces/deploy.sh --space patrickdeutsch/cliniq-eicr-fhir
#   bash spaces/deploy.sh --space owner/name --out /tmp/space
#   bash spaces/deploy.sh --no-push                          # stage only
#
# By default this script DOES NOT push (no auth assumed locally) — it
# stages a commit in the bundle directory and prints the push command to
# run. Pass --push to actually run `git push origin main`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- args ----
SPACE_REPO="patrickdeutsch/cliniq-eicr-fhir"
OUT_DIR=""
DO_PUSH=0
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --space)      SPACE_REPO="$2"; shift 2 ;;
    --out)        OUT_DIR="$2"; shift 2 ;;
    --push)       DO_PUSH=1; shift ;;
    --no-push)    DO_PUSH=0; shift ;;
    --message|-m) COMMIT_MSG="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/out/space}"
SPACE_URL="https://huggingface.co/spaces/${SPACE_REPO}"
SPACE_GIT_URL="https://huggingface.co/spaces/${SPACE_REPO}"
COMMIT_MSG="${COMMIT_MSG:-Deploy from $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

echo "==> Build flat bundle"
bash "${REPO_ROOT}/spaces/build.sh" "${OUT_DIR}"

cd "${OUT_DIR}"

echo "==> Initialize / verify local git repo at ${OUT_DIR}"
if [[ ! -d .git ]]; then
  git init -q -b main
fi

# Ensure default branch is `main` (HF Spaces expects main)
CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo main)"
if [[ "${CURRENT_BRANCH}" != "main" ]]; then
  git branch -M main
fi

echo "==> Configure remote ${SPACE_GIT_URL}"
if git remote get-url origin >/dev/null 2>&1; then
  EXISTING="$(git remote get-url origin)"
  if [[ "${EXISTING}" != "${SPACE_GIT_URL}" ]]; then
    echo "    updating origin: ${EXISTING} -> ${SPACE_GIT_URL}"
    git remote set-url origin "${SPACE_GIT_URL}"
  else
    echo "    origin already set"
  fi
else
  git remote add origin "${SPACE_GIT_URL}"
fi

echo "==> Verify Hugging Face auth"
if command -v huggingface-cli >/dev/null 2>&1; then
  if huggingface-cli whoami >/dev/null 2>&1; then
    HF_USER="$(huggingface-cli whoami 2>/dev/null | head -1 || echo unknown)"
    echo "    logged in as: ${HF_USER}"
  else
    echo "    NOT logged in. Run: huggingface-cli login"
    echo "    (continuing — will stage commit, but skip push)"
    DO_PUSH=0
  fi
else
  echo "    huggingface-cli not found on PATH. Install with: pip install huggingface_hub"
  echo "    (continuing — will stage commit, but skip push)"
  DO_PUSH=0
fi

echo "==> Stage + commit"
git add -A
if git diff --cached --quiet; then
  echo "    no changes to commit"
else
  git commit -q -m "${COMMIT_MSG}"
  echo "    committed: ${COMMIT_MSG}"
fi

if [[ "${DO_PUSH}" -eq 1 ]]; then
  echo "==> Push to ${SPACE_GIT_URL}"
  git push origin main
  echo
  echo "Deployed: ${SPACE_URL}"
else
  echo
  echo "==> DRY RUN — not pushing. To push:"
  echo "    cd ${OUT_DIR}"
  echo "    huggingface-cli login    # if not already"
  echo "    git push origin main"
  echo
  echo "Target Space: ${SPACE_URL}"
fi
