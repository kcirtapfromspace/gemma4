#!/usr/bin/env bash
# Fetch Google's stock gemma-4-E2B-it.litertlm from HuggingFace. Used by the
# Swift unit tests via the LITERTLM_MODEL_PATH env var.
#
# The file is ~2.5 GB and should NOT be committed. We download straight to
# /tmp so it doesn't touch the git repo.

set -euo pipefail

DEST="${1:-/tmp/gemma-4-E2B-it.litertlm}"
REPO="litert-community/gemma-4-E2B-it-litert-lm"
FILE="gemma-4-E2B-it.litertlm"

if [ -f "$DEST" ]; then
  echo "==> $DEST already exists; skipping download (delete to refetch)"
  exit 0
fi

if command -v huggingface-cli >/dev/null 2>&1; then
  TMP="$(mktemp -d)"
  trap "rm -rf $TMP" EXIT
  huggingface-cli download "$REPO" "$FILE" --local-dir "$TMP"
  mv "$TMP/$FILE" "$DEST"
elif command -v curl >/dev/null 2>&1; then
  # Anonymous HTTPS fallback. May hit HF rate limits.
  URL="https://huggingface.co/$REPO/resolve/main/$FILE?download=true"
  curl --fail --location --output "$DEST" "$URL"
else
  echo "error: need either huggingface-cli or curl" >&2
  exit 1
fi

echo "==> wrote $DEST ($(du -sh "$DEST" | cut -f1))"
