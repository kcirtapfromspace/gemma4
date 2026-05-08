#!/usr/bin/env bash
# Pull v63 outputs from Kaggle and print the metric delta vs v62 baseline.
# Run after kernel status flips to COMPLETE.
set -euo pipefail

# KGAT token must already be exported in the env (see handoff-2026-04-27.md
# for the canonical KAGGLE_API_TOKEN value; not duplicated here).
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN before running (KGAT_...) — see handoff-2026-04-27.md}"

OUT=/tmp/v63-out
mkdir -p "$OUT"
kaggle kernels output \
  patrickdeutsch/cliniq-gemma4-unsloth-v63-experiment \
  -p "$OUT" -q

echo
echo "=== v63 inline bench (from kernel log) ==="
log="$OUT/cliniq-gemma4-unsloth-v63-experiment.log"
if [ -f "$log" ]; then
  awk '/Inline bench summary/{flag=1} flag' "$log" | head -30
else
  echo "no .log file in $OUT — listing pulled artifacts:"
  ls -la "$OUT"
fi

echo
echo "=== v62 baseline (shipped submission) ==="
cat <<'EOF'
micro_f1            : 0.823
micro_precision     : 0.979
micro_recall        : 0.710
json_valid_rate     : 0.86
cases_above_f1_0_70 : 162 / 200
train_wall_clock    : 1h 04m
EOF

echo
echo "=== LoRA artifact ==="
ls -lh "$OUT"/cliniq_lora* 2>/dev/null || echo "no LoRA dir at root of output — check $OUT"
