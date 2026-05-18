#!/usr/bin/env bash
# record_ios_demo.sh — capture an 8-beat ClinIQ demo from the iPhone simulator.
#
# Drives the app through 8 beats by relaunching with different CLINIQ_*
# environment hooks between each beat; records each beat as a separate
# .mov clip so the transition flashes don't bleed into the final video.
# Concatenated and muxed with TTS narration downstream by mux_demo.sh.
#
# Usage:
#   ./scripts/record_ios_demo.sh [--out-dir demo-video/raw]
#
# Prereqs:
#   - iPhone17ProDemo simulator already booted
#   - ClinIQ.app installed (see apps/mobile/ios-app/BUILD.md)
#   - Gemma 4 GGUF seeded in the app's Documents/ dir

set -euo pipefail

UDID="${CLINIQ_SIM_UDID:-CADA1806-F64D-4B02-B983-B75F197D1EF3}"
BUNDLE_ID="com.cliniq.ClinIQ"
OUT_DIR="${1:-demo-video/raw}"

mkdir -p "$OUT_DIR"

# Each beat: (filename, dwell-seconds, env-vars-as-string)
# Env vars are passed via SIMCTL_CHILD_* prefix to propagate into the app.
beats=(
  "beat-01-offline-cases:10:CLINIQ_SIMULATE_OFFLINE=1 CLINIQ_TAB=cases"
  "beat-02-new-case-prefilled:12:CLINIQ_SIMULATE_OFFLINE=1 CLINIQ_OPEN_NEW_CASE=1 CLINIQ_PREFILL_NEW_CASE=1"
  "beat-03-extraction-running:18:CLINIQ_SIMULATE_OFFLINE=1 CLINIQ_OPEN_DRAFT_REVIEW=1 CLINIQ_AUTO_EXTRACT=1"
  "beat-04-review-curate:14:CLINIQ_SIMULATE_OFFLINE=1 CLINIQ_OPEN_REVIEW=1"
  "beat-05-outbox-queued:10:CLINIQ_SIMULATE_OFFLINE=1 CLINIQ_TAB=outbox"
  "beat-06-longitudinal-timeline:14:CLINIQ_OPEN_TIMELINE=1"
  "beat-07-whats-new-banner:14:CLINIQ_OPEN_LONGITUDINAL_REVIEW=1"
  "beat-08-history-audit:10:CLINIQ_TAB=history"
)

record_beat() {
  local name="$1" dwell="$2" env_pairs="$3"
  local out="$OUT_DIR/$name.mov"

  echo ">> beat $name  (${dwell}s)  env: $env_pairs"

  # simctl launch reads env vars from the calling shell with the
  # SIMCTL_CHILD_ prefix and strips the prefix before passing into the
  # app. Translate "CLINIQ_FOO=1 CLINIQ_BAR=2" into the prefixed form.
  local child_env=""
  for pair in $env_pairs; do
    child_env+=" SIMCTL_CHILD_${pair}"
  done

  # Kill any leftover app
  xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true
  sleep 0.5

  # Start screen recording in background
  xcrun simctl io "$UDID" recordVideo --codec=h264 --force "$out" &
  local rec_pid=$!
  sleep 1.5  # let recorder warm up

  # Launch the app with this beat's env vars (prefixed)
  # shellcheck disable=SC2086
  env $child_env xcrun simctl launch --terminate-running-process "$UDID" "$BUNDLE_ID" >/dev/null

  # Dwell — the user-visible duration of this beat
  sleep "$dwell"

  # Stop the recorder cleanly (SIGINT so it flushes the .mov)
  kill -INT $rec_pid 2>/dev/null || true
  wait $rec_pid 2>/dev/null || true

  echo "   wrote $(du -h "$out" | cut -f1)  $out"
}

# Make sure the simulator is booted
xcrun simctl boot "$UDID" 2>/dev/null || true
xcrun simctl bootstatus "$UDID" -b >/dev/null

# Bring the simulator window to the front so we capture clean frames
open -ga Simulator

for entry in "${beats[@]}"; do
  IFS=':' read -r name dwell env <<<"$entry"
  record_beat "$name" "$dwell" "$env"
done

# Final teardown
xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true

echo
echo "All beats captured in $OUT_DIR/"
ls -la "$OUT_DIR/"
