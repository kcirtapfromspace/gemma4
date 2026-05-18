#!/usr/bin/env bash
# generate_say_narration.sh — local-TTS fallback for the ClinIQ demo narration.
# Uses macOS `say` (Samantha) to produce per-beat MP3s matching the beat names
# expected by mux_demo_video.sh.

set -euo pipefail

OUT_DIR="${OUT_DIR:-demo-video/raw/audio}"
VOICE="${VOICE:-Samantha}"
RATE="${RATE:-185}"   # words per minute; 185 is a comfortable clinical pace

mkdir -p "$OUT_DIR"

# Beat narrations (must mirror scripts/record_ios_demo.sh beat names + dwell).
declare -a beats=(
  "beat-01-offline-cases|A clinician is at a remote clinic with no internet. On her phone, she opens ClinIQ. The yellow bar tells her she is offline. Cases stay on the device until the network comes back."
  "beat-02-new-case-prefilled|She sees a patient with a possible notifiable disease. She taps the plus button and enters demographics with a narrative describing fever, cough, and a positive respiratory test."
  "beat-03-extraction-running|She taps Review with AI. Gemma 4 runs entirely on the phone. Tokens stream in at around eight per second. No data ever leaves the device, no patient health information in flight."
  "beat-04-review-curate|The model proposes a condition, a lab, a medication, and the vitals. Each row shows the SNOMED, LOINC, or RxNorm code underneath for audit. She confirms each one with a tap."
  "beat-05-outbox-queued|She queues the case. It lands in the Outbox. Still no network, so the status stays queued, ready to drain the moment Wi-Fi returns."
  "beat-06-longitudinal-timeline|Five days later, Maria Santos returns. ClinIQ recognizes her from prior visits: three e-C-Rs across ten days, on-device, no patient data leaving the phone."
  "beat-07-whats-new-banner|The What is New banner shows exactly what changed since her last visit: Dengue confirmed and unchanged, a new chest X-ray, an elevated blood pressure. One row of new findings instead of two whole e-C-Rs."
  "beat-08-history-audit|Back online, ClinIQ drains the outbox automatically. Every submission lands in History with a receipt, filterable by condition or status for follow-up."
)

for entry in "${beats[@]}"; do
  name="${entry%%|*}"
  text="${entry#*|}"
  aiff="$OUT_DIR/$name.aiff"
  mp3="$OUT_DIR/$name.mp3"

  echo "  say  $name"
  say -v "$VOICE" -r "$RATE" -o "$aiff" "$text"
  ffmpeg -y -loglevel error -i "$aiff" -ar 44100 -ac 2 -b:a 192k "$mp3"
  rm -f "$aiff"
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$mp3")
  echo "        ${dur}s -> $mp3"
done

echo
echo "Done. Per-beat MP3s in $OUT_DIR/"
