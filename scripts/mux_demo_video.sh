#!/usr/bin/env bash
# mux_demo_video.sh — assemble the final ClinIQ demo .mp4
#
# Inputs (under demo-video/raw/):
#   beat-NN-*.mov                per-beat screen recordings from record_ios_demo.sh
#   audio/beat-NN-*.mp3          per-beat TTS narration from generate_tts_narration.py
#   audio/narration.mp3          full concatenated narration (sanity check)
#
# Output:
#   demo-video/cliniq-demo.mp4   judges-facing video (h264 + aac, 1280x... if scaled)
#
# Pipeline per beat:
#   1. Pad audio to match video clip duration (silence at end) so audio
#      doesn't underrun the visual.
#   2. Combine: ffmpeg -i clip.mov -i audio.mp3 -shortest -map 0:v -map 1:a
#   3. Append all combined clips with concat demuxer.

set -euo pipefail

RAW_DIR="${RAW_DIR:-demo-video/raw}"
AUDIO_DIR="${AUDIO_DIR:-$RAW_DIR/audio}"
OUT="${OUT:-demo-video/cliniq-demo.mp4}"
WORK_DIR="$RAW_DIR/work"

mkdir -p "$WORK_DIR"

beats=(
  beat-01-offline-cases
  beat-02-new-case-prefilled
  beat-03-extraction-running
  beat-04-review-curate
  beat-05-outbox-queued
  beat-06-longitudinal-timeline
  beat-07-whats-new-banner
  beat-08-history-audit
)

concat_list="$WORK_DIR/concat.txt"
: > "$concat_list"

for name in "${beats[@]}"; do
  vid="$RAW_DIR/$name.mov"
  aud="$AUDIO_DIR/$name.mp3"
  padded_aud="$WORK_DIR/$name-padded.mp3"
  out_clip="$WORK_DIR/$name.mp4"

  if [[ ! -f "$vid" ]]; then
    echo "skip $name: $vid missing"
    continue
  fi
  if [[ ! -f "$aud" ]]; then
    echo "skip $name: $aud missing"
    continue
  fi

  vdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$vid")
  adur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$aud")

  # Pad audio to >= video duration (silence trailing) so -shortest cuts at video end.
  ffmpeg -y -loglevel error -i "$aud" \
    -af "apad=whole_dur=${vdur}" \
    -ar 44100 -ac 2 -b:a 192k \
    "$padded_aud"

  # Combine. Re-encode to h264 + aac for clean concat downstream.
  # The raw simulator capture has variable frame rate (idle screens drop to
  # ~6 fps), which makes the re-encoded clip shorter than the source. Force
  # 30 fps via -vf fps=30 so the video covers the full clip duration.
  ffmpeg -y -loglevel error \
    -i "$vid" -i "$padded_aud" \
    -map 0:v -map 1:a \
    -vf "fps=30" \
    -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
    -c:a aac -b:a 192k -ar 44100 \
    -t "$vdur" -movflags +faststart \
    "$out_clip"

  echo "file '$name.mp4'" >> "$concat_list"
  echo "  packed $name  v=${vdur}s a=${adur}s -> $out_clip"
done

# Final concat
ffmpeg -y -loglevel error \
  -f concat -safe 0 -i "$concat_list" \
  -c copy -movflags +faststart \
  "$OUT"

dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")
size=$(du -h "$OUT" | cut -f1)
echo
echo "Final: $OUT  duration=${dur}s  size=${size}"
