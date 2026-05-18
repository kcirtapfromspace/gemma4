#!/usr/bin/env python3
"""Generate per-beat narration audio for the ClinIQ demo using OpenAI TTS.

Beat list mirrors scripts/record_ios_demo.sh. Each beat gets its own .mp3
sized to fit the recorded clip duration (with ~1s breathing room at start
and end). Audio model defaults to tts-1-hd (~$0.03 for the full clip).

Outputs:
  demo-video/raw/audio/beat-NN-*.mp3   per-beat narration
  demo-video/raw/audio/narration.mp3   concatenated full track

Requires OPENAI_API_KEY in env.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import urllib.request
import json

BEATS = [
    (
        "beat-01-offline-cases",
        9.0,
        "A clinician is at a remote clinic with no internet. On her phone, "
        "she opens ClinIQ. The yellow bar tells her she's offline — cases "
        "stay on the device until the network comes back.",
    ),
    (
        "beat-02-new-case-prefilled",
        11.0,
        "She sees a patient with a possible notifiable disease. She taps the "
        "plus button and enters demographics with a narrative describing "
        "fever, cough, and a positive respiratory test.",
    ),
    (
        "beat-03-extraction-running",
        17.0,
        "She taps Review with AI. Gemma 4 runs entirely on the phone. Tokens "
        "stream in at around eight per second — no data ever leaves the "
        "device, no PHI in flight.",
    ),
    (
        "beat-04-review-curate",
        13.0,
        "The model proposes a condition, a lab, a medication, and the vitals. "
        "Each row shows the SNOMED, LOINC, or RxNorm code underneath for "
        "audit. She confirms each one with a tap.",
    ),
    (
        "beat-05-outbox-queued",
        9.0,
        "She queues the case. It lands in the Outbox — still no network, so "
        "the status stays queued, ready to drain the moment Wi-Fi returns.",
    ),
    (
        "beat-06-longitudinal-timeline",
        13.0,
        "Five days later, Maria Santos returns. ClinIQ recognizes her from "
        "prior visits — three eCRs across ten days, on-device, no patient "
        "data leaving the phone.",
    ),
    (
        "beat-07-whats-new-banner",
        13.0,
        "The What's New banner shows exactly what changed since her last "
        "visit: Dengue confirmed and unchanged, a new chest X-ray, an elevated "
        "blood pressure. One row of new findings instead of two whole eCRs.",
    ),
    (
        "beat-08-history-audit",
        9.0,
        "Back online, ClinIQ drains the outbox automatically. Every "
        "submission lands in History with a receipt — filterable by "
        "condition or status for follow-up.",
    ),
]


def synthesize(text: str, voice: str, model: str, api_key: str) -> bytes:
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": "mp3",
            "speed": 1.05,
        }).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="nova")
    parser.add_argument("--model", default="tts-1-hd")
    parser.add_argument("--out-dir", default="demo-video/raw/audio")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in env", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    list_file = out / "concat.txt"
    list_lines: list[str] = []

    for name, target_sec, text in BEATS:
        path = out / f"{name}.mp3"
        print(f"  TTS  {name}  ({len(text)} chars, target {target_sec:.0f}s)")
        audio = synthesize(text, voice=args.voice, model=args.model, api_key=api_key)
        path.write_bytes(audio)

        actual = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
        ).strip())
        print(f"       wrote {path.name}: {actual:.2f}s actual / {target_sec:.0f}s target")

        # If TTS came back shorter than the target dwell, pad with silence so
        # downstream mux can put each beat's audio exactly under its video.
        if actual < target_sec - 0.3:
            padded = out / f"{name}-padded.mp3"
            pad_sec = target_sec - actual
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                "-af", f"apad=pad_dur={pad_sec:.2f}",
                "-t", f"{target_sec:.2f}",
                str(padded),
            ], check=True)
            path = padded

        list_lines.append(f"file '{path.name}'")

    list_file.write_text("\n".join(list_lines) + "\n")

    final = out / "narration.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(final),
    ], check=True)

    total = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(final)]
    ).strip())
    print(f"\n  -> {final} ({total:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
