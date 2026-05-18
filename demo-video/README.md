# ClinIQ demo video — judges' guide

`cliniq-demo.mp4` is a 1 min 54 s screen recording of the ClinIQ iOS app on
the iPhone 17 Pro simulator, with synthesized voiceover (macOS Samantha).
The recording follows the 8-beat flow in
[`../apps/mobile/ios-app/DEMO_SCRIPT.md`](../apps/mobile/ios-app/DEMO_SCRIPT.md).
Each beat was captured by relaunching the app with a different
`CLINIQ_*` environment hook between segments, then assembled by
[`../scripts/mux_demo_video.sh`](../scripts/mux_demo_video.sh).

## Beat list

| Time | Beat | What you see |
|---:|---|---|
| 0:00 | Set the scene | Offline banner, Cases list with mixed Draft / Submitted statuses |
| 0:11 | Create a case | New Case sheet pre-filled with a COVID-19 narrative |
| 0:25 | Run AI extraction | Gemma 4 Q3_K_M streams extraction on-device |
| 0:44 | Review and curate | SNOMED / LOINC / RxNorm codes underneath each row, clinician confirms |
| 1:00 | Queue to outbox | Outbox tab, queued report waiting for network |
| 1:11 | Longitudinal "what's new" (c21) | Maria Santos returns — three eCRs across ten days, color-coded diff |
| 1:27 | What's New banner | "1 new, 2 resolved since last eCR" — on-device diff vs prior visit |
| 1:43 | History + audit | Submitted reports filterable by condition / status |

## Reproduce

```bash
# 1. Boot simulator, install fresh build, seed GGUF (see apps/mobile/ios-app/BUILD.md)
# 2. Capture per-beat .mov clips by relaunching with CLINIQ_* env vars:
./scripts/record_ios_demo.sh demo-video/raw

# 3. Generate per-beat narration (macOS Samantha, no API keys required):
./scripts/generate_say_narration.sh

# 4. Assemble the final .mp4:
./scripts/mux_demo_video.sh
```

The recording script drives the simulator entirely through the app's built-in
`CLINIQ_*` env hooks (see `apps/mobile/ios-app/ClinIQ/ClinIQ/Views/Cases/CasesTab.swift`),
so no XCUITest harness or UI automation library is required — each beat is
a clean app relaunch with a different env var set.

## Voiceover

Synthesized with macOS `say` (Samantha voice at 185 wpm). The script
([`../scripts/generate_say_narration.sh`](../scripts/generate_say_narration.sh))
also supports OpenAI TTS via [`../scripts/generate_tts_narration.py`](../scripts/generate_tts_narration.py)
if you set `OPENAI_API_KEY`; switch by running the Python script before
`mux_demo_video.sh`.

## Talking points

- Runs fully offline — inference + storage + queueing happen on the
  device. No PHI in flight.
- Clinician sees clinical names; audit codes (SNOMED / LOINC / RxNorm)
  stay visible but muted.
- Every submission is logged with endpoint + response for audit.
- The phone IS the longitudinal source of truth for the patients the
  clinician has seen. The c21 diff between case versions for the same
  patient runs on-device, no Verato, exact identity-hash match only.
- The sync endpoint is configurable (`SyncConfig.swift`). Real
  public-health interop (mTLS, jurisdiction routing) is documented as
  out of scope for this PoC.
