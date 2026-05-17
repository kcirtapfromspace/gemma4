# ClinIQ demo video — judges' guide

`cliniq-demo.mp4` is a 64-second screen capture of the ClinIQ iOS app on
the iPhone 17 Pro simulator. No voiceover; read along with the narration
below. The capture follows the original 7-beat demo flow.

> **Note for judges:** the video was recorded 2026-04-13, before the c21
> longitudinal "what's new" view shipped (added 2026-05-15). The current
> demo script in [`../apps/mobile/ios-app/DEMO_SCRIPT.md`](../apps/mobile/ios-app/DEMO_SCRIPT.md)
> includes an 8th beat for the longitudinal feature; the video does not.
> The longitudinal feature is visible in the
> [`screenshot-c21-longitudinal-timeline.png`](../apps/mobile/ios-app/screenshot-c21-longitudinal-timeline.png)
> and [`screenshot-c21-longitudinal-diff-banner.png`](../apps/mobile/ios-app/screenshot-c21-longitudinal-diff-banner.png)
> stills in the repo root and is wired through `PatientTimelineView.swift`.

## Narration (read along with the .mp4)

| Time | Beat | Narration |
|---:|---|---|
| 0:00 | Set the scene | A clinician is in a remote clinic with no cellular or Wi-Fi during her shift. On her phone she opens ClinIQ. The yellow bar at the top tells her she's offline — cases stay on the device until the network comes back. |
| 0:10 | Create a case | She sees a patient with a possible notifiable disease. She taps the plus button, enters demographics, and types or pastes a narrative describing fever, cough, and a positive respiratory test. |
| 0:20 | Run AI extraction | She taps Review with AI. When the GGUF model is present, Gemma 4 runs entirely on the phone. If the model is missing, ClinIQ labels the deterministic fallback instead of pretending the model ran. No data leaves the device. |
| 0:30 | Review and curate | The model proposes a condition, a lab result, a medication, and the vitals. Each row shows the human-readable name, a review state chip, and the SNOMED, LOINC, or RxNorm code underneath for audit. She taps the green check to confirm each one — or the red minus to remove anything incorrect. |
| 0:40 | Queue to outbox | Happy with the extraction, she queues the case. It lands in the Outbox. The report isn't sent yet — there's still no network, so the status stays "Queued". |
| 0:50 | Network returns | Back at the district hospital her phone picks up Wi-Fi. ClinIQ detects the network, drains the outbox automatically, and stamps each report with a submitted badge and a receipt reference. |
| 1:00 | Audit trail | The History tab now shows the submission alongside past ones — filterable by condition or status for follow-up. |

## Key talking points (for a live demo)

- Runs fully offline — inference + storage + queueing.
- Clinician sees clinical names; audit codes stay visible but muted.
- Every submission is logged with endpoint + response for audit.
- Sync endpoint is configurable (`SyncConfig.swift`). Real public-health
  interop (mTLS, jurisdiction routing) is documented as out of scope for
  this PoC.

For the most up-to-date 8-beat script including the longitudinal "what's
new" view, see [`../apps/mobile/ios-app/DEMO_SCRIPT.md`](../apps/mobile/ios-app/DEMO_SCRIPT.md).
