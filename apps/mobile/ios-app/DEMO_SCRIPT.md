# ClinIQ — 60-second demo narration

Read this verbatim while holding the simulator / device in view. Each
section is timed for ~10-15 s at a moderate pace.

## 0:00 — Set the scene

> "A clinician is in a remote clinic with no cellular or Wi-Fi during her
> shift. On her phone she opens ClinIQ. The yellow bar at the top tells
> her she's offline — cases stay on the device until the network comes
> back."

(Open the app. You see the **Cases list** with a mix of Draft, Queued,
and Submitted cases, plus the offline banner.)

## 0:10 — Create a case

> "She sees a patient with a possible notifiable disease. She taps the
> plus button, enters demographics, and types or pastes a narrative
> describing fever, cough, and a positive respiratory test."

(Tap the plus button top-right. The **New Case** sheet presents patient
fields and a free-text clinical narrative. Demonstrate the Sample menu
to show a prefilled narrative if typing would be slow on camera.)

## 0:20 — Run AI extraction

> "She taps Review with AI. Gemma 4 runs entirely on the phone. No data
> leaves the device. A live tokens-per-second counter shows inference
> progressing — on a real iPhone this is under ten seconds."

(Tap **Review with AI**. For the demo, the running state shows the
streaming tok/s readout; the sample cases run through the deterministic
fallback so the review view appears promptly.)

## 0:30 — Review and curate

> "The model proposes a condition, a lab result, a medication, and the
> vitals. Each row shows the human-readable name, a review state chip,
> and the SNOMED, LOINC, or RxNorm code underneath for audit. She taps
> the green check to confirm each one — or the red minus to remove
> anything incorrect."

(Scroll the **AI Review** sheet. Point at the SNOMED / LOINC / RxNorm
codes in the muted subtitle lines. Tap a few check marks.)

## 0:40 — Queue to outbox

> "Happy with the extraction, she queues the case. It lands in the
> Outbox. The report isn't sent yet — there's still no network, so the
> status stays 'Queued'."

(Tap **Queue to Outbox**. Switch to the **Outbox** tab. Show the queued
row and the disabled Sync button. The Outbox tab badge shows the
pending count.)

## 0:50 — Network returns, auto-sync

> "Back at the district hospital her phone picks up Wi-Fi. ClinIQ
> detects the network, drains the outbox automatically, and stamps each
> report with a submitted badge and a receipt reference. The History
> tab now shows it alongside past submissions — filterable by condition
> or status for follow-up."

(Flip the "Simulate offline for demo" toggle in Settings off, or
disconnect the simulator's offline mode. Watch the banner turn green,
the Outbox count go to zero, and the badge on the submitted case
switch to green. Then switch to **History** to show the row with an
audit record.)

## Key talking points

- Runs fully offline — inference + storage + queueing.
- Clinician sees clinical names; audit codes stay visible but muted.
- Every submission is logged with endpoint + response for audit.
- The sync endpoint is configurable (`SyncConfig.swift`). Real public
  health interop (mTLS, jurisdiction routing) is explicitly out of scope
  for this PoC — documented, not wired.
