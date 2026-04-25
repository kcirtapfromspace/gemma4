// ReviewFlowView.swift
// Inline AI review sheet. Handles three states:
//
//   1. intro   — shows the narrative preview + "Start AI review" CTA
//   2. running — streaming tok counter + partial output log
//   3. review  — structured list of proposed entities with Accept/Edit/Reject
//                per-row, and a big "Queue to Outbox" terminal button
//
// Keeps state local; persistence to SwiftData happens when the user taps
// Apply or Queue.

import SwiftUI
import SwiftData

struct ReviewFlowView: View {
    @Bindable var clinicalCase: ClinicalCase
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var context
    @EnvironmentObject private var sync: SyncService
    // C15: pulled up to an @EnvironmentObject so Settings' backend
    // picker can reload the engine on the same instance.
    @EnvironmentObject private var service: ExtractionService
    // Observe the shared inference metrics so the runningView's phase
    // chip + ETA copy update live as the engine transitions through
    // load/prefill/decode/finalize.
    @ObservedObject private var inferenceMetrics: InferenceMetrics = .shared

    @State private var phase: Phase = .intro
    @State private var draft = ReviewDraft()
    /// Toggles the "View FHIR Bundle" sheet on the Review screen. The
    /// sheet renders BundleBuilder.bundleJSON(from:) — a structurally
    /// R4-valid Bundle assembled on-device from the current draft. Lets
    /// a judge tap and inspect the same shape that score_fhir.py
    /// validates 35/35 on combined-27 + adv4.
    @State private var showingBundleSheet = false

    enum Phase {
        case intro      // waiting to start
        case running    // model is streaming
        case review     // user is reviewing draft entities
        case error
    }

    var body: some View {
        NavigationStack {
            Group {
                switch phase {
                case .intro: introView
                case .running: runningView
                case .review: reviewView
                case .error: errorView
                }
            }
            .navigationTitle("AI Review")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .onAppear {
                // If the case has existing extracted entities, skip the AI
                // intro and go straight to editing what's already there.
                if hasExistingDraft {
                    draft = ReviewDraft.from(case: clinicalCase)
                    phase = .review
                } else if ProcessInfo.processInfo.environment["CLINIQ_AUTO_EXTRACT"] == "1" {
                    // Demo hook: auto-fire the extraction so we can drive
                    // the flow headlessly from `xcrun simctl launch` with
                    // env vars (taps don't register reliably on iOS 26.4
                    // simulator via idb/simctl).
                    Task { await runExtraction() }
                }
            }
            // Make the live LLM status pill visible while this sheet is
            // presented (it also shows in the root TabView below, but the
            // sheet covers that). `InferenceStatusBar` is a shared observer
            // so this reflects the same metrics.
            .safeAreaInset(edge: .bottom, spacing: 0) {
                InferenceStatusBar()
            }
        }
    }

    private var hasExistingDraft: Bool {
        !clinicalCase.conditions.isEmpty || !clinicalCase.labs.isEmpty || !clinicalCase.medications.isEmpty
    }

    // MARK: - Intro

    private var introView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                infoBanner
                card(title: "Narrative") {
                    Text(clinicalCase.narrative)
                        .font(.callout)
                        .lineSpacing(2)
                }
                Button {
                    Task { await runExtraction() }
                } label: {
                    Label("Start AI review", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                if service.isStubEngine {
                    Text("No on-device model found — using the deterministic rule-based fallback so the PoC flow remains end-to-end.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(16)
        }
        .background(ClinIQTheme.pageBackground)
    }

    private var infoBanner: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Image(systemName: "lock.shield")
                    .foregroundStyle(ClinIQTheme.accent)
                Text("All inference runs on this device. Nothing leaves until you queue a case.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                    .foregroundStyle(ClinIQTheme.accent)
                Text("Backend: \(service.activeBackendLabel)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Running

    private var runningView: some View {
        // Observe the live phase so the heading + chip update as the
        // engine progresses through Loading → Prefilling → Decoding →
        // Finalizing. Without this, "Extracting entities..." sits frozen
        // for 30–90 s and looks like a crash to a non-engineer.
        let metrics = inferenceMetrics
        return VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 12) {
                ProgressView().controlSize(.regular)
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(phaseHeading(metrics.phase))
                            .font(.callout.weight(.medium))
                        phaseChip(for: metrics.phase)
                    }
                    Text(phaseDetail(metrics: metrics))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 12))
            .animation(.easeInOut(duration: 0.2), value: metrics.phase)

            // Tier expectation. A small calming line so the demo viewer
            // knows simulator latency is normal and a physical iPhone is
            // faster. Hardware-tagged based on measured numbers in
            // VALIDATION.md.
            HStack(spacing: 8) {
                Image(systemName: "timer")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(etaCopy(elapsed: metrics.elapsedSeconds, phase: metrics.phase))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
            }

            card(title: "Live extraction") {
                ScrollView {
                    Text(service.streamedOutput.isEmpty
                         ? "Waiting for first token..."
                         : Self.humanReadablePreview(service.streamedOutput))
                        .font(.footnote)
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 180)
            }

            Spacer()
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(ClinIQTheme.pageBackground)
    }

    /// One-line heading shown next to the spinner. Updates as the engine
    /// transitions phases so the user sees forward progress instead of a
    /// frozen "Extracting entities..." string.
    private func phaseHeading(_ phase: InferenceMetrics.Phase) -> String {
        switch phase {
        case .idle:       return "Starting up..."
        case .loading:    return "Loading on-device model..."
        case .prefilling: return "Reading the narrative..."
        case .decoding:   return "Extracting entities..."
        case .finalizing: return "Finalizing extraction..."
        case .error:      return "Inference error"
        }
    }

    /// Right-side detail line under the heading. Includes tok/s when we
    /// have a meaningful number to show and falls back to elapsed-only
    /// during model load / prefill where tok/s is 0.
    private func phaseDetail(metrics: InferenceMetrics) -> String {
        let elapsed = String(format: "%.1f s", max(0, metrics.elapsedSeconds))
        switch metrics.phase {
        case .decoding, .finalizing:
            return String(format: "%@ · %.1f tok/s · %d tok",
                          elapsed, metrics.instantTokensPerSecond, metrics.outputTokens)
        case .loading, .prefilling, .idle:
            return "\(elapsed) · waiting for first token"
        case .error:
            return metrics.lastError ?? "Failed"
        }
    }

    /// Phase chip — same color family as InferenceStatusBar so the two
    /// visualisations agree. Lets a glance from across the room confirm
    /// "the model is doing something" even before the first token lands.
    @ViewBuilder
    private func phaseChip(for phase: InferenceMetrics.Phase) -> some View {
        let (label, color) = phaseChipStyle(phase)
        Text(label)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }

    private func phaseChipStyle(_ phase: InferenceMetrics.Phase) -> (String, Color) {
        switch phase {
        case .idle:       return ("READY", Color.secondary)
        case .loading:    return ("LOAD", Color(red: 0.10, green: 0.30, blue: 0.55))
        case .prefilling: return ("PREFILL", Color(red: 0.10, green: 0.30, blue: 0.55))
        case .decoding:   return ("DECODE", ClinIQTheme.accent)
        case .finalizing: return ("FINAL", ClinIQTheme.accent)
        case .error:      return ("ERROR", Color(red: 0.66, green: 0.15, blue: 0.12))
        }
    }

    /// Soft expectation-setting copy. Updates from "first inference is the
    /// slowest" to "still working" if we cross 60 s — useful for the demo
    /// presenter who otherwise has to guess whether to keep talking.
    private func etaCopy(elapsed: Double, phase: InferenceMetrics.Phase) -> String {
        switch phase {
        case .loading:
            return "First-time model load takes ~15 s on simulator (mmap + KV setup); a few seconds on a real iPhone."
        case .prefilling:
            return "Prefilling the narrative. Typically 15–60 s on simulator CPU; under 1 s on iPhone with Metal."
        case .decoding:
            if elapsed > 60 {
                return "Still decoding — simulator CPU is ~10× slower than iPhone Metal. The stream above shows live tokens."
            }
            return "Streaming tokens. On simulator CPU expect 15–90 s total; on iPhone with Metal, 5–15 s."
        case .finalizing:
            return "Closing the JSON envelope and validating extracted entities."
        case .idle, .error:
            return "Standing by."
        }
    }

    private static func humanReadablePreview(_ raw: String) -> String {
        // Strip obvious JSON noise for the watching clinician. Full parse
        // happens downstream; this is only for the live feed.
        raw
            .replacingOccurrences(of: "\"code\":", with: "code ")
            .replacingOccurrences(of: "\"display\":", with: "display ")
            .replacingOccurrences(of: "\"system\":", with: "system ")
            .replacingOccurrences(of: "{", with: "")
            .replacingOccurrences(of: "}", with: "")
            .replacingOccurrences(of: "\"", with: "")
    }

    // MARK: - Review

    private var reviewView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                metaBanner

                // Show the tier legend whenever at least one entity has
                // provenance attached. Without this a judge sees colored
                // INLINE/CDA/LOOKUP/RAG chips with no context — the c17
                // killer feature becomes invisible.
                if draft.hasAnyProvenance {
                    ProvenanceLegend()
                }

                if !draft.conditions.isEmpty {
                    card(title: "Proposed conditions") {
                        ForEach(draft.conditions.indices, id: \.self) { idx in
                            EntityReviewRow(
                                title: draft.conditions[idx].display,
                                subtitle: "SNOMED · \(draft.conditions[idx].code)",
                                reviewState: draft.conditions[idx].reviewState,
                                onAccept: { draft.conditions[idx].reviewState = .confirmed },
                                onReject: { draft.conditions[idx].reviewState = .rejected },
                                onEdit: {
                                    draft.conditions[idx].reviewState = .edited
                                },
                                provenance: draft.conditions[idx].provenance
                            )
                            if idx != draft.conditions.count - 1 {
                                Divider().padding(.vertical, 2)
                            }
                        }
                    }
                }

                if !draft.labs.isEmpty {
                    card(title: "Proposed labs") {
                        ForEach(draft.labs.indices, id: \.self) { idx in
                            LabReviewRow(lab: $draft.labs[idx])
                            if idx != draft.labs.count - 1 {
                                Divider().padding(.vertical, 2)
                            }
                        }
                    }
                }

                if !draft.medications.isEmpty {
                    card(title: "Proposed medications") {
                        ForEach(draft.medications.indices, id: \.self) { idx in
                            EntityReviewRow(
                                title: draft.medications[idx].display,
                                subtitle: "RxNorm · \(draft.medications[idx].code)",
                                reviewState: draft.medications[idx].reviewState,
                                onAccept: { draft.medications[idx].reviewState = .confirmed },
                                onReject: { draft.medications[idx].reviewState = .rejected },
                                onEdit: { draft.medications[idx].reviewState = .edited },
                                provenance: draft.medications[idx].provenance
                            )
                            if idx != draft.medications.count - 1 {
                                Divider().padding(.vertical, 2)
                            }
                        }
                    }
                }

                if let vitals = draft.vitals {
                    card(title: "Proposed vitals") {
                        VitalsDraftView(vitals: vitals)
                    }
                }

                if draft.isEmpty {
                    card(title: "No entities detected") {
                        Text("The on-device model did not find any reportable entities. Edit the narrative or report manually.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }

                if !draft.isEmpty {
                    Button {
                        showingBundleSheet = true
                    } label: {
                        Label("View FHIR Bundle", systemImage: "doc.text.magnifyingglass")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .padding(.top, 2)
                }

                HStack(spacing: 10) {
                    Button(role: .cancel) {
                        dismiss()
                    } label: {
                        Text("Keep as draft")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button {
                        applyAndQueue()
                    } label: {
                        Label("Queue to Outbox", systemImage: "tray.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(draft.isEmpty)
                }
                .padding(.top, 4)
            }
            .padding(16)
        }
        .background(ClinIQTheme.pageBackground)
        .sheet(isPresented: $showingBundleSheet) {
            FHIRBundleSheet(json: BundleBuilder.bundleJSON(from: draft))
        }
    }

    private var metaBanner: some View {
        HStack(spacing: 12) {
            Label("\(draft.acceptedCount) accepted",
                  systemImage: "checkmark.seal.fill")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Color(red: 0.13, green: 0.36, blue: 0.22))
            if draft.needsReviewCount > 0 {
                Label("\(draft.needsReviewCount) pending",
                      systemImage: "circle.dashed")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Color(red: 0.45, green: 0.30, blue: 0.00))
            }
            Spacer()
            Text(String(format: "%.1f tok/s", service.tokensPerSecond > 0 ? service.tokensPerSecond : clinicalCase.tokensPerSecond))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Error

    private var errorView: some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(Color(red: 0.66, green: 0.15, blue: 0.12))
            Text("Inference failed")
                .font(.title3.weight(.semibold))
            if let msg = service.errorMessage {
                Text(msg)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Button("Retry") {
                phase = .intro
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(24)
    }

    // MARK: - Actions

    private func runExtraction() async {
        phase = .running
        guard let parsed = await service.run(narrative: clinicalCase.narrative) else {
            phase = .error
            return
        }
        clinicalCase.tokensGenerated = service.lastTokens
        clinicalCase.elapsedSeconds = service.lastElapsed
        clinicalCase.tokensPerSecond = service.tokensPerSecond
        draft = ReviewDraft.from(parsed: parsed)
        phase = .review
    }

    private func applyAndQueue() {
        // Wipe the previous extraction rows and replace with the reviewed
        // draft. We deliberately overwrite so repeat extractions don't
        // leave stale entities.
        for old in clinicalCase.conditions { context.delete(old) }
        for old in clinicalCase.labs { context.delete(old) }
        for old in clinicalCase.medications { context.delete(old) }
        clinicalCase.conditions = []
        clinicalCase.labs = []
        clinicalCase.medications = []
        if let oldVitals = clinicalCase.vitals {
            context.delete(oldVitals)
            clinicalCase.vitals = nil
        }

        for cond in draft.conditions where cond.reviewState != .rejected {
            clinicalCase.conditions.append(ExtractedCondition(
                code: cond.code,
                system: cond.system,
                displayName: cond.display,
                reviewState: cond.reviewState == .needsReview ? .confirmed : cond.reviewState))
        }
        for lab in draft.labs where lab.reviewState != .rejected {
            clinicalCase.labs.append(ExtractedLab(
                code: lab.code,
                system: lab.system,
                displayName: lab.display,
                interpretation: lab.interpretation,
                value: lab.value,
                unit: lab.unit,
                reviewState: lab.reviewState == .needsReview ? .confirmed : lab.reviewState))
        }
        for med in draft.medications where med.reviewState != .rejected {
            clinicalCase.medications.append(ExtractedMedication(
                code: med.code,
                system: med.system,
                displayName: med.display,
                reviewState: med.reviewState == .needsReview ? .confirmed : med.reviewState))
        }
        if let v = draft.vitals {
            clinicalCase.vitals = Vitals(
                tempC: v.tempC,
                heartRate: v.heartRate,
                respRate: v.respRate,
                spo2: v.spo2,
                bpSystolic: v.bpSystolic)
        }
        clinicalCase.updatedAt = Date()
        sync.queue(clinicalCase)
        try? context.save()
        dismiss()
    }

    // MARK: - Helpers

    @ViewBuilder
    private func card<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.footnote.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Draft + supporting rows

struct ReviewDraft {
    var conditions: [DraftCondition] = []
    var labs: [DraftLab] = []
    var medications: [DraftMedication] = []
    var vitals: DraftVitals?

    var isEmpty: Bool {
        conditions.isEmpty && labs.isEmpty && medications.isEmpty && vitals == nil
    }

    var acceptedCount: Int {
        conditions.filter { $0.reviewState != .rejected }.count
            + labs.filter { $0.reviewState != .rejected }.count
            + medications.filter { $0.reviewState != .rejected }.count
    }

    var needsReviewCount: Int {
        conditions.filter { $0.reviewState == .needsReview }.count
            + labs.filter { $0.reviewState == .needsReview }.count
            + medications.filter { $0.reviewState == .needsReview }.count
    }

    /// True if any draft entity has provenance attached. Used by the
    /// review screen to decide whether to render the tier legend — there's
    /// no point showing it for cases that flow through the LLM-only path
    /// without provenance metadata.
    var hasAnyProvenance: Bool {
        conditions.contains { $0.provenance != nil }
            || labs.contains { $0.provenance != nil }
            || medications.contains { $0.provenance != nil }
    }

    static func from(parsed: ParsedExtraction) -> ReviewDraft {
        var d = ReviewDraft()
        // Build a (bucket, code) → provenance index so we can attach the
        // tier badge + source span to each draft entry. Provenance comes
        // from EicrPreparser.extractWithProvenance (called inside
        // ExtractionService when det.hasAnyDeterministic is true) or from
        // the agent's tool trace; both populate parsed.matches.
        var provIndex: [String: CodeProvenance] = [:]
        for p in parsed.matches {
            provIndex["\(p.bucket):\(p.code)"] = p
        }
        d.conditions = parsed.conditions.map {
            DraftCondition(code: $0.code, system: $0.system, display: $0.display,
                           provenance: provIndex["snomed:\($0.code)"])
        }
        d.labs = parsed.labs.map {
            DraftLab(code: $0.code,
                     system: $0.system,
                     display: $0.display,
                     interpretation: $0.interpretation,
                     value: $0.value,
                     unit: $0.unit,
                     provenance: provIndex["loinc:\($0.code)"])
        }
        d.medications = parsed.medications.map {
            DraftMedication(code: $0.code, system: $0.system, display: $0.display,
                            provenance: provIndex["rxnorm:\($0.code)"])
        }
        if let v = parsed.vitals {
            d.vitals = DraftVitals(tempC: v.tempC,
                                   heartRate: v.heartRate,
                                   respRate: v.respRate,
                                   spo2: v.spo2,
                                   bpSystolic: v.bpSystolic)
        }
        return d
    }

    static func from(case c: ClinicalCase) -> ReviewDraft {
        var d = ReviewDraft()
        d.conditions = c.conditions.map {
            DraftCondition(code: $0.code, system: $0.system, display: $0.displayName, reviewState: $0.reviewState)
        }
        d.labs = c.labs.map {
            DraftLab(code: $0.code,
                     system: $0.system,
                     display: $0.displayName,
                     interpretation: $0.interpretation,
                     value: $0.value,
                     unit: $0.unit,
                     reviewState: $0.reviewState)
        }
        d.medications = c.medications.map {
            DraftMedication(code: $0.code, system: $0.system, display: $0.displayName, reviewState: $0.reviewState)
        }
        if let v = c.vitals, !v.isEmpty {
            d.vitals = DraftVitals(tempC: v.tempC,
                                   heartRate: v.heartRate,
                                   respRate: v.respRate,
                                   spo2: v.spo2,
                                   bpSystolic: v.bpSystolic)
        }
        return d
    }
}

struct DraftCondition {
    var code: String
    var system: String
    var display: String
    var reviewState: ReviewState = .needsReview
    /// Provenance for the deterministic / agent path. Lost on persistence
    /// (SwiftData layer doesn't carry it) — only present on freshly-extracted
    /// drafts. The reviewer sees the tier badge + source span; subsequent
    /// edits zero this out.
    var provenance: CodeProvenance?
}

struct DraftLab {
    var code: String
    var system: String
    var display: String
    var interpretation: String?
    var value: Double?
    var unit: String?
    var reviewState: ReviewState = .needsReview
    var provenance: CodeProvenance?

    var resultSummary: String {
        if let v = value, let u = unit {
            let formatted = v.rounded() == v ? String(Int(v)) : String(format: "%.1f", v)
            return "\(formatted) \(u)"
        }
        return (interpretation ?? "Pending").capitalized
    }
}

struct DraftMedication {
    var code: String
    var system: String
    var display: String
    var reviewState: ReviewState = .needsReview
    var provenance: CodeProvenance?
}

struct DraftVitals {
    var tempC: Double?
    var heartRate: Int?
    var respRate: Int?
    var spo2: Int?
    var bpSystolic: Int?
}

struct EntityReviewRow: View {
    let title: String
    let subtitle: String
    let reviewState: ReviewState
    let onAccept: () -> Void
    let onReject: () -> Void
    let onEdit: () -> Void
    /// Optional provenance produced by the deterministic preparser or the
    /// agent loop. Renders the tier badge + tap-to-expand source span.
    var provenance: CodeProvenance? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(title)
                            .font(.body.weight(.medium))
                            .fixedSize(horizontal: false, vertical: true)
                        ReviewStateChip(state: reviewState)
                    }
                    Text(subtitle)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(ClinIQTheme.auditMuted)
                }
                Spacer()
                HStack(spacing: 6) {
                    Button(action: onAccept) {
                        Image(systemName: reviewState == .confirmed ? "checkmark.circle.fill" : "checkmark.circle")
                            .font(.title3)
                            .foregroundStyle(Color(red: 0.13, green: 0.36, blue: 0.22))
                    }
                    Button(action: onReject) {
                        Image(systemName: reviewState == .rejected ? "minus.circle.fill" : "minus.circle")
                            .font(.title3)
                            .foregroundStyle(Color(red: 0.66, green: 0.15, blue: 0.12))
                    }
                }
            }
            if let prov = provenance {
                ProvenanceBadge(provenance: prov)
            }
        }
        .padding(.vertical, 6)
        .contextMenu {
            Button("Mark edited") { onEdit() }
        }
    }
}

struct LabReviewRow: View {
    @Binding var lab: DraftLab

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(lab.display)
                            .font(.body.weight(.medium))
                            .fixedSize(horizontal: false, vertical: true)
                        ReviewStateChip(state: lab.reviewState)
                    }
                    Text("LOINC · \(lab.code)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(ClinIQTheme.auditMuted)
                }
                Spacer()
                Text(lab.resultSummary)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(resultColor)
            }
            HStack(spacing: 6) {
                Spacer()
                Button {
                    lab.reviewState = .confirmed
                } label: {
                    Image(systemName: lab.reviewState == .confirmed ? "checkmark.circle.fill" : "checkmark.circle")
                        .font(.title3)
                        .foregroundStyle(Color(red: 0.13, green: 0.36, blue: 0.22))
                }
                Button {
                    lab.reviewState = .rejected
                } label: {
                    Image(systemName: lab.reviewState == .rejected ? "minus.circle.fill" : "minus.circle")
                        .font(.title3)
                        .foregroundStyle(Color(red: 0.66, green: 0.15, blue: 0.12))
                }
            }
            if let prov = lab.provenance {
                ProvenanceBadge(provenance: prov)
            }
        }
        .padding(.vertical, 4)
    }

    private var resultColor: Color {
        let text = (lab.interpretation ?? "").lowercased()
        if text.contains("not detected") || text.contains("negative") {
            return Color(red: 0.13, green: 0.36, blue: 0.22)
        }
        if text.contains("detected") || text.contains("positive") {
            return Color(red: 0.66, green: 0.15, blue: 0.12)
        }
        return .primary
    }
}

struct VitalsDraftView: View {
    let vitals: DraftVitals

    var body: some View {
        let columns: [GridItem] = [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())]
        LazyVGrid(columns: columns, spacing: 10) {
            if let t = vitals.tempC {
                VitalCell(label: "Temp", value: String(format: "%.1f C", t), icon: "thermometer.medium")
            }
            if let h = vitals.heartRate {
                VitalCell(label: "HR", value: "\(h) bpm", icon: "heart")
            }
            if let r = vitals.respRate {
                VitalCell(label: "RR", value: "\(r) /min", icon: "wind")
            }
            if let s = vitals.spo2 {
                VitalCell(label: "SpO₂", value: "\(s)%", icon: "lungs")
            }
            if let b = vitals.bpSystolic {
                VitalCell(label: "BP", value: "\(b)", icon: "gauge")
            }
        }
    }
}
