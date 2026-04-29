// NewCaseView.swift
// Full-screen sheet for starting a new case. Collects minimum demographics
// + a narrative, then hands the narrative to the Review flow. The
// clinician can also choose a seeded sample narrative — useful for demos
// and when the same patient comes in repeatedly.

import SwiftUI
import SwiftData

struct NewCaseView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var context

    @State private var givenName: String = ""
    @State private var familyName: String = ""
    @State private var gender: String = "U"
    @State private var hasBirthDate: Bool = false
    @State private var birthDate: Date = Calendar.current.date(byAdding: .year, value: -40, to: Date()) ?? Date()
    @State private var postalCode: String = ""
    @State private var facility: String = "Field Clinic, Remote Site 04"
    @State private var narrative: String = ""

    @State private var showingTemplates = false
    @State private var reviewTarget: ClinicalCase?

    /// Cached prior cases for the demographics currently entered. Recomputed
    /// whenever name+DOB change (debounced via .onChange). Drives the
    /// "Returning patient" card.
    @State private var matchingPriors: [ClinicalCase] = []
    /// Pushed when the clinician taps "View timeline" on the returning-
    /// patient card. The destination is keyed off the cached identity hash.
    @State private var timelinePush: String?

    var body: some View {
        NavigationStack {
            Form {
                if !matchingPriors.isEmpty {
                    Section {
                        ReturningPatientCard(priors: matchingPriors,
                                             onViewTimeline: {
                            timelinePush = matchingPriors.first?.patientIdentityHash
                        })
                    }
                }
                Section("Patient") {
                    HStack {
                        TextField("Given name", text: $givenName)
                            .textContentType(.givenName)
                        TextField("Family name", text: $familyName)
                            .textContentType(.familyName)
                    }
                    Picker("Gender", selection: $gender) {
                        Text("Female").tag("F")
                        Text("Male").tag("M")
                        Text("Unspecified").tag("U")
                    }
                    Toggle("Date of birth known", isOn: $hasBirthDate)
                    if hasBirthDate {
                        DatePicker("Date of birth",
                                   selection: $birthDate,
                                   in: ...Date(),
                                   displayedComponents: .date)
                    }
                    HStack {
                        TextField("Postal code", text: $postalCode)
                            .textContentType(.postalCode)
                        TextField("Facility", text: $facility)
                    }
                }

                Section {
                    DictationButton(narrative: $narrative)
                    TextEditor(text: $narrative)
                        .frame(minHeight: 170)
                        .font(.callout)
                        .scrollContentBackground(.hidden)
                        .overlay(alignment: .topLeading) {
                            if narrative.isEmpty {
                                Text("Type or dictate the visit: symptoms, exposures, labs, treatment...")
                                    .foregroundStyle(.tertiary)
                                    .font(.callout)
                                    .padding(.top, 10)
                                    .padding(.leading, 4)
                                    .allowsHitTesting(false)
                            }
                        }
                } header: {
                    HStack {
                        Text("Clinical narrative")
                        Spacer()
                        Button {
                            showingTemplates = true
                        } label: {
                            Label("Sample", systemImage: "sparkles")
                                .font(.caption.weight(.medium))
                        }
                    }
                } footer: {
                    Text("Speech is transcribed on-device. On-device AI then extracts reportable conditions, labs, medications, and vitals. Nothing leaves the device until you review and queue it.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("New Case")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        startReview()
                    } label: {
                        Label("Review with AI", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(narrative.trimmingCharacters(in: .whitespacesAndNewlines).count < 12)
                }
            }
            .confirmationDialog("Sample narratives",
                                isPresented: $showingTemplates,
                                titleVisibility: .visible) {
                ForEach(NarrativeTemplate.all) { t in
                    Button(t.label) {
                        applyTemplate(t)
                    }
                }
                Button("Cancel", role: .cancel) {}
            }
            .sheet(item: $reviewTarget) { target in
                ReviewFlowView(clinicalCase: target)
            }
            .navigationDestination(item: $timelinePush) { hash in
                PatientTimelineView(patientIdentityHash: hash,
                                    patientName: [givenName, familyName]
                                        .filter { !$0.isEmpty }
                                        .joined(separator: " "))
            }
            .onAppear {
                // Screenshot harness: prefill with a sensible demo case.
                let env = ProcessInfo.processInfo.environment
                if env["CLINIQ_PREFILL_NEW_CASE"] == "1",
                   let t = NarrativeTemplate.all.first(where: { $0.id == "covid" })
                {
                    applyTemplate(t)
                }
                if env["CLINIQ_PREFILL_RETURNING"] == "1" {
                    applyReturningPatientPrefill()
                }
                refreshMatchingPriors()
            }
            .onChange(of: givenName) { _, _ in refreshMatchingPriors() }
            .onChange(of: familyName) { _, _ in refreshMatchingPriors() }
            .onChange(of: birthDate) { _, _ in refreshMatchingPriors() }
            .onChange(of: hasBirthDate) { _, _ in refreshMatchingPriors() }
        }
    }

    /// Recompute the (given, family, dob) identity hash and look up any
    /// prior cases for the same patient. Cleared as soon as any required
    /// field is empty.
    private func refreshMatchingPriors() {
        guard hasBirthDate,
              !givenName.trimmed.isEmpty,
              !familyName.trimmed.isEmpty
        else {
            matchingPriors = []
            return
        }
        let hash = LocalPatient.identityHash(given: givenName,
                                             family: familyName,
                                             dob: birthDate)
        matchingPriors = PersistenceController.priorCases(
            for: hash, before: Date.distantFuture, in: context)
    }

    /// Pre-fill the form with Maria Santos so the screenshot harness +
    /// demo presenter can land directly on the returning-patient card.
    private func applyReturningPatientPrefill() {
        givenName = "Maria"
        familyName = "Santos"
        gender = "F"
        hasBirthDate = true
        let cal = Calendar(identifier: .gregorian)
        if let dob = cal.date(from: DateComponents(year: 1985, month: 3, day: 12)) {
            birthDate = dob
        }
        postalCode = "33101"
        narrative = """
Day 14 follow-up. Patient feeling well. Repeat exam normal. Continuing
home BP monitoring. No new complaints.
"""
    }

    private func applyTemplate(_ t: NarrativeTemplate) {
        givenName = t.givenName
        familyName = t.familyName
        gender = t.gender
        hasBirthDate = t.birthDate != nil
        if let bd = t.birthDate { birthDate = bd }
        postalCode = t.postalCode
        narrative = t.narrative
    }

    private func startReview() {
        let patient = Patient(givenName: givenName.trimmed,
                              familyName: familyName.trimmed,
                              gender: gender,
                              birthDate: hasBirthDate ? birthDate : nil,
                              postalCode: postalCode.nonEmpty,
                              facilityName: facility.nonEmpty)
        let c = ClinicalCase(narrative: narrative, status: .draft)
        c.patient = patient
        // Wire the longitudinal join key. Falls back to "" when DOB is
        // absent (anonymous patient — no longitudinal grouping).
        if let hash = LocalPatient.identityHash(from: patient) {
            c.patientIdentityHash = hash
        }
        context.insert(c)
        try? context.save()
        reviewTarget = c
    }
}

// MARK: - Returning patient card

/// Inline card shown above the demographics fields when the entered
/// (given, family, dob) matches priors on this device. Two affordances:
/// view the timeline, or just continue — either way the new case will be
/// auto-linked because the same identity hash is recomputed at save.
struct ReturningPatientCard: View {
    let priors: [ClinicalCase]
    let onViewTimeline: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "person.crop.circle.badge.clock")
                    .foregroundStyle(ClinIQTheme.accent)
                Text("Returning patient")
                    .font(.callout.weight(.semibold))
                Spacer()
            }
            Text("\(priors.count) prior eCR\(priors.count == 1 ? "" : "s") on this device since \(oldestDate)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(action: onViewTimeline) {
                Label("View timeline", systemImage: "list.bullet.rectangle.portrait")
                    .font(.caption.weight(.semibold))
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ClinIQTheme.accent.opacity(0.10),
                    in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(ClinIQTheme.accent.opacity(0.35), lineWidth: 1)
        )
    }

    private var oldestDate: String {
        guard let oldest = priors.map(\.createdAt).min() else { return "—" }
        return oldest.formatted(date: .abbreviated, time: .omitted)
    }
}

private extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
    var nonEmpty: String? {
        let t = trimmed
        return t.isEmpty ? nil : t
    }
}

// MARK: - Sample narratives

struct NarrativeTemplate: Identifiable {
    let id: String
    let label: String
    let givenName: String
    let familyName: String
    let gender: String
    let birthDate: Date?
    let postalCode: String
    let narrative: String

    static let all: [NarrativeTemplate] = {
        let cal = Calendar(identifier: .gregorian)
        return [
            NarrativeTemplate(
                id: "covid",
                label: "COVID-19 with respiratory workup",
                givenName: "Aisha",
                familyName: "Patel",
                gender: "F",
                birthDate: cal.date(from: DateComponents(year: 1990, month: 4, day: 2)),
                postalCode: "97201",
                narrative: """
36 y/o F with 6 days of fever, dry cough, progressive shortness of breath.
Temp 39.1 C, HR 102, RR 24, SpO2 93% on room air. BP 124/78.
SARS-CoV-2 RNA respiratory swab — Detected.
Started oral nirmatrelvir/ritonavir. Counseled on isolation.
"""
            ),
            NarrativeTemplate(
                id: "measles",
                label: "Measles, household cluster",
                givenName: "Luis",
                familyName: "Ramirez",
                gender: "M",
                birthDate: cal.date(from: DateComponents(year: 2019, month: 9, day: 11)),
                postalCode: "80202",
                narrative: """
6 y/o M presented with 4 days of high fever, Koplik spots, coryza, and a
maculopapular rash spreading from face downward. Unvaccinated (MMR refused).
Two siblings in same household now symptomatic.
Measles IgM — Positive. Temp 39.7 C, HR 124, SpO2 96%.
Supportive care, vitamin A. Isolation per jurisdictional guidance.
"""
            ),
            NarrativeTemplate(
                id: "tb",
                label: "Suspected pulmonary TB",
                givenName: "Amara",
                familyName: "Okafor",
                gender: "F",
                birthDate: cal.date(from: DateComponents(year: 1973, month: 11, day: 5)),
                postalCode: "60601",
                narrative: """
52 y/o F with 5 weeks productive cough, weight loss, night sweats. Recent
contact with a known TB patient. Chest X-ray: right-upper-lobe cavitation.
Sputum AFB smear — Positive (3+).
Temp 38.4 C, HR 98, RR 20, SpO2 95%, BP 118/74.
Started 4-drug RIPE regimen. Reported to TB program.
"""
            ),
        ]
    }()
}
