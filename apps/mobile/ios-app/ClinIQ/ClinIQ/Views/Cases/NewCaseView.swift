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

    var body: some View {
        NavigationStack {
            Form {
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
            .onAppear {
                // Screenshot harness: prefill with a sensible demo case.
                let env = ProcessInfo.processInfo.environment
                if env["CLINIQ_PREFILL_NEW_CASE"] == "1",
                   let t = NarrativeTemplate.all.first(where: { $0.id == "covid" })
                {
                    applyTemplate(t)
                }
            }
        }
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
        context.insert(c)
        try? context.save()
        reviewTarget = c
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
