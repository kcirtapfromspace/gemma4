// LongitudinalSeedData.swift
// Three-visit Maria Santos demo seed for the longitudinal "what's new"
// feature. The Python edge agent ships the same patient in
// `scripts/test_cases_longitudinal.jsonl`; the cases here are pre-extracted
// so the demo doesn't need to pipe through the LLM. Keep the SNOMED/LOINC/
// RxNorm codes aligned with the Python file.
//
// Visit 1 (Day 0,  2026-04-15): Fever + cough → Dengue, CBC, acetaminophen
// Visit 2 (Day 5,  2026-04-20): Dengue confirmed (unchanged), CBC unchanged,
//                              CXR added (LOINC 30746-2), acetaminophen unchanged
// Visit 3 (Day 10, 2026-04-25): Dengue, CBC, BP added (LOINC 8480-6),
//                              acetaminophen REMOVED — discontinued

import Foundation

enum LongitudinalSeedData {

    /// Stable patient identity for the three Maria Santos cases. Computed
    /// once and used by both the seed and any UI lookup that needs to
    /// resolve "the demo timeline patient" without re-hashing.
    static let mariaSantosIdentityHash: String = {
        let cal = Calendar(identifier: .gregorian)
        let dob = cal.date(from: DateComponents(year: 1985, month: 3, day: 12)) ?? Date()
        return LocalPatient.identityHash(given: "Maria", family: "Santos", dob: dob)
    }()

    /// Three pre-populated `ClinicalCase` rows for the same patient,
    /// ordered oldest-first. Caller inserts each into the model context.
    static func build() -> [ClinicalCase] {
        let cal = Calendar(identifier: .gregorian)
        let dob = cal.date(from: DateComponents(year: 1985, month: 3, day: 12)) ?? Date()

        let visit1Date = cal.date(from: DateComponents(year: 2026, month: 4, day: 15, hour: 9))
            ?? Date()
        let visit2Date = cal.date(from: DateComponents(year: 2026, month: 4, day: 20, hour: 10))
            ?? Date()
        let visit3Date = cal.date(from: DateComponents(year: 2026, month: 4, day: 25, hour: 11))
            ?? Date()

        let identityHash = mariaSantosIdentityHash

        // --- Visit 1 — initial presentation ---
        let v1 = ClinicalCase(narrative: """
Patient: Maria Santos, 41 y/o F, seen 2026-04-15 at Field Clinic Remote Site 04.
Three-day history of high fever (39.4 C), retro-orbital headache, dry cough,
and myalgia. Recently returned from coastal travel. Started acetaminophen
650 mg q6h. CBC drawn — workup for dengue.
""",
                              status: .submitted,
                              createdAt: visit1Date)
        v1.patient = makePatient(dob: dob)
        v1.patientIdentityHash = identityHash
        v1.conditions = [
            ExtractedCondition(code: "38362002",
                               displayName: "Dengue fever",
                               reviewState: .confirmed)
        ]
        v1.labs = [
            ExtractedLab(code: "58410-2",
                         displayName: "Complete blood count panel",
                         interpretation: "Pending",
                         reviewState: .confirmed)
        ]
        v1.medications = [
            ExtractedMedication(code: "161",
                                displayName: "Acetaminophen 650 mg",
                                reviewState: .confirmed)
        ]
        v1.vitals = Vitals(tempC: 39.4, heartRate: 102, respRate: 20, spo2: 97)

        // --- Visit 2 — five days later, dengue confirmed, chest X-ray added ---
        let v2 = ClinicalCase(narrative: """
Patient: Maria Santos, returns Day 5 (2026-04-20). Persistent fever (38.7 C),
new productive cough, no chest pain. Dengue NS1 confirmed positive. CBC
unchanged from prior. Chest X-ray ordered — patchy right-lower-lobe
infiltrate, consistent with bacterial superinfection. Continuing
acetaminophen for fever control.
""",
                              status: .submitted,
                              createdAt: visit2Date)
        v2.patient = makePatient(dob: dob)
        v2.patientIdentityHash = identityHash
        v2.conditions = [
            ExtractedCondition(code: "38362002",
                               displayName: "Dengue fever",
                               reviewState: .confirmed)
        ]
        v2.labs = [
            ExtractedLab(code: "58410-2",
                         displayName: "Complete blood count panel",
                         interpretation: "Stable",
                         reviewState: .confirmed),
            ExtractedLab(code: "30746-2",
                         displayName: "Chest X-ray",
                         interpretation: "Patchy infiltrate",
                         reviewState: .confirmed)
        ]
        v2.medications = [
            ExtractedMedication(code: "161",
                                displayName: "Acetaminophen 650 mg",
                                reviewState: .confirmed)
        ]
        v2.vitals = Vitals(tempC: 38.7, heartRate: 96, respRate: 22, spo2: 95)

        // --- Visit 3 — ten days, BP elevated, acetaminophen discontinued ---
        let v3 = ClinicalCase(narrative: """
Patient: Maria Santos, follow-up Day 10 (2026-04-25). Afebrile. Cough
resolving. Newly elevated BP 148 mmHg systolic — first time recorded;
likely post-viral, will recheck in 1 week. Discontinuing acetaminophen
(no longer febrile). Dengue resolving, no warning signs. CBC trending
toward baseline.
""",
                              status: .draft,
                              createdAt: visit3Date)
        v3.patient = makePatient(dob: dob)
        v3.patientIdentityHash = identityHash
        v3.conditions = [
            ExtractedCondition(code: "38362002",
                               displayName: "Dengue fever",
                               reviewState: .confirmed)
        ]
        v3.labs = [
            ExtractedLab(code: "58410-2",
                         displayName: "Complete blood count panel",
                         interpretation: "Improving",
                         reviewState: .confirmed)
        ]
        // No medications — acetaminophen has been discontinued (REMOVED axis).
        v3.medications = []
        v3.vitals = Vitals(tempC: 36.8, heartRate: 78, respRate: 16, spo2: 99,
                           bpSystolic: 148, bpDiastolic: 92)

        return [v1, v2, v3]
    }

    private static func makePatient(dob: Date) -> Patient {
        Patient(givenName: "Maria",
                familyName: "Santos",
                gender: "F",
                birthDate: dob,
                postalCode: "33101",
                facilityName: "Field Clinic, Remote Site 04")
    }
}
