// DemoSeed.swift
// Seeds the SwiftData store with 4 realistic case examples on first launch
// so the PoC demo has visible data without waiting for an AI extraction run.
// Mix of draft / pending / submitted states gives the screenshot a mix.

import Foundation

enum DemoSeed {
    /// Returns fully-populated `ClinicalCase` instances ready to insert into
    /// the model context.
    static func build() -> [ClinicalCase] {
        let cal = Calendar(identifier: .gregorian)
        let now = Date()

        // --- Submitted: COVID ---
        let covid = ClinicalCase(narrative: """
Patient: Maria Garcia, 40 y/o F, seen 2026-03-15 at Denver Health.
Five days of dry cough, fever (39.2C), fatigue, shortness of breath.
SARS-CoV-2 RNA respiratory — Detected. SpO2 94% on room air.
Started nirmatrelvir/ritonavir. Home isolation instructions given.
""",
                                 status: .submitted,
                                 createdAt: cal.date(byAdding: .hour, value: -6, to: now) ?? now)
        covid.patient = Patient(givenName: "Maria",
                                familyName: "Garcia",
                                gender: "F",
                                birthDate: cal.date(from: DateComponents(year: 1985, month: 6, day: 14)),
                                postalCode: "80202",
                                facilityName: "Denver Health Medical Center")
        covid.conditions = [ExtractedCondition(code: "840539006",
                                               displayName: "COVID-19",
                                               reviewState: .confirmed)]
        covid.labs = [ExtractedLab(code: "94500-6",
                                   displayName: "SARS-CoV-2 RNA (Resp)",
                                   interpretation: "Detected",
                                   reviewState: .confirmed)]
        covid.medications = [ExtractedMedication(code: "2599543",
                                                 displayName: "Nirmatrelvir 150 mg / Ritonavir 100 mg",
                                                 reviewState: .confirmed)]
        covid.vitals = Vitals(tempC: 39.2, heartRate: 98, respRate: 22, spo2: 94, bpSystolic: 128, bpDiastolic: 82)
        covid.tokensGenerated = 342
        covid.elapsedSeconds = 41.2
        covid.tokensPerSecond = 8.3
        covid.syncHistory = [SyncRecord(attemptedAt: cal.date(byAdding: .hour, value: -25, to: now) ?? now,
                                        succeeded: true,
                                        endpoint: "http://localhost:8080/reports",
                                        message: "202 Accepted — ref PH-2026-0316-A01")]

        // --- Pending: meningococcal meningitis, outbox ---
        let meningitis = ClinicalCase(narrative: """
Patient: Daniel Johnson, 48 y/o M, presents 2026-04-22 with 10-day history of
high fever, severe headache, stiff neck, petechial rash, and photophobia.
CSF preliminary: Neisseria meningitidis DNA detected. BP 160/96, HR 95.
Empiric ceftriaxone 500 mg IV initiated. Isolation precautions, family contact
tracing in progress. Reportable — notifiable within 24 h.
""",
                                      status: .pending,
                                      createdAt: cal.date(byAdding: .hour, value: -3, to: now) ?? now)
        meningitis.patient = Patient(givenName: "Daniel",
                                     familyName: "Johnson",
                                     gender: "M",
                                     birthDate: cal.date(from: DateComponents(year: 1977, month: 3, day: 20)),
                                     postalCode: "33101",
                                     facilityName: "Field Clinic, Remote Site 04")
        meningitis.conditions = [ExtractedCondition(code: "23511006",
                                                    displayName: "Meningococcal disease",
                                                    reviewState: .confirmed)]
        meningitis.labs = [ExtractedLab(code: "49672-8",
                                        displayName: "N. meningitidis DNA (CSF)",
                                        interpretation: "Detected",
                                        reviewState: .confirmed)]
        meningitis.medications = [ExtractedMedication(code: "1665021",
                                                      displayName: "Ceftriaxone 500 mg Injection",
                                                      reviewState: .confirmed)]
        meningitis.vitals = Vitals(tempC: 38.9, heartRate: 95, respRate: 24, spo2: 97, bpSystolic: 160, bpDiastolic: 96)
        meningitis.tokensGenerated = 380
        meningitis.elapsedSeconds = 48.9
        meningitis.tokensPerSecond = 7.8

        // --- Submitted: HIV new diagnosis ---
        let hiv = ClinicalCase(narrative: """
Patient: Michael Martinez, 68 y/o M. Five weeks of weight loss, night sweats,
low-grade fevers, lymphadenopathy, oral thrush. Labs: HIV Ag+Ab positive.
CD4 180 cells/uL. WBC 2.1. Started bictegravir/emtricitabine/tenofovir AF.
Fluconazole 200 mg for thrush. Reportable HIV infection, new diagnosis.
""",
                               status: .submitted,
                               createdAt: cal.date(byAdding: .day, value: -4, to: now) ?? now)
        hiv.patient = Patient(givenName: "Michael",
                              familyName: "Martinez",
                              gender: "M",
                              birthDate: cal.date(from: DateComponents(year: 1958, month: 3, day: 16)),
                              postalCode: "60601",
                              facilityName: "Hennepin Healthcare (CHW outreach)")
        hiv.conditions = [ExtractedCondition(code: "86406008",
                                             displayName: "HIV infection",
                                             reviewState: .confirmed)]
        hiv.labs = [
            ExtractedLab(code: "75622-1",
                         displayName: "HIV 1+2 Ag+Ab (Serum)",
                         interpretation: "Positive",
                         reviewState: .confirmed),
            ExtractedLab(code: "24467-3",
                         displayName: "CD4+ T cells",
                         value: 180, unit: "cells/uL",
                         reviewState: .confirmed),
        ]
        hiv.medications = [
            ExtractedMedication(code: "1999563",
                                displayName: "Bictegravir/Emtricitabine/Tenofovir AF",
                                reviewState: .confirmed),
            ExtractedMedication(code: "197696",
                                displayName: "Fluconazole 200 mg",
                                reviewState: .confirmed),
        ]
        hiv.vitals = Vitals(tempC: 40.0, heartRate: 89, respRate: 18, spo2: 90, bpSystolic: 97, bpDiastolic: 62)
        hiv.tokensGenerated = 452
        hiv.elapsedSeconds = 56.1
        hiv.tokensPerSecond = 8.1
        hiv.syncHistory = [SyncRecord(attemptedAt: cal.date(byAdding: .day, value: -4, to: now) ?? now,
                                      succeeded: true,
                                      endpoint: "http://localhost:8080/reports",
                                      message: "202 Accepted — ref PH-2026-0419-B17")]

        // --- Draft: user started typing, hasn't extracted yet ---
        let drafting = ClinicalCase(narrative: """
Patient: Jennifer Brown, 40 y/o F. Forty-week history of fatigue,
abdominal discomfort, nausea, mild jaundice. Travel to endemic region.
""",
                                    status: .draft,
                                    createdAt: cal.date(byAdding: .minute, value: -18, to: now) ?? now)
        drafting.patient = Patient(givenName: "Jennifer",
                                   familyName: "Brown",
                                   gender: "F",
                                   birthDate: cal.date(from: DateComponents(year: 1985, month: 10, day: 5)),
                                   postalCode: "97201",
                                   facilityName: "Field Clinic, Remote Site 04")

        return [covid, meningitis, hiv, drafting]
    }
}
