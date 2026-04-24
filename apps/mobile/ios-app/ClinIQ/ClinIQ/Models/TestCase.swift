// TestCase.swift
// The five bundled test cases from `scripts/test_cases.jsonl` that we run
// through the extractor during simulator validation. Copying the JSONL in as
// a literal struct avoids reliance on bundled resources at build time — any
// change to the test cases is a one-file diff here.

import Foundation

struct TestCase: Identifiable, Hashable {
    let caseId: String
    let description: String
    let user: String
    let expectedConditions: [String]
    let expectedLoincs: [String]
    let expectedRxnorms: [String]

    var id: String { caseId }
}

extension TestCase {
    static let bundled: [TestCase] = [
        TestCase(
            caseId: "bench_minimal",
            description: "Minimal: single condition, no vitals",
            user: """
Patient: Wei Brown
Gender: M
DOB: 1958-08-07
Race: Asian
Ethnicity: Hispanic or Latino
Location: Seattle, WA 98101
Phone: +1-995-555-1144
Facility: Denver Health Medical Center (NPI: 1234567800)
Encounter: 2026-12-05
Reason: painless chancre for 39.0 days, regional lymphadenopathy, maculopapular rash on palms and soles
Dx: Syphilis (SNOMED 76272004)
Lab: Treponema pallidum Ab [Presence] in Serum by Immunoassay (LOINC 20507-0) - Positive [Serum, final]
Meds: penicillin G benzathine 2400000 UNT/injection (RxNorm 105220)
""",
            expectedConditions: ["76272004"],
            expectedLoincs: ["20507-0"],
            expectedRxnorms: ["105220"]
        ),
        TestCase(
            caseId: "bench_typical_covid",
            description: "Typical: COVID case with vitals, labs, meds",
            user: """
Patient: Maria Garcia
Gender: F
DOB: 1985-06-14
Race: White
Ethnicity: Hispanic or Latino
Location: Denver, CO 80202
Phone: +1-303-555-0142
Facility: Denver Health Medical Center (NPI: 1234567800)
Encounter: 2026-03-15
Reason: fever (39.2C), dry cough for 5 days, fatigue, shortness of breath
Dx: COVID-19 (SNOMED 840539006)
Lab: SARS-CoV-2 RNA NAA+probe Ql Resp (LOINC 94500-6) - Detected [Respiratory, final]
Vitals: Temp 39.2C, HR 98, RR 22, SpO2 94%, BP 128
Meds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543)
""",
            expectedConditions: ["840539006"],
            expectedLoincs: ["94500-6"],
            expectedRxnorms: ["2599543"]
        ),
        TestCase(
            caseId: "bench_complex_multi",
            description: "Complex: multi-condition, multi-lab, multi-med",
            user: """
Patient: Michael Martinez
Gender: M
DOB: 1958-03-16
Race: White
Ethnicity: Hispanic or Latino
Location: Chicago, IL 60601
Phone: +1-923-555-3884
Facility: Hennepin Healthcare (NPI: 1234567809)
Encounter: 2026-06-24
Reason: fever (40.2C), weight loss, night sweats for 5 weeks, lymphadenopathy, oral thrush
Dx: HIV infection (SNOMED 86406008)
Lab: HIV 1 and 2 Ag+Ab [Presence] in Serum by Immunoassay (LOINC 75622-1) - Positive [Serum, final]
Lab: Complete blood count (LOINC 57021-8) - WBC 2.1 x10^3/uL [Blood, final]
Lab: CD4+ T cells [#/volume] in Blood (LOINC 24467-3) - 180 cells/uL [Blood, final]
Vitals: Temp 40.0C, HR 89, RR 18, SpO2 90%, BP 97
Meds: bictegravir 50 MG / emtricitabine 200 MG / tenofovir alafenamide 25 MG (RxNorm 1999563)
Meds: fluconazole 200 MG Oral Tablet (RxNorm 197696)
""",
            expectedConditions: ["86406008"],
            expectedLoincs: ["75622-1", "57021-8", "24467-3"],
            expectedRxnorms: ["1999563", "197696"]
        ),
        TestCase(
            caseId: "bench_meningitis",
            description: "Urgent: meningococcal with CSF results",
            user: """
Patient: Daniel Johnson
Gender: M
DOB: 1977-03-20
Race: American Indian or Alaska Native
Ethnicity: Hispanic or Latino
Location: Miami, FL 33101
Phone: +1-438-555-1530
Facility: Northwestern Memorial Hospital (NPI: 1234567801)
Encounter: 2025-02-28
Reason: high fever (38.9C), severe headache, stiff neck for 10 days, petechial rash, photophobia
Dx: Meningococcal disease (SNOMED 23511006)
Lab: Neisseria meningitidis DNA [Presence] in Specimen by NAA (LOINC 49672-8) - Detected [CSF, preliminary]
Vitals: Temp 38.3C, HR 95, RR 24, SpO2 97%, BP 160
Meds: ceftriaxone 500 MG Injection (RxNorm 1665021)
""",
            expectedConditions: ["23511006"],
            expectedLoincs: ["49672-8"],
            expectedRxnorms: ["1665021"]
        ),
        TestCase(
            caseId: "bench_negative_lab",
            description: "Edge case: negative lab result with condition suspected",
            user: """
Patient: Jennifer Brown
Gender: F
DOB: 1985-10-05
Race: American Indian or Alaska Native
Ethnicity: Non Hispanic or Latino
Location: Portland, OR 97201
Phone: +1-616-555-1494
Facility: Mass General Brigham (NPI: 1234567806)
Encounter: 2026-12-10
Reason: fatigue, abdominal discomfort, nausea for 40.2 weeks, mild jaundice
Dx: Hepatitis C (SNOMED 50711007)
Lab: Hepatitis C virus Ab [Presence] in Serum (LOINC 11259-9) - Not detected [Serum, final]
Vitals: Temp 39.7C, HR 113, RR 27, SpO2 97%, BP 96
Meds: sofosbuvir 400 MG / velpatasvir 100 MG (RxNorm 1940261)
""",
            expectedConditions: ["50711007"],
            expectedLoincs: ["11259-9"],
            expectedRxnorms: ["1940261"]
        ),
    ]
}
