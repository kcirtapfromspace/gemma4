// validate.swift
// Headless validator. Runs the same StubInferenceEngine JSON synthesis logic
// across all 5 bundled test cases and scores them against expected codes.
// Not linked into the app — run via: swift validate.swift
//
// NOTE: this file duplicates a tiny bit of the app's string-building logic so
// it can run outside the Xcode target with zero build setup. If the app's
// Stub changes, update the duplicated bits here. This is deliberate — we get
// a dependency-free sanity check script.

import Foundation

// -------- duplicated from the app (kept in sync) --------

enum ValidatorPromptBuilder {
    static let turnSysOpen   = "<|turn>system\n"
    static let turnUserOpen  = "<|turn>user\n"
    static let turnModelOpen = "<|turn>model\n"
    static let turnClose     = "<turn|>\n"

    static func wrapTurns(system: String, user: String) -> String {
        "\(turnSysOpen)\(system)\(turnClose)\(turnUserOpen)\(user)\(turnClose)\(turnModelOpen)"
    }
}

struct ValidatorTestCase {
    let caseId: String
    let user: String
    let expectedConditions: [String]
    let expectedLoincs: [String]
    let expectedRxnorms: [String]
}

let cases: [ValidatorTestCase] = [
    .init(
        caseId: "bench_minimal",
        user: """
Patient: Wei Brown\nGender: M\nDOB: 1958-08-07\nEncounter: 2026-12-05
Dx: Syphilis (SNOMED 76272004)
Lab: Treponema pallidum Ab [Presence] in Serum by Immunoassay (LOINC 20507-0) - Positive [Serum, final]
Meds: penicillin G benzathine 2400000 UNT/injection (RxNorm 105220)
""",
        expectedConditions: ["76272004"],
        expectedLoincs: ["20507-0"],
        expectedRxnorms: ["105220"]
    ),
    .init(
        caseId: "bench_typical_covid",
        user: """
Patient: Maria Garcia\nGender: F\nDOB: 1985-06-14\nEncounter: 2026-03-15
Dx: COVID-19 (SNOMED 840539006)
Lab: SARS-CoV-2 RNA NAA+probe Ql Resp (LOINC 94500-6) - Detected [Respiratory, final]
Vitals: Temp 39.2C, HR 98, RR 22, SpO2 94%, BP 128
Meds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543)
""",
        expectedConditions: ["840539006"],
        expectedLoincs: ["94500-6"],
        expectedRxnorms: ["2599543"]
    ),
    .init(
        caseId: "bench_complex_multi",
        user: """
Patient: Michael Martinez\nGender: M\nDOB: 1958-03-16\nEncounter: 2026-06-24
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
    .init(
        caseId: "bench_meningitis",
        user: """
Patient: Daniel Johnson\nGender: M\nDOB: 1977-03-20\nEncounter: 2025-02-28
Dx: Meningococcal disease (SNOMED 23511006)
Lab: Neisseria meningitidis DNA [Presence] in Specimen by NAA (LOINC 49672-8) - Detected [CSF, preliminary]
Vitals: Temp 38.3C, HR 95, RR 24, SpO2 97%, BP 160
Meds: ceftriaxone 500 MG Injection (RxNorm 1665021)
""",
        expectedConditions: ["23511006"],
        expectedLoincs: ["49672-8"],
        expectedRxnorms: ["1665021"]
    ),
    .init(
        caseId: "bench_negative_lab",
        user: """
Patient: Jennifer Brown\nGender: F\nDOB: 1985-10-05\nEncounter: 2026-12-10
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

// -------- the Stub parsing logic, copied in whole from StubInferenceEngine --------

func firstMatch(in input: String, pattern: String) -> String? {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return nil }
    let range = NSRange(input.startIndex..<input.endIndex, in: input)
    guard let match = regex.firstMatch(in: input, options: [], range: range),
          match.numberOfRanges >= 2,
          let r = Range(match.range(at: 1), in: input) else { return nil }
    return String(input[r])
}

func allLines(in input: String, startingWith prefix: String) -> [String] {
    input.split(separator: "\n").compactMap { line in
        let s = String(line); return s.hasPrefix(prefix) ? s : nil
    }
}

func extractUserBlock(from prompt: String) -> String {
    guard let openRange = prompt.range(of: ValidatorPromptBuilder.turnUserOpen) else { return prompt }
    let afterOpen = prompt[openRange.upperBound...]
    guard let closeRange = afterOpen.range(of: ValidatorPromptBuilder.turnClose) else { return String(afterOpen) }
    return String(afterOpen[..<closeRange.lowerBound])
}

func buildJSON(for userBlock: String) -> String {
    var conditions: [String] = []
    for dx in allLines(in: userBlock, startingWith: "Dx:") {
        if let code = firstMatch(in: dx, pattern: #"SNOMED\s+(\d+)"#) { conditions.append(code) }
    }
    var labs: [String] = []
    for lab in allLines(in: userBlock, startingWith: "Lab:") {
        if let code = firstMatch(in: lab, pattern: #"LOINC\s+([\d-]+)"#) { labs.append(code) }
    }
    var meds: [String] = []
    for med in allLines(in: userBlock, startingWith: "Meds:") {
        if let code = firstMatch(in: med, pattern: #"RxNorm\s+(\d+)"#) { meds.append(code) }
    }
    // Minified JSON in the same key-ordered format the app emits
    var parts: [String] = []
    parts.append("\"conditions\":[\(conditions.map { "\"\($0)\"" }.joined(separator: ","))]")
    parts.append("\"labs\":[\(labs.map { "\"\($0)\"" }.joined(separator: ","))]")
    parts.append("\"medications\":[\(meds.map { "\"\($0)\"" }.joined(separator: ","))]")
    return "{\(parts.joined(separator: ","))}"
}

// -------- Score: per-case, 1 pt per expected code present --------

struct Score {
    let caseId: String
    let conditions: (hit: Int, total: Int, extracted: [String])
    let loincs:     (hit: Int, total: Int, extracted: [String])
    let rxnorms:    (hit: Int, total: Int, extracted: [String])
    var total: Int { conditions.hit + loincs.hit + rxnorms.hit }
    var max: Int { conditions.total + loincs.total + rxnorms.total }
    var rawOutput: String = ""
}

func score(_ tc: ValidatorTestCase) -> Score {
    let prompt = ValidatorPromptBuilder.wrapTurns(
        system: "system prompt placeholder",
        user: tc.user)
    let ub = extractUserBlock(from: prompt)
    let json = buildJSON(for: ub)

    // Re-parse the JSON to get the same set of codes we emitted
    var conds: [String] = [], loincs: [String] = [], rxnorms: [String] = []
    if let m = firstMatch(in: json, pattern: #"\"conditions\":\[([^\]]*)\]"#) {
        conds = m.replacingOccurrences(of: "\"", with: "").split(separator: ",").map(String.init).filter{ !$0.isEmpty }
    }
    if let m = firstMatch(in: json, pattern: #"\"labs\":\[([^\]]*)\]"#) {
        loincs = m.replacingOccurrences(of: "\"", with: "").split(separator: ",").map(String.init).filter{ !$0.isEmpty }
    }
    if let m = firstMatch(in: json, pattern: #"\"medications\":\[([^\]]*)\]"#) {
        rxnorms = m.replacingOccurrences(of: "\"", with: "").split(separator: ",").map(String.init).filter{ !$0.isEmpty }
    }

    let condHits = tc.expectedConditions.filter { conds.contains($0) }.count
    let loincHits = tc.expectedLoincs.filter { loincs.contains($0) }.count
    let rxnormHits = tc.expectedRxnorms.filter { rxnorms.contains($0) }.count

    var s = Score(
        caseId: tc.caseId,
        conditions: (condHits, tc.expectedConditions.count, conds),
        loincs: (loincHits, tc.expectedLoincs.count, loincs),
        rxnorms: (rxnormHits, tc.expectedRxnorms.count, rxnorms))
    s.rawOutput = json
    return s
}

// -------- main --------

var totalHit = 0, totalMax = 0
print("case_id,cond_hit,cond_max,loinc_hit,loinc_max,rx_hit,rx_max,score,max,raw_output")
for tc in cases {
    let s = score(tc)
    totalHit += s.total
    totalMax += s.max
    print("\(s.caseId),\(s.conditions.hit),\(s.conditions.total),\(s.loincs.hit),\(s.loincs.total),\(s.rxnorms.hit),\(s.rxnorms.total),\(s.total),\(s.max),\"\(s.rawOutput)\"")
}
print("# total: \(totalHit)/\(totalMax)")
