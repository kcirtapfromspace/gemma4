// validate_preparser.swift
// Headless 14-case bench for EicrPreparser. Mirrors the Python bench in
// apps/mobile/convert/regex_preparser.py — both should agree at 42/42.
//
// Run via:
//   swift apps/mobile/ios-app/validate_preparser.swift \
//        apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/EicrPreparser.swift \
//        apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/LookupTable.swift \
//        scripts/test_cases.jsonl scripts/test_cases_adversarial.jsonl
//
// Stripped-down ParsedExtraction siblings live below — just the fields
// EicrPreparser writes to, no SwiftUI/SwiftData dependencies.

import Foundation

// MARK: - Minimal Parsed* types matching the app's
// EicrPreparser only writes (code, system, display) per entity, so the test
// harness can stub the rest as nil/no-ops.

struct ParsedCondition {
    var code: String
    var system: String
    var display: String
}

struct ParsedLab {
    var code: String
    var system: String
    var display: String
    var interpretation: String?
    var value: Double?
    var unit: String?
}

struct ParsedMedication {
    var code: String
    var system: String
    var display: String
}

struct ParsedVitals {
    var tempC: Double?
    var heartRate: Int?
    var respRate: Int?
    var spo2: Int?
    var bpSystolic: Int?
}

struct ParsedExtraction {
    var patientGender: String?
    var patientBirthDate: Date?
    var encounterDate: Date?
    var conditions: [ParsedCondition] = []
    var labs: [ParsedLab] = []
    var medications: [ParsedMedication] = []
    var vitals: ParsedVitals?
    var raw: String = ""
}

// MARK: - Test harness

struct BenchCase {
    let caseId: String
    let user: String
    let expectedConditions: [String]
    let expectedLoincs: [String]
    let expectedRxnorms: [String]
}

func loadCases(from path: String) -> [BenchCase] {
    guard let data = FileManager.default.contents(atPath: path),
          let text = String(data: data, encoding: .utf8) else {
        FileHandle.standardError.write("ERR: cannot read \(path)\n".data(using: .utf8)!)
        return []
    }
    var out: [BenchCase] = []
    for ln in text.split(separator: "\n") {
        let s = String(ln).trimmingCharacters(in: .whitespaces)
        if s.isEmpty { continue }
        guard let d = s.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { continue }
        out.append(BenchCase(
            caseId: obj["case_id"] as? String ?? "?",
            user: obj["user"] as? String ?? "",
            expectedConditions: obj["expected_conditions"] as? [String] ?? [],
            expectedLoincs: obj["expected_loincs"] as? [String] ?? [],
            expectedRxnorms: obj["expected_rxnorms"] as? [String] ?? []
        ))
    }
    return out
}

func score(_ extracted: ParsedExtraction, _ tc: BenchCase) -> (matched: Int, expected: Int) {
    let condCodes = Set(extracted.conditions.map { $0.code })
    let labCodes = Set(extracted.labs.map { $0.code })
    let medCodes = Set(extracted.medications.map { $0.code })
    let m = tc.expectedConditions.filter { condCodes.contains($0) }.count
        + tc.expectedLoincs.filter { labCodes.contains($0) }.count
        + tc.expectedRxnorms.filter { medCodes.contains($0) }.count
    let e = tc.expectedConditions.count + tc.expectedLoincs.count + tc.expectedRxnorms.count
    return (m, e)
}

// MARK: - Main

@main
enum ValidatePreparserCLI {
    static func main() {
        let args = CommandLine.arguments.dropFirst()
        guard !args.isEmpty else {
            FileHandle.standardError.write("usage: <jsonl> [<jsonl> ...]\n".data(using: .utf8)!)
            exit(2)
        }

        var cases: [BenchCase] = []
        for p in args { cases.append(contentsOf: loadCases(from: p)) }
        print("Loaded \(cases.count) cases\n")

        var totalMatched = 0
        var totalExpected = 0
        var perfect = 0
        for (i, tc) in cases.enumerated() {
            let res = EicrPreparser.extract(tc.user)
            let s = score(res, tc)
            totalMatched += s.matched
            totalExpected += s.expected
            if s.expected > 0 && s.matched == s.expected { perfect += 1 }
            let mark = s.matched == s.expected ? "OK " : "MISS"
            print("  \(mark) \(i + 1)/\(cases.count) \(tc.caseId)  \(s.matched)/\(s.expected)")
            if s.matched != s.expected {
                let condCodes = res.conditions.map { $0.code }
                let labCodes = res.labs.map { $0.code }
                let medCodes = res.medications.map { $0.code }
                print("        extracted SNOMED: \(condCodes)")
                print("        extracted LOINC : \(labCodes)")
                print("        extracted RxNorm: \(medCodes)")
            }
        }

        let agg = totalExpected > 0 ? Double(totalMatched) / Double(totalExpected) : 0.0
        print(String(format: "\nAggregate: %d/%d = %.3f (%d/%d cases perfect)",
                     totalMatched, totalExpected, agg, perfect, cases.count))
        exit(totalMatched == totalExpected ? 0 : 1)
    }
}
