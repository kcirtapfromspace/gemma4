// ExtractionParser.swift
// Converts the model's minified JSON output into strongly-typed structured
// entities the UI can review. Tolerant: if the model emits a trailing
// `<turn|>` delimiter, slight markdown fences, or extra prose, we still
// attempt to find a JSON object inside the output.
//
// This is pure text-in / value-out; side effects (writing to the store)
// live in ExtractionService.apply(...) so this can be unit-tested in
// isolation later.

import Foundation

struct ParsedExtraction {
    var patientGender: String?     // "M" / "F" / "U"
    var patientBirthDate: Date?
    var encounterDate: Date?
    var conditions: [ParsedCondition] = []
    var labs: [ParsedLab] = []
    var medications: [ParsedMedication] = []
    var vitals: ParsedVitals?
    var raw: String = ""
}

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

    var isEmpty: Bool {
        tempC == nil && heartRate == nil && respRate == nil && spo2 == nil && bpSystolic == nil
    }
}

enum ExtractionParser {
    /// Best-effort parse: returns whatever we could recover. Empty
    /// extraction (no conditions / labs / meds) is a valid outcome.
    static func parse(_ rawOutput: String) -> ParsedExtraction {
        var parsed = ParsedExtraction()
        parsed.raw = rawOutput

        guard let obj = Self.findJSONObject(in: rawOutput) else {
            return parsed
        }

        // patient
        if let p = obj["patient"] as? [String: Any] {
            parsed.patientGender = (p["gender"] as? String)?.uppercased()
            if let s = p["birth_date"] as? String {
                parsed.patientBirthDate = Self.parseDate(s)
            }
        }
        if let s = obj["encounter_date"] as? String {
            parsed.encounterDate = Self.parseDate(s)
        }

        // conditions
        if let arr = obj["conditions"] as? [[String: Any]] {
            parsed.conditions = arr.compactMap { entry in
                guard let code = entry["code"] as? String,
                      !code.isEmpty,
                      let display = entry["display"] as? String else { return nil }
                return ParsedCondition(code: code,
                                       system: (entry["system"] as? String) ?? "SNOMED",
                                       display: display)
            }
        }

        // labs
        if let arr = obj["labs"] as? [[String: Any]] {
            parsed.labs = arr.compactMap { entry in
                guard let code = entry["code"] as? String,
                      !code.isEmpty,
                      let display = entry["display"] as? String else { return nil }
                let v: Double? = (entry["value"] as? Double)
                    ?? (entry["value"] as? NSNumber)?.doubleValue
                return ParsedLab(code: code,
                                 system: (entry["system"] as? String) ?? "LOINC",
                                 display: display,
                                 interpretation: (entry["interpretation"] as? String)?.capitalized,
                                 value: v,
                                 unit: entry["unit"] as? String)
            }
        }

        // medications
        if let arr = obj["medications"] as? [[String: Any]] {
            parsed.medications = arr.compactMap { entry in
                guard let code = entry["code"] as? String,
                      !code.isEmpty,
                      let display = entry["display"] as? String else { return nil }
                return ParsedMedication(code: code,
                                        system: (entry["system"] as? String) ?? "RxNorm",
                                        display: display)
            }
        }

        // vitals
        if let v = obj["vitals"] as? [String: Any] {
            let vitals = ParsedVitals(
                tempC: (v["temp_c"] as? NSNumber)?.doubleValue,
                heartRate: (v["hr"] as? NSNumber)?.intValue,
                respRate: (v["rr"] as? NSNumber)?.intValue,
                spo2: (v["spo2"] as? NSNumber)?.intValue,
                bpSystolic: (v["bp_systolic"] as? NSNumber)?.intValue)
            parsed.vitals = vitals.isEmpty ? nil : vitals
        }

        return parsed
    }

    // MARK: - Helpers

    private static func findJSONObject(in text: String) -> [String: Any]? {
        // Look for the outermost balanced `{...}`. Model output may include
        // extra tokens (turn delimiter, trailing whitespace).
        let trimmed = text
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .replacingOccurrences(of: "<turn|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let chars = Array(trimmed)
        guard let start = chars.firstIndex(of: "{") else { return nil }

        var depth = 0
        var endIndex: Int?
        var inString = false
        var escape = false
        for i in start..<chars.count {
            let c = chars[i]
            if escape { escape = false; continue }
            if c == "\\" { escape = true; continue }
            if c == "\"" { inString.toggle(); continue }
            if inString { continue }
            if c == "{" { depth += 1 }
            if c == "}" {
                depth -= 1
                if depth == 0 { endIndex = i; break }
            }
        }
        guard let end = endIndex else { return nil }
        let jsonSlice = String(chars[start...end])
        guard let data = jsonSlice.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func parseDate(_ s: String) -> Date? {
        let fmt = DateFormatter()
        fmt.calendar = Calendar(identifier: .gregorian)
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.dateFormat = "yyyy-MM-dd"
        return fmt.date(from: s)
    }
}
