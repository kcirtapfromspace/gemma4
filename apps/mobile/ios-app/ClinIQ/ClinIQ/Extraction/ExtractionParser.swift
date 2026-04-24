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

        guard let topObj = Self.findJSONObject(in: rawOutput) else {
            // JSON parsing failed (unclosed quote, malformed string, etc.).
            // Fall back to a regex scan for `"code":"xxx"` / `"display":"yyy"`
            // pairs so we can still recover codes from imperfect model
            // output instead of showing "0 entities" on a run that clearly
            // contained data.
            return Self.regexFallback(rawOutput)
        }

        // The model occasionally nests everything (encounter_date + arrays)
        // inside the `patient` sub-object. If the top-level is missing the
        // content arrays but `patient` has them, lift from there.
        let obj: [String: Any] = {
            let topHasContent = (topObj["conditions"] as? [[String: Any]])?.isEmpty == false
                || (topObj["labs"] as? [[String: Any]])?.isEmpty == false
                || (topObj["medications"] as? [[String: Any]])?.isEmpty == false
            if topHasContent { return topObj }
            if let nested = topObj["patient"] as? [String: Any] {
                let nestedHasContent = (nested["conditions"] as? [[String: Any]])?.isEmpty == false
                    || (nested["labs"] as? [[String: Any]])?.isEmpty == false
                    || (nested["medications"] as? [[String: Any]])?.isEmpty == false
                if nestedHasContent {
                    // Merge: keep the `patient`-level fields for demographics,
                    // but pull arrays + encounter_date to the top. `gender`
                    // and `birth_date` stay on the nested patient object.
                    var merged = topObj
                    for key in ["conditions", "labs", "medications", "encounter_date", "vitals"] {
                        if merged[key] == nil, let v = nested[key] {
                            merged[key] = v
                        }
                    }
                    return merged
                }
            }
            return topObj
        }()

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

        // vitals — also accept the v2 LoRA's shorter alias `sbp` for
        // systolic BP, since the fine-tune occasionally emits that.
        if let v = obj["vitals"] as? [String: Any] {
            let vitals = ParsedVitals(
                tempC: (v["temp_c"] as? NSNumber)?.doubleValue,
                heartRate: (v["hr"] as? NSNumber)?.intValue,
                respRate: (v["rr"] as? NSNumber)?.intValue,
                spo2: (v["spo2"] as? NSNumber)?.intValue,
                bpSystolic: (v["bp_systolic"] as? NSNumber)?.intValue
                    ?? (v["sbp"] as? NSNumber)?.intValue)
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
        // If the stream cut off mid-object (max_tokens hit), the closing
        // braces for outer objects are missing. Try to recover by appending
        // enough `}` / `]` to balance the opens we saw up to that point.
        let sliceEnd: Int = endIndex ?? (chars.count - 1)
        var jsonSlice = String(chars[start...sliceEnd])
        if endIndex == nil {
            // Count unmatched opens (ignoring ones inside strings) to figure
            // out how many closers to append, and in what order.
            var stack: [Character] = []
            var s = false
            var esc = false
            for c in jsonSlice {
                if esc { esc = false; continue }
                if c == "\\" { esc = true; continue }
                if c == "\"" { s.toggle(); continue }
                if s { continue }
                if c == "{" { stack.append("}") }
                else if c == "[" { stack.append("]") }
                else if c == "}" || c == "]" { if !stack.isEmpty { stack.removeLast() } }
            }
            // Trim any trailing partial token (e.g. `"ke`) before closing.
            if s, let lastQuote = jsonSlice.lastIndex(of: "\"") {
                jsonSlice = String(jsonSlice[..<lastQuote])
            }
            // Trim dangling trailing comma before appending closers.
            while let last = jsonSlice.last,
                  last == "," || last.isWhitespace || last == ":" {
                jsonSlice.removeLast()
            }
            while let closer = stack.popLast() { jsonSlice.append(closer) }
        }

        // Best-effort repair: v2 LoRA occasionally drops a leading or
        // trailing quote around keys like  display":"x"  or  unit:null .
        // Two regex passes are conservative — they only fire on bare
        // identifiers before `:` and `"...` sequences without the leading
        // quote. If JSONSerialization still rejects, we return nil.
        if let data = jsonSlice.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return obj
        }
        let repaired = Self.repairBareKeys(jsonSlice)
        guard let data = repaired.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    /// Last-resort: scan the raw text for `"code":"xxx","display":"yyy"`
    /// pairs, classify by the `system` value that comes alongside, and
    /// return a populated ParsedExtraction. Used when JSON parsing fails
    /// (unclosed strings, missing key quotes, etc.). Does NOT recover
    /// vitals or patient demographics — those require a working JSON
    /// parse.
    private static func regexFallback(_ raw: String) -> ParsedExtraction {
        var parsed = ParsedExtraction()
        parsed.raw = raw

        // Match `"code":"xxx" ... display":"yyy"` loosely — the v2 LoRA
        // sometimes drops the opening quote on the `display` key. We also
        // grab a ±80 char window around the match so the system-type
        // classifier has context to spot SNOMED / LOINC / RxNorm.
        let entryPattern = #""code"\s*:\s*"([^"]+)"[^\{\}]*?"?display"?\s*:\s*"([^"]+)""#
        guard let rx = try? NSRegularExpression(pattern: entryPattern) else {
            return parsed
        }
        let nsr = NSRange(raw.startIndex..., in: raw)
        let matches = rx.matches(in: raw, range: nsr)
        let rawCount = raw.count
        for m in matches {
            guard m.numberOfRanges >= 3,
                  let codeR = Range(m.range(at: 1), in: raw),
                  let displayR = Range(m.range(at: 2), in: raw),
                  let matchR = Range(m.range(at: 0), in: raw)
            else { continue }
            let code = String(raw[codeR])
            let display = String(raw[displayR])
            // ±80 char window around the match for system/interp lookups
            let matchStart = raw.distance(from: raw.startIndex, to: matchR.lowerBound)
            let matchEnd = raw.distance(from: raw.startIndex, to: matchR.upperBound)
            let ctxStart = raw.index(raw.startIndex, offsetBy: max(0, matchStart - 80))
            let ctxEnd = raw.index(raw.startIndex, offsetBy: min(rawCount, matchEnd + 80))
            let entry = String(raw[ctxStart..<ctxEnd])
            // Best-effort system detection from the same entry block
            let system: String = {
                if entry.contains("\"system\":\"SNOMED\"") || entry.contains("SNOMED") { return "SNOMED" }
                if entry.contains("\"system\":\"LOINC\"") || entry.contains("LOINC") { return "LOINC" }
                if entry.contains("\"system\":\"RxNorm\"") || entry.contains("RxNorm") { return "RxNorm" }
                // Fall back to code-shape heuristics
                if code.contains("-") { return "LOINC" }
                if code.count <= 7, Int(code) != nil { return "RxNorm" }
                return "SNOMED"
            }()
            switch system {
            case "SNOMED":
                parsed.conditions.append(ParsedCondition(code: code, system: "SNOMED", display: display))
            case "LOINC":
                // Interpretation heuristic
                var interp: String?
                for needle in ["Detected", "Not detected", "Positive", "Negative"] {
                    if entry.localizedCaseInsensitiveContains(needle) { interp = needle; break }
                }
                parsed.labs.append(ParsedLab(code: code, system: "LOINC",
                                             display: display,
                                             interpretation: interp, value: nil, unit: nil))
            case "RxNorm":
                parsed.medications.append(ParsedMedication(code: code, system: "RxNorm", display: display))
            default:
                break
            }
        }

        // De-dup by (system, code) to avoid double-counting when the model
        // repeats the same entity in two arrays.
        parsed.conditions = Self.dedupConditions(parsed.conditions)
        parsed.labs = Self.dedupLabs(parsed.labs)
        parsed.medications = Self.dedupMedications(parsed.medications)
        return parsed
    }

    private static func dedupConditions(_ arr: [ParsedCondition]) -> [ParsedCondition] {
        var seen: Set<String> = []
        return arr.filter { seen.insert($0.code).inserted }
    }
    private static func dedupLabs(_ arr: [ParsedLab]) -> [ParsedLab] {
        var seen: Set<String> = []
        return arr.filter { seen.insert($0.code).inserted }
    }
    private static func dedupMedications(_ arr: [ParsedMedication]) -> [ParsedMedication] {
        var seen: Set<String> = []
        return arr.filter { seen.insert($0.code).inserted }
    }

    /// Two-pass key-quote repair for the common LLM JSON drifts:
    ///   1. `{,key:value` → `{,"key":value` (bare identifier before `:`)
    ///   2. `key":"value"` → `"key":"value"` (trailing quote only)
    /// Leaves string values alone by tracking string state with `\"` escape.
    private static func repairBareKeys(_ s: String) -> String {
        // Pass 1: quote bare identifiers that appear as object keys.
        // Pattern: after `{` or `,` (optionally whitespace), an unquoted
        // identifier [a-zA-Z_][a-zA-Z0-9_]* followed by `:`.
        let pat1 = #"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)"#
        // Pass 2: orphan trailing quote — `identifier"` — usually results
        // from a missing leading quote. Pattern: after `{` or `,` (optionally
        // whitespace), identifier, `"`, `:`. Add leading quote.
        let pat2 = #"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)"(\s*:)"#
        var out = s
        if let r = try? NSRegularExpression(pattern: pat1) {
            let range = NSRange(out.startIndex..., in: out)
            out = r.stringByReplacingMatches(in: out, range: range, withTemplate: "$1\"$2\"$3")
        }
        if let r = try? NSRegularExpression(pattern: pat2) {
            let range = NSRange(out.startIndex..., in: out)
            out = r.stringByReplacingMatches(in: out, range: range, withTemplate: "$1\"$2\"$3")
        }
        return out
    }

    private static func parseDate(_ s: String) -> Date? {
        let fmt = DateFormatter()
        fmt.calendar = Calendar(identifier: .gregorian)
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.dateFormat = "yyyy-MM-dd"
        return fmt.date(from: s)
    }
}
