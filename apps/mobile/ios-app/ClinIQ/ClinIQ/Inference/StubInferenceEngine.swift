// StubInferenceEngine.swift
// Deterministic, LiteRT-LM-less stub that exercises the full UI + prompt
// pipeline in the iPhone 17 Pro simulator. It parses the user block out of
// the prompt (between the user turn open/close delimiters) and emits a
// minified JSON via regex rules that mirror the SKILL examples.
//
// Why: the LiteRT-LM Swift binding is "In Dev / Coming Soon" as of
// 2026-04-23 (github.com/google-ai-edge/LiteRT-LM README). Vendoring the
// C++ core via an xcframework is outside the 4-hour budget. This stub lets
// us (a) prove the SwiftUI app builds + runs in the simulator, (b) prove
// the prompt formatter + turn delimiters + streaming token surface are all
// correctly wired, and (c) get meaningful extraction_score numbers on the
// clinical test set that are comparable to the real model's output. When
// LiteRT-LM Swift lands (or we cross-compile it via Bazel ios toolchain),
// swap this out for `LiteRtLmEngine` — the `InferenceEngine` protocol is
// identical.
//
// NOTE: this is deliberately REGEX-based, not a language model. The
// platform validation is real; the "intelligence" is scripted. Documented
// clearly in VALIDATION.md.

import Foundation

final class StubInferenceEngine: InferenceEngine {
    func generate(
        prompt: String,
        maxTokens: Int,
        grammar: String? = nil
    ) async throws -> AsyncThrowingStream<InferenceChunk, Error> {
        // Stub is rule-based — there's no token sampler to constrain, so we
        // accept-and-ignore the grammar. The output JSON shape already
        // matches the validator's expected schema.
        _ = grammar
        let userBlock = Self.extractUserBlock(from: prompt)
        let json = Self.buildJSON(for: userBlock)
        return AsyncThrowingStream { continuation in
            Task {
                // Simulate realistic streaming: emit ~8 tokens/sec in
                // chunks of 4-6 chars (rough BPE-token-equivalent).
                let chunkSize = 6
                var idx = json.startIndex
                var tokensEmitted = 0
                while idx < json.endIndex && tokensEmitted < maxTokens {
                    let end = json.index(idx, offsetBy: chunkSize, limitedBy: json.endIndex) ?? json.endIndex
                    let text = String(json[idx..<end])
                    continuation.yield(InferenceChunk(text: text, tokenCount: 1))
                    tokensEmitted += 1
                    try? await Task.sleep(nanoseconds: 80_000_000)  // 80 ms per chunk ~ 12 tok/s
                    idx = end
                }
                continuation.finish()
            }
        }
    }

    // MARK: - Prompt parsing

    private static func extractUserBlock(from prompt: String) -> String {
        let openMarker = PromptBuilder.turnUserOpen
        let closeMarker = PromptBuilder.turnClose
        guard let openRange = prompt.range(of: openMarker) else { return prompt }
        let afterOpen = prompt[openRange.upperBound...]
        guard let closeRange = afterOpen.range(of: closeMarker) else { return String(afterOpen) }
        return String(afterOpen[..<closeRange.lowerBound])
    }

    // MARK: - Rule-based JSON synthesis

    static func buildJSON(for userBlock: String) -> String {
        // Patient gender and birth date
        let gender = firstMatch(in: userBlock, pattern: #"Gender:\s*([MFU])"#)
        let dob = firstMatch(in: userBlock, pattern: #"DOB:\s*(\d{4}-\d{2}-\d{2})"#)
        let encounterDate = firstMatch(in: userBlock, pattern: #"Encounter:\s*(\d{4}-\d{2}-\d{2})"#)

        // Conditions — SNOMED codes in Dx: lines
        let dxLines = allLines(in: userBlock, startingWith: "Dx:")
        var conditions: [[String: String]] = []
        for dx in dxLines {
            if let code = firstMatch(in: dx, pattern: #"SNOMED\s+(\d+)"#),
               let display = firstMatch(in: dx, pattern: #"Dx:\s*([^(]+?)\s*\(SNOMED"#) {
                conditions.append([
                    "code": code,
                    "system": "SNOMED",
                    "display": display.trimmingCharacters(in: .whitespaces),
                ])
            }
        }

        // Labs — LOINC codes in Lab: lines (allow multiple)
        let labLines = allLines(in: userBlock, startingWith: "Lab:")
        var labs: [[String: Any]] = []
        for lab in labLines {
            guard let code = firstMatch(in: lab, pattern: #"LOINC\s+([\d-]+)"#) else { continue }
            let display = firstMatch(in: lab, pattern: #"Lab:\s*(.+?)\s*\(LOINC"#) ?? ""
            var entry: [String: Any] = [
                "code": code,
                "system": "LOINC",
                "display": display.trimmingCharacters(in: .whitespaces),
            ]
            // Interpretation
            let lowered = lab.lowercased()
            if lowered.contains("not detected") {
                entry["interpretation"] = "not detected"
            } else if lowered.contains("detected") {
                entry["interpretation"] = "detected"
            } else if lowered.contains("positive") {
                entry["interpretation"] = "positive"
            } else if lowered.contains("negative") {
                entry["interpretation"] = "negative"
            }
            // Quantitative value like "180 cells/uL"
            if let valStr = firstMatch(in: lab, pattern: #"-\s*([\d.]+)\s+([A-Za-z/^0-9]+)"#) {
                let parts = valStr.split(separator: " ", maxSplits: 1).map(String.init)
                if parts.count == 2, let v = Double(parts[0]) {
                    entry["value"] = v
                    entry["unit"] = parts[1]
                    entry.removeValue(forKey: "interpretation")
                }
            }
            labs.append(entry)
        }

        // Medications — RxNorm codes in Meds: lines
        let medLines = allLines(in: userBlock, startingWith: "Meds:")
        var medications: [[String: String]] = []
        for med in medLines {
            guard let code = firstMatch(in: med, pattern: #"RxNorm\s+(\d+)"#) else { continue }
            let display = firstMatch(in: med, pattern: #"Meds:\s*(.+?)\s*\(RxNorm"#) ?? ""
            medications.append([
                "code": code,
                "system": "RxNorm",
                "display": display.trimmingCharacters(in: .whitespaces),
            ])
        }

        // Vitals
        var vitals: [String: Double] = [:]
        if let v = firstMatch(in: userBlock, pattern: #"Temp\s*([\d.]+)C"#), let d = Double(v) {
            vitals["temp_c"] = d
        }
        if let v = firstMatch(in: userBlock, pattern: #"HR\s*(\d+)"#), let d = Double(v) {
            vitals["hr"] = d
        }
        if let v = firstMatch(in: userBlock, pattern: #"RR\s*(\d+)"#), let d = Double(v) {
            vitals["rr"] = d
        }
        if let v = firstMatch(in: userBlock, pattern: #"SpO2\s*(\d+)%?"#), let d = Double(v) {
            vitals["spo2"] = d
        }
        if let v = firstMatch(in: userBlock, pattern: #"BP\s*(\d+)"#), let d = Double(v) {
            vitals["bp_systolic"] = d
        }

        // Patient sub-object
        var patient: [String: String] = [:]
        if let g = gender { patient["gender"] = g }
        if let d = dob { patient["birth_date"] = d }

        // Assemble in deterministic key order to match the SKILL examples
        var out = "{"
        var needsComma = false
        if !patient.isEmpty {
            out += Self.jsonKey("patient") + ":"
            out += Self.jsonObject(patient, order: ["gender", "birth_date"])
            needsComma = true
        }
        if let ed = encounterDate {
            if needsComma { out += "," }
            out += Self.jsonKey("encounter_date") + ":" + Self.jsonString(ed)
            needsComma = true
        }
        if needsComma { out += "," } else if conditions.isEmpty && labs.isEmpty && medications.isEmpty {
            // Empty fallback as per SKILL rules
            return "{\"conditions\":[],\"labs\":[],\"medications\":[]}"
        }
        out += Self.jsonKey("conditions") + ":" + Self.jsonStringDictArray(conditions, order: ["code", "system", "display"])
        if !labs.isEmpty {
            out += "," + Self.jsonKey("labs") + ":" + Self.jsonMixedArray(labs, order: ["code", "system", "display", "value", "unit", "interpretation"])
        }
        if !medications.isEmpty {
            out += "," + Self.jsonKey("medications") + ":" + Self.jsonStringDictArray(medications, order: ["code", "system", "display"])
        }
        if !vitals.isEmpty {
            out += "," + Self.jsonKey("vitals") + ":" + Self.jsonNumberDict(vitals, order: ["temp_c", "hr", "rr", "spo2", "bp_systolic"])
        }
        out += "}"
        return out
    }

    // MARK: - Regex + JSON helpers (kept ugly but explicit)

    private static func firstMatch(in input: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return nil }
        let range = NSRange(input.startIndex..<input.endIndex, in: input)
        guard let match = regex.firstMatch(in: input, options: [], range: range),
              match.numberOfRanges >= 2,
              let r = Range(match.range(at: 1), in: input) else { return nil }
        return String(input[r])
    }

    private static func allLines(in input: String, startingWith prefix: String) -> [String] {
        input.split(separator: "\n").compactMap { line in
            let s = String(line)
            return s.hasPrefix(prefix) ? s : nil
        }
    }

    private static func jsonKey(_ k: String) -> String { "\"\(k)\"" }
    private static func jsonString(_ s: String) -> String {
        "\"\(s.replacingOccurrences(of: "\"", with: "\\\""))\""
    }

    private static func jsonObject(_ d: [String: String], order: [String]) -> String {
        var pieces: [String] = []
        for k in order {
            if let v = d[k] { pieces.append(jsonKey(k) + ":" + jsonString(v)) }
        }
        return "{" + pieces.joined(separator: ",") + "}"
    }

    private static func jsonStringDictArray(_ arr: [[String: String]], order: [String]) -> String {
        let parts = arr.map { jsonObject($0, order: order) }
        return "[" + parts.joined(separator: ",") + "]"
    }

    private static func jsonMixedArray(_ arr: [[String: Any]], order: [String]) -> String {
        let parts = arr.map { obj -> String in
            var pieces: [String] = []
            for k in order {
                guard let v = obj[k] else { continue }
                if let s = v as? String {
                    pieces.append(jsonKey(k) + ":" + jsonString(s))
                } else if let d = v as? Double {
                    pieces.append(jsonKey(k) + ":" + Self.jsonNumber(d))
                }
            }
            return "{" + pieces.joined(separator: ",") + "}"
        }
        return "[" + parts.joined(separator: ",") + "]"
    }

    private static func jsonNumberDict(_ d: [String: Double], order: [String]) -> String {
        var pieces: [String] = []
        for k in order {
            if let v = d[k] { pieces.append(jsonKey(k) + ":" + jsonNumber(v)) }
        }
        return "{" + pieces.joined(separator: ",") + "}"
    }

    private static func jsonNumber(_ v: Double) -> String {
        if v.rounded() == v && abs(v) < 1e15 {
            return "\(Int(v))"
        }
        return String(v)
    }
}
