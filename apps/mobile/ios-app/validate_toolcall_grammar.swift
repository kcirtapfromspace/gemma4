// validate_toolcall_grammar.swift
// Swift CLI smoke test for the tool-call parser's strict-mode contract
// (proposals-2026-04-25.md Rank 4). Mirrors the validate_rag.swift pattern.
//
// Build:
//   swiftc -O validate_toolcall_grammar.swift \
//       ClinIQ/ClinIQ/Extraction/ToolCallParser.swift \
//       -parse-as-library -o /tmp/validate_toolcall_grammar
//   /tmp/validate_toolcall_grammar
//
// Doesn't load a model — purely exercises the parser's success / failure
// branches on representative inputs the agent loop sees in practice. The
// grammar itself is exercised by the iOS llama.cpp build at run-time;
// this file just guards `parseStrict` against silent regressions.

import Foundation

@main
enum ValidateToolCallGrammarCLI {
    static func main() {
        let probes: [(String, String, ProbeExpect)] = [
            // (label, output, expectation)
            ("single_tool_call_sentinel",
             "<|tool_call>call:extract_codes_from_text{text:<|\"|>Patient has flu<|\"|>}<tool_call|>",
             .successCount(1)),
            ("two_tool_calls",
             "<|tool_call>call:extract_codes_from_text{text:<|\"|>narrative<|\"|>}<tool_call|>" +
             "<|tool_call>call:lookup_reportable_conditions{query:<|\"|>marburg<|\"|>,top_k:3}<tool_call|>",
             .successCount(2)),
            ("json_quote_drift",
             "<|tool_call>call:validate_fhir_extraction{extraction:{conditions:[\"840539006\"],loincs:[],rxnorms:[]}}<tool_call|>",
             .successCount(1)),
            ("final_answer_only",
             "{\"conditions\":[\"840539006\"],\"loincs\":[],\"rxnorms\":[]}",
             .successCount(0)),
            ("unterminated_open",
             "<|tool_call>call:extract_codes_from_text{text:<|\"|>oh no",
             .failureUnterminated),
            ("unknown_tool",
             "<|tool_call>call:make_coffee{kind:<|\"|>cortado<|\"|>}<tool_call|>",
             .failureUnknownName),
            ("malformed_body",
             "<|tool_call>completely garbled: not a call at all<tool_call|>",
             .failureMalformed),
        ]

        var pass = 0
        for (label, output, expect) in probes {
            let result = ToolCallParser.parseStrict(output)
            let ok = matches(result, expect)
            print("  \(ok ? "PASS" : "FAIL") \(label)  expect=\(expect)  got=\(formatResult(result))")
            if ok { pass += 1 }
        }
        // Cross-check: tolerant `parse` should always succeed on the success
        // probes and silently swallow the failure ones (legacy behaviour).
        var tolerantPass = 0
        for (label, output, expect) in probes {
            let calls = ToolCallParser.parse(output)
            let ok: Bool
            switch expect {
            case .successCount(let n): ok = calls.count == n
            // Failure cases under tolerant parse don't error; they return
            // whatever salvageable calls existed (often 0). We just want
            // confirmation the tolerant parser doesn't crash.
            case .failureUnterminated, .failureMalformed, .failureUnknownName:
                ok = true
            }
            print("  \(ok ? "PASS" : "FAIL") tolerant-\(label)  calls=\(calls.count)")
            if ok { tolerantPass += 1 }
        }

        print("\n=== summary ===")
        print("strict:   \(pass)/\(probes.count) pass")
        print("tolerant: \(tolerantPass)/\(probes.count) pass")
        let total = pass + tolerantPass
        let denom = probes.count * 2
        print("overall:  \(total)/\(denom)")
        exit(total == denom ? 0 : 1)
    }

    enum ProbeExpect: CustomStringConvertible {
        case successCount(Int)
        case failureUnterminated
        case failureMalformed
        case failureUnknownName

        var description: String {
            switch self {
            case .successCount(let n): return "ok(\(n))"
            case .failureUnterminated: return "fail(unterminated)"
            case .failureMalformed: return "fail(malformed)"
            case .failureUnknownName: return "fail(unknownName)"
            }
        }
    }

    static func matches(
        _ result: Result<[GemmaToolCall], ToolCallParseError>,
        _ expect: ProbeExpect
    ) -> Bool {
        switch (result, expect) {
        case (.success(let calls), .successCount(let n)):
            return calls.count == n
        case (.failure(.unterminatedToolCall), .failureUnterminated):
            return true
        case (.failure(.malformedBody), .failureMalformed):
            return true
        case (.failure(.unknownToolName), .failureUnknownName):
            return true
        default:
            return false
        }
    }

    static func formatResult(_ r: Result<[GemmaToolCall], ToolCallParseError>) -> String {
        switch r {
        case .success(let calls):
            return "ok(\(calls.count): \(calls.map { $0.name }.joined(separator: ",")))"
        case .failure(let err):
            return "fail(\(err))"
        }
    }
}
