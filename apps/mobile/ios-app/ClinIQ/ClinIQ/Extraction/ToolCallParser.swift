// ToolCallParser.swift
// Parses Gemma 4 native tool-call tokens out of a raw model output string.
//
// Wire format (matches the bundle template's emitter):
//   <|tool_call>call:NAME{key:<|"|>value<|"|>,nested:{...},arr:[<|"|>v<|"|>]}<tool_call|>
//
// Multiple <|tool_call>...<tool_call|> blocks may appear in one turn.
// Returns an empty list if no tool call is present (the model is producing
// a final answer instead of orchestrating tools).
//
// Robustness notes:
//   - The LLM occasionally drifts to standard JSON quotes ("foo") instead of
//     the trained <|"|> sentinel. We accept both.
//   - Nested objects/arrays parse via a small recursive descent, not regex.
//   - On malformed argument syntax we return the call with empty arguments
//     rather than dropping the call entirely — the agent loop will then
//     report an error from the tool's empty-args branch.

import Foundation

struct GemmaToolCall {
    let name: String
    let arguments: [String: Any]
    let raw: String
}

/// Strict-mode parse failure modes (proposals-2026-04-25.md Rank 4).
/// Surfaced via `ToolCallParser.parseStrict` so the agent loop can
/// distinguish "model emitted a final answer" from "model emitted broken
/// tool-call syntax" — the silent-fallback path was hiding the second case.
enum ToolCallParseError: Error, CustomStringConvertible {
    /// Open tag found but no matching close tag in the rest of the buffer.
    case unterminatedToolCall(prefix: String)
    /// Open + close tags found, but the body doesn't match `call:NAME{...}`.
    case malformedBody(body: String)
    /// Tool name in the body isn't one of the registered names. With the
    /// GBNF active this should be unreachable, so a hit means the grammar
    /// is being bypassed — usually a misconfiguration.
    case unknownToolName(name: String)

    var description: String {
        switch self {
        case .unterminatedToolCall(let prefix):
            return "unterminated <|tool_call> block (no <tool_call|> closer in \(prefix.count) chars)"
        case .malformedBody(let body):
            return "malformed tool-call body: \(body.prefix(80))…"
        case .unknownToolName(let name):
            return "unknown tool name '\(name)'"
        }
    }
}

enum ToolCallParser {
    private static let openTag = "<|tool_call>"
    private static let closeTag = "<tool_call|>"
    /// Both sides of a Gemma 4 string-arg use the same 5-char sentinel
    /// `<|"|>`. The bundle template emits `<|"|>foo<|"|>` for a string `foo`.
    static let stringSentinel = "<|\"|>"

    /// The 4 tool names registered with the agent. Used by `parseStrict` to
    /// catch grammar bypasses; the tolerant `parse(...)` accepts any name
    /// (the agent loop's tool dispatcher already has an "unknown tool"
    /// branch, so unfamiliar names error out at execution time).
    static let registeredToolNames: Set<String> = [
        "extract_codes_from_text",
        "lookup_reportable_conditions",
        "validate_fhir_extraction",
        "lookup_displayname",
    ]

    static func parse(_ output: String) -> [GemmaToolCall] {
        var calls: [GemmaToolCall] = []
        var cursor = output.startIndex
        while let openRange = output.range(of: openTag, range: cursor..<output.endIndex) {
            cursor = openRange.upperBound
            guard let closeRange = output.range(of: closeTag, range: cursor..<output.endIndex) else {
                break
            }
            let body = String(output[openRange.upperBound..<closeRange.lowerBound])
            cursor = closeRange.upperBound
            if let call = parseOne(body) {
                calls.append(call)
            }
        }
        return calls
    }

    /// Strict-mode parse for grammar-on agent turns. Returns `.failure` on
    /// any structural problem instead of silently dropping calls. Emits
    /// `.success([])` only when the output legitimately contains no tool
    /// calls (final-answer turn) — that's how the AgentRunner distinguishes
    /// "JSON-final-answer turn, stop the loop" from "broken tool call".
    ///
    /// Behaviour contract (used by AgentRunner.run):
    ///   - 0 open tags + non-empty body → `.success([])` (final answer; the
    ///     ExtractionParser path takes over).
    ///   - Open tag without close tag → `.failure(.unterminatedToolCall)`
    ///     even if the body so far parses; truncation is a parse failure.
    ///   - Body doesn't start with `call:NAME{` → `.failure(.malformedBody)`.
    ///   - Tool name not in `registeredToolNames` → `.failure(.unknownToolName)`.
    static func parseStrict(_ output: String) -> Result<[GemmaToolCall], ToolCallParseError> {
        var calls: [GemmaToolCall] = []
        var cursor = output.startIndex
        while let openRange = output.range(of: openTag, range: cursor..<output.endIndex) {
            cursor = openRange.upperBound
            guard let closeRange = output.range(of: closeTag, range: cursor..<output.endIndex) else {
                let prefix = String(output[openRange.upperBound..<output.endIndex])
                return .failure(.unterminatedToolCall(prefix: prefix))
            }
            let body = String(output[openRange.upperBound..<closeRange.lowerBound])
            cursor = closeRange.upperBound
            guard let call = parseOne(body) else {
                return .failure(.malformedBody(body: body))
            }
            if !registeredToolNames.contains(call.name) {
                return .failure(.unknownToolName(name: call.name))
            }
            calls.append(call)
        }
        return .success(calls)
    }

    /// Parse a single body like "call:NAME{key:value,...}".
    private static func parseOne(_ body: String) -> GemmaToolCall? {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("call:") else { return nil }
        let after = String(trimmed.dropFirst("call:".count))
        guard let braceIdx = after.firstIndex(of: "{") else { return nil }
        let name = String(after[..<braceIdx]).trimmingCharacters(in: .whitespaces)
        let argsBody = String(after[after.index(after: braceIdx)...])
        // Drop a trailing "}" if present; tolerate omission.
        let argsTrimmed: String
        if argsBody.hasSuffix("}") {
            argsTrimmed = String(argsBody.dropLast())
        } else {
            argsTrimmed = argsBody
        }
        var parser = ArgParser(input: argsTrimmed)
        let dict = parser.parseObjectBody()
        return GemmaToolCall(name: name, arguments: dict, raw: body)
    }
}

// MARK: - Argument parser (Gemma-format-aware)
//
// Tiny recursive-descent over key:value pairs. Values can be strings (in
// either quote form), numbers, true/false/null, arrays, or objects.

private struct ArgParser {
    let input: String
    var idx: String.Index

    init(input: String) {
        self.input = input
        self.idx = input.startIndex
    }

    mutating func parseObjectBody() -> [String: Any] {
        var out: [String: Any] = [:]
        skipWhitespace()
        while idx < input.endIndex {
            skipWhitespace()
            if peek() == "}" { advance(); break }
            guard let key = readKey() else { break }
            skipWhitespace()
            if !consume(":") { break }
            skipWhitespace()
            let value = readValue()
            out[key] = value
            skipWhitespace()
            if !consume(",") { break }
        }
        return out
    }

    /// Keys are unquoted identifier-ish strings up to ":".
    private mutating func readKey() -> String? {
        // Some emitters quote keys with the sentinel; tolerate both.
        if peekString(ToolCallParser.stringSentinel) {
            return readSentinelString()
        }
        if peek() == "\"" {
            return readJsonString()
        }
        let start = idx
        while idx < input.endIndex,
              !"\":".contains(input[idx]),
              !input[idx].isWhitespace {
            advance()
        }
        let key = String(input[start..<idx]).trimmingCharacters(in: .whitespaces)
        return key.isEmpty ? nil : key
    }

    private mutating func readValue() -> Any {
        skipWhitespace()
        if idx == input.endIndex { return NSNull() }
        let c = input[idx]
        if c == "{" {
            advance()
            return parseObjectBody()
        }
        if c == "[" {
            advance()
            return readArray()
        }
        if peekString(ToolCallParser.stringSentinel) {
            return readSentinelString()
        }
        if c == "\"" {
            return readJsonString()
        }
        // Bareword: number, true/false/null
        let start = idx
        while idx < input.endIndex,
              !",}]".contains(input[idx]) {
            advance()
        }
        let token = String(input[start..<idx]).trimmingCharacters(in: .whitespaces)
        return interpretBareword(token)
    }

    private mutating func readArray() -> [Any] {
        var arr: [Any] = []
        skipWhitespace()
        while idx < input.endIndex {
            skipWhitespace()
            if peek() == "]" { advance(); break }
            arr.append(readValue())
            skipWhitespace()
            if !consume(",") {
                // closing ']' or end
                _ = consume("]")
                break
            }
        }
        return arr
    }

    private mutating func readSentinelString() -> String {
        // Sentinel is the full 5-char `<|"|>`; both sides use the same string.
        let sentinel = ToolCallParser.stringSentinel
        guard let openEnd = input.index(idx, offsetBy: sentinel.count, limitedBy: input.endIndex)
        else { return "" }
        idx = openEnd
        guard let r = input.range(of: sentinel, range: idx..<input.endIndex) else {
            // Unterminated; consume to end
            let s = String(input[idx..<input.endIndex])
            idx = input.endIndex
            return s
        }
        let s = String(input[idx..<r.lowerBound])
        idx = r.upperBound
        return s
    }

    private mutating func readJsonString() -> String {
        // ascii double-quoted, no escape handling beyond \"
        advance() // consume opening "
        var out = ""
        while idx < input.endIndex {
            let c = input[idx]
            if c == "\\" {
                advance()
                if idx < input.endIndex {
                    out.append(input[idx])
                    advance()
                }
                continue
            }
            if c == "\"" {
                advance()
                break
            }
            out.append(c)
            advance()
        }
        return out
    }

    private func interpretBareword(_ token: String) -> Any {
        let t = token.lowercased()
        if t == "true" { return true }
        if t == "false" { return false }
        if t == "null" { return NSNull() }
        if let i = Int(token) { return i }
        if let d = Double(token) { return d }
        return token
    }

    // MARK: - Cursor helpers

    private mutating func skipWhitespace() {
        while idx < input.endIndex, input[idx].isWhitespace { advance() }
    }

    private func peek() -> Character? {
        idx < input.endIndex ? input[idx] : nil
    }

    private func peekString(_ s: String) -> Bool {
        guard let end = input.index(idx, offsetBy: s.count, limitedBy: input.endIndex) else {
            return false
        }
        return input[idx..<end] == s
    }

    private mutating func consume(_ c: Character) -> Bool {
        guard idx < input.endIndex, input[idx] == c else { return false }
        advance()
        return true
    }

    private mutating func advance() {
        idx = input.index(after: idx)
    }
}

