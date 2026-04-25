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

enum ToolCallParser {
    private static let openTag = "<|tool_call>"
    private static let closeTag = "<tool_call|>"
    /// Both sides of a Gemma 4 string-arg use the same 5-char sentinel
    /// `<|"|>`. The bundle template emits `<|"|>foo<|"|>` for a string `foo`.
    static let stringSentinel = "<|\"|>"

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

