// InferenceEngine.swift
// Pluggable protocol for the token stream. The app uses this surface and
// never imports a runtime directly, so we can swap between:
//
//   1. StubInferenceEngine         — simulator smoke test (always available)
//   2. LiteRtLmEngine              — when Google ships the Swift package
//   3. LlamaCppEngine              — fallback via vendored llama.cpp .xcframework
//
// The protocol mirrors what LiteRT-LM's Kotlin Conversation.sendMessage()
// exposes, promoted to Swift with AsyncThrowingStream for per-token UI
// updates.

import Foundation

struct InferenceChunk {
    let text: String
    let tokenCount: Int
}

protocol InferenceEngine {
    /// Run inference against the provided *already-formatted* prompt (with
    /// turn delimiters). Returns an async stream of chunks the UI can render
    /// as they arrive. Implementations should set `tokenCount` to the number
    /// of tokens in the chunk (or best approximation).
    ///
    /// `grammar` — optional GBNF text. When non-nil, the engine attempts to
    /// constrain decoding to grammar-legal tokens for this single call. The
    /// constraint is applied to the *current* turn only (the engine resets
    /// to its default sampler after the call completes). Engines that don't
    /// support grammar (stub, LiteRT-LM) accept-and-ignore so callers don't
    /// need backend-specific branches. See `apps/mobile/convert/cliniq_toolcall.gbnf`
    /// for the canonical agent-mode grammar.
    func generate(
        prompt: String,
        maxTokens: Int,
        grammar: String?
    ) async throws -> AsyncThrowingStream<InferenceChunk, Error>
}

extension InferenceEngine {
    /// Convenience overload: no grammar (most call sites). Mirrors the
    /// pre-c18 surface so existing callers keep working unchanged.
    func generate(prompt: String, maxTokens: Int) async throws -> AsyncThrowingStream<InferenceChunk, Error> {
        try await generate(prompt: prompt, maxTokens: maxTokens, grammar: nil)
    }
}

enum InferenceError: LocalizedError {
    case modelNotFound(String)
    case backendUnavailable(String)
    case runtimeFailure(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let path): return "Model file not found: \(path)"
        case .backendUnavailable(let reason): return "Backend unavailable: \(reason)"
        case .runtimeFailure(let reason): return "Runtime failure: \(reason)"
        }
    }
}
