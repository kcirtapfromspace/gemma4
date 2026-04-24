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
    func generate(prompt: String, maxTokens: Int) async throws -> AsyncThrowingStream<InferenceChunk, Error>
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
