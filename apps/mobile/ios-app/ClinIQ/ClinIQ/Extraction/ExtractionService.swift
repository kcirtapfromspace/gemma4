// ExtractionService.swift
// Orchestrates: narrative -> prompt -> model stream -> parsed entities ->
// review-ready SwiftData objects. Observes the `InferenceEngine`
// implementation injected at construction (defaults to
// `LlamaCppInferenceEngine` when a GGUF is on disk, else StubEngine).
//
// The service is a @MainActor observable so views can bind directly to
// `isRunning`, `tokensPerSecond`, `streamedOutput` for the live tok counter.

import Foundation
import SwiftUI

@MainActor
final class ExtractionService: ObservableObject {
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var tokensPerSecond: Double = 0
    @Published private(set) var lastTokens: Int = 0
    @Published private(set) var lastElapsed: Double = 0
    @Published private(set) var streamedOutput: String = ""
    @Published private(set) var errorMessage: String?

    private let engine: any InferenceEngine
    private let usingStub: Bool

    init(engine: (any InferenceEngine)? = nil) {
        if let engine = engine {
            self.engine = engine
            self.usingStub = engine is StubInferenceEngine
        } else if LlamaCppInferenceEngine.resolveModelPath() != nil {
            self.engine = LlamaCppInferenceEngine()
            self.usingStub = false
        } else {
            self.engine = StubInferenceEngine()
            self.usingStub = true
        }
    }

    /// True if the running engine is the regex stub (no GGUF found). The UI
    /// surfaces this as a soft chip so the demoer knows.
    var isStubEngine: Bool { usingStub }

    /// Run extraction for the given narrative. Returns a `ParsedExtraction`
    /// that `ReviewViewModel` can turn into SwiftData rows, plus timing.
    func run(narrative: String) async -> ParsedExtraction? {
        isRunning = true
        errorMessage = nil
        streamedOutput = ""
        lastTokens = 0
        lastElapsed = 0
        tokensPerSecond = 0

        let prompt = PromptBuilder.wrapTurns(system: SystemPrompt.clinicalExtraction,
                                             user: narrative)
        let start = Date()
        var accum = ""
        var tokens = 0

        do {
            let stream = try await engine.generate(prompt: prompt, maxTokens: 512)
            for try await chunk in stream {
                accum += chunk.text
                tokens += chunk.tokenCount
                let elapsed = Date().timeIntervalSince(start)
                if elapsed > 0 {
                    tokensPerSecond = Double(tokens) / elapsed
                }
                streamedOutput = accum
            }
            let elapsed = Date().timeIntervalSince(start)
            lastElapsed = elapsed
            lastTokens = tokens
            tokensPerSecond = elapsed > 0 ? Double(tokens) / elapsed : 0
            isRunning = false
            return ExtractionParser.parse(accum)
        } catch {
            errorMessage = "Inference error: \(error.localizedDescription)"
            isRunning = false
            return nil
        }
    }
}
