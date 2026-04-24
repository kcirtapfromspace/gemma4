// ExtractionViewModel.swift
// View model driving the ContentView. Owns the inference engine, the
// streaming state, and all timing counters.

import Foundation
import Combine

@MainActor
final class ExtractionViewModel: ObservableObject {
    @Published var inputEicr: String = TestCase.bundled.first(where: { $0.caseId == "bench_typical_covid" })?.user ?? ""
    @Published var output: String = ""
    @Published var isExtracting: Bool = false
    @Published var tokensPerSecond: Double = 0
    @Published var lastTokensGenerated: Int = 0
    @Published var lastElapsedSeconds: Double = 0
    @Published var errorMessage: String?

    // Inject the engine. Default is the stub in SIMULATOR builds so the
    // project compiles cleanly in the iPhone 17 Pro simulator without the
    // LiteRT-LM static library. On device (when we have the .xcframework)
    // we'd swap to `LiteRtLmEngine()` — the protocol is identical.
    private let engine: any InferenceEngine

    init(engine: (any InferenceEngine)? = nil) {
        if let engine = engine {
            self.engine = engine
        } else {
            self.engine = Self.makeDefaultEngine()
        }
    }

    private static func makeDefaultEngine() -> any InferenceEngine {
        // LiteRT-LM Swift package is "In Dev" as of 2026-04-23 (confirmed on
        // github.com/google-ai-edge/LiteRT-LM README — language table lists
        // Swift as "In Dev / Coming Soon"). Until the .xcframework ships,
        // simulator runs use the deterministic stub that exercises the full
        // UI + prompt wrapping pipeline but returns a canned JSON.
        #if targetEnvironment(simulator)
        return StubInferenceEngine()
        #else
        return StubInferenceEngine()
        #endif
    }

    func loadCase(_ tc: TestCase) {
        inputEicr = tc.user
        output = ""
        errorMessage = nil
    }

    func extract() async {
        isExtracting = true
        output = ""
        errorMessage = nil
        tokensPerSecond = 0
        lastTokensGenerated = 0

        let prompt = PromptBuilder.wrapTurns(
            system: SystemPrompt.clinicalExtraction,
            user: inputEicr)

        let start = Date()
        var accum = ""
        var tokenCount = 0

        do {
            let stream = try await engine.generate(prompt: prompt, maxTokens: 512)
            for try await chunk in stream {
                accum += chunk.text
                tokenCount += chunk.tokenCount
                let elapsed = Date().timeIntervalSince(start)
                if elapsed > 0 {
                    tokensPerSecond = Double(tokenCount) / elapsed
                }
                // Push to UI
                output = accum
            }
            let elapsed = Date().timeIntervalSince(start)
            lastElapsedSeconds = elapsed
            lastTokensGenerated = tokenCount
            tokensPerSecond = elapsed > 0 ? Double(tokenCount) / elapsed : 0
        } catch {
            errorMessage = "Inference error: \(error.localizedDescription)"
        }
        isExtracting = false
    }
}
