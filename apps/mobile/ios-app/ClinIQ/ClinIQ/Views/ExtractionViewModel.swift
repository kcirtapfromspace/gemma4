// ExtractionViewModel.swift
// View model driving the ContentView. Owns the inference engine, the
// streaming state, and all timing counters.

import Foundation
import Combine

@MainActor
final class ExtractionViewModel: ObservableObject {
    @Published var inputEicr: String = TestCase.bundled.first(where: { $0.caseId == "bench_typical_covid" })?.user ?? ""
    /// Human-readable case label for the currently-loaded case (set by
    /// `loadCase(_:)`). Persisted alongside the extracted JSON.
    @Published private(set) var currentCaseID: String = "bench_typical_covid"
    @Published var output: String = ""
    @Published var isExtracting: Bool = false
    @Published var tokensPerSecond: Double = 0
    @Published var lastTokensGenerated: Int = 0
    @Published var lastElapsedSeconds: Double = 0
    @Published var errorMessage: String?

    // Inject the engine. Default (C12) is `LlamaCppInferenceEngine` when a
    // GGUF is discoverable on disk, falling back to the stub regex engine
    // for CI / SwiftUI Previews. The runtime decision lives in
    // `makeDefaultEngine()` below so the owner can still inject a custom
    // engine in unit tests.
    private let engine: any InferenceEngine

    init(engine: (any InferenceEngine)? = nil) {
        if let engine = engine {
            self.engine = engine
        } else {
            self.engine = Self.makeDefaultEngine()
        }
    }

    private static func makeDefaultEngine() -> any InferenceEngine {
        // Team C12 — real inference via the vendored llama.cpp xcframework
        // (see `ClinIQ/Frameworks/llama.xcframework/`). Works on both
        // simulator (CPU-only — `n_gpu_layers=0`) and physical iPhone
        // (Metal). Falls back to `StubInferenceEngine` only if no GGUF
        // can be resolved via the bundle / Documents / tmp search path —
        // keeps CI + Previews functional without a 2-3 GB model file.
        if LlamaCppInferenceEngine.resolveModelPath() != nil {
            return LlamaCppInferenceEngine()
        }
        return StubInferenceEngine()
    }

    func loadCase(_ tc: TestCase) {
        inputEicr = tc.user
        output = ""
        errorMessage = nil
        currentCaseID = tc.caseId
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

            // C12: persist the final output alongside metadata so headless
            // runs + screenshot harnesses can audit the exact JSON the model
            // produced. Writes a timestamped file under Documents so it
            // survives app relaunch.
            Self.persistExtraction(
                caseID: currentCaseID,
                output: accum,
                tokens: tokenCount,
                elapsed: elapsed,
                tps: tokensPerSecond)
        } catch {
            errorMessage = "Inference error: \(error.localizedDescription)"
        }
        isExtracting = false
    }

    private static func persistExtraction(caseID: String, output: String, tokens: Int, elapsed: Double, tps: Double) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let ts = ISO8601DateFormatter().string(from: Date())
        let entry = """
### case=\(caseID) @ \(ts)
tokens=\(tokens) elapsed=\(String(format: "%.2f", elapsed))s tps=\(String(format: "%.3f", tps))
OUTPUT:
\(output)
---

"""
        let url = docs.appendingPathComponent("extractions.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(entry.utf8))
        } else {
            try? entry.data(using: .utf8)?.write(to: url)
        }
    }
}
