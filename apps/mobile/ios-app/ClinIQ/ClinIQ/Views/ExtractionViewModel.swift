// ExtractionViewModel.swift
// View model driving the ContentView. Owns the inference engine, the
// streaming state, and all timing counters.
//
// Backend selection (C15): reads the `ClinIQ.Backend` @AppStorage key so
// the user can flip between the fine-tuned llama.cpp path and the stock
// Gemma 4 LiteRT-LM path at runtime. The engine is cached; flipping the
// toggle invalidates the cache (see `reloadEngine()`).

import Foundation
import Combine
import SwiftUI

/// Available inference backends exposed to the Settings UI.
/// Persisted as the raw String value under `ClinIQ.Backend`.
enum InferenceBackend: String, CaseIterable, Identifiable {
    case llamacpp = "llamacpp"
    case litertlm = "litertlm"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .llamacpp: return "llama.cpp (fine-tune)"
        case .litertlm: return "LiteRT-LM (base)"
        }
    }

    static let appStorageKey = "ClinIQ.Backend"
    static let `default`: InferenceBackend = .llamacpp
}

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
    /// Human-readable label for the backend actually serving this run
    /// (after any graceful fallback). Surfaced in the UI so the demoer
    /// knows which path produced the JSON.
    @Published private(set) var activeBackendLabel: String = "llama.cpp"

    // Inject the engine. Default is `makeDefaultEngine()` which honours
    // the `ClinIQ.Backend` AppStorage key (C15). Unit tests can still
    // inject a stub.
    private var engine: any InferenceEngine

    init(engine: (any InferenceEngine)? = nil) {
        if let engine = engine {
            self.engine = engine
            self.activeBackendLabel = Self.label(for: engine)
        } else {
            let (e, label) = Self.makeDefaultEngine()
            self.engine = e
            self.activeBackendLabel = label
        }
    }

    /// Flip the active engine (called from SettingsTab when the picker
    /// changes). Safe to invoke mid-session — the previous engine is
    /// dropped and the next `extract()` call will use the new one.
    func reloadEngine() {
        let (e, label) = Self.makeDefaultEngine()
        self.engine = e
        self.activeBackendLabel = label
    }

    /// Returns an engine + its human-readable label, honouring the
    /// `ClinIQ.Backend` AppStorage selection. Falls back to llama.cpp
    /// (and ultimately the regex stub) if the requested backend's model
    /// file isn't present on disk — we never crash on a missing model.
    static func makeDefaultEngine() -> (any InferenceEngine, String) {
        let raw = UserDefaults.standard.string(forKey: InferenceBackend.appStorageKey)
        let requested = InferenceBackend(rawValue: raw ?? "") ?? .default

        switch requested {
        case .litertlm:
            if LiteRtLmInferenceEngine.resolveModelPath() != nil {
                return (LiteRtLmInferenceEngine(), "LiteRT-LM (base)")
            }
            // Graceful fallback: .litertlm not seeded; log + drop to llama.cpp.
            NSLog("[ClinIQ] LiteRT-LM backend requested but no .litertlm model found; falling back to llama.cpp")
            fallthrough
        case .llamacpp:
            if LlamaCppInferenceEngine.resolveModelPath() != nil {
                return (LlamaCppInferenceEngine(), "llama.cpp (fine-tune)")
            }
            NSLog("[ClinIQ] No GGUF found either; falling back to rule-based stub")
            return (StubInferenceEngine(), "Rule-based stub")
        }
    }

    private static func label(for engine: any InferenceEngine) -> String {
        if engine is LiteRtLmInferenceEngine { return "LiteRT-LM (base)" }
        if engine is LlamaCppInferenceEngine { return "llama.cpp (fine-tune)" }
        if engine is StubInferenceEngine { return "Rule-based stub" }
        return String(describing: type(of: engine))
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
            // C15: backend label prefixed so downstream scoring can segment
            // llama.cpp (fine-tune) vs LiteRT-LM (base) runs.
            Self.persistExtraction(
                caseID: currentCaseID,
                output: accum,
                tokens: tokenCount,
                elapsed: elapsed,
                tps: tokensPerSecond,
                backend: activeBackendLabel)
        } catch {
            errorMessage = "Inference error: \(error.localizedDescription)"
        }
        isExtracting = false
    }

    private static func persistExtraction(caseID: String, output: String, tokens: Int, elapsed: Double, tps: Double, backend: String) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let ts = ISO8601DateFormatter().string(from: Date())
        let entry = """
### backend=\(backend) case=\(caseID) @ \(ts)
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
