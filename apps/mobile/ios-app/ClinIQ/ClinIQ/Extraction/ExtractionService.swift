// ExtractionService.swift
// Orchestrates: narrative -> prompt -> model stream -> parsed entities ->
// review-ready SwiftData objects. Observes the `InferenceEngine`
// implementation injected at construction (defaults to whichever backend
// the user has selected in Settings — see `ExtractionViewModel.makeDefaultEngine`).
//
// C15: engine is a `var`, not a `let`, so a Settings toggle flip can
// invalidate it via `reloadEngine()` without tearing down the whole
// service. The service is a @MainActor observable so views can bind
// directly to `isRunning`, `tokensPerSecond`, `streamedOutput` for the
// live tok counter.

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
    /// Human-readable backend name (e.g. "LiteRT-LM (base)"). Surfaces
    /// in the UI + sync log so reviewers know which path produced a row.
    @Published private(set) var activeBackendLabel: String = "llama.cpp"

    private var engine: any InferenceEngine
    private var usingStub: Bool

    init(engine: (any InferenceEngine)? = nil) {
        if let engine = engine {
            self.engine = engine
            self.usingStub = engine is StubInferenceEngine
            self.activeBackendLabel = Self.label(for: engine)
        } else {
            let (e, label) = ExtractionViewModel.makeDefaultEngine()
            self.engine = e
            self.activeBackendLabel = label
            self.usingStub = e is StubInferenceEngine
        }
    }

    /// Flip the active engine (called from SettingsTab when the picker
    /// changes). No-op if the target backend is unavailable — we fall
    /// back to the next best option per `makeDefaultEngine()`.
    func reloadEngine() {
        let (e, label) = ExtractionViewModel.makeDefaultEngine()
        self.engine = e
        self.activeBackendLabel = label
        self.usingStub = e is StubInferenceEngine
    }

    private static func label(for engine: any InferenceEngine) -> String {
        if engine is LiteRtLmInferenceEngine { return "LiteRT-LM (base)" }
        if engine is LlamaCppInferenceEngine { return "llama.cpp (fine-tune)" }
        if engine is StubInferenceEngine { return "Rule-based stub" }
        return String(describing: type(of: engine))
    }

    /// True if the running engine is the regex stub (no model found). The UI
    /// surfaces this as a soft chip so the demoer knows.
    var isStubEngine: Bool { usingStub }

    /// Run extraction for the given narrative. Returns a `ParsedExtraction`
    /// that `ReviewViewModel` can turn into SwiftData rows, plus timing.
    ///
    /// Tier order:
    ///   1. `EicrPreparser` — deterministic regex/CDA/lookup. If it finds
    ///      any code, return immediately (single-digit ms, no LLM).
    ///   1b. **c19 fast-path**: tier-1 empty + `RagSearch.fastPathHit()`
    ///       returns a top hit ≥0.70 with at least one non-negated
    ///       occurrence of the matched phrase → fabricate the extraction
    ///       from the RAG hit (no LLM). Drops the long-tail agent-tier
    ///       median latency from ~13 s to <1 s. Provenance tier
    ///       `.ragFast`. Per proposals-2026-04-25 Rank 2.
    ///   2. LLM via the active engine (LiteRT-LM, llama.cpp, or stub),
    ///      with the engine's existing tolerant JSON parser on the output.
    ///
    /// On the 14-case combined bench the deterministic preparser hits 1.00,
    /// so the LLM only fires on inputs that have no codes/CDA/aliases —
    /// the residual prose-narrative tail.
    func run(narrative: String) async -> ParsedExtraction? {
        isRunning = true
        errorMessage = nil
        streamedOutput = ""
        lastTokens = 0
        lastElapsed = 0
        tokensPerSecond = 0

        // Tier 1 — deterministic preparser. Short-circuits the LLM whenever
        // the input contains inline parenthesized codes, CDA XML attrs, or
        // displayNames in the curated lookup table. Carries per-code
        // provenance (tier + source span + confidence) into the review UI.
        let detStart = Date()
        let detResult = EicrPreparser.extractWithProvenance(narrative)
        var det = detResult.extraction
        det.matches = detResult.provenance
        let detElapsed = Date().timeIntervalSince(detStart)
        if det.hasAnyDeterministic {
            lastElapsed = detElapsed
            lastTokens = 0
            tokensPerSecond = 0
            isRunning = false
            activeBackendLabel = "Deterministic preparser"
            return det
        }

        // Tier 1b — c19 single-turn fast-path. Tier-1 came back empty;
        // before paying the agent's multi-turn LLM cost, ask RAG whether
        // the narrative names a known reportable condition with high
        // confidence and a non-negated mention. If so, synthesize an
        // extraction directly from the RAG hit. Mirrors Python's
        // agent_pipeline.py --fast-path-rag-threshold gate.
        let fpStart = Date()
        if let fp = RagSearch.fastPathHit(narrative: narrative) {
            var fast = ParsedExtraction()
            fast.raw = narrative
            fast.conditions = [
                ParsedCondition(code: fp.hit.code,
                                system: fp.hit.system,
                                display: fp.hit.display)
            ]
            fast.matches = [
                CodeProvenance(
                    code: fp.hit.code,
                    display: fp.hit.display,
                    system: fp.hit.system,
                    bucket: "snomed",
                    tier: .ragFast,
                    confidence: max(EicrPreparser.tierConfidenceRagFastFloor,
                                    fp.hit.score),
                    sourceText: fp.span.text,
                    sourceOffset: fp.span.location,
                    sourceLength: fp.span.length,
                    alias: fp.hit.matchedPhrase,
                    sourceURL: fp.hit.sourceURL
                )
            ]
            lastElapsed = Date().timeIntervalSince(fpStart) + detElapsed
            lastTokens = 0
            tokensPerSecond = 0
            isRunning = false
            activeBackendLabel = "RAG fast-path"
            return fast
        }

        // Tier 2 — Gemma 4 agent loop. The model orchestrates the same
        // deterministic stack as tools, plus RAG over CDC NNDSS / WHO IDSR
        // for codes outside the curated lookup. On Mac llama-server this
        // hits F1=0.986 across 27 cases (originals + adv1 + adv2 + adv3).
        if !usingStub {
            let agentStart = Date()
            let runner = AgentRunner(engine: engine)
            do {
                let trace = try await runner.run(narrative: narrative)
                if let extraction = trace.finalExtraction,
                   extraction.hasAnyDeterministic {
                    lastElapsed = Date().timeIntervalSince(agentStart)
                    lastTokens = 0
                    tokensPerSecond = 0
                    isRunning = false
                    activeBackendLabel = "Gemma 4 agent + RAG"
                    return extraction
                }
            } catch {
                // Agent loop failed — fall through to legacy raw-LLM path
                // rather than dropping the request.
            }
        }

        // Tier 3 — legacy raw LLM. Single-shot prompt, no tool calling.
        // Kept as a final fallback for engines that can't sustain the
        // multi-turn agent (e.g. the regex stub).
        activeBackendLabel = Self.label(for: engine)

        let prompt = PromptBuilder.wrapTurns(system: SystemPrompt.clinicalExtraction,
                                             user: narrative)
        let start = Date()
        var accum = ""
        var tokens = 0

        // Publish live telemetry to the status-bar overlay.
        let backendName = usingStub
            ? "Stub (rule-based)"
            : String(describing: type(of: engine)).replacingOccurrences(of: "InferenceEngine", with: "")
        let modelName: String = {
            if engine is LlamaCppInferenceEngine {
                return (LlamaCppInferenceEngine.resolveModelPath() as NSString?)?
                    .lastPathComponent ?? "on-device model"
            }
            return usingStub ? "—" : "on-device model"
        }()
        // Bumped from 512 after observing the model needs ~700-900 tokens
        // to emit the full schema for a vitals-heavy eICR. Truncated JSON
        // makes the parser bail and the review sheet shows "0 entities".
        let maxTokens = 1024
        InferenceMetrics.shared.begin(backend: backendName,
                                      model: modelName,
                                      promptChars: prompt.count,
                                      maxTokens: maxTokens)

        do {
            let stream = try await engine.generate(prompt: prompt, maxTokens: maxTokens)
            for try await chunk in stream {
                accum += chunk.text
                tokens += chunk.tokenCount
                let elapsed = Date().timeIntervalSince(start)
                if elapsed > 0 {
                    tokensPerSecond = Double(tokens) / elapsed
                }
                streamedOutput = accum
                InferenceMetrics.shared.record(chunkTokens: max(1, chunk.tokenCount))
            }
            InferenceMetrics.shared.finalize()
            let elapsed = Date().timeIntervalSince(start)
            lastElapsed = elapsed
            lastTokens = tokens
            tokensPerSecond = elapsed > 0 ? Double(tokens) / elapsed : 0
            isRunning = false
            InferenceMetrics.shared.end()
            // DEBUG: persist the raw model output so simulator runs can be
            // post-mortemed from the container without a live tap-through.
            if let docs = FileManager.default.urls(for: .documentDirectory,
                                                    in: .userDomainMask).first {
                let path = docs.appendingPathComponent("last-extraction-raw.txt")
                let header = "backend=\(backendName) model=\(modelName) tokens=\(tokens) elapsed=\(String(format: "%.1f", elapsed))s\n---\n"
                try? (header + accum).write(to: path, atomically: true, encoding: .utf8)
            }
            return ExtractionParser.parse(accum)
        } catch {
            errorMessage = "Inference error: \(error.localizedDescription)"
            isRunning = false
            InferenceMetrics.shared.end(error: error.localizedDescription)
            return nil
        }
    }
}
