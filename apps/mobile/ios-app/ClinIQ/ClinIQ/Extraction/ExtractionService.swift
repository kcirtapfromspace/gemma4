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

    private var llmReviewMode: LLMReviewMode {
        let raw = UserDefaults.standard.string(forKey: LLMReviewMode.appStorageKey)
            ?? LLMReviewMode.default.rawValue
        return LLMReviewMode(rawValue: raw) ?? .default
    }

    private var shouldForceLLMAudit: Bool {
        !usingStub && llmReviewMode == .always
    }

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

        // Tier 1 — deterministic preparser. Carries per-code provenance
        // (tier + source span + confidence) into the review UI.
        //
        // Short-circuit gate (c20 Candidate D, see
        // tools/autoresearch/c20-llm-tuning-2026-04-25.md "Live finding from
        // grammar bench"): the prior gate `det.hasAnyDeterministic` fired on
        // any code, so cases like adv3_rmsf_rag where deterministic only
        // recovered the RxNorm via lookup (and missed the SNOMED that lives
        // in the RAG database) would short-circuit and the agent could
        // never fill the gap. The agent path recovered both targets at
        // 2/2 when invoked.
        //
        // Refined gate fires when EITHER:
        //   (a) at least one code came from an explicit-assertion tier —
        //       inline parenthesized "(SNOMED ...)" or CDA <code/> XML —
        //       i.e. the author named the code outright, low FP risk; or
        //   (b) deterministic populated >=2 of the 3 buckets
        //       (conditions / loincs / rxnorms), so we already have a
        //       multi-axis answer and the marginal agent recall isn't worth
        //       the latency.
        // Otherwise (single-bucket lookup-only result), fall through to the
        // fast-path / agent so the LLM can backfill the missing axis. Keeps
        // F1=1.000 on combined-27 unchanged (every case there hits at least
        // one inline/CDA tier or fills >=2 buckets) and recovers
        // adv3_rmsf_rag + adv3_valley_fever_rag from 1/2 → 2/2.
        let detStart = Date()
        let detResult = EicrPreparser.extractWithProvenance(narrative)
        var det = detResult.extraction
        det.matches = detResult.provenance
        let detElapsed = Date().timeIntervalSince(detStart)
        let forceLLMAudit = shouldForceLLMAudit
        var baseline = det
        if det.shortCircuitsLLM && !forceLLMAudit {
            lastElapsed = detElapsed
            lastTokens = 0
            tokensPerSecond = 0
            isRunning = false
            activeBackendLabel = "Deterministic preparser"
            return det
        }
        if !det.conditions.isEmpty && !forceLLMAudit {
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
        // Inject EicrPreparser.isNegated as the negation predicate so the
        // fast-path uses the same NegEx rule as Tier-3 lookup. The closure
        // hop (instead of RagSearch importing EicrPreparser directly)
        // keeps RagSearch self-contained for the validate_rag.swift CLI.
        if let fp = RagSearch.fastPathHit(
            narrative: narrative,
            isNegated: { txt, s, e in
                EicrPreparser.isNegated(in: txt, matchStart: s, matchEnd: e)
            }
        ) {
            // c20 adv6 fix (Bug 5 follow-on): START from det and merge the
            // RAG hit. Det may carry lookup-tier matches (e.g. a drug
            // alias) that should NOT be discarded just because RAG
            // backfilled the missing condition. `adv3_rmsf_rag` style
            // cases drop from 2/2 to 1/2 if we replace.
            var fast = det
            fast.raw = narrative
            let sysu = fp.hit.system.uppercased()
            let codeAlreadyPresent: Bool = {
                switch sysu {
                case "SNOMED":
                    return fast.conditions.contains { $0.code == fp.hit.code }
                case "LOINC":
                    return fast.labs.contains { $0.code == fp.hit.code }
                case "RXNORM":
                    return fast.medications.contains { $0.code == fp.hit.code }
                default:
                    return true
                }
            }()
            if !codeAlreadyPresent {
                switch sysu {
                case "SNOMED":
                    fast.conditions.append(
                        ParsedCondition(code: fp.hit.code,
                                        system: fp.hit.system,
                                        display: fp.hit.display)
                    )
                case "LOINC":
                    fast.labs.append(
                        ParsedLab(code: fp.hit.code,
                                  system: fp.hit.system,
                                  display: fp.hit.display,
                                  interpretation: nil,
                                  value: nil,
                                  unit: nil)
                    )
                case "RXNORM":
                    fast.medications.append(
                        ParsedMedication(code: fp.hit.code,
                                         system: fp.hit.system,
                                         display: fp.hit.display)
                    )
                default:
                    break
                }
                let bucket: String = {
                    switch sysu {
                    case "SNOMED": return "snomed"
                    case "LOINC": return "loinc"
                    case "RXNORM": return "rxnorm"
                    default: return "snomed"
                    }
                }()
                fast.matches.append(
                    CodeProvenance(
                        code: fp.hit.code,
                        display: fp.hit.display,
                        system: fp.hit.system,
                        bucket: bucket,
                        tier: .ragFast,
                        confidence: max(EicrPreparser.tierConfidenceRagFastFloor,
                                        fp.hit.score),
                        sourceText: fp.span.text,
                        sourceOffset: fp.span.location,
                        sourceLength: fp.span.length,
                        alias: fp.hit.matchedPhrase,
                        sourceURL: fp.hit.sourceURL
                    )
                )
            }
            baseline = fast
            if !forceLLMAudit {
                lastElapsed = Date().timeIntervalSince(fpStart) + detElapsed
                lastTokens = 0
                tokensPerSecond = 0
                isRunning = false
                activeBackendLabel = "RAG fast-path"
                return fast
            }
        }
        if baseline.hasAnyDeterministic && !forceLLMAudit {
            lastElapsed = detElapsed
            lastTokens = 0
            tokensPerSecond = 0
            isRunning = false
            activeBackendLabel = "Deterministic preparser"
            return baseline
        }

        // Mobile-first LLM audit. When the deterministic/RAG stack already
        // has a credible baseline, ask the model to audit a compact prompt
        // before paying the full multi-turn tool-agent prefill cost. This is
        // still real LLM usage, but it avoids putting tool declarations into
        // the hot simulator demo path.
        if forceLLMAudit && baseline.hasAnyDeterministic && !usingStub {
            let backendName = "\(Self.label(for: engine)) compact audit"
            let modelName = Self.modelName(for: engine, usingStub: usingStub)
            if let audited = await runCompactLLMAudit(
                narrative: narrative,
                baseline: baseline,
                backendName: backendName,
                modelName: modelName
            ) {
                return audited
            }
        }

        // Tier 2 — Gemma 4 agent loop. The model orchestrates the same
        // deterministic stack as tools, plus RAG over CDC NNDSS / WHO IDSR
        // for codes outside the curated lookup. On Mac llama-server this
        // hits F1=0.986 across 27 cases (originals + adv1 + adv2 + adv3).
        if !usingStub {
            let agentStart = Date()
            let runner = AgentRunner(engine: engine)
            let backendName = "\(Self.label(for: engine)) agent"
            let modelName = Self.modelName(for: engine, usingStub: usingStub)
            var agentOutput = ""
            var agentTokens = 0
            let initialPromptChars = AgentRunner.initialPromptCharacterCount(narrative: narrative)
            InferenceMetrics.shared.begin(backend: backendName,
                                          model: modelName,
                                          promptChars: initialPromptChars,
                                          maxTokens: 2048 * 10)
            do {
                let trace = try await runner.run(narrative: narrative) { [weak self] text, chunkTokens in
                    guard let self else { return }
                    agentOutput += text
                    agentTokens += chunkTokens
                    let elapsed = Date().timeIntervalSince(agentStart)
                    self.tokensPerSecond = elapsed > 0 ? Double(agentTokens) / elapsed : 0
                    self.streamedOutput = agentOutput
                    InferenceMetrics.shared.record(chunkTokens: chunkTokens)
                }
                InferenceMetrics.shared.finalize()
                let elapsed = Date().timeIntervalSince(agentStart)
                lastElapsed = elapsed
                lastTokens = max(agentTokens, trace.outputTokens)
                tokensPerSecond = elapsed > 0 ? Double(lastTokens) / elapsed : 0

                if let extraction = trace.finalExtraction {
                    let merged = Self.merge(modelExtraction: extraction, baseline: baseline)
                    if Self.hasClinicalContent(merged) {
                        isRunning = false
                        activeBackendLabel = forceLLMAudit
                            ? "Gemma 4 agent audit + RAG"
                            : "Gemma 4 agent + RAG"
                        InferenceMetrics.shared.end()
                        return merged
                    }
                }

                InferenceMetrics.shared.end()
                if forceLLMAudit && baseline.hasAnyDeterministic {
                    isRunning = false
                    activeBackendLabel = "Gemma 4 agent audit + deterministic baseline"
                    return baseline
                }
            } catch {
                // Agent loop failed — fall through to legacy raw-LLM path
                // rather than dropping the request.
                InferenceMetrics.shared.end(error: error.localizedDescription)
                if forceLLMAudit && baseline.hasAnyDeterministic {
                    isRunning = false
                    activeBackendLabel = "LLM audit failed; deterministic baseline"
                    return baseline
                }
            }
        }

        if baseline.hasAnyDeterministic {
            isRunning = false
            activeBackendLabel = "Deterministic preparser"
            return baseline
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
        let modelName = Self.modelName(for: engine, usingStub: usingStub)
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

    private func runCompactLLMAudit(
        narrative: String,
        baseline: ParsedExtraction,
        backendName: String,
        modelName: String
    ) async -> ParsedExtraction? {
        let user = """
Narrative:
\(narrative)

Baseline extraction from local deterministic/RAG tools:
\(Self.compactBaselineJSON(baseline))

Audit the baseline against the narrative. Keep supported codes, remove unsupported codes, and add obvious missing SNOMED/LOINC/RxNorm codes. Return only the final minified JSON object.
"""
        let prompt = PromptBuilder.wrapTurns(system: Self.compactAuditSystemPrompt,
                                             user: user)
        let maxTokens = 512
        let start = Date()
        var accum = ""
        var tokens = 0

        activeBackendLabel = "Gemma 4 compact audit + baseline"
        InferenceMetrics.shared.begin(backend: backendName,
                                      model: modelName,
                                      promptChars: prompt.count,
                                      maxTokens: maxTokens)

        do {
            let stream = try await engine.generate(prompt: prompt, maxTokens: maxTokens)
            for try await chunk in stream {
                accum += chunk.text
                tokens += max(1, chunk.tokenCount)
                let elapsed = Date().timeIntervalSince(start)
                tokensPerSecond = elapsed > 0 ? Double(tokens) / elapsed : 0
                streamedOutput = accum
                InferenceMetrics.shared.record(chunkTokens: max(1, chunk.tokenCount))
            }
            InferenceMetrics.shared.finalize()
            let elapsed = Date().timeIntervalSince(start)
            lastElapsed = elapsed
            lastTokens = tokens
            tokensPerSecond = elapsed > 0 ? Double(tokens) / elapsed : 0

            let parsed = ExtractionParser.parse(accum)
            let merged = Self.merge(modelExtraction: parsed, baseline: baseline)
            guard Self.hasClinicalContent(merged) else {
                InferenceMetrics.shared.end()
                return nil
            }

            isRunning = false
            activeBackendLabel = "Gemma 4 compact audit + RAG"
            InferenceMetrics.shared.end()
            Self.writeRawDebugOutput(
                backendName: backendName,
                modelName: modelName,
                tokens: tokens,
                elapsed: elapsed,
                raw: accum
            )
            return merged
        } catch {
            InferenceMetrics.shared.end(error: error.localizedDescription)
            return nil
        }
    }

    private static func modelName(for engine: any InferenceEngine, usingStub: Bool) -> String {
        if engine is LlamaCppInferenceEngine {
            return (LlamaCppInferenceEngine.resolveModelPath() as NSString?)?
                .lastPathComponent ?? "on-device model"
        }
        if engine is LiteRtLmInferenceEngine {
            return (LiteRtLmInferenceEngine.resolveModelPath() as NSString?)?
                .lastPathComponent ?? "on-device model"
        }
        return usingStub ? "—" : "on-device model"
    }

    private static let compactAuditSystemPrompt: String = """
You are an on-device clinical extraction auditor. Given a narrative and a \
baseline extraction, return one minified JSON object using this schema: \
patient.gender, patient.birth_date, encounter_date, \
conditions[{code,system:"SNOMED",display}], \
labs[{code,system:"LOINC",display,value?,unit?,interpretation?}], \
medications[{code,system:"RxNorm",display}], \
vitals.{temp_c,hr,rr,spo2,bp_systolic}. \
Keep baseline codes only when supported by the narrative. Add obvious missing \
codes. Return JSON only, with conditions/labs/medications arrays present.
"""

    private static func compactBaselineJSON(_ extraction: ParsedExtraction) -> String {
        let conditions = extraction.conditions.map {
            ["code": $0.code, "system": $0.system, "display": $0.display]
        }
        let labs = extraction.labs.map {
            ["code": $0.code, "system": $0.system, "display": $0.display]
        }
        let medications = extraction.medications.map {
            ["code": $0.code, "system": $0.system, "display": $0.display]
        }
        let object: [String: Any] = [
            "conditions": conditions,
            "labs": labs,
            "medications": medications,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: []) else {
            return "{\"conditions\":[],\"labs\":[],\"medications\":[]}"
        }
        return String(data: data, encoding: .utf8)
            ?? "{\"conditions\":[],\"labs\":[],\"medications\":[]}"
    }

    private static func writeRawDebugOutput(
        backendName: String,
        modelName: String,
        tokens: Int,
        elapsed: Double,
        raw: String
    ) {
        if let docs = FileManager.default.urls(for: .documentDirectory,
                                                in: .userDomainMask).first {
            let path = docs.appendingPathComponent("last-extraction-raw.txt")
            let header = "backend=\(backendName) model=\(modelName) tokens=\(tokens) elapsed=\(String(format: "%.1f", elapsed))s\n---\n"
            try? (header + raw).write(to: path, atomically: true, encoding: .utf8)
        }
    }

    private static func hasClinicalContent(_ extraction: ParsedExtraction) -> Bool {
        extraction.hasAnyDeterministic
            || extraction.vitals != nil
            || extraction.patientGender != nil
            || extraction.patientBirthDate != nil
            || extraction.encounterDate != nil
    }

    private static func merge(modelExtraction: ParsedExtraction, baseline: ParsedExtraction) -> ParsedExtraction {
        var merged = hasClinicalContent(modelExtraction) ? modelExtraction : baseline

        if merged.patientGender == nil { merged.patientGender = baseline.patientGender }
        if merged.patientBirthDate == nil { merged.patientBirthDate = baseline.patientBirthDate }
        if merged.encounterDate == nil { merged.encounterDate = baseline.encounterDate }
        if merged.vitals == nil { merged.vitals = baseline.vitals }
        if merged.raw.isEmpty {
            merged.raw = modelExtraction.raw.isEmpty ? baseline.raw : modelExtraction.raw
        }

        var conditionCodes = Set(merged.conditions.map(\.code))
        for condition in baseline.conditions where conditionCodes.insert(condition.code).inserted {
            merged.conditions.append(condition)
        }

        var labCodes = Set(merged.labs.map(\.code))
        for lab in baseline.labs where labCodes.insert(lab.code).inserted {
            merged.labs.append(lab)
        }

        var medicationCodes = Set(merged.medications.map(\.code))
        for medication in baseline.medications where medicationCodes.insert(medication.code).inserted {
            merged.medications.append(medication)
        }

        var provenanceKeys = Set(merged.matches.map { "\($0.system)|\($0.code)|\($0.sourceOffset)" })
        for match in baseline.matches {
            let key = "\(match.system)|\(match.code)|\(match.sourceOffset)"
            if provenanceKeys.insert(key).inserted {
                merged.matches.append(match)
            }
        }

        return merged
    }
}
