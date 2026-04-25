// AgentRunner.swift
// Swift mirror of apps/mobile/convert/agent_pipeline.py.
//
// Architecture:
//   InferenceEngine (raw prompt-in / token-stream-out) is the bottom layer.
//   AgentRunner sits ABOVE: it formats the Gemma 4 tool-calling chat
//   template, parses <|tool_call> tokens from the model's stream, executes
//   the named tool (via Tool.execute), appends the tool response to the
//   prompt, and re-runs generate() until the model emits a final turn that
//   doesn't include a tool call.
//
// Tools available to the agent — same surface as the Python agent:
//   - extract_codes_from_text(text)        → EicrPreparser deterministic 3-tier
//   - lookup_displayname(name, codeset)    → single-name lookup against curated table
//   - lookup_reportable_conditions(query)  → RAG over CDC NNDSS / WHO IDSR
//   - validate_fhir_extraction(extraction) → structural sanity check
//
// On the 27-case combined bench the Mac llama-server agent hits F1=0.986.
// Once this Swift port is wired into ExtractionService, the iOS app can
// run the same loop on-device against the LlamaCppInferenceEngine.

import Foundation

// MARK: - Tool surface

struct AgentTool {
    let name: String
    let description: String
    let parametersJSON: String   // Gemma 4 tool-declaration parameter schema
    let execute: ([String: Any]) -> [String: Any]
}

// MARK: - Trace

struct AgentTrace {
    var turns: [AgentTurn] = []
    var toolEvents: [AgentToolEvent] = []
    var finalExtraction: ParsedExtraction?
}

struct AgentTurn {
    let index: Int
    let rawOutput: String
    let toolCalls: [GemmaToolCall]
    let elapsedSeconds: Double
}

struct AgentToolEvent {
    let turnIndex: Int
    let toolName: String
    let arguments: [String: Any]
    let result: [String: Any]
}

// MARK: - Built-in tools (regex + RAG, no model)

enum AgentTools {
    /// Default tool set — wraps the deterministic stack as agent tools so a
    /// truly on-device agent can use the same architecture as the Python
    /// reference implementation.
    static func defaults() -> [AgentTool] {
        return [
            AgentTool(
                name: "extract_codes_from_text",
                description: """
                    Extract SNOMED, LOINC, and RxNorm codes from clinical \
                    narrative text using a deterministic 3-tier extractor \
                    (inline parenthesized codes + CDA XML attribute parsing \
                    + curated displayName lookup with NegEx negation \
                    suppression). Returns three arrays + per-code provenance. \
                    Run this FIRST.
                    """,
                parametersJSON: """
                    {"properties":{"text":{"description":"Raw clinical narrative.","type":"STRING"}},"required":["text"],"type":"OBJECT"}
                    """,
                execute: { args in
                    let text = (args["text"] as? String) ?? ""
                    let (extraction, prov) = EicrPreparser.extractWithProvenance(text)
                    return [
                        "conditions": extraction.conditions.map { $0.code },
                        "loincs": extraction.labs.map { $0.code },
                        "rxnorms": extraction.medications.map { $0.code },
                        "matches": prov.map { p -> [String: Any] in
                            var d: [String: Any] = [
                                "code": p.code,
                                "display": p.display,
                                "system": p.system,
                                "tier": p.tier.rawValue,
                                "confidence": p.confidence,
                                "source_text": p.sourceText,
                                "source_offset": p.sourceOffset,
                                "source_length": p.sourceLength,
                            ]
                            if let a = p.alias { d["alias"] = a }
                            if let u = p.sourceURL { d["source_url"] = u }
                            return d
                        },
                    ]
                }
            ),
            AgentTool(
                name: "lookup_reportable_conditions",
                description: """
                    Search the curated reportable-conditions database (CDC \
                    NNDSS + WHO IDSR, ~50 entries) for candidate codes when \
                    the deterministic extractor and the displayName lookup \
                    miss. Returns top_k candidates with score, source, and \
                    source_url. Trust results with score >= 0.4.
                    """,
                parametersJSON: """
                    {"properties":{"query":{"description":"Disease name or short clinical description.","type":"STRING"},"top_k":{"description":"Max results (default 3).","type":"INTEGER"}},"required":["query"],"type":"OBJECT"}
                    """,
                execute: { args in
                    let query = (args["query"] as? String) ?? ""
                    let topK = (args["top_k"] as? Int) ?? 3
                    let hits = RagSearch.search(query: query, topK: topK)
                    return [
                        "query": query,
                        "results": hits.map { h -> [String: Any] in
                            var d: [String: Any] = [
                                "code": h.code,
                                "system": h.system,
                                "display": h.display,
                                "score": h.score,
                                "source": h.source,
                                "source_url": h.sourceURL,
                                "alt_names": h.altNames,
                            ]
                            if let m = h.matchedPhrase { d["matched_phrase"] = m }
                            if let c = h.category { d["category"] = c }
                            return d
                        },
                    ]
                }
            ),
            AgentTool(
                name: "validate_fhir_extraction",
                description: """
                    Validate the structure of a final extraction object. \
                    Confirms keys conditions/loincs/rxnorms exist and are \
                    arrays. Run this LAST before producing the final answer.
                    """,
                parametersJSON: """
                    {"properties":{"extraction":{"properties":{"conditions":{"items":{"type":"STRING"},"type":"ARRAY"},"loincs":{"items":{"type":"STRING"},"type":"ARRAY"},"rxnorms":{"items":{"type":"STRING"},"type":"ARRAY"}},"required":["conditions","loincs","rxnorms"],"type":"OBJECT"}},"required":["extraction"],"type":"OBJECT"}
                    """,
                execute: { args in
                    var issues: [String] = []
                    let extraction = args["extraction"] as? [String: Any] ?? [:]
                    for key in ["conditions", "loincs", "rxnorms"] {
                        if extraction[key] == nil {
                            issues.append("missing key '\(key)'")
                        } else if !(extraction[key] is [Any]) {
                            issues.append("'\(key)' must be an array")
                        }
                    }
                    return [
                        "valid": issues.isEmpty,
                        "issues": issues,
                    ]
                }
            ),
        ]
    }
}

// MARK: - Runner

@MainActor
final class AgentRunner {
    private let engine: any InferenceEngine
    private let tools: [AgentTool]
    private let toolByName: [String: AgentTool]
    private let systemPrompt: String
    private let maxTurns: Int
    private let maxTokensPerTurn: Int

    init(
        engine: any InferenceEngine,
        tools: [AgentTool] = AgentTools.defaults(),
        systemPrompt: String = AgentRunner.defaultSystemPrompt,
        maxTurns: Int = 10,
        maxTokensPerTurn: Int = 2048
    ) {
        self.engine = engine
        self.tools = tools
        self.toolByName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name, $0) })
        self.systemPrompt = systemPrompt
        self.maxTurns = maxTurns
        self.maxTokensPerTurn = maxTokensPerTurn
    }

    static let defaultSystemPrompt: String = """
        You are a clinical NLP agent. Given an eICR narrative, produce a JSON \
        object with three keys: 'conditions' (SNOMED), 'loincs' (LOINC), and \
        'rxnorms' (RxNorm).

        MANDATORY workflow — execute steps in order:
        1. Call extract_codes_from_text(text) ONCE on the full narrative.
        2. If 'conditions' is EMPTY in the result AND the narrative mentions \
        ANY disease name (formal, colloquial, abbreviation), you MUST call \
        lookup_reportable_conditions(query=<disease name>). Take the top \
        result if score >= 0.4 and add its code to conditions.
        3. Call validate_fhir_extraction once on your final JSON.
        4. Reply with ONLY the validated JSON object — no extra prose.

        Do NOT call extract_codes_from_text more than once.
        """

    /// Run the agent loop against a clinical narrative. Returns the parsed
    /// final extraction plus a full trace of turns and tool events.
    func run(narrative: String) async throws -> AgentTrace {
        var trace = AgentTrace()
        var prompt = GemmaToolTemplate.renderInitial(
            system: systemPrompt,
            tools: tools,
            user: narrative
        )

        for turnIndex in 0..<maxTurns {
            let turnStart = Date()
            var rawOutput = ""
            let stream = try await engine.generate(prompt: prompt, maxTokens: maxTokensPerTurn)
            for try await chunk in stream {
                rawOutput += chunk.text
            }
            let elapsed = Date().timeIntervalSince(turnStart)
            let calls = ToolCallParser.parse(rawOutput)
            trace.turns.append(AgentTurn(
                index: turnIndex,
                rawOutput: rawOutput,
                toolCalls: calls,
                elapsedSeconds: elapsed
            ))

            if calls.isEmpty {
                // Final answer — try to parse JSON object from the output.
                trace.finalExtraction = ExtractionParser.parse(rawOutput)
                return trace
            }

            // Append the assistant's turn (containing the tool calls) to the
            // prompt, then execute each tool and append the responses.
            prompt += GemmaToolTemplate.renderAssistantToolCalls(calls)
            for call in calls {
                let result = (toolByName[call.name]?.execute(call.arguments)) ?? [
                    "error": "unknown tool '\(call.name)'",
                ]
                trace.toolEvents.append(AgentToolEvent(
                    turnIndex: turnIndex,
                    toolName: call.name,
                    arguments: call.arguments,
                    result: result
                ))
                prompt += GemmaToolTemplate.renderToolResponse(name: call.name, response: result)
            }
            // Open the next assistant turn so the model continues.
            prompt += PromptBuilder.turnModelOpen
        }

        // Hit max_turns without a final answer; salvage whatever JSON we have.
        if let lastRaw = trace.turns.last?.rawOutput {
            trace.finalExtraction = ExtractionParser.parse(lastRaw)
        }
        return trace
    }
}
