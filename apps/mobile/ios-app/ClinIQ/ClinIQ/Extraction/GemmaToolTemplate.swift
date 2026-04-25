// GemmaToolTemplate.swift
// Hand-rolled Gemma 4 tool-calling chat template renderer.
//
// Reverse-engineered from the Jinja template embedded in the C16 .litertlm
// bundle (see apps/mobile/convert/build/litertlm/cliniq-gemma4-e2b.litertlm
// `jinja_prompt_template`). Key tokens:
//
//   <|turn>system\n ... <turn|>          system + tool declarations
//   <|turn>user\n ... <turn|>            user message
//   <|turn>model\n                       assistant turn opener (no closer until done)
//   <|tool>declaration:NAME{...}<tool|>  one per tool, inside system turn
//   <|tool_call>call:NAME{k:v}<tool_call|>   model emits to invoke a tool
//   <|tool_response>response:NAME{k:v}<tool_response|>  we emit back
//
// Argument values use the bundle template's <|"|> string-quote sentinel
// rather than real JSON quotes; that's deliberate — the model was trained
// to recognize that token. We use the same in both directions.

import Foundation

enum GemmaToolTemplate {
    private static let stringQuote = "<|\"|>"

    /// Initial prompt: system + tool declarations + user, ready for the
    /// model to start its first model-turn. Mirrors the Conversation.
    static func renderInitial(
        system: String,
        tools: [AgentTool],
        user: String
    ) -> String {
        var out = ""
        out += PromptBuilder.turnSysOpen
        out += system.trimmingCharacters(in: .whitespacesAndNewlines)
        for tool in tools {
            out += "<|tool>" + renderDeclaration(tool) + "<tool|>"
        }
        out += PromptBuilder.turnClose
        out += PromptBuilder.turnUserOpen
        out += user.trimmingCharacters(in: .whitespacesAndNewlines)
        out += PromptBuilder.turnClose
        out += PromptBuilder.turnModelOpen
        return out
    }

    /// Append the assistant's tool calls to the running prompt. The model
    /// emitted these — we reflect them back so the next prefill sees a
    /// complete history.
    static func renderAssistantToolCalls(_ calls: [GemmaToolCall]) -> String {
        var out = ""
        for call in calls {
            out += "<|tool_call>call:" + call.name + "{"
            out += renderArguments(call.arguments)
            out += "}<tool_call|>"
        }
        return out
    }

    /// Render a tool execution result back into the prompt context.
    static func renderToolResponse(name: String, response: [String: Any]) -> String {
        var out = "<|tool_response>response:" + name + "{"
        out += renderObjectBody(response)
        out += "}<tool_response|>"
        return out
    }

    // MARK: - Internal formatters

    private static func renderDeclaration(_ tool: AgentTool) -> String {
        // Format: declaration:NAME{description:<|"|>...<|"|>,parameters:{...},type:OBJECT}
        // The bundle template generates this from the OpenAI tool-call dict.
        // We hand-write it to match the wire format the model was trained on.
        var s = "declaration:" + tool.name + "{"
        s += "description:" + stringQuote + escape(tool.description) + stringQuote
        s += ",parameters:"
        s += tool.parametersJSON.replacingOccurrences(of: "\"", with: stringQuote)
        s += ",type:" + stringQuote + "OBJECT" + stringQuote
        s += "}"
        return s
    }

    private static func renderArguments(_ args: [String: Any]) -> String {
        // call(args) ⇒ key:formatArg, key2:formatArg2 — keys NOT quoted
        let sortedKeys = args.keys.sorted()
        return sortedKeys.map { k in "\(k):\(renderArgValue(args[k]!))" }
            .joined(separator: ",")
    }

    private static func renderObjectBody(_ obj: [String: Any]) -> String {
        let sortedKeys = obj.keys.sorted()
        return sortedKeys.map { k in "\(k):\(renderArgValue(obj[k]!))" }
            .joined(separator: ",")
    }

    private static func renderArgValue(_ value: Any) -> String {
        if let s = value as? String {
            return stringQuote + escape(s) + stringQuote
        }
        if let b = value as? Bool {
            return b ? "true" : "false"
        }
        if let i = value as? Int { return String(i) }
        if let d = value as? Double {
            // Avoid trailing ".0" when value is integral
            if d.truncatingRemainder(dividingBy: 1) == 0 {
                return String(Int(d))
            }
            return String(d)
        }
        if let arr = value as? [Any] {
            return "[" + arr.map { renderArgValue($0) }.joined(separator: ",") + "]"
        }
        if let dict = value as? [String: Any] {
            return "{" + renderObjectBody(dict) + "}"
        }
        if value is NSNull { return "null" }
        // Fallback to JSON-like string
        return stringQuote + escape(String(describing: value)) + stringQuote
    }

    private static func escape(_ s: String) -> String {
        // Strip the sentinel from user-supplied content so it can't break out.
        return s.replacingOccurrences(of: stringQuote, with: "")
    }
}
