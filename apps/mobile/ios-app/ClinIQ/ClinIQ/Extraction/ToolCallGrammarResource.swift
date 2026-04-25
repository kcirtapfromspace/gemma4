// ToolCallGrammarResource.swift
// Mirror of `apps/mobile/convert/cliniq_toolcall.gbnf` — the source-of-truth
// GBNF grammar lives there and is the file Python's `agent_pipeline.py`
// reads via `--tool-call-grammar`. We inline a copy here because adding a
// bundled .gbnf resource to the iOS target would mean editing
// `ClinIQ.xcodeproj/project.pbxproj` (Resources phase) — a riskier change
// than a Swift string literal that the compiler validates.
//
// **Sync rule:** when editing this grammar, edit the `.gbnf` file FIRST
// (Python uses it directly), then paste the updated rules into the literal
// below. The `validate_grammar_resource.py` tool (apps/mobile/convert) can
// be wired into pre-commit later if drift becomes an issue. For now, the
// agent loop will surface a parse failure (via
// `LlamaCppInferenceEngine.applyGrammar` returning false) if we ship a
// broken grammar — preferable to silent regression.

import Foundation

enum ToolCallGrammarResource {
    /// Verbatim copy of `apps/mobile/convert/cliniq_toolcall.gbnf`.
    /// llama.cpp's GBNF parser ignores `#` comments + blank lines.
    static let gbnf: String = """
    # GBNF grammar for ClinIQ tool-call wire format (Gemma 4 native function-calling).
    # See apps/mobile/convert/cliniq_toolcall.gbnf for the canonical version
    # and rationale comments. This Swift mirror is loaded by AgentRunner on
    # tool-response turns; it must stay byte-identical to the .gbnf file.

    root            ::= tool-call | final-answer

    # ----------------------------------------------------------------------------
    # (a) Tool-call wire format

    tool-call       ::= "<|tool_call>" "call:" tool-name "{" arg-list? "}" "<tool_call|>"

    tool-name       ::= "extract_codes_from_text"
                      | "lookup_reportable_conditions"
                      | "validate_fhir_extraction"
                      | "lookup_displayname"

    arg-list        ::= arg-pair ("," arg-pair)*
    arg-pair        ::= ws key ws ":" ws value ws

    key             ::= identifier | sentinel-string | json-string
    identifier      ::= [a-zA-Z_] [a-zA-Z0-9_]*

    value           ::= sentinel-string | json-string | number | bool | null | array | object
    array           ::= "[" ws (value (ws "," ws value)*)? ws "]"
    object          ::= "{" ws (arg-pair (ws "," ws arg-pair)*)? ws "}"

    number          ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)?
    bool            ::= "true" | "false"
    null            ::= "null"

    sentinel-string ::= "<|\\"|>" sentinel-body "<|\\"|>"
    sentinel-body   ::= ([^<] | "<" [^|] | "<|" [^"])*
    json-string     ::= "\\"" json-body "\\""
    json-body       ::= ([^"\\\\] | "\\\\" .)*

    # ----------------------------------------------------------------------------
    # (b) Final-answer JSON

    final-answer    ::= ws "{" ws "\\"conditions\\"" ws ":" ws code-array ws ","
                           ws "\\"loincs\\""     ws ":" ws code-array ws ","
                           ws "\\"rxnorms\\""    ws ":" ws code-array ws "}" ws

    code-array      ::= "[" ws (code-string (ws "," ws code-string)*)? ws "]"
    code-string     ::= "\\"" [0-9] [0-9-]* "\\""

    # ----------------------------------------------------------------------------
    # Whitespace

    ws              ::= [ \\t\\n]*
    """
}
