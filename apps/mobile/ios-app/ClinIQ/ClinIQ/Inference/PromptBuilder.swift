// PromptBuilder.swift
// Wraps the (system, user) pair in the unsloth gemma-4 turn delimiters that
// our fine-tune was trained against. Matches `validate_litertlm.py` and our
// Kaggle training config. See apps/mobile/convert/validate_litertlm.py lines
// 100-110 for the provenance of these exact delimiter strings.

import Foundation

enum PromptBuilder {
    static let turnSysOpen   = "<|turn>system\n"
    static let turnUserOpen  = "<|turn>user\n"
    static let turnModelOpen = "<|turn>model\n"
    static let turnClose     = "<turn|>\n"

    static func wrapTurns(system: String, user: String) -> String {
        "\(turnSysOpen)\(system)\(turnClose)\(turnUserOpen)\(user)\(turnClose)\(turnModelOpen)"
    }
}

enum SystemPrompt {
    // A deliberately compact version of the skill's system prompt. Long
    // system prompts on a 2.5 GB model + simulator's ~4 GB RAM cap will
    // blow memory. See SKILL.md for the full version used in training.
    //
    // Token budget for this prompt: ~180 tokens (BPE-approx). Leaves room
    // for ~800-token eICRs and ~300-token JSON outputs within a 2048-token
    // context window.
    static let clinicalExtraction: String = """
You are a clinical NLP assistant. Given an eICR summary, emit a single \
minified JSON object with these fields when present (omit if absent):
patient.gender (M/F/U), patient.birth_date (YYYY-MM-DD), \
encounter_date (YYYY-MM-DD), \
conditions[{code,system:\"SNOMED\",display}], \
labs[{code,system:\"LOINC\",display,value?,unit?,interpretation?}], \
medications[{code,system:\"RxNorm\",display}], \
vitals.{temp_c,hr,rr,spo2,bp_systolic}.
Rules: use existing codes in parentheses verbatim; return minified JSON only, \
no prose, no markdown fences; interpretation is one of positive/negative/\
detected/not detected/normal/abnormal. Empty input → \
{\"conditions\":[],\"labs\":[],\"medications\":[]}.
"""
}
