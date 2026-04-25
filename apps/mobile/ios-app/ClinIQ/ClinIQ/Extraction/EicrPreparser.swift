// EicrPreparser.swift
// Deterministic eICR code extractor: three tiers, no LLM.
//
//   Tier 1  inline regex             "(SNOMED 76272004)" → 76272004
//   Tier 2  CDA XML attributes       <code code="..." codeSystem="2.16.840.1.113883.6.96"/>
//   Tier 3  curated displayName map  "amoxicillin" → 723
//
// On the 14-case combined bench (apps/mobile/convert/{test_cases.jsonl,test_cases_adversarial.jsonl})
// the Python prototype hits 42/42 = 1.00 with all three tiers enabled.
//
// `ExtractionService.run()` calls `EicrPreparser.extract(_:)` first and
// short-circuits the LLM whenever any code is recovered, dropping demo
// latency from ~46 s (iPhone 14 GPU LLM) to single-digit milliseconds.
//
// This is an exact port of the Python module at
// apps/mobile/convert/regex_preparser.py — keep them in sync; the Python
// version is the bench-tested ground truth.

import Foundation

/// Per-code provenance record. One per emitted code; lets the UI explain
/// "I extracted 50711007 because line 4 says 'Hep C' (lookup tier, conf 0.85)".
/// Confidence floors mirror apps/mobile/convert/regex_preparser.py.
struct CodeProvenance {
    enum Tier: String { case inline, cda, lookup, rag }
    let code: String
    let display: String
    let system: String      // "SNOMED" / "LOINC" / "RxNorm"
    let bucket: String      // "snomed" / "loinc" / "rxnorm"
    let tier: Tier
    let confidence: Double
    let sourceText: String
    let sourceOffset: Int
    let sourceLength: Int
    let alias: String?
    let sourceURL: String?
}

enum EicrPreparser {
    static let tierConfidenceInline: Double = 0.99
    static let tierConfidenceCDA: Double = 0.99
    static let tierConfidenceLookup: Double = 0.85

    // MARK: - Tier 1 — inline parenthesized codes
    //
    // Captures the optional display text preceding "(SNOMED CODE)" so the
    // UI can show "Hepatitis C" alongside "50711007" rather than a bare
    // numeric. The display capture is non-greedy and stops at ":" / "\n"
    // / "(" so multi-clause lines like "Dx: Hep C (SNOMED ...)" don't
    // bleed earlier text into the display.
    private static let inlineSnomed = NSRegularExpression.cached(
        pattern: #"(?:^|[\n,])\s*(?:[A-Za-z][A-Za-z ]{0,30}?:\s*)?([^:\n(]+?)?\s*\(\s*SNOMED[\s:]+(\d{6,9})\s*\)"#,
        options: [.caseInsensitive]
    )
    private static let inlineLoinc = NSRegularExpression.cached(
        pattern: #"(?:^|[\n,])\s*(?:[A-Za-z][A-Za-z ]{0,30}?:\s*)?([^:\n(]+?)?\s*\(\s*LOINC[\s:]+(\d{1,7}-\d)\s*\)"#,
        options: [.caseInsensitive]
    )
    private static let inlineRxNorm = NSRegularExpression.cached(
        pattern: #"(?:^|[\n,])\s*(?:[A-Za-z][A-Za-z ]{0,30}?:\s*)?([^:\n(]+?)?\s*\(\s*RxNorm[\s:]+(\d{2,9})\s*\)"#,
        options: [.caseInsensitive]
    )

    // MARK: - Tier 2 — CDA XML attribute pairs
    //
    // CDA <code code="..." codeSystem="..."/> attribute order is unspecified;
    // we match each tag containing a known codeSystem OID and pull the code
    // attr from inside the same tag. OIDs:
    //   SNOMED CT: 2.16.840.1.113883.6.96
    //   LOINC:     2.16.840.1.113883.6.1
    //   RxNorm:    2.16.840.1.113883.6.88
    private static let cdaTag = NSRegularExpression.cached(
        pattern: #"<[^>]*\bcodeSystem="([^"]+)"[^>]*>"#,
        options: []
    )
    private static let codeAttr = NSRegularExpression.cached(
        pattern: #"\bcode="([^"]+)""#,
        options: []
    )
    private static let displayNameAttr = NSRegularExpression.cached(
        pattern: #"\bdisplayName="([^"]+)""#,
        options: []
    )
    private static let cdaOidToBucket: [String: ParsedKind] = [
        "2.16.840.1.113883.6.96": .condition,
        "2.16.840.1.113883.6.1": .lab,
        "2.16.840.1.113883.6.88": .medication,
    ]

    // MARK: - NegEx-style negation detection (Tier 3 only)
    //
    // Tier 1 (parenthesized) and Tier 2 (CDA XML) are immune because their
    // patterns require explicit code structure that "ruled out" wouldn't be
    // inside. Tier 3 lookup matches a bare displayName alias and so can
    // false-positive on phrases like "Tuberculosis ruled out".
    //
    // Approach: scan a window of ~60 chars before and ~30 chars after the
    // match for negation triggers. Window is clipped at clause terminators
    // so "negative for COVID, positive for influenza" doesn't suppress
    // influenza too. Patterns mirror apps/mobile/convert/regex_preparser.py.
    private static let negTriggers = NSRegularExpression.cached(
        pattern: #"\b(?:ruled\s+out|negative\s+for|no\s+evidence\s+of|no\s+signs?\s+of|no\s+history\s+of|denies|without|absent|not\s+detected|not\s+positive\s+for|not\s+suspected|exclud(?:e|ed|es|ing)|differential\s+(?:diagnosis|dx))\b"#,
        options: [.caseInsensitive]
    )
    private static let negTerminators = NSRegularExpression.cached(
        pattern: #"(?:\bbut\b|\bhowever\b|\balthough\b|\bexcept\b|\.\s|;|\n|<)"#,
        options: [.caseInsensitive]
    )
    private static let negWindowBefore = 60
    private static let negWindowAfter = 30

    /// True if the alias span at [matchStart, matchEnd) sits in negation scope.
    static func isNegated(in text: String, matchStart: Int, matchEnd: Int) -> Bool {
        let utf16 = text.utf16
        let total = utf16.count

        // Backward window — clip at the nearest terminator on the left.
        let leftStart = max(0, matchStart - negWindowBefore)
        let leftRange = NSRange(location: leftStart, length: matchStart - leftStart)
        var preStart = leftStart
        var lastTermEnd = -1
        for tm in negTerminators.matches(in: text, range: leftRange) {
            lastTermEnd = tm.range.location + tm.range.length
        }
        if lastTermEnd >= 0 { preStart = lastTermEnd }
        let preRange = NSRange(location: preStart, length: matchStart - preStart)
        if preRange.length > 0,
           negTriggers.firstMatch(in: text, range: preRange) != nil {
            return true
        }

        // Forward window — clip at the nearest terminator on the right.
        let rightEnd = min(total, matchEnd + negWindowAfter)
        let rightRange = NSRange(location: matchEnd, length: rightEnd - matchEnd)
        var postEnd = rightEnd
        if let firstTerm = negTerminators.firstMatch(in: text, range: rightRange) {
            postEnd = firstTerm.range.location
        }
        let postRange = NSRange(location: matchEnd, length: postEnd - matchEnd)
        if postRange.length > 0,
           negTriggers.firstMatch(in: text, range: postRange) != nil {
            return true
        }

        return false
    }

    // MARK: - Public API

    /// Returns a `ParsedExtraction` populated from the deterministic tiers.
    /// `hasAny` is true iff at least one code was recovered. Convenience
    /// wrapper around `extractWithProvenance` for callers that don't need
    /// per-code source tracking.
    static func extract(_ text: String) -> ParsedExtraction {
        return extractWithProvenance(text).extraction
    }

    /// Like `extract(_:)` but also returns a per-code provenance list with
    /// the source span, tier, alias, and confidence. Used by the iOS UI to
    /// render "I extracted X because line Y says Z" tap-throughs and by the
    /// Gemma 4 agent's `extract_codes_from_text` tool to surface the same
    /// data to the model.
    static func extractWithProvenance(_ text: String)
        -> (extraction: ParsedExtraction, provenance: [CodeProvenance])
    {
        var out = ParsedExtraction()
        out.raw = text

        var seen = SeenCodes()
        var prov: [CodeProvenance] = []

        // Tier 1 — inline parenthesized labels
        for hit in inlineSnomed.allMatches(in: text) {
            guard let code = hit.group(2) else { continue }
            let display = hit.group(1)?.trimmingChars() ?? code
            if seen.addCondition(code: code, system: "SNOMED", display: display, into: &out) {
                let r = hit.result.range
                prov.append(CodeProvenance(
                    code: code, display: display, system: "SNOMED",
                    bucket: "snomed", tier: .inline,
                    confidence: tierConfidenceInline,
                    sourceText: hit.fullMatch,
                    sourceOffset: r.location, sourceLength: r.length,
                    alias: nil, sourceURL: nil
                ))
            }
        }
        for hit in inlineLoinc.allMatches(in: text) {
            guard let code = hit.group(2) else { continue }
            let display = hit.group(1)?.trimmingChars() ?? code
            if seen.addLab(code: code, system: "LOINC", display: display, into: &out) {
                let r = hit.result.range
                prov.append(CodeProvenance(
                    code: code, display: display, system: "LOINC",
                    bucket: "loinc", tier: .inline,
                    confidence: tierConfidenceInline,
                    sourceText: hit.fullMatch,
                    sourceOffset: r.location, sourceLength: r.length,
                    alias: nil, sourceURL: nil
                ))
            }
        }
        for hit in inlineRxNorm.allMatches(in: text) {
            guard let code = hit.group(2) else { continue }
            let display = hit.group(1)?.trimmingChars() ?? code
            if seen.addMedication(code: code, system: "RxNorm", display: display, into: &out) {
                let r = hit.result.range
                prov.append(CodeProvenance(
                    code: code, display: display, system: "RxNorm",
                    bucket: "rxnorm", tier: .inline,
                    confidence: tierConfidenceInline,
                    sourceText: hit.fullMatch,
                    sourceOffset: r.location, sourceLength: r.length,
                    alias: nil, sourceURL: nil
                ))
            }
        }

        // Tier 2 — CDA XML attribute pairs
        for tag in cdaTag.allMatches(in: text) {
            guard let oid = tag.group(1), let kind = cdaOidToBucket[oid] else { continue }
            let tagText = tag.fullMatch
            guard let codeMatch = codeAttr.firstMatch(in: tagText),
                  let code = codeMatch.group(1) else { continue }
            let display = displayNameAttr.firstMatch(in: tagText)?.group(1) ?? code
            let r = tag.result.range
            let added: Bool
            let bucket: String
            let system: String
            switch kind {
            case .condition:
                added = seen.addCondition(code: code, system: "SNOMED", display: display, into: &out)
                bucket = "snomed"; system = "SNOMED"
            case .lab:
                added = seen.addLab(code: code, system: "LOINC", display: display, into: &out)
                bucket = "loinc"; system = "LOINC"
            case .medication:
                added = seen.addMedication(code: code, system: "RxNorm", display: display, into: &out)
                bucket = "rxnorm"; system = "RxNorm"
            }
            if added {
                prov.append(CodeProvenance(
                    code: code, display: display, system: system,
                    bucket: bucket, tier: .cda,
                    confidence: tierConfidenceCDA,
                    sourceText: tagText,
                    sourceOffset: r.location, sourceLength: r.length,
                    alias: nil, sourceURL: nil
                ))
            }
        }

        // Tier 3 — displayName lookup (with NegEx suppression)
        for entry in LookupTable.snomed {
            if let alias = entry.firstAssertedAlias(in: text) {
                if seen.addCondition(code: entry.code, system: "SNOMED", display: alias, into: &out) {
                    let span = entry.firstAssertedSpan(in: text)
                    prov.append(CodeProvenance(
                        code: entry.code, display: alias, system: "SNOMED",
                        bucket: "snomed", tier: .lookup,
                        confidence: tierConfidenceLookup,
                        sourceText: span?.text ?? alias,
                        sourceOffset: span?.location ?? 0,
                        sourceLength: span?.length ?? alias.count,
                        alias: alias, sourceURL: nil
                    ))
                }
            }
        }
        for entry in LookupTable.loincs {
            if let alias = entry.firstAssertedAlias(in: text) {
                if seen.addLab(code: entry.code, system: "LOINC", display: alias, into: &out) {
                    let span = entry.firstAssertedSpan(in: text)
                    prov.append(CodeProvenance(
                        code: entry.code, display: alias, system: "LOINC",
                        bucket: "loinc", tier: .lookup,
                        confidence: tierConfidenceLookup,
                        sourceText: span?.text ?? alias,
                        sourceOffset: span?.location ?? 0,
                        sourceLength: span?.length ?? alias.count,
                        alias: alias, sourceURL: nil
                    ))
                }
            }
        }
        for entry in LookupTable.rxnorms {
            if let alias = entry.firstAssertedAlias(in: text) {
                if seen.addMedication(code: entry.code, system: "RxNorm", display: alias, into: &out) {
                    let span = entry.firstAssertedSpan(in: text)
                    prov.append(CodeProvenance(
                        code: entry.code, display: alias, system: "RxNorm",
                        bucket: "rxnorm", tier: .lookup,
                        confidence: tierConfidenceLookup,
                        sourceText: span?.text ?? alias,
                        sourceOffset: span?.location ?? 0,
                        sourceLength: span?.length ?? alias.count,
                        alias: alias, sourceURL: nil
                    ))
                }
            }
        }

        return (out, prov)
    }
}

// MARK: - Helpers

private enum ParsedKind { case condition, lab, medication }

private struct SeenCodes {
    var conditions = Set<String>()
    var labs = Set<String>()
    var meds = Set<String>()

    /// Returns true iff this code is new (and was therefore appended). Lets
    /// the caller emit a matching CodeProvenance record only on the winning
    /// (earliest-tier) hit.
    mutating func addCondition(code: String, system: String, display: String, into out: inout ParsedExtraction) -> Bool {
        guard conditions.insert(code).inserted else { return false }
        out.conditions.append(ParsedCondition(code: code, system: system, display: display))
        return true
    }
    mutating func addLab(code: String, system: String, display: String, into out: inout ParsedExtraction) -> Bool {
        guard labs.insert(code).inserted else { return false }
        out.labs.append(ParsedLab(code: code, system: system, display: display,
                                  interpretation: nil, value: nil, unit: nil))
        return true
    }
    mutating func addMedication(code: String, system: String, display: String, into out: inout ParsedExtraction) -> Bool {
        guard meds.insert(code).inserted else { return false }
        out.medications.append(ParsedMedication(code: code, system: system, display: display))
        return true
    }
}

private extension String {
    func trimmingChars() -> String {
        trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: ":,;-")))
    }
}

extension ParsedExtraction {
    /// True if the deterministic preparser found at least one code in any category.
    var hasAnyDeterministic: Bool {
        !conditions.isEmpty || !labs.isEmpty || !medications.isEmpty
    }
}

// MARK: - NSRegularExpression conveniences

private struct RegexMatch {
    let result: NSTextCheckingResult
    let source: String
    var fullMatch: String { Self.substring(source, range: result.range) ?? "" }
    func group(_ idx: Int) -> String? {
        guard idx <= result.numberOfRanges - 1 else { return nil }
        let r = result.range(at: idx)
        guard r.location != NSNotFound else { return nil }
        return Self.substring(source, range: r)
    }
    static func substring(_ s: String, range: NSRange) -> String? {
        guard range.location != NSNotFound,
              let r = Range(range, in: s) else { return nil }
        return String(s[r])
    }
}

private extension NSRegularExpression {
    static func cached(pattern: String, options: NSRegularExpression.Options) -> NSRegularExpression {
        // Force-try is acceptable: patterns are static, errors caught in dev.
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: pattern, options: options)
    }

    func allMatches(in source: String) -> [RegexMatch] {
        let range = NSRange(source.startIndex..., in: source)
        return matches(in: source, range: range).map {
            RegexMatch(result: $0, source: source)
        }
    }

    func firstMatch(in source: String) -> RegexMatch? {
        let range = NSRange(source.startIndex..., in: source)
        return firstMatch(in: source, range: range).map {
            RegexMatch(result: $0, source: source)
        }
    }
}
