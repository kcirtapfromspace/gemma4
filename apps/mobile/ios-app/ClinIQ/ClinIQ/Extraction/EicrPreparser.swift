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
    /// `.ragFast` is the c19 single-turn agent fast-path tier: tier-1 was
    /// empty, RAG returned a high-confidence hit (≥0.7), and the matched
    /// phrase is not in NegEx scope, so we skip the agent loop entirely
    /// and emit the RAG hit directly. Reuses the `.rag` purple in the UI
    /// per design but kept distinct so bench rows can disambiguate
    /// fast-path vs. agent-mediated RAG.
    enum Tier: String { case inline, cda, lookup, rag, ragFast = "rag_fast" }
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
    /// Floor used when the c19 single-turn fast-path emits a synthetic
    /// extraction from a RAG hit. Actual confidence is `hit.score`,
    /// clipped to ≥ this floor so the chip doesn't render an unhelpful
    /// "70%" when hit.score is exactly the trigger threshold.
    static let tierConfidenceRagFastFloor: Double = 0.70

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

    // c20 Bug 7: clinical-statement label prefixes that legitimize an inline
    // `(SNOMED 12345)` annotation. When NONE of these labels appears within
    // `inlineLabelWindow` chars before the inline code AND the code is not
    // in the curated lookup table, the inline tier refuses to emit. Mirror
    // of `_INLINE_LABEL_RE` in apps/mobile/convert/regex_preparser.py.
    private static let inlineLabelRe = NSRegularExpression.cached(
        pattern: #"\b(?:Dx|Diagnosis|Diagnoses|Final|Final\s+admission\s+diagnosis|Reason(?:\s+for\s+(?:visit|ED\s+visit|admission))?|Assessment|Assessment\s+and\s+Plan|A/P|Plan|Impression|Clinical\s+impression|Clinical\s+assessment|Lab|Labs|Laboratory|Workup|Med|Meds|Medication|Medications|Rx|Treatment|Imaging|Microbiology|Pathology|Procedure|Problem(?:\s+list)?|History|PMH|Encounter|Vitals|Vital\s+signs)\s*:"#,
        options: [.caseInsensitive]
    )
    private static let inlineLabelWindow = 60

    /// Returns true if the inline code at [matchStart, matchEnd) sits within
    /// `inlineLabelWindow` chars of a clinical-statement label
    /// (`Dx:`, `Lab:`, `Reason for visit:`, ...). Used as the c20 Bug 7
    /// gate alongside the curated-code allowlist.
    static func hasInlineLabelPrefix(in text: String, matchStart: Int) -> Bool {
        let windowStart = max(0, matchStart - inlineLabelWindow)
        let range = NSRange(location: windowStart, length: matchStart - windowStart)
        if range.length <= 0 { return false }
        return inlineLabelRe.firstMatch(in: text, range: range) != nil
    }

    // c20 cross-axis bleed fix: when scanning for lookup-tier matches, mask
    // CDA `<code .../>` attribute spans (and verbatim narrative duplicates
    // of their displayName) with whitespace so an alt_name like "Zika"
    // doesn't fire on a LOINC-display attribute that already belongs to a
    // different-axis explicit code.
    private static let displayNameAttrInTag = NSRegularExpression.cached(
        pattern: #"\bdisplayName="([^"]+)""#,
        options: []
    )

    /// Returns `text` with the contents of every CDA `<code .../>` tag (and
    /// any same-string narrative duplicates of its `displayName` attr)
    /// replaced by length-equal whitespace, so the lookup tier scans only
    /// freeform narrative. Mirror of `_mask_cda_attributes` in
    /// apps/mobile/convert/regex_preparser.py.
    static func maskCdaAttributes(in text: String) -> String {
        // Build a UTF-16 byte buffer so we can mask in place at NSRange
        // offsets. Reassemble as a String at the end via the same encoding.
        var utf16: [UInt16] = Array(text.utf16)
        let nsText = text as NSString
        let total = nsText.length
        let fullRange = NSRange(location: 0, length: total)
        var maskedDisplays: [String] = []
        let space: UInt16 = 0x20

        for tag in cdaTag.matches(in: text, range: fullRange) {
            let r = tag.range
            let oidRange = tag.range(at: 1)
            guard oidRange.location != NSNotFound else { continue }
            let oid = nsText.substring(with: oidRange)
            guard cdaOidToBucket[oid] != nil else { continue }
            let tagText = nsText.substring(with: r)
            // Tag must carry a non-empty `code="..."` attribute for masking
            // to apply (otherwise lookup needs to fire on the displayName,
            // cf. `adv5_cda_displayname_no_code_attr`).
            guard let codeMatch = codeAttr.firstMatch(in: tagText),
                  let codeStr = codeMatch.group(1),
                  !codeStr.isEmpty
            else { continue }
            if let dispMatch = displayNameAttrInTag.firstMatch(in: tagText),
               let disp = dispMatch.group(1),
               disp.count >= 8 {
                maskedDisplays.append(disp)
            }
            // Mask the inside of the tag (keep the angle brackets).
            let innerStart = r.location + 1
            let innerEnd = r.location + r.length - 1
            if innerEnd > innerStart {
                for i in innerStart..<innerEnd where i < utf16.count {
                    utf16[i] = space
                }
            }
        }

        // Second pass: blank out same-string duplicates in narrative cells.
        if !maskedDisplays.isEmpty {
            let sep = #"(?:[^A-Za-z0-9]|<[^>]{0,16}>){1,32}?"#
            // Reassemble current state for regex scanning.
            let tokenRe = try? NSRegularExpression(
                pattern: #"[A-Za-z0-9][A-Za-z0-9\-]*"#,
                options: []
            )
            for disp in maskedDisplays {
                guard let tokenRe = tokenRe else { continue }
                let dispNS = disp as NSString
                let dispRange = NSRange(location: 0, length: dispNS.length)
                let tokenMatches = tokenRe.matches(in: disp, range: dispRange)
                guard tokenMatches.count >= 3 else { continue }
                let tokens: [String] = tokenMatches.map { tm in
                    NSRegularExpression.escapedPattern(for: dispNS.substring(with: tm.range))
                }
                let pattern = #"\b"# + tokens.joined(separator: sep) + #"\b"#
                guard let dupRe = try? NSRegularExpression(pattern: pattern, options: []) else {
                    continue
                }
                // Scan against the CURRENT masked buffer state.
                let current = String(utf16CodeUnits: utf16, count: utf16.count) as String? ?? ""
                let currentNS = current as NSString
                let curRange = NSRange(location: 0, length: currentNS.length)
                for m in dupRe.matches(in: current, range: curRange) {
                    let mr = m.range
                    let from = mr.location
                    let to = mr.location + mr.length
                    for i in from..<min(to, utf16.count) {
                        utf16[i] = space
                    }
                }
            }
        }
        return String(utf16CodeUnits: utf16, count: utf16.count) as String? ?? text
    }

    /// True when a short uppercase acronym alias (≤4 chars) is being used
    /// as a section/data label rather than a clinical assertion. Example:
    /// `CBC: WBC 4.2 (low)` — `CBC` introduces lab values, not a clinical
    /// assertion. Mirror of `_is_label_header_use` in
    /// apps/mobile/convert/regex_preparser.py.
    static func isLabelHeaderUse(alias: String, in text: String, matchEnd: Int) -> Bool {
        if alias.count > 4 || alias != alias.uppercased() { return false }
        let utf16NS = text as NSString
        let lookahead = min(4, utf16NS.length - matchEnd)
        if lookahead <= 0 { return false }
        let tail = utf16NS.substring(with: NSRange(location: matchEnd, length: lookahead))
        // Allow up to 2 chars of whitespace then `:`.
        guard let re = try? NSRegularExpression(pattern: #"^\s{0,2}:"#, options: []) else {
            return tail.first == ":"
        }
        return re.firstMatch(in: tail, range: NSRange(location: 0, length: tail.utf16.count)) != nil
    }

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
    // c20 adv7 fix: drug-context contraindication negation
    // ("avoid doxycycline", "doxycycline contraindicated in pregnancy",
    // "explicit avoidance of NSAIDs"). Pre-window catches "avoid X";
    // post-window catches "X contraindicated".
    //
    // c20 adv7 fix #2: incidental-finding suppression
    // ("Incidental finding on admission screening swab: SARS-CoV-2 RNA",
    // death-with-disease narratives). Catches cases where a code is
    // mentioned only as an incidental screen result, not the active dx.
    private static let negTriggers = NSRegularExpression.cached(
        pattern: #"\b(?:ruled\s+out|negative\s+for|no\s+evidence\s+of|no\s+current\s+evidence\s+of|no\s+signs?\s+of|no\s+history\s+of|denies|without|absent|not\s+detected|not\s+positive\s+for|not\s+suspected|not\s+invoking|do(?:es)?\s+not\s+have|did\s+not\s+have|not\s+eligible\s+for|avoid\w*|contraindicated|incidental\s+finding|incidental\s+screen\w*|incidental\s+(?:noted?|detect\w*|appear\w*|observ\w*)|exclud(?:e|ed|es|ing)|differential\s+(?:diagnosis|dx|includ(?:ed|es|ing)))\b"#,
        options: [.caseInsensitive]
    )
    // c20 adv6 fix: comma added so "no history of stroke, history of HIV"
    // does NOT leak the "no history of" trigger across the comma into the
    // following positive clause. Mirror of `_NEG_TERMINATORS` in
    // apps/mobile/convert/regex_preparser.py.
    private static let negTerminators = NSRegularExpression.cached(
        pattern: #"(?:\bbut\b|\bhowever\b|\balthough\b|\bexcept\b|\.\s|;|\n|<|,)"#,
        options: [.caseInsensitive]
    )
    private static let negWindowBefore = 60
    private static let negWindowAfter = 30

    // Post-hoc NegEx triggers — phrases where the disease name precedes the
    // negation marker ("Zika serology came back negative", "Influenza A
    // returned negative"). These multi-word constructions can be more than
    // 30 chars after the alias due to intervening lab/method tokens, so we
    // scan a wider clause-bounded window (capped at 80 chars). Each pattern
    // is highly discriminative so over-firing is minimal.
    // Mirror of `_POSTHOC_NEG_TRIGGERS` in
    // apps/mobile/convert/regex_preparser.py.
    // c20 adv7 fix: post-hoc incidental + death-with-disease.
    // `... result was an incidental screening finding`,
    // `... did not contribute to the cause of death`. Multi-encounter
    // cases that legitimately mention "incidental but clinically
    // significant" sit >100 chars from their curated aliases (and behind
    // a clause terminator), so these triggers don't regress on them.
    private static let postHocNegTriggers = NSRegularExpression.cached(
        pattern: #"\b(?:came\s+back\s+negative|returned\s+negative|(?:was|is|were|are)\s+negative|reported\s+negative|results?\s+(?:was|were|is|are)\s+negative|IgM\s+negative|IgG\s+negative|(?:was|is|were)\s+(?:an?\s+)?incidental(?:\s+\w+){0,2}\s+(?:finding|screening|note\w*|swab\w*)|did\s+not\s+contribute(?:\s+to)?)\b"#,
        options: [.caseInsensitive]
    )
    private static let postHocTerminators = NSRegularExpression.cached(
        pattern: #"(?:\.\s|;|\n|,)"#,
        options: [.caseInsensitive]
    )
    private static let postHocWindowAfter = 80

    // c20 final pass: vaccine-context negation. "No varicella series" /
    // "never received MMR booster" — disease/agent name is named only as
    // a vaccine target, NOT as an active diagnosis. Mirror of
    // `_VAX_PRE_TRIGGERS` and `_VAX_POST_NOUNS` in
    // apps/mobile/convert/regex_preparser.py.
    private static let vaxPreTriggers = NSRegularExpression.cached(
        pattern: #"\b(?:no(?:t)?(?:\s+up\s+to\s+date(?:\s+on)?)?|never(?:\s+received|\s+had|\s+got)?|did\s+not\s+(?:receive|have|get)|missed|declined|refused|skipped)\b"#,
        options: [.caseInsensitive]
    )
    private static let vaxPostNouns = NSRegularExpression.cached(
        pattern: #"^\W{0,3}\S{0,30}?\s*\b(?:series|booster|vaccine|vaccination|immuniz\w*|shot|dose|MMR\b)"#,
        options: [.caseInsensitive]
    )

    /// True when the alias is named in a "No <X> series/booster/vaccine"
    /// context. Pre-window (12 chars): one of `no`/`never`/`missed`/etc.
    /// Post-window (40 chars from match_end): a vaccine-context noun
    /// (series/booster/vaccine/vaccination/immunization/shot/dose). Both
    /// must be present — either alone is too loose.
    static func isVaccineContextNegation(in text: String,
                                         matchStart: Int,
                                         matchEnd: Int) -> Bool {
        let utf16Count = text.utf16.count
        let preStart = max(0, matchStart - 12)
        let preRange = NSRange(location: preStart, length: matchStart - preStart)
        if preRange.length <= 0 { return false }
        if vaxPreTriggers.firstMatch(in: text, range: preRange) == nil {
            return false
        }
        let postEnd = min(utf16Count, matchEnd + 40)
        let postRange = NSRange(location: matchEnd, length: postEnd - matchEnd)
        if postRange.length <= 0 { return false }
        return vaxPostNouns.firstMatch(in: text, range: postRange) != nil
    }

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

        // Post-hoc forward scan — wider clause-bounded window for very
        // specific multi-word negation constructions ("X came back negative").
        let postHocEnd = min(total, matchEnd + postHocWindowAfter)
        let postHocRange = NSRange(location: matchEnd, length: postHocEnd - matchEnd)
        var postHocSpanEnd = postHocEnd
        if let firstTerm = postHocTerminators.firstMatch(in: text, range: postHocRange) {
            postHocSpanEnd = firstTerm.range.location
        }
        let postHocSpan = NSRange(location: matchEnd, length: postHocSpanEnd - matchEnd)
        if postHocSpan.length > 0,
           postHocNegTriggers.firstMatch(in: text, range: postHocSpan) != nil {
            return true
        }

        // c20 final pass: vaccine-context negation
        // ("No varicella series", "never received MMR vaccine"). Long
        // admission notes enumerate immunization history in this form;
        // without this rule they leak vaccine-target diseases as if
        // actively diagnosed. Mirror of `_is_vaccine_context_negation`.
        if isVaccineContextNegation(in: text,
                                    matchStart: matchStart,
                                    matchEnd: matchEnd) {
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
    static func extractWithProvenance(_ rawText: String)
        -> (extraction: ParsedExtraction, provenance: [CodeProvenance])
    {
        // c20 adv6 fix: NFKC normalize so non-breaking space (U+00A0),
        // smart quotes, em/en dashes etc. collapse to ASCII forms before
        // regex matching. `precomposedStringWithCompatibilityMapping` is
        // Swift's NFKC equivalent. Mirror of `unicodedata.normalize("NFKC", ...)`
        // in apps/mobile/convert/regex_preparser.py extract().
        let text = rawText.precomposedStringWithCompatibilityMapping
        var out = ParsedExtraction()
        out.raw = text

        var seen = SeenCodes()
        var prov: [CodeProvenance] = []

        // Tier 1 — inline parenthesized labels.
        //
        // c20 Bug 7 gate: emit ONLY when (a) the code is curated in
        // `LookupTable` (we already know about it) OR (b) a clinical
        // statement label appears within `inlineLabelWindow` chars before
        // the match. Adversarial bait substrings like "(SNOMED 99999999)"
        // inside quoted internet-rumor prose carry no preceding label, so
        // they fall through. Mirror of `_extract_inline` in
        // apps/mobile/convert/regex_preparser.py.
        for hit in inlineSnomed.allMatches(in: text) {
            guard let code = hit.group(2) else { continue }
            let display = hit.group(1)?.trimmingChars() ?? code
            // Locate the SNOMED token's position so we can window-check
            // for a clinical-statement label preceding it.
            // Window-check anchored at the END of the match (just past the
            // closing paren). The fullMatch already swallows any preceding
            // label so we look further back to be sure of catching the
            // label even when the match span starts at the label itself.
            let codeEnd = hit.result.range.location + hit.result.range.length
            let curated = LookupTable.snomed.contains { $0.code == code }
            let hasLabel = hasInlineLabelPrefix(in: text, matchStart: codeEnd)
            if !curated && !hasLabel { continue }
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
            let codeEnd = hit.result.range.location + hit.result.range.length
            let curated = LookupTable.loincs.contains { $0.code == code }
            let hasLabel = hasInlineLabelPrefix(in: text, matchStart: codeEnd)
            if !curated && !hasLabel { continue }
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
            let codeEnd = hit.result.range.location + hit.result.range.length
            let curated = LookupTable.rxnorms.contains { $0.code == code }
            let hasLabel = hasInlineLabelPrefix(in: text, matchStart: codeEnd)
            if !curated && !hasLabel { continue }
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

        // Tier 3 — displayName lookup (with NegEx suppression).
        //
        // c20 cross-axis bleed fix: scan a masked copy of the narrative
        // where CDA `<code .../>` attribute spans (and any verbatim
        // duplicates of their `displayName`) are replaced with whitespace.
        // This prevents "Zika" alt_name from firing on a LOINC's
        // `displayName="Zika virus envelope..."` already accounted for by
        // the CDA tier. Spans returned point into the original `text` for
        // provenance fidelity.
        let scanText = maskCdaAttributes(in: text)
        for entry in LookupTable.snomed {
            if let alias = entry.firstAssertedAlias(in: scanText) {
                if seen.addCondition(code: entry.code, system: "SNOMED", display: alias, into: &out) {
                    let span = entry.firstAssertedSpan(in: scanText)
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
            if let alias = entry.firstAssertedAlias(in: scanText) {
                if seen.addLab(code: entry.code, system: "LOINC", display: alias, into: &out) {
                    let span = entry.firstAssertedSpan(in: scanText)
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
            if let alias = entry.firstAssertedAlias(in: scanText) {
                if seen.addMedication(code: entry.code, system: "RxNorm", display: alias, into: &out) {
                    let span = entry.firstAssertedSpan(in: scanText)
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
    /// True if the deterministic preparser found at least one code in any
    /// category. Used by the agent path to confirm a non-empty result; NOT
    /// used as the LLM short-circuit gate any more — see `shortCircuitsLLM`.
    var hasAnyDeterministic: Bool {
        !conditions.isEmpty || !labs.isEmpty || !medications.isEmpty
    }

    /// True iff the deterministic result is strong enough to bypass the
    /// LLM/RAG agent loop. c20 final cleanup — Cand D refinement (see
    /// `tools/autoresearch/c20-llm-tuning-2026-04-25.md`).
    ///
    /// Fires ONLY when at least one match came from an explicit-assertion
    /// tier (inline `(SNOMED 12345)` or CDA `<code .../>`). Lookup-tier-only
    /// results — including multi-bucket lookup-only — fall through to the
    /// fast-path / agent so the LLM can verify them against context.
    ///
    /// Why drop the old `bucketCount >= 2` clause? Bug 5 from adv6:
    /// `adv6_long_form_admission_note` had two lookup-tier FPs (varicella
    /// from "no varicella series", CBC from "CBC: WBC...") that span 2
    /// buckets, short-circuiting before the agent / RAG fast-path could
    /// recover the actual diagnosis (measles). Lookup-tier matches are
    /// inherently ambiguous — alias→code mapping carries no contextual
    /// confidence — so they should not gate the LLM out.
    ///
    /// For previously-short-circuiting cases with correct lookup-only
    /// results (e.g. adv2_h5n1_avian_flu, adv2_mpox), the fast-path
    /// merge logic in `ExtractionService` preserves the lookup matches
    /// and adds the RAG hit, so the extraction stays correct.
    var shortCircuitsLLM: Bool {
        return matches.contains { m in
            m.tier == .inline || m.tier == .cda
        }
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
