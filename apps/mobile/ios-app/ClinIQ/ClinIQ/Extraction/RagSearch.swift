// RagSearch.swift
// Tier 4 of the deterministic stack: keyword/phrase search over the curated
// ReportableConditions database. Mirror of apps/mobile/convert/rag_search.py
// — same scoring, same apostrophe normalization, same sort order.
//
//   exact-phrase match on display or any alt_name → +1.0
//   token-overlap F1 (fraction match)              → 0.0–0.5
//   long-token bonus (≥6 chars, per matched token) → +0.1 each
//
// Returns top-k results scoring above min_score, sorted descending. Each
// hit carries source + sourceURL for provenance display in the iOS UI.
//
// Complexity: O(N entries × M alt_names) — fine for ~50 entries × ~3 aliases
// each. No embeddings, no on-device ML inference. Easy to debug; easy to
// expand by appending to ReportableConditions.all.

import Foundation

struct RagHit {
    let code: String
    let system: String
    let display: String
    let score: Double
    let matchedPhrase: String?
    let source: String
    let sourceURL: String
    let category: String?
    let altNames: [String]
}

enum RagSearch {
    static let defaultMinScore: Double = 0.2
    static let defaultTopK: Int = 3
    /// Threshold used by the c19 single-turn fast-path. Mirror of the Python
    /// `--fast-path-rag-threshold` default in `apps/mobile/convert/agent_pipeline.py`.
    /// Per proposals-2026-04-25 Rank 2.
    static let fastPathThreshold: Double = 0.70

    /// c20 final cleanup: short alt_names like 'CRE', 'RSV', 'TB', 'HUS'
    /// are 3-4 char acronyms that substring-match into unrelated English
    /// words ("CRE" → "saCREd", "DM" → "Madam"). When the matched candidate
    /// is a short alt_name, require a word-bounded match in the narrative
    /// so "Providence Sacred Heart" no longer triggers the Carbapenem-
    /// resistant Enterobacteriaceae phrase bonus. Mirror of `_SHORT_ALT_MAX_LEN`
    /// in apps/mobile/convert/rag_search.py.
    static let shortAltMaxLen: Int = 4

    private static let tokenRe: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: #"\b[a-z0-9][a-z0-9-]+\b"#, options: [.caseInsensitive])
    }()

    private static let apostropheRe: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: #"['‘’]s?\b"#, options: [.caseInsensitive])
    }()

    /// Strip apostrophes + possessives so "Legionnaires'" matches "Legionnaires".
    static func normalize(_ text: String) -> String {
        let range = NSRange(text.startIndex..., in: text)
        return apostropheRe.stringByReplacingMatches(
            in: text, range: range, withTemplate: ""
        )
    }

    /// Word-bounded case-insensitive search for short alt_name acronyms.
    /// Both args MUST already be lower-cased and apostrophe-normalized.
    /// Used to filter out spurious 3-4 char acronym substring matches
    /// inside longer words ("CRE" inside "Sacred", "DM" inside "Madam").
    /// Mirror of `_short_alt_word_bounded` in apps/mobile/convert/rag_search.py.
    private static func wordBoundedContains(_ needle: String, in haystack: String) -> Bool {
        let escaped = NSRegularExpression.escapedPattern(for: needle)
        let pattern = "\\b" + escaped + "\\b"
        guard let regex = try? NSRegularExpression(pattern: pattern,
                                                   options: [.caseInsensitive])
        else {
            return haystack.contains(needle)  // fall back to substring on regex error
        }
        let range = NSRange(haystack.startIndex..., in: haystack)
        return regex.firstMatch(in: haystack, range: range) != nil
    }

    /// Case-SENSITIVE word-bounded search for short uppercase acronym
    /// alt_names. Curated alt_names like "CRE", "RSV", "MRSA", "MERS",
    /// "WNV" are uppercase acronyms by convention. Lowercase tokens that
    /// happen to spell the same letters MUST NOT match — they're never
    /// the intended clinical concept. Mirror of
    /// `_short_alt_uppercase_word` in apps/mobile/convert/rag_search.py.
    private static func uppercaseWordBounded(_ needle: String, in originalQuery: String) -> Bool {
        if needle != needle.uppercased() { return false }
        let escaped = NSRegularExpression.escapedPattern(for: needle)
        let pattern = "\\b" + escaped + "\\b"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [])
        else { return false }
        let range = NSRange(originalQuery.startIndex..., in: originalQuery)
        return regex.firstMatch(in: originalQuery, range: range) != nil
    }

    private static func tokens(_ text: String) -> Set<String> {
        let range = NSRange(text.startIndex..., in: text)
        var out = Set<String>()
        for m in tokenRe.matches(in: text, range: range) {
            if let r = Range(m.range, in: text) {
                out.insert(String(text[r]).lowercased())
            }
        }
        return out
    }

    /// Search the reportable-conditions DB. Returns up to `topK` hits with
    /// score ≥ `minScore`, sorted descending then by display name.
    static func search(
        query: String,
        topK: Int = defaultTopK,
        minScore: Double = defaultMinScore
    ) -> [RagHit] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let qNorm = normalize(trimmed)
        let qLower = qNorm.lowercased()
        let qTokens = tokens(qNorm)

        var scored: [(Double, String?, ReportableCondition)] = []
        for entry in ReportableConditions.all {
            let altNames = entry.altNames

            // 1. Exact-phrase bonus
            // Walk display first, then altNames. Display is the canonical
            // name (always long enough not to false-positive on substrings);
            // altNames may be short acronyms, so apply a word-boundary check
            // for those (≤ shortAltMaxLen chars). Mirror of the Python rule
            // in apps/mobile/convert/rag_search.py.
            var matchedPhrase: String? = nil
            var phraseBonus = 0.0
            let candidates: [(String, Bool)] =
                [(entry.display, false)] + altNames.map { ($0, true) }
            for (candidate, isAlt) in candidates {
                let candNorm = normalize(candidate).lowercased()
                let matched: Bool
                if isAlt && candNorm.count <= shortAltMaxLen {
                    // c20 final pass: uppercase acronyms (CRE, RSV, MRSA,
                    // MERS, WNV) require a case-sensitive word match
                    // against the ORIGINAL narrative so lowercase tokens
                    // never slip through. Lowercase short alt_names (e.g.
                    // "lues" for syphilis) keep the prior word-bounded
                    // case-insensitive behaviour.
                    if candidate == candidate.uppercased() {
                        matched = uppercaseWordBounded(candidate, in: query)
                    } else {
                        matched = wordBoundedContains(candNorm, in: qLower)
                    }
                } else {
                    matched = qLower.contains(candNorm)
                }
                if matched {
                    if 1.0 > phraseBonus { phraseBonus = 1.0 }
                    if matchedPhrase == nil { matchedPhrase = candidate }
                }
            }

            // 2. Token-overlap F1 (×0.5 weight)
            var eTokens = tokens(normalize(entry.display))
            for alt in altNames { eTokens.formUnion(tokens(normalize(alt))) }
            let overlap = qTokens.intersection(eTokens)
            var tokenScore = 0.0
            if !eTokens.isEmpty {
                let recall = Double(overlap.count) / Double(max(qTokens.count, 1))
                let precision = Double(overlap.count) / Double(max(eTokens.count, 1))
                if recall + precision > 0 {
                    tokenScore = (2.0 * recall * precision / (recall + precision)) * 0.5
                }
            }

            // 3. Long-token bonus
            let longTokenBonus = 0.1 * Double(overlap.filter { $0.count >= 6 }.count)

            let score = phraseBonus + tokenScore + longTokenBonus
            if score >= minScore {
                scored.append((score, matchedPhrase, entry))
            }
        }

        scored.sort { (a, b) in
            if a.0 != b.0 { return a.0 > b.0 }
            return a.2.display < b.2.display
        }

        return scored.prefix(topK).map { (score, matched, entry) in
            RagHit(
                code: entry.code,
                system: entry.system,
                display: entry.display,
                score: score,
                matchedPhrase: matched,
                source: entry.source,
                sourceURL: entry.sourceURL,
                category: entry.category,
                altNames: entry.altNames
            )
        }
    }

    // MARK: - Fast-path hit (c19 Rank 2)
    //
    // The single-turn agent fast-path: when tier-1 is empty AND RAG has a
    // high-confidence hit AND the matched phrase isn't in NegEx scope,
    // skip the agent loop and synthesise the extraction directly from the
    // RAG hit. Drops the agent-tier-only median latency from ~13 s to
    // <1 s on the long-tail cases where deterministic was previously
    // empty (Legionnaires, C diff, valley fever, Marburg, RMSF, etc.).
    //
    // Mirror of the Python gate in `apps/mobile/convert/agent_pipeline.py`.
    // Keep them in sync; precision must stay 1.000 — any new false positive
    // kills the proposal per Rank 2's kill criterion.

    /// Where in the narrative did the fast-path candidate phrase land? The
    /// caller stamps this onto the synthesised CodeProvenance so the UI
    /// can highlight the source span on tap-to-expand.
    struct FastPathSpan {
        let text: String        // literal substring, e.g. "Legionnaires' disease"
        let location: Int       // UTF-16 offset
        let length: Int
    }

    /// One fast-path hit + the literal matched span. Returned only when:
    ///   1. top RAG score ≥ `threshold` (default 0.70)
    ///   2. there exists at least one occurrence of the matched phrase in
    ///      the narrative that is NOT in NegEx scope.
    /// Caller must ensure `det.hasAnyDeterministic == false` before calling.
    struct FastPathHit {
        let hit: RagHit
        let span: FastPathSpan
    }

    /// Negation predicate. Caller provides this so RagSearch stays
    /// self-contained for the standalone `validate_rag.swift` CLI.
    /// In-app: `ExtractionService` injects a closure that delegates to
    /// `EicrPreparser.isNegated(in:matchStart:matchEnd:)`. Bench / CLI
    /// can pass `{ _,_,_ in false }` to skip NegEx, or its own
    /// implementation to test a different rule.
    typealias IsNegated = (_ text: String, _ matchStart: Int, _ matchEnd: Int) -> Bool

    /// A no-op negation predicate that always returns false. Useful for
    /// tests / CLIs that don't want to pull in EicrPreparser.
    static let neverNegated: IsNegated = { _, _, _ in false }

    /// Find the first non-negated occurrence in `narrative` of the hit's
    /// `matchedPhrase` and return it. Negation is judged via the
    /// caller-provided `isNegated` closure — keeps the bar consistent with
    /// Tier-3 lookup ("ruled out Legionnaires" stays suppressed in both
    /// paths) without RagSearch.swift directly depending on EicrPreparser.
    ///
    /// Why only `matchedPhrase` (not altNames + display)? altName fallback
    /// is too loose: a short alias like "Cocci" (for Coccidioidomycosis)
    /// matches the genus prefix in "Coccidioides serology negative for
    /// valley fever" at an unrelated offset, defeating NegEx on the
    /// actual diagnosis. `matchedPhrase` is the exact phrase that earned
    /// RAG's score in the first place — if THAT span is suppressed, the
    /// hit is suppressed. If matchedPhrase is nil (score came from
    /// token-overlap alone, never an exact phrase match) the fast-path
    /// declines and falls through to the agent.
    static func firstAssertedSpan(in narrative: String,
                                  for hit: RagHit,
                                  isNegated: IsNegated = neverNegated) -> FastPathSpan? {
        guard let matched = hit.matchedPhrase, !matched.isEmpty else {
            return nil
        }
        return firstNonNegatedSpan(of: matched,
                                   in: narrative,
                                   isNegated: isNegated)
    }

    /// Scan `narrative` for every case-insensitive occurrence of `phrase`.
    /// Return the first occurrence that is NOT in NegEx scope per the
    /// caller-provided predicate. Returns nil when every occurrence is
    /// suppressed ("ruled out Legionnaires" with no other mention).
    private static func firstNonNegatedSpan(of phrase: String,
                                            in narrative: String,
                                            isNegated: IsNegated) -> FastPathSpan? {
        let trimmed = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // Use literal range search rather than regex so phrases containing
        // metacharacters ("C. trachomatis", "C. diff", "Plasmodium malariae
        // malaria") match without escaping.
        let ns = narrative as NSString
        var searchStart = 0
        let total = ns.length
        while searchStart < total {
            let searchRange = NSRange(location: searchStart, length: total - searchStart)
            let r = ns.range(of: trimmed,
                             options: [.caseInsensitive],
                             range: searchRange)
            if r.location == NSNotFound { break }
            let end = r.location + r.length
            if !isNegated(narrative, r.location, end) {
                let spanText = ns.substring(with: r)
                return FastPathSpan(text: spanText,
                                    location: r.location,
                                    length: r.length)
            }
            searchStart = end
        }
        return nil
    }

    /// Top-level gate. Returns the fast-path hit if every condition holds:
    ///   - top RAG score ≥ `threshold`
    ///   - at least one non-negated occurrence of a candidate phrase per
    ///     the caller-provided `isNegated` predicate
    /// Otherwise nil → caller falls through to the agent loop.
    static func fastPathHit(
        narrative: String,
        threshold: Double = fastPathThreshold,
        isNegated: IsNegated = neverNegated
    ) -> FastPathHit? {
        let hits = search(query: narrative, topK: 1, minScore: threshold)
        guard let top = hits.first, top.score >= threshold else { return nil }
        guard let span = firstAssertedSpan(in: narrative,
                                           for: top,
                                           isNegated: isNegated) else {
            return nil
        }
        return FastPathHit(hit: top, span: span)
    }
}
