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
            var matchedPhrase: String? = nil
            var phraseBonus = 0.0
            for candidate in [entry.display] + altNames {
                let candNorm = normalize(candidate).lowercased()
                if qLower.contains(candNorm) {
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

    /// Find the first non-negated occurrence in `narrative` of any candidate
    /// phrase from the hit (matchedPhrase first, then altNames, then the
    /// canonical display) and return it. Negation is judged via the same
    /// `EicrPreparser.isNegated` used by Tier-3 lookup — keeps the bar
    /// consistent ("ruled out Legionnaires" stays suppressed in both paths).
    static func firstAssertedSpan(in narrative: String, for hit: RagHit) -> FastPathSpan? {
        // Try the matched phrase first; falling back to all altNames + display
        // gives us a chance even when the score came from token-overlap rather
        // than an exact-phrase match (e.g. "valley fever" wins on tokens).
        var candidates: [String] = []
        if let matched = hit.matchedPhrase, !matched.isEmpty {
            candidates.append(matched)
        }
        for alt in hit.altNames where !candidates.contains(alt) {
            candidates.append(alt)
        }
        if !candidates.contains(hit.display) {
            candidates.append(hit.display)
        }

        for candidate in candidates {
            if let span = firstNonNegatedSpan(of: candidate, in: narrative) {
                return span
            }
        }
        return nil
    }

    /// Scan `narrative` for every case-insensitive occurrence of `phrase`.
    /// Return the first occurrence that is NOT in NegEx scope per
    /// `EicrPreparser.isNegated`. Returns nil when every occurrence is
    /// suppressed ("ruled out Legionnaires" with no other mention).
    private static func firstNonNegatedSpan(of phrase: String,
                                            in narrative: String) -> FastPathSpan? {
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
            if !EicrPreparser.isNegated(in: narrative,
                                        matchStart: r.location,
                                        matchEnd: end) {
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
    ///   - at least one non-negated occurrence of a candidate phrase
    /// Otherwise nil → caller falls through to the agent loop.
    static func fastPathHit(
        narrative: String,
        threshold: Double = fastPathThreshold
    ) -> FastPathHit? {
        let hits = search(query: narrative, topK: 1, minScore: threshold)
        guard let top = hits.first, top.score >= threshold else { return nil }
        guard let span = firstAssertedSpan(in: narrative, for: top) else {
            return nil
        }
        return FastPathHit(hit: top, span: span)
    }
}
