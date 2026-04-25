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
}
