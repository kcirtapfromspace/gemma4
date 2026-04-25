// validate_rag.swift
// Sanity-check the Swift RAG search produces the same top-1 hits as the
// Python rag_search.py for a representative query set, plus the c19
// single-turn agent fast-path gate (RagSearch.fastPathHit, threshold
// 0.70, NegEx-suppressed-aware).
//
// Usage:
//   swiftc -O validate_rag.swift \
//       ClinIQ/ClinIQ/Extraction/RagSearch.swift \
//       ClinIQ/ClinIQ/Extraction/ReportableConditions.swift \
//       -parse-as-library -o /tmp/validate_rag
//   /tmp/validate_rag

import Foundation

@main
enum ValidateRagCLI {
    static func main() {
        let topkPass = runTopkProbes()
        let fastPass = runFastPathProbes()
        let total = topkPass.0 + fastPass.0
        let denom = topkPass.1 + fastPass.1
        print("\n=== summary ===")
        print("top-k:      \(topkPass.0)/\(topkPass.1) pass")
        print("fast-path:  \(fastPass.0)/\(fastPass.1) pass")
        print("overall:    \(total)/\(denom) pass")
        exit(total == denom ? 0 : 1)
    }

    // MARK: - Top-1 probe (existing bench)

    private static func runTopkProbes() -> (Int, Int) {
        // (query, expected top-1 code, expected match phrase)
        let cases: [(String, String, String?)] = [
            ("Lyme disease",                                "37117007",  nil),
            ("Rocky Mountain spotted fever",                "186788009", "Rocky Mountain spotted fever"),
            ("Vibrio cholerae",                             "1857005",   "Cholera"),
            ("VRE infection",                               "21639007",  "VRE"),
            ("Salmonella enteritis after eating eggs",      "302229004", "salmonella enteritis"),
            ("MERS coronavirus",                            "240370005", "MERS"),
            ("patient has C diff colitis",                  "186431008", "C diff"),
            ("valley fever from California desert",         "37436014",  "valley fever"),
            ("Legionnaires' disease",                       "37117007",  nil),
            ("Marburg hemorrhagic fever outbreak",          "418182002", nil),
            ("Plasmodium malariae malaria",                 "186946009", "Plasmodium malariae malaria"),
        ]

        print("=== top-k probes ===")
        var pass = 0, fail = 0
        for (i, item) in cases.enumerated() {
            let (query, expectedCode, expectedPhrase) = item
            FileHandle.standardError.write("probe \(i): \(query)\n".data(using: .utf8)!)
            let hits = RagSearch.search(query: query, topK: 3)
            guard let top = hits.first else {
                print("  MISS  \(query) — no hits at all")
                fail += 1; continue
            }
            let topMatches = top.code == expectedCode
            let phraseOk = expectedPhrase == nil || top.matchedPhrase == expectedPhrase
            let q44 = query.padding(toLength: 44, withPad: " ", startingAt: 0)
            let scoreStr = String(format: "%.3f", top.score)
            let phrase = top.matchedPhrase ?? "—"
            if topMatches && phraseOk {
                print("  OK    \(q44) top=\(top.code) score=\(scoreStr) phrase=\(phrase)")
                pass += 1
            } else {
                print("  MISS  \(q44) expected=\(expectedCode) got=\(top.code) score=\(scoreStr) phrase=\(phrase)")
                fail += 1
            }
        }
        return (pass, pass + fail)
    }

    // MARK: - Fast-path gate probes (c19 Rank 2)
    //
    // Each case asserts whether `RagSearch.fastPathHit` should fire AND
    // (when it should) which SNOMED code it picks. Two negation cases
    // ("ruled out X" / "no evidence of X") use a hand-rolled NegEx
    // predicate matching the EicrPreparser rules, so the CLI can stay
    // self-contained without pulling in EicrPreparser.swift.

    private static let negTriggers = [
        "ruled out", "negative for", "no evidence of",
        "no signs of", "no sign of", "no history of",
        "denies", "without", "absent", "not detected",
        "not positive for", "not suspected",
        "exclude", "excluded", "excludes", "excluding",
        "differential diagnosis", "differential dx",
    ]
    private static let cliIsNegated: RagSearch.IsNegated = { text, matchStart, _ in
        // Look at the 60-character window immediately before the match
        // for any of the canonical NegEx triggers. Crude vs. the real
        // EicrPreparser scanner (no terminator clipping), but enough to
        // exercise the fast-path gate's "negation suppresses fast-path"
        // contract for the bench cases below.
        let ns = text as NSString
        let start = max(0, matchStart - 60)
        let len = matchStart - start
        if len <= 0 { return false }
        let window = ns.substring(with: NSRange(location: start, length: len)).lowercased()
        return negTriggers.contains { window.contains($0) }
    }

    private struct FastCase {
        let label: String
        let narrative: String
        let shouldFire: Bool
        let expectedCode: String?  // when shouldFire == true
    }

    private static func runFastPathProbes() -> (Int, Int) {
        let cases: [FastCase] = [
            FastCase(label: "valley fever (asserted)",
                     narrative: "Patient with classic valley fever from a California desert vacation. Cough, fatigue.",
                     shouldFire: true,
                     expectedCode: "37436014"),
            FastCase(label: "Marburg outbreak (asserted)",
                     narrative: "Returning traveler from Uganda, suspected Marburg hemorrhagic fever, isolation initiated.",
                     shouldFire: true,
                     expectedCode: "418182002"),
            FastCase(label: "C diff colitis (asserted)",
                     narrative: "Severe diarrhea after broad-spectrum antibiotics. C diff colitis confirmed by toxin assay.",
                     shouldFire: true,
                     expectedCode: "186431008"),
            FastCase(label: "Legionnaires (token-overlap)",
                     narrative: "Outbreak investigation suggests Legionnaires' disease via cooling tower aerosol.",
                     shouldFire: false,        // 0.486 score → below 0.70 threshold by design
                     expectedCode: nil),
            FastCase(label: "ruled out Legionnaires (negated)",
                     narrative: "Legionnaires' disease ruled out per negative urinary antigen.",
                     shouldFire: false,        // negated AND below threshold; double-no
                     expectedCode: nil),
            FastCase(label: "negative for valley fever (negated)",
                     // "negative for" sits in the 60-char before-window of
                     // "valley fever" with no terminator between them, so
                     // both the real EicrPreparser scanner and the CLI
                     // approximation suppress this. The earlier
                     // "X but ruled out Y" phrasing is NOT considered
                     // negated by EicrPreparser because the `but`
                     // terminator clips the forward window before
                     // "ruled out" — that's a NegEx-philosophy choice,
                     // not a fast-path bug.
                     narrative: "Coccidioides serology negative for valley fever; alternate workup pending.",
                     shouldFire: false,
                     expectedCode: nil),
            FastCase(label: "no fast-path on bare narrative without hit",
                     narrative: "Patient reports headache and fatigue. No specific exposures identified.",
                     shouldFire: false,
                     expectedCode: nil),
            FastCase(label: "Plasmodium malariae (asserted)",
                     narrative: "Returning traveler with intermittent fevers and Plasmodium malariae malaria diagnosed.",
                     shouldFire: true,
                     expectedCode: "186946009"),
        ]

        print("\n=== fast-path probes (threshold \(RagSearch.fastPathThreshold)) ===")
        var pass = 0, fail = 0
        for (i, c) in cases.enumerated() {
            FileHandle.standardError.write("fp probe \(i): \(c.label)\n".data(using: .utf8)!)
            let label = c.label.padding(toLength: 40, withPad: " ", startingAt: 0)
            let fired = RagSearch.fastPathHit(narrative: c.narrative,
                                              isNegated: cliIsNegated)
            switch (c.shouldFire, fired) {
            case (true, .some(let fp)):
                let codeOk = c.expectedCode == nil || c.expectedCode == fp.hit.code
                let scoreStr = String(format: "%.3f", fp.hit.score)
                if codeOk {
                    print("  OK    \(label) fired code=\(fp.hit.code) score=\(scoreStr) span='\(fp.span.text)'")
                    pass += 1
                } else {
                    print("  MISS  \(label) fired code=\(fp.hit.code) expected=\(c.expectedCode ?? "?") score=\(scoreStr)")
                    fail += 1
                }
            case (false, .none):
                print("  OK    \(label) did not fire (correct)")
                pass += 1
            case (true, .none):
                print("  MISS  \(label) expected fire but got nil")
                fail += 1
            case (false, .some(let fp)):
                print("  MISS  \(label) FALSE POSITIVE: fired \(fp.hit.code) score=\(String(format: "%.3f", fp.hit.score))")
                fail += 1
            }
        }
        return (pass, pass + fail)
    }
}
