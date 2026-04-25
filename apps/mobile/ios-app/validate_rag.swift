// validate_rag.swift
// Sanity-check the Swift RAG search produces the same top-1 hits as the
// Python rag_search.py for a representative query set.
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
        // (query, expected top-1 code, expected match phrase)
        let cases: [(String, String, String?)] = [
            ("Lyme disease",                                "37117007",  nil),       // not in DB; fallback to legionella/leptospira tokens
            ("Rocky Mountain spotted fever",                "186788009", "Rocky Mountain spotted fever"),
            ("Vibrio cholerae",                             "1857005",   "Cholera"),
            ("VRE infection",                               "21639007",  "VRE"),
            ("Salmonella enteritis after eating eggs",      "302229004", "salmonella enteritis"),
            ("MERS coronavirus",                            "240370005", "MERS"),
            ("patient has C diff colitis",                  "186431008", "C diff"),
            ("valley fever from California desert",         "37436014",  "valley fever"),
            ("Legionnaires' disease",                       "37117007",  nil),       // no exact alt has the apostrophe
            ("Marburg hemorrhagic fever outbreak",          "418182002", nil),
            ("Plasmodium malariae malaria",                 "186946009", "Plasmodium malariae malaria"),
        ]

        var pass = 0
        var fail = 0
        for (i, item) in cases.enumerated() {
            let (query, expectedCode, expectedPhrase) = item
            FileHandle.standardError.write("probe \(i): \(query)\n".data(using: .utf8)!)
            let hits = RagSearch.search(query: query, topK: 3)
            guard let top = hits.first else {
                print("  MISS  \(query) — no hits at all")
                fail += 1
                continue
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

        print("\n\(pass)/\(pass + fail) pass")
        exit(fail == 0 ? 0 : 1)
    }
}
