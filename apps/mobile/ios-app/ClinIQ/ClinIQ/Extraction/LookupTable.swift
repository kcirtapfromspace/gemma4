// LookupTable.swift
// Curated displayName → code dictionary (Tier 3 of EicrPreparser).
//
// This is the Swift mirror of apps/mobile/convert/lookup_table.json. Keep
// in sync. Aliases match case-insensitively against word-bounded substrings;
// any-match-wins per code (one alias hit is enough). Order is most-specific
// first because the Python and Swift parsers report the first matched alias
// as the display name.
//
// Scope is hackathon bench coverage + a small base of common reportable
// conditions. Not a comprehensive UMLS replacement — extend by appending
// to the relevant array (and keep lookup_table.json in lock-step so the
// Python bench still verifies the data).

import Foundation

struct LookupEntry {
    let code: String
    let aliases: [String]
    private let patterns: [NSRegularExpression]

    init(code: String, aliases: [String]) {
        self.code = code
        self.aliases = aliases
        self.patterns = aliases.map { alias in
            // Word-bounded, case-insensitive. Patterns are static; force-try
            // is fine — any failure is a programmer error caught in dev.
            // swiftlint:disable:next force_try
            try! NSRegularExpression(
                pattern: #"\b"# + NSRegularExpression.escapedPattern(for: alias) + #"\b"#,
                options: [.caseInsensitive]
            )
        }
    }

    /// Return the first alias whose pattern matches `text`, else nil.
    /// Pure pattern match — no negation handling.
    func firstMatchingAlias(in text: String) -> String? {
        let range = NSRange(text.startIndex..., in: text)
        for (i, p) in patterns.enumerated() {
            if p.firstMatch(in: text, range: range) != nil {
                return aliases[i]
            }
        }
        return nil
    }

    /// Return the first alias whose match is asserted (not in negation scope).
    func firstAssertedAlias(in text: String) -> String? {
        return firstAssertedSpan(in: text)?.alias
    }

    /// Source-span record for an asserted alias hit. The provenance pipeline
    /// uses this to record exactly which substring proved the code.
    struct AssertedSpan {
        let alias: String
        let text: String
        let location: Int
        let length: Int
    }

    func firstAssertedSpan(in text: String) -> AssertedSpan? {
        let fullRange = NSRange(text.startIndex..., in: text)
        for (i, p) in patterns.enumerated() {
            for m in p.matches(in: text, range: fullRange) {
                let r = m.range
                if EicrPreparser.isNegated(in: text,
                                           matchStart: r.location,
                                           matchEnd: r.location + r.length) {
                    continue
                }
                // c20 final pass: skip short uppercase acronym aliases
                // when they're used as data-label headers (`CBC:`, `CMP:`).
                // Mirror of `_is_label_header_use` in
                // apps/mobile/convert/regex_preparser.py.
                if EicrPreparser.isLabelHeaderUse(alias: aliases[i],
                                                  in: text,
                                                  matchEnd: r.location + r.length) {
                    continue
                }
                let nsText = text as NSString
                let span = (r.location + r.length <= nsText.length)
                    ? nsText.substring(with: r)
                    : aliases[i]
                return AssertedSpan(
                    alias: aliases[i],
                    text: span,
                    location: r.location,
                    length: r.length
                )
            }
        }
        return nil
    }
}

enum LookupTable {
    static let snomed: [LookupEntry] = [
        LookupEntry(code: "3928002",   aliases: ["Zika virus infection", "Zika"]),
        LookupEntry(code: "43878008",  aliases: ["streptococcal pharyngitis", "strep pharyngitis", "strep throat", "group A Streptococcus pharyngitis", "group A strep"]),
        LookupEntry(code: "38907003",  aliases: ["chickenpox", "varicella"]),
        LookupEntry(code: "66071002",  aliases: ["chronic viral hepatitis B", "chronic hepatitis B"]),
        LookupEntry(code: "27836007",  aliases: ["whooping cough", "pertussis"]),
        LookupEntry(code: "50711007",  aliases: ["chronic hepatitis C", "hepatitis C"]),
        LookupEntry(code: "840539006", aliases: ["COVID-19", "SARS-CoV-2 infection", "COVID 19", "coronavirus disease 2019"]),
        LookupEntry(code: "86406008",  aliases: ["HIV infection", "human immunodeficiency virus infection", "human immunodeficiency virus disease", "HIV disease"]),
        LookupEntry(code: "23511006",  aliases: ["meningococcal disease", "meningococcal infection"]),
        LookupEntry(code: "76272004",  aliases: ["syphilis"]),
        LookupEntry(code: "23502006",  aliases: ["Lyme disease", "Lyme borreliosis"]),
        LookupEntry(code: "398565003", aliases: ["Shiga toxin-producing E. coli infection", "STEC infection", "Shiga toxin E. coli"]),
        LookupEntry(code: "56717001",  aliases: ["tuberculosis", "active pulmonary TB", "pulmonary TB", "TB"]),
        LookupEntry(code: "38362002",  aliases: ["dengue", "dengue fever"]),
        LookupEntry(code: "442695009", aliases: ["Avian influenza A H5N1 infection", "Avian influenza A(H5N1)", "H5N1 infection", "H5N1"]),
        LookupEntry(code: "22253000",  aliases: ["mpox", "monkeypox virus infection", "monkeypox"]),
        LookupEntry(code: "55735004",  aliases: ["Respiratory syncytial virus infection", "RSV infection", "RSV"]),
        LookupEntry(code: "302229004", aliases: ["Salmonella gastroenteritis", "Salmonellosis"]),
    ]

    static let loincs: [LookupEntry] = [
        LookupEntry(code: "78929-3",  aliases: ["Zika virus IgM Ab", "Zika virus IgM antibody"]),
        LookupEntry(code: "78012-2",  aliases: ["Strep A Ag rapid", "rapid antigen detection test for group A Streptococcus", "rapid antigen detection test", "rapid strep test", "strep rapid antigen", "strep antigen", "Strep test"]),
        LookupEntry(code: "41513-3",  aliases: ["Varicella zoster virus DNA NAA", "VZV DNA NAA"]),
        LookupEntry(code: "5193-8",   aliases: ["Hepatitis B surface Ag", "HBV surface Ag", "HBsAg"]),
        LookupEntry(code: "71773-2",  aliases: ["Bordetella pertussis DNA NAA"]),
        LookupEntry(code: "11259-9",  aliases: ["Hepatitis C virus Ab", "HCV Ab"]),
        LookupEntry(code: "94500-6",  aliases: ["SARS-CoV-2 RNA NAA+probe Ql Resp", "SARS-CoV-2 RNA"]),
        LookupEntry(code: "75622-1",  aliases: ["HIV 1 and 2 Ag+Ab"]),
        LookupEntry(code: "57021-8",  aliases: ["Complete blood count", "CBC"]),
        LookupEntry(code: "24467-3",  aliases: ["CD4+ T cells"]),
        LookupEntry(code: "49672-8",  aliases: ["Neisseria meningitidis DNA NAA"]),
        LookupEntry(code: "20507-0",  aliases: ["Treponema pallidum Ab", "T. pallidum Ab"]),
        LookupEntry(code: "5061-1",   aliases: ["Borrelia burgdorferi Ab"]),
        LookupEntry(code: "16832-8",  aliases: ["Escherichia coli O157 Ag"]),
        LookupEntry(code: "38379-4",  aliases: ["Mycobacterium tuberculosis complex DNA", "Xpert MTB/RIF", "MTB/RIF"]),
        LookupEntry(code: "6386-1",   aliases: ["Dengue virus IgM Ab"]),
        LookupEntry(code: "100343-3", aliases: ["Influenza A H5 RNA", "Influenza A H5", "H5 RNA NAA", "H5N1 RNA NAA", "H5N1 RNA", "H5 RNA"]),
        LookupEntry(code: "96741-4",  aliases: ["Monkeypox virus DNA by PCR", "Monkeypox virus DNA"]),
        LookupEntry(code: "31933-7",  aliases: ["RSV antigen by direct fluorescent antibody", "Respiratory syncytial virus Ag", "RSV antigen"]),
        LookupEntry(code: "589-7",    aliases: ["Salmonella culture", "Salmonella sp identified"]),
        LookupEntry(code: "56888-1",  aliases: ["HIV-1/2 antigen/antibody combination immunoassay", "HIV-1 and 2 antigen/antibody combination", "HIV-1/2 Ag+Ab combination"]),
    ]

    static let rxnorms: [LookupEntry] = [
        LookupEntry(code: "723",     aliases: ["amoxicillin"]),
        LookupEntry(code: "197612",  aliases: ["acyclovir 800 MG Oral Tablet", "acyclovir 800 MG", "acyclovir"]),
        LookupEntry(code: "105220",  aliases: ["penicillin G benzathine 2400000 UNT", "penicillin G benzathine"]),
        LookupEntry(code: "2599543", aliases: ["nirmatrelvir 150 MG / ritonavir 100 MG", "Paxlovid", "nirmatrelvir"]),
        LookupEntry(code: "1999563", aliases: ["bictegravir 50 MG / emtricitabine 200 MG / tenofovir alafenamide 25 MG", "bictegravir/emtricitabine/tenofovir alafenamide", "bictegravir"]),
        LookupEntry(code: "197696",  aliases: ["fluconazole 200 MG Oral Tablet", "fluconazole"]),
        // Note: bare "ceftriaxone" intentionally omitted — it false-positives
        // on inputs with non-500 MG doses (see adv2_drug_dose_variant where
        // the input was "ceftriaxone 1 g IV"). Match only the exact dose form.
        LookupEntry(code: "1665021", aliases: ["ceftriaxone 500 MG Injection"]),
        LookupEntry(code: "1940261", aliases: ["sofosbuvir 400 MG / velpatasvir 100 MG", "Epclusa", "sofosbuvir"]),
        LookupEntry(code: "197984",  aliases: ["doxycycline 100 MG Oral Tablet", "doxycycline"]),
        LookupEntry(code: "197622",  aliases: ["isoniazid 300 MG Oral Tablet", "isoniazid"]),
        LookupEntry(code: "199279",  aliases: ["rifampin 300 MG Oral Capsule", "rifampin"]),
        LookupEntry(code: "105078",  aliases: ["entecavir 0.5 MG Oral Tablet", "entecavir"]),
        LookupEntry(code: "261244",  aliases: ["oseltamivir 75 MG", "oseltamivir"]),
        LookupEntry(code: "1923432", aliases: ["tecovirimat 600 MG", "tecovirimat"]),
        LookupEntry(code: "309309",  aliases: ["ciprofloxacin 500 MG", "ciprofloxacin"]),
    ]
}
