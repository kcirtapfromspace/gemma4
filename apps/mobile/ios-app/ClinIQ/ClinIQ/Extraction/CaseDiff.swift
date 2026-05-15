// CaseDiff.swift
// Longitudinal "what's new vs prior eCR" diff — Swift mirror of the Python
// edge agent's diff. The two implementations MUST stay aligned so cases
// produced by either pipeline align in the timeline.
//
// Mirrors ClinIQ's flat "what's new vs prior" export. On-device, no
// Verato, exact-match identity only.
//
// Algorithm (per axis):
//   priorKeys    = { "<system>|<code>" for each entry in prior }
//   currentKeys  = { "<system>|<code>" for each entry in current }
//   added        = current - priorKeys
//   removed      = prior - currentKeys
//   unchanged    = current ∩ priorKeys

import Foundation

/// One axis of the FHIR R4 extraction. The bucketing tag carried on each
/// diff entry so the UI can colorize per-axis chips.
enum ExtractionAxis: String, Codable, CaseIterable {
    case condition
    case lab
    case medication
    case vital
}

/// One concrete entity that appears in either the prior or the current
/// case. The key for diffing is `(codeSystem, code)` — the display string
/// is informational only and is taken from the *current* row when present
/// so the user sees the freshest label.
struct DiffEntry: Identifiable, Hashable {
    let id: UUID
    let axis: ExtractionAxis
    let codeSystem: String   // "http://snomed.info/sct", "http://loinc.org", "http://www.nlm.nih.gov/research/umls/rxnorm"
    let code: String
    let display: String
    let reviewStateRaw: String?

    init(id: UUID = UUID(),
         axis: ExtractionAxis,
         codeSystem: String,
         code: String,
         display: String,
         reviewStateRaw: String? = nil) {
        self.id = id
        self.axis = axis
        self.codeSystem = codeSystem
        self.code = code
        self.display = display
        self.reviewStateRaw = reviewStateRaw
    }

    /// Diff key — must be identical between Swift and Python.
    var diffKey: String { "\(codeSystem)|\(code)" }
}

/// The structural result of comparing two cases for the same patient.
/// The view layer renders three sections (added / removed / unchanged) and
/// optionally a one-line summary banner.
struct CaseDiff: Hashable {
    let patientIdentityHash: String
    let priorCaseId: UUID
    let currentCaseId: UUID
    let priorDate: Date
    let currentDate: Date
    let added: [DiffEntry]
    let removed: [DiffEntry]
    let unchanged: [DiffEntry]

    var hasChanges: Bool { !added.isEmpty || !removed.isEmpty }
    var totalNew: Int { added.count }
    var daysBetween: Int {
        let comps = Calendar(identifier: .gregorian)
            .dateComponents([.day], from: priorDate, to: currentDate)
        return comps.day ?? 0
    }

    /// Compose a concise summary for the banner: "3 new findings since last
    /// eCR (5 days ago)". Pluralisation handled here so the view stays a
    /// pure renderer.
    var summary: String {
        let n = added.count
        let r = removed.count
        let days = daysBetween
        let dayCopy: String
        switch days {
        case 0: dayCopy = "today"
        case 1: dayCopy = "1 day ago"
        default: dayCopy = "\(days) days ago"
        }
        switch (n, r) {
        case (0, 0):
            return "No changes since last eCR (\(dayCopy))"
        case (let a, 0):
            return "\(a) new finding\(a == 1 ? "" : "s") since last eCR (\(dayCopy))"
        case (0, let d):
            return "\(d) finding\(d == 1 ? "" : "s") resolved since last eCR (\(dayCopy))"
        case (let a, let d):
            return "\(a) new, \(d) resolved since last eCR (\(dayCopy))"
        }
    }
}

// MARK: - Builder

enum CaseDiffBuilder {
    /// Per-axis system URIs. We use the canonical FHIR system URIs in the
    /// diff layer so it stays interop-neutral; the existing SwiftData rows
    /// store short-form ("SNOMED" / "LOINC" / "RxNorm") which we translate
    /// here. Keep this aligned with `BundleBuilder.swift`.
    static let snomedSystem = "http://snomed.info/sct"
    static let loincSystem = "http://loinc.org"
    static let rxnormSystem = "http://www.nlm.nih.gov/research/umls/rxnorm"
    static let vitalsSystem = "http://loinc.org" // vitals are LOINC-coded too

    /// Compute a `CaseDiff` between two `ClinicalCase` rows. Caller is
    /// responsible for ensuring both belong to the same patient (matching
    /// `patientIdentityHash`); we don't enforce here so the seed data /
    /// preview machinery can call directly.
    static func diff(prior: ClinicalCase, current: ClinicalCase) -> CaseDiff {
        let priorEntries = entries(from: prior)
        let currentEntries = entries(from: current)
        return diff(prior: priorEntries,
                    current: currentEntries,
                    patientIdentityHash: current.patientIdentityHash,
                    priorCaseId: prior.id,
                    currentCaseId: current.id,
                    priorDate: prior.createdAt,
                    currentDate: current.createdAt)
    }

    /// Pure form — operates on already-flattened entry lists. Useful for
    /// unit tests and previews.
    static func diff(prior: [DiffEntry],
                     current: [DiffEntry],
                     patientIdentityHash: String = "",
                     priorCaseId: UUID,
                     currentCaseId: UUID,
                     priorDate: Date,
                     currentDate: Date) -> CaseDiff {
        let priorKeys = Set(prior.map { $0.diffKey })
        let currentKeys = Set(current.map { $0.diffKey })

        let added = current.filter { !priorKeys.contains($0.diffKey) }
        let removed = prior.filter { !currentKeys.contains($0.diffKey) }
        let unchanged = current.filter { priorKeys.contains($0.diffKey) }

        return CaseDiff(patientIdentityHash: patientIdentityHash,
                        priorCaseId: priorCaseId,
                        currentCaseId: currentCaseId,
                        priorDate: priorDate,
                        currentDate: currentDate,
                        added: added,
                        removed: removed,
                        unchanged: unchanged)
    }

    /// Flatten a `ClinicalCase` into axis-tagged `DiffEntry` rows. Vitals
    /// are emitted as one entry per non-nil vital with the canonical LOINC.
    static func entries(from c: ClinicalCase) -> [DiffEntry] {
        var out: [DiffEntry] = []
        for cond in c.conditions where cond.reviewState != .rejected {
            out.append(DiffEntry(axis: .condition,
                                 codeSystem: snomedSystem,
                                 code: cond.code,
                                 display: cond.displayName,
                                 reviewStateRaw: cond.reviewStateRaw))
        }
        for lab in c.labs where lab.reviewState != .rejected {
            out.append(DiffEntry(axis: .lab,
                                 codeSystem: loincSystem,
                                 code: lab.code,
                                 display: lab.displayName,
                                 reviewStateRaw: lab.reviewStateRaw))
        }
        for med in c.medications where med.reviewState != .rejected {
            out.append(DiffEntry(axis: .medication,
                                 codeSystem: rxnormSystem,
                                 code: med.code,
                                 display: med.displayName,
                                 reviewStateRaw: med.reviewStateRaw))
        }
        if let v = c.vitals, !v.isEmpty {
            // Canonical LOINC codes for the five vitals we capture.
            if v.tempC != nil {
                out.append(DiffEntry(axis: .vital, codeSystem: vitalsSystem,
                                     code: "8310-5", display: "Body temperature",
                                     reviewStateRaw: ReviewState.confirmed.rawValue))
            }
            if v.heartRate != nil {
                out.append(DiffEntry(axis: .vital, codeSystem: vitalsSystem,
                                     code: "8867-4", display: "Heart rate",
                                     reviewStateRaw: ReviewState.confirmed.rawValue))
            }
            if v.respRate != nil {
                out.append(DiffEntry(axis: .vital, codeSystem: vitalsSystem,
                                     code: "9279-1", display: "Respiratory rate",
                                     reviewStateRaw: ReviewState.confirmed.rawValue))
            }
            if v.spo2 != nil {
                out.append(DiffEntry(axis: .vital, codeSystem: vitalsSystem,
                                     code: "59408-5", display: "SpO2",
                                     reviewStateRaw: ReviewState.confirmed.rawValue))
            }
            if v.bpSystolic != nil {
                out.append(DiffEntry(axis: .vital, codeSystem: vitalsSystem,
                                     code: "8480-6", display: "Systolic blood pressure",
                                     reviewStateRaw: ReviewState.confirmed.rawValue))
            }
        }
        return out
    }
}

// MARK: - ClinIQ flat export

enum CaseDiffExport {
    static func dictionary(from diff: CaseDiff) -> [String: Any] {
        let includeOnlyAdded = JurisdictionProfile.includeOnlyAddedFindings
        var rows: [[String: Any]] = []
        rows.append(contentsOf: exportRows(diff.added, change: "added"))
        if !includeOnlyAdded {
            rows.append(contentsOf: exportRows(diff.removed, change: "removed"))
            rows.append(contentsOf: exportRows(diff.unchanged, change: "unchanged"))
        }
        return [
            "schema": "cliniq.flat_diff.v1",
            "jurisdiction": JurisdictionProfile.currentName,
            "patientIdentityHash": diff.patientIdentityHash,
            "priorCaseId": diff.priorCaseId.uuidString,
            "currentCaseId": diff.currentCaseId.uuidString,
            "priorDate": ISO8601DateFormatter().string(from: diff.priorDate),
            "currentDate": ISO8601DateFormatter().string(from: diff.currentDate),
            "includeOnlyAddedFindings": includeOnlyAdded,
            "rows": rows,
        ]
    }

    static func jsonString(from diff: CaseDiff) -> String {
        let payload = dictionary(from: diff)
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload,
                                                      options: [.prettyPrinted, .sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return #"{"error":"flat diff export serialization failed"}"#
        }
        return text
    }

    private static func exportRows(_ entries: [DiffEntry], change: String) -> [[String: Any]] {
        entries.map { entry in
            let state = ReviewState(rawValue: entry.reviewStateRaw ?? "")
            let decision = JurisdictionProfile.decision(axis: entry.axis,
                                                        code: entry.code,
                                                        reviewState: state)
            return [
                "change": change,
                "axis": entry.axis.rawValue,
                "system": entry.codeSystem,
                "code": entry.code,
                "display": entry.display,
                "reviewState": entry.reviewStateRaw ?? "unknown",
                "ruleDecision": decision.rawValue,
            ]
        }
    }
}
