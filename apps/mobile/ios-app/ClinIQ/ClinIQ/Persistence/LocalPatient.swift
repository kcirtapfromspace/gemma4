// LocalPatient.swift
// Patient-identity model for the longitudinal "what's new" feature.
//
// At the edge there is no Verato or master patient index; the clinician
// confirms identity at intake. To recognise a returning patient across
// shifts we hash (given, family, dob) into a stable 16-hex-char identity
// hash. Exact match only — no probabilistic matching.
//
// Mirrors the Python edge agent's identity hash so cases produced by both
// pipelines align in the timeline.

import Foundation
import CryptoKit

/// Lightweight patient identity used to chain cases together for the same
/// person across visits. Not persisted as its own SwiftData row — the hash
/// lives on `ClinicalCase.patientIdentityHash`. The struct is the canonical
/// builder + comparator.
struct LocalPatient: Hashable, Codable {
    let id: UUID
    let given: String
    let family: String
    let dob: Date

    init(id: UUID = UUID(), given: String, family: String, dob: Date) {
        self.id = id
        self.given = given
        self.family = family
        self.dob = dob
    }

    /// Stable identity hash: first 16 hex chars of
    /// `sha256(lower(given) + "|" + lower(family) + "|" + iso8601(dob))`.
    /// Using only the first 16 hex chars (64 bits) keeps the value short
    /// enough for log lines while remaining collision-resistant for the
    /// O(100) patients a clinician sees in their lifetime on this device.
    var identityHash: String {
        Self.identityHash(given: given, family: family, dob: dob)
    }

    /// Static builder so callers (intake form, seed data) can compute the
    /// hash without instantiating a `LocalPatient` first.
    static func identityHash(given: String, family: String, dob: Date) -> String {
        let g = given.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let f = family.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let iso = Self.isoFormatter.string(from: dob)
        let payload = "\(g)|\(f)|\(iso)"
        let digest = SHA256.hash(data: Data(payload.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(16))
    }

    /// Convenience: derive a hash from an existing SwiftData `Patient` row.
    /// Returns nil if the row is missing a birthDate — without one we cannot
    /// produce a stable identity, and the case stays unlinked.
    static func identityHash(from patient: Patient) -> String? {
        guard let dob = patient.birthDate,
              !patient.givenName.isEmpty,
              !patient.familyName.isEmpty
        else { return nil }
        return identityHash(given: patient.givenName,
                            family: patient.familyName,
                            dob: dob)
    }

    private static let isoFormatter: DateFormatter = {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()
}
