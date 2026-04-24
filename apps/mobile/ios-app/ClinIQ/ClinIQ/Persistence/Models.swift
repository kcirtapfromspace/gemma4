// Models.swift
// SwiftData persistence models for the ClinIQ clinician PoC.
//
// Relationships:
//   Case -> Patient (1:1, owned)
//   Case -> ExtractedCondition* (1:many, cascade delete)
//   Case -> ExtractedLab* (1:many, cascade delete)
//   Case -> ExtractedMedication* (1:many, cascade delete)
//   Case -> Vitals? (0:1, cascade delete)
//   Case -> SyncRecord? (0:1 latest attempt)
//
// Storage: SwiftData ModelContainer with cloud disabled, app support dir,
// data-protection set to completeUntilFirstUserAuthentication. See
// PersistenceController.swift for configuration.

import Foundation
import SwiftData

// MARK: - Status enum

enum CaseStatus: String, Codable, CaseIterable {
    case draft
    case pending      // queued in outbox, waiting for network
    case syncing      // POST in flight
    case submitted    // mock endpoint accepted it
    case failed       // mock endpoint rejected / toggle flipped to fail
}

enum ReviewState: String, Codable {
    case needsReview  // AI produced it, clinician has not confirmed
    case confirmed    // clinician accepted
    case edited       // clinician modified text
    case rejected     // clinician removed it (soft-hide; stays in audit)
}

// MARK: - Core entities

@Model
final class ClinicalCase {
    // Identity
    @Attribute(.unique) var id: UUID
    var createdAt: Date
    var updatedAt: Date

    // Narrative input — what the clinician typed or pasted
    var narrative: String

    // Lifecycle
    var statusRaw: String = CaseStatus.draft.rawValue
    var status: CaseStatus {
        get { CaseStatus(rawValue: statusRaw) ?? .draft }
        set { statusRaw = newValue.rawValue }
    }

    // Inference telemetry (populated after extraction)
    var tokensGenerated: Int = 0
    var elapsedSeconds: Double = 0
    var tokensPerSecond: Double = 0

    // Relationships
    @Relationship(deleteRule: .cascade) var patient: Patient?
    @Relationship(deleteRule: .cascade, inverse: \ExtractedCondition.clinicalCase) var conditions: [ExtractedCondition] = []
    @Relationship(deleteRule: .cascade, inverse: \ExtractedLab.clinicalCase) var labs: [ExtractedLab] = []
    @Relationship(deleteRule: .cascade, inverse: \ExtractedMedication.clinicalCase) var medications: [ExtractedMedication] = []
    @Relationship(deleteRule: .cascade) var vitals: Vitals?
    @Relationship(deleteRule: .cascade, inverse: \SyncRecord.clinicalCase) var syncHistory: [SyncRecord] = []

    init(id: UUID = UUID(),
         narrative: String = "",
         status: CaseStatus = .draft,
         createdAt: Date = Date()) {
        self.id = id
        self.narrative = narrative
        self.statusRaw = status.rawValue
        self.createdAt = createdAt
        self.updatedAt = createdAt
    }

    // MARK: - Computed

    /// The canonical display label used across list / detail screens.
    var displayTitle: String {
        let name = patient?.fullName ?? "Unknown patient"
        let cond = conditions.first?.displayName ?? "Unclassified"
        return "\(name) — \(cond)"
    }

    var primaryConditionDisplay: String {
        conditions.first?.displayName ?? "Pending review"
    }

    var hasUnreviewedEntities: Bool {
        conditions.contains { $0.reviewState == .needsReview }
            || labs.contains { $0.reviewState == .needsReview }
            || medications.contains { $0.reviewState == .needsReview }
    }

    var acceptedCount: Int {
        conditions.filter { $0.reviewState != .rejected }.count
            + labs.filter { $0.reviewState != .rejected }.count
            + medications.filter { $0.reviewState != .rejected }.count
    }
}

@Model
final class Patient {
    var givenName: String
    var familyName: String
    var genderRaw: String  // "M" / "F" / "U"
    var birthDate: Date?
    var postalCode: String?
    var facilityName: String?

    init(givenName: String = "",
         familyName: String = "",
         gender: String = "U",
         birthDate: Date? = nil,
         postalCode: String? = nil,
         facilityName: String? = nil) {
        self.givenName = givenName
        self.familyName = familyName
        self.genderRaw = gender
        self.birthDate = birthDate
        self.postalCode = postalCode
        self.facilityName = facilityName
    }

    var fullName: String {
        let name = [givenName, familyName].filter { !$0.isEmpty }.joined(separator: " ")
        return name.isEmpty ? "Unnamed patient" : name
    }

    var ageDescription: String {
        guard let bd = birthDate else { return "—" }
        let years = Calendar.current.dateComponents([.year], from: bd, to: Date()).year ?? 0
        return "\(years) y"
    }

    var genderDisplay: String {
        switch genderRaw.uppercased() {
        case "M": return "Male"
        case "F": return "Female"
        default: return "Unspecified"
        }
    }
}

@Model
final class ExtractedCondition {
    @Attribute(.unique) var id: UUID
    var code: String         // e.g. "840539006"
    var system: String       // "SNOMED"
    var displayName: String  // "COVID-19"
    var reviewStateRaw: String = ReviewState.needsReview.rawValue
    var reviewState: ReviewState {
        get { ReviewState(rawValue: reviewStateRaw) ?? .needsReview }
        set { reviewStateRaw = newValue.rawValue }
    }
    var clinicalCase: ClinicalCase?

    init(id: UUID = UUID(),
         code: String,
         system: String = "SNOMED",
         displayName: String,
         reviewState: ReviewState = .needsReview) {
        self.id = id
        self.code = code
        self.system = system
        self.displayName = displayName
        self.reviewStateRaw = reviewState.rawValue
    }
}

@Model
final class ExtractedLab {
    @Attribute(.unique) var id: UUID
    var code: String         // LOINC
    var system: String       // "LOINC"
    var displayName: String  // "SARS-CoV-2 RNA NAA+probe Ql Resp"
    var interpretation: String?  // "Detected" / "Not detected" / "Positive" / "Negative"
    var value: Double?
    var unit: String?
    var reviewStateRaw: String = ReviewState.needsReview.rawValue
    var reviewState: ReviewState {
        get { ReviewState(rawValue: reviewStateRaw) ?? .needsReview }
        set { reviewStateRaw = newValue.rawValue }
    }
    var clinicalCase: ClinicalCase?

    init(id: UUID = UUID(),
         code: String,
         system: String = "LOINC",
         displayName: String,
         interpretation: String? = nil,
         value: Double? = nil,
         unit: String? = nil,
         reviewState: ReviewState = .needsReview) {
        self.id = id
        self.code = code
        self.system = system
        self.displayName = displayName
        self.interpretation = interpretation
        self.value = value
        self.unit = unit
        self.reviewStateRaw = reviewState.rawValue
    }

    /// Human-readable result summary.
    var resultSummary: String {
        if let v = value, let u = unit {
            return "\(formatNumber(v)) \(u)"
        }
        return (interpretation ?? "Pending").capitalized
    }

    private func formatNumber(_ v: Double) -> String {
        if v.rounded() == v { return String(Int(v)) }
        return String(format: "%.1f", v)
    }
}

@Model
final class ExtractedMedication {
    @Attribute(.unique) var id: UUID
    var code: String         // RxNorm
    var system: String       // "RxNorm"
    var displayName: String
    var reviewStateRaw: String = ReviewState.needsReview.rawValue
    var reviewState: ReviewState {
        get { ReviewState(rawValue: reviewStateRaw) ?? .needsReview }
        set { reviewStateRaw = newValue.rawValue }
    }
    var clinicalCase: ClinicalCase?

    init(id: UUID = UUID(),
         code: String,
         system: String = "RxNorm",
         displayName: String,
         reviewState: ReviewState = .needsReview) {
        self.id = id
        self.code = code
        self.system = system
        self.displayName = displayName
        self.reviewStateRaw = reviewState.rawValue
    }
}

@Model
final class Vitals {
    var tempC: Double?
    var heartRate: Int?
    var respRate: Int?
    var spo2: Int?
    var bpSystolic: Int?
    var bpDiastolic: Int?

    init(tempC: Double? = nil,
         heartRate: Int? = nil,
         respRate: Int? = nil,
         spo2: Int? = nil,
         bpSystolic: Int? = nil,
         bpDiastolic: Int? = nil) {
        self.tempC = tempC
        self.heartRate = heartRate
        self.respRate = respRate
        self.spo2 = spo2
        self.bpSystolic = bpSystolic
        self.bpDiastolic = bpDiastolic
    }

    var isEmpty: Bool {
        tempC == nil && heartRate == nil && respRate == nil && spo2 == nil && bpSystolic == nil
    }
}

@Model
final class SyncRecord {
    @Attribute(.unique) var id: UUID
    var attemptedAt: Date
    var succeeded: Bool
    var endpoint: String
    var message: String?
    var clinicalCase: ClinicalCase?

    init(id: UUID = UUID(),
         attemptedAt: Date = Date(),
         succeeded: Bool,
         endpoint: String,
         message: String? = nil) {
        self.id = id
        self.attemptedAt = attemptedAt
        self.succeeded = succeeded
        self.endpoint = endpoint
        self.message = message
    }
}
