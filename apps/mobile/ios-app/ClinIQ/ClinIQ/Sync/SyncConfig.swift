// SyncConfig.swift
// Configuration for the mock public-health endpoint.
//
// For the 25-day hackathon demo we POST to a local endpoint at
// `http://localhost:8080/reports`. The production pathway (not wired) would
// target a mutually-authenticated TLS endpoint configured per-jurisdiction
// (state / tribal public-health agency). Auth strategy is out of scope here.
//
// Demo narrative control: the `ClinIQ.MockSyncSucceeds` UserDefaults flag
// lets the demoer flip between "success" and "failure" without hitting the
// network. `true` = simulate 202 Accepted, `false` = simulate 503 failure.
//
// When `ClinIQ.UseLocalEndpoint` is true AND the network is online AND no
// UserDefaults override for `MockSyncSucceeds` is set, the sync service will
// actually POST to `localhost:8080`. If nothing is listening the POST fails
// naturally — this is useful when the demo host has the mock server up.

import Foundation

enum SyncConfig {
    static let defaultEndpoint = URL(string: "http://localhost:8080/reports")!

    static var currentEndpoint: URL {
        if let override = UserDefaults.standard.string(forKey: "ClinIQ.EndpointURL"),
           let url = URL(string: override) {
            return url
        }
        return defaultEndpoint
    }

    /// When unset, the sync service operates in fully mocked mode (no HTTP
    /// traffic). When set, an actual POST is attempted; on any transport
    /// failure we fall back to the mock result.
    static var shouldAttemptRealPost: Bool {
        UserDefaults.standard.bool(forKey: "ClinIQ.UseLocalEndpoint")
    }

    /// Toggle that controls the mocked-success narrative.
    static var mockSucceeds: Bool {
        // Default to `true` the first time the app runs — the demo starts in
        // the happy path.
        if UserDefaults.standard.object(forKey: "ClinIQ.MockSyncSucceeds") == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: "ClinIQ.MockSyncSucceeds")
    }

    static func setMockSucceeds(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "ClinIQ.MockSyncSucceeds")
    }

    /// When true, the Outbox payload is a FHIR R4 Bundle (built on-device
    /// via `BundleBuilder`); otherwise the legacy flat extraction dict.
    /// Defaults to `true` — judges expect the wire format to match the
    /// "View FHIR Bundle" sheet shown on the Review screen. The legacy
    /// path stays available behind the toggle so a non-FHIR partner
    /// endpoint can still receive a report.
    static var useFhirBundlePayload: Bool {
        if UserDefaults.standard.object(forKey: "ClinIQ.UseFhirBundlePayload") == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: "ClinIQ.UseFhirBundlePayload")
    }

    static func setUseFhirBundlePayload(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "ClinIQ.UseFhirBundlePayload")
    }
}

// MARK: - Jurisdiction rules

enum JurisdictionRuleDecision: String, Codable, Hashable {
    case reportable
    case needsReview
    case notIncluded

    var displayName: String {
        switch self {
        case .reportable: return "Reportable"
        case .needsReview: return "Needs review"
        case .notIncluded: return "Not included"
        }
    }
}

struct JurisdictionCategoryOption: Identifiable {
    let id: String
    let label: String
}

enum JurisdictionProfile {
    static let defaultName = "Arizona Demo PHA"
    static let nameKey = "ClinIQ.Jurisdiction.Name"
    static let enabledCategoriesKey = "ClinIQ.Jurisdiction.EnabledCategories"
    static let requireReviewKey = "ClinIQ.Jurisdiction.RequireClinicianReview"
    static let includeOnlyAddedKey = "ClinIQ.Jurisdiction.IncludeOnlyAddedFindings"

    static let categoryOptions: [JurisdictionCategoryOption] = [
        JurisdictionCategoryOption(id: "outbreak_priority", label: "Outbreak priority"),
        JurisdictionCategoryOption(id: "vector_borne", label: "Vector-borne"),
        JurisdictionCategoryOption(id: "respiratory", label: "Respiratory"),
        JurisdictionCategoryOption(id: "vaccine_preventable", label: "Vaccine-preventable"),
        JurisdictionCategoryOption(id: "foodborne", label: "Foodborne"),
        JurisdictionCategoryOption(id: "fungal", label: "Fungal"),
        JurisdictionCategoryOption(id: "zoonotic", label: "Zoonotic"),
        JurisdictionCategoryOption(id: "stis", label: "STIs"),
        JurisdictionCategoryOption(id: "healthcare_associated", label: "Healthcare-associated"),
        JurisdictionCategoryOption(id: "select_agent", label: "Select agent"),
    ]

    static let defaultEnabledCategories: Set<String> = Set(categoryOptions.map(\.id))

    static var defaultEnabledCategoriesCSV: String {
        defaultEnabledCategories.sorted().joined(separator: ",")
    }

    static var currentName: String {
        let raw = UserDefaults.standard.string(forKey: nameKey) ?? defaultName
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? defaultName : trimmed
    }

    static var enabledCategories: Set<String> {
        let raw = UserDefaults.standard.string(forKey: enabledCategoriesKey)
            ?? defaultEnabledCategoriesCSV
        let values = raw.split(separator: ",")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return values.isEmpty ? defaultEnabledCategories : Set(values)
    }

    static func setEnabledCategories(_ categories: Set<String>) {
        let next = categories.isEmpty ? defaultEnabledCategories : categories
        UserDefaults.standard.set(next.sorted().joined(separator: ","), forKey: enabledCategoriesKey)
    }

    static var requireClinicianReview: Bool {
        if UserDefaults.standard.object(forKey: requireReviewKey) == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: requireReviewKey)
    }

    static var includeOnlyAddedFindings: Bool {
        if UserDefaults.standard.object(forKey: includeOnlyAddedKey) == nil {
            return false
        }
        return UserDefaults.standard.bool(forKey: includeOnlyAddedKey)
    }

    static func category(forSNOMED code: String) -> String? {
        ReportableConditions.all.first { $0.code == code }?.category
    }

    static func categoryLabel(_ id: String?) -> String {
        guard let id else { return "Unmapped" }
        return categoryOptions.first { $0.id == id }?.label ?? id
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }

    static func decision(axis: ExtractionAxis,
                         code: String,
                         reviewState: ReviewState?) -> JurisdictionRuleDecision {
        if reviewState == .rejected {
            return .notIncluded
        }
        if requireClinicianReview && reviewState == .needsReview {
            return .needsReview
        }
        guard axis == .condition else {
            return .reportable
        }
        guard let category = category(forSNOMED: code) else {
            return .needsReview
        }
        return enabledCategories.contains(category) ? .reportable : .notIncluded
    }
}
