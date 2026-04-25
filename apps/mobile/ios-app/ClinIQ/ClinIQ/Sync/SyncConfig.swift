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
