// SyncService.swift
// Drains `pending` cases to the mock public-health endpoint. Observes the
// NetworkMonitor and triggers drains when we come back online. Exposes a
// `drainNow()` coroutine for manual sync button.
//
// Architecture:
//   * `actor` owns a reference to the ModelContainer's main context.
//   * Cases transition: pending -> syncing -> submitted | failed.
//   * Each attempt appends a `SyncRecord` so audit history survives.
//
// Swift concurrency notes: we operate on the `@MainActor` context from this
// actor by hopping via `Task { @MainActor in ... }`. This mirrors Apple's
// guidance for SwiftData model contexts.

import Foundation
import SwiftData
import SwiftUI

@MainActor
final class SyncService: ObservableObject {
    @Published private(set) var isDraining: Bool = false
    @Published private(set) var lastDrainedAt: Date?
    @Published private(set) var lastMessage: String = ""

    private weak var monitor: NetworkMonitor?
    private var container: ModelContainer?
    private var autoDrainTask: Task<Void, Never>?

    init() {}

    func configure(container: ModelContainer, monitor: NetworkMonitor) {
        self.container = container
        self.monitor = monitor
        observe()
    }

    private func observe() {
        // Auto-drain when coming online.
        autoDrainTask?.cancel()
        guard let monitor = monitor else { return }
        autoDrainTask = Task { @MainActor [weak self] in
            var last = monitor.isOnline
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                let cur = monitor.isOnline
                if cur && !last {
                    self?.lastMessage = "Network restored. Draining outbox..."
                    await self?.drainNow()
                }
                last = cur
            }
        }
    }

    /// Move one pending case to `.pending` and persist. Used when the user
    /// taps "Add to outbox" during review.
    func queue(_ clinicalCase: ClinicalCase) {
        clinicalCase.status = .pending
        clinicalCase.updatedAt = Date()
        saveContext()
    }

    /// Attempt to POST every pending case. Each case gets a SyncRecord
    /// entry regardless of outcome. Non-blocking for callers — kicks off a
    /// task if one isn't already in flight.
    func drainNow() async {
        guard !isDraining else { return }
        guard let container = container else { return }
        isDraining = true
        defer { isDraining = false }

        let context = container.mainContext
        let predicate = #Predicate<ClinicalCase> { $0.statusRaw == "pending" }
        let fetch = FetchDescriptor<ClinicalCase>(predicate: predicate,
                                                  sortBy: [SortDescriptor(\.createdAt)])
        guard let pending = try? context.fetch(fetch) else {
            lastMessage = "Nothing to sync."
            return
        }
        if pending.isEmpty {
            lastMessage = "Outbox is empty."
            return
        }

        var succeeded = 0
        var failed = 0
        for c in pending {
            if let monitor = monitor, !monitor.isOnline { break }
            c.status = .syncing
            c.updatedAt = Date()
            saveContext()

            let result = await postCase(c)
            let record = SyncRecord(attemptedAt: Date(),
                                    succeeded: result.success,
                                    endpoint: SyncConfig.currentEndpoint.absoluteString,
                                    message: result.message)
            c.syncHistory.append(record)
            c.status = result.success ? .submitted : .failed
            c.updatedAt = Date()
            if result.success { succeeded += 1 } else { failed += 1 }
            saveContext()
        }

        lastDrainedAt = Date()
        lastMessage = "Synced \(succeeded) case\(succeeded == 1 ? "" : "s")" + (failed > 0 ? ", \(failed) failed" : ".")
    }

    // MARK: - Transport

    private struct PostResult {
        let success: Bool
        let message: String
    }

    private func postCase(_ c: ClinicalCase) async -> PostResult {
        let payload = ReportPayload.from(c)
        // Real transport path (only if opted in and we're actually online).
        if SyncConfig.shouldAttemptRealPost, let monitor = monitor, monitor.isOnline {
            do {
                let data = try JSONSerialization.data(withJSONObject: payload, options: [])
                var request = URLRequest(url: SyncConfig.currentEndpoint)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = data
                let (_, response) = try await URLSession.shared.data(for: request)
                if let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) {
                    return .init(success: true, message: "HTTP \(http.statusCode) from \(SyncConfig.currentEndpoint.absoluteString)")
                }
                return .init(success: false, message: "Endpoint rejected the report.")
            } catch {
                // Fall through to mock result — still persist the attempt.
                return .init(success: SyncConfig.mockSucceeds,
                             message: SyncConfig.mockSucceeds
                                 ? "Mock accept after transport error: \(error.localizedDescription)"
                                 : "Mock fail: \(error.localizedDescription)")
            }
        }

        // Fully mocked demo path.
        // Give the UI a moment so the badge actually ticks through `syncing`.
        try? await Task.sleep(nanoseconds: 900_000_000)
        if SyncConfig.mockSucceeds {
            let ref = "PH-\(Int.random(in: 10_000...99_999))"
            return .init(success: true,
                         message: "202 Accepted — ref \(ref)")
        } else {
            return .init(success: false,
                         message: "503 Service Unavailable — upstream timeout.")
        }
    }

    private func saveContext() {
        guard let ctx = container?.mainContext else { return }
        do { try ctx.save() } catch {
            NSLog("[ClinIQ] sync context save failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - Payload

/// Builds the JSON we would POST to a real public-health endpoint. Exposed
/// as a separate type so tests (future work) can diff payloads without
/// spinning up the whole sync service.
enum ReportPayload {
    static func from(_ c: ClinicalCase) -> [String: Any] {
        var payload: [String: Any] = [:]
        payload["caseId"] = c.id.uuidString
        payload["createdAt"] = ISO8601DateFormatter().string(from: c.createdAt)

        if let p = c.patient {
            var patient: [String: Any] = [:]
            patient["gender"] = p.genderRaw
            if let bd = p.birthDate {
                let fmt = DateFormatter()
                fmt.dateFormat = "yyyy-MM-dd"
                fmt.locale = Locale(identifier: "en_US_POSIX")
                patient["birthDate"] = fmt.string(from: bd)
            }
            if let pc = p.postalCode { patient["postalCode"] = pc }
            if let f = p.facilityName { patient["facility"] = f }
            payload["patient"] = patient
        }

        payload["conditions"] = c.conditions
            .filter { $0.reviewState != .rejected }
            .map {
                ["code": $0.code, "system": $0.system, "display": $0.displayName]
            }

        payload["labs"] = c.labs
            .filter { $0.reviewState != .rejected }
            .map { lab -> [String: Any] in
                var entry: [String: Any] = [
                    "code": lab.code,
                    "system": lab.system,
                    "display": lab.displayName,
                ]
                if let i = lab.interpretation { entry["interpretation"] = i }
                if let v = lab.value { entry["value"] = v }
                if let u = lab.unit { entry["unit"] = u }
                return entry
            }

        payload["medications"] = c.medications
            .filter { $0.reviewState != .rejected }
            .map {
                ["code": $0.code, "system": $0.system, "display": $0.displayName]
            }

        if let v = c.vitals, !v.isEmpty {
            var vitals: [String: Any] = [:]
            if let t = v.tempC { vitals["temp_c"] = t }
            if let h = v.heartRate { vitals["hr"] = h }
            if let r = v.respRate { vitals["rr"] = r }
            if let s = v.spo2 { vitals["spo2"] = s }
            if let bs = v.bpSystolic { vitals["bp_systolic"] = bs }
            if let bd = v.bpDiastolic { vitals["bp_diastolic"] = bd }
            payload["vitals"] = vitals
        }

        return payload
    }
}
