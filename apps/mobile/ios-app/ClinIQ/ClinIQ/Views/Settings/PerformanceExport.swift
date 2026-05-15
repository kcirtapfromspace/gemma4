// PerformanceExport.swift
// Builds a reviewer-friendly JSON artifact from on-device runs. The export
// is intentionally PHI-light: it records narrative length and hash, not the
// raw narrative text.

import CryptoKit
import Foundation
import SwiftData
#if canImport(UIKit)
import UIKit
#endif

@MainActor
enum PerformanceExportBuilder {
    static func write(cases: [ClinicalCase], activeBackendLabel: String) throws -> URL {
        let artifact = makeArtifact(cases: cases, activeBackendLabel: activeBackendLabel)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(artifact)
        let stamp = ISO8601DateFormatter()
            .string(from: artifact.generatedAt)
            .replacingOccurrences(of: ":", with: "")
            .replacingOccurrences(of: ".", with: "")
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let url = directory.appendingPathComponent("cliniq-device-performance-\(stamp).json")
        try data.write(to: url, options: .atomic)
        return url
    }

    private static func makeArtifact(cases: [ClinicalCase], activeBackendLabel: String) -> PerformanceArtifact {
        let sortedCases = cases.sorted { $0.createdAt < $1.createdAt }
        let rows = sortedCases.map(CaseRun.init)

        return PerformanceArtifact(
            schemaVersion: 1,
            generatedAt: Date(),
            privacy: PrivacyNote(
                rawNarrativesIncluded: false,
                patientNamesIncluded: false,
                note: "Narratives are represented only by character count and SHA-256 hash. Export assumes synthetic demo data or user-approved sharing."
            ),
            app: AppSnapshot(),
            device: DeviceSnapshot(),
            configuration: ConfigurationSnapshot(activeBackendLabel: activeBackendLabel),
            liveMetricsSnapshot: LiveMetricsSnapshot(metrics: InferenceMetrics.shared),
            aggregate: AggregateStats(rows: rows),
            cases: rows
        )
    }
}

struct PerformanceArtifact: Codable {
    let schemaVersion: Int
    let generatedAt: Date
    let privacy: PrivacyNote
    let app: AppSnapshot
    let device: DeviceSnapshot
    let configuration: ConfigurationSnapshot
    let liveMetricsSnapshot: LiveMetricsSnapshot
    let aggregate: AggregateStats
    let cases: [CaseRun]
}

struct PrivacyNote: Codable {
    let rawNarrativesIncluded: Bool
    let patientNamesIncluded: Bool
    let note: String
}

struct AppSnapshot: Codable {
    let bundleIdentifier: String
    let displayName: String
    let version: String
    let build: String

    init(bundle: Bundle = .main) {
        bundleIdentifier = bundle.bundleIdentifier ?? "unknown"
        displayName = bundle.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String
            ?? bundle.object(forInfoDictionaryKey: "CFBundleName") as? String
            ?? "ClinIQ"
        version = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        build = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
    }
}

struct DeviceSnapshot: Codable {
    let hardwareIdentifier: String
    let systemName: String
    let systemVersion: String
    let model: String
    let physicalMemoryBytes: UInt64
    let processorCount: Int
    let activeProcessorCount: Int
    let thermalState: String
    let lowPowerModeEnabled: Bool
    let batteryLevel: Float?
    let batteryState: String?

    init() {
        hardwareIdentifier = Self.hardwareIdentifier()
        physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
        processorCount = ProcessInfo.processInfo.processorCount
        activeProcessorCount = ProcessInfo.processInfo.activeProcessorCount
        thermalState = Self.thermalState(ProcessInfo.processInfo.thermalState)
        lowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled

        #if canImport(UIKit)
        UIDevice.current.isBatteryMonitoringEnabled = true
        systemName = UIDevice.current.systemName
        systemVersion = UIDevice.current.systemVersion
        model = UIDevice.current.model
        let level = UIDevice.current.batteryLevel
        batteryLevel = level >= 0 ? level : nil
        batteryState = Self.batteryState(UIDevice.current.batteryState)
        #else
        systemName = "unknown"
        systemVersion = ProcessInfo.processInfo.operatingSystemVersionString
        model = "unknown"
        batteryLevel = nil
        batteryState = nil
        #endif
    }

    private static func hardwareIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return Mirror(reflecting: systemInfo.machine).children.reduce(into: "") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            identifier.append(String(UnicodeScalar(UInt8(value))))
        }
    }

    private static func thermalState(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    #if canImport(UIKit)
    private static func batteryState(_ state: UIDevice.BatteryState) -> String {
        switch state {
        case .unknown: return "unknown"
        case .unplugged: return "unplugged"
        case .charging: return "charging"
        case .full: return "full"
        @unknown default: return "unknown"
        }
    }
    #endif
}

struct ConfigurationSnapshot: Codable {
    let selectedBackendRaw: String
    let activeBackendLabel: String
    let llamaCppModelPresent: Bool
    let llamaCppModelName: String?
    let liteRtLmModelPresent: Bool
    let liteRtLmModelName: String?
    let llmReviewMode: String
    let syncPayloadKind: String
    let syncMode: String

    init(activeBackendLabel: String) {
        selectedBackendRaw = UserDefaults.standard.string(forKey: InferenceBackend.appStorageKey)
            ?? InferenceBackend.default.rawValue
        self.activeBackendLabel = activeBackendLabel

        let gguf = LlamaCppInferenceEngine.resolveModelPath()
        llamaCppModelPresent = gguf != nil
        llamaCppModelName = gguf.map { URL(fileURLWithPath: $0).lastPathComponent }

        let litert = LiteRtLmInferenceEngine.resolveModelPath()
        liteRtLmModelPresent = litert != nil
        liteRtLmModelName = litert.map { URL(fileURLWithPath: $0).lastPathComponent }
        let rawReviewMode = UserDefaults.standard.string(forKey: LLMReviewMode.appStorageKey)
            ?? LLMReviewMode.default.rawValue
        llmReviewMode = LLMReviewMode(rawValue: rawReviewMode)?.displayName
            ?? LLMReviewMode.default.displayName

        syncPayloadKind = SyncConfig.useFhirBundlePayload ? "FHIR R4 Bundle" : "legacy extraction"
        syncMode = SyncConfig.shouldAttemptRealPost ? "real POST with mock fallback" : "mock"
    }
}

struct LiveMetricsSnapshot: Codable {
    let phase: String
    let backend: String
    let modelName: String
    let promptChars: Int
    let promptTokensApprox: Int
    let outputTokens: Int
    let elapsedSeconds: Double
    let firstTokenLatencySeconds: Double?
    let instantTokensPerSecond: Double
    let averageTokensPerSecond: Double
    let peakTokensPerSecond: Double
    let residentMemoryMB: Double
    let threadCount: Int
    let lastError: String?

    @MainActor
    init(metrics: InferenceMetrics) {
        phase = metrics.phase.rawValue
        backend = metrics.backend
        modelName = metrics.modelName
        promptChars = metrics.promptChars
        promptTokensApprox = metrics.promptTokensApprox
        outputTokens = metrics.outputTokens
        elapsedSeconds = metrics.elapsedSeconds
        firstTokenLatencySeconds = metrics.firstTokenLatencySeconds
        instantTokensPerSecond = metrics.instantTokensPerSecond
        averageTokensPerSecond = metrics.avgTokensPerSecond
        peakTokensPerSecond = metrics.peakTokensPerSecond
        residentMemoryMB = metrics.residentMemoryMB
        threadCount = metrics.threadCount
        lastError = metrics.lastError
    }
}

struct AggregateStats: Codable {
    let totalCases: Int
    let measuredCases: Int
    let modelTokenCases: Int
    let zeroTokenPipelineCases: Int
    let statusCounts: [String: Int]
    let entityCounts: EntityCounts
    let elapsedSeconds: NumericStats
    let tokensPerSecond: NumericStats
    let tokensGenerated: NumericStats

    init(rows: [CaseRun]) {
        totalCases = rows.count
        measuredCases = rows.filter { $0.metrics.elapsedSeconds > 0 }.count
        modelTokenCases = rows.filter { $0.metrics.tokensGenerated > 0 }.count
        zeroTokenPipelineCases = rows.filter { $0.metrics.elapsedSeconds > 0 && $0.metrics.tokensGenerated == 0 }.count
        statusCounts = Dictionary(grouping: rows, by: { $0.status }).mapValues(\.count)
        entityCounts = EntityCounts(
            conditions: rows.reduce(0) { $0 + $1.extracted.conditions.count },
            labs: rows.reduce(0) { $0 + $1.extracted.labs.count },
            medications: rows.reduce(0) { $0 + $1.extracted.medications.count },
            casesWithVitals: rows.filter { $0.extracted.vitals != nil }.count
        )
        elapsedSeconds = NumericStats(rows.map(\.metrics.elapsedSeconds).filter { $0 > 0 })
        tokensPerSecond = NumericStats(rows.map(\.metrics.tokensPerSecond).filter { $0 > 0 })
        tokensGenerated = NumericStats(rows.map { Double($0.metrics.tokensGenerated) }.filter { $0 > 0 })
    }
}

struct NumericStats: Codable {
    let count: Int
    let min: Double?
    let mean: Double?
    let median: Double?
    let p95: Double?
    let max: Double?

    init(_ values: [Double]) {
        let sorted = values.sorted()
        count = sorted.count
        min = sorted.first
        max = sorted.last
        mean = sorted.isEmpty ? nil : sorted.reduce(0, +) / Double(sorted.count)
        median = Self.percentile(sorted, fraction: 0.50)
        p95 = Self.percentile(sorted, fraction: 0.95)
    }

    private static func percentile(_ sorted: [Double], fraction: Double) -> Double? {
        guard !sorted.isEmpty else { return nil }
        if sorted.count == 1 { return sorted[0] }
        let position = Swift.max(0, Swift.min(sorted.count - 1, Int((Double(sorted.count - 1) * fraction).rounded())))
        return sorted[position]
    }
}

struct EntityCounts: Codable {
    let conditions: Int
    let labs: Int
    let medications: Int
    let casesWithVitals: Int
}

struct CaseRun: Codable {
    let caseId: String
    let createdAt: Date
    let updatedAt: Date
    let status: String
    let narrativeChars: Int
    let narrativeSHA256: String
    let patient: PatientSummary?
    let metrics: CaseMetrics
    let extracted: ExtractedSummary
    let syncHistory: [SyncAttemptSummary]

    init(_ clinicalCase: ClinicalCase) {
        caseId = clinicalCase.id.uuidString
        createdAt = clinicalCase.createdAt
        updatedAt = clinicalCase.updatedAt
        status = clinicalCase.status.rawValue
        narrativeChars = clinicalCase.narrative.count
        narrativeSHA256 = Self.sha256(clinicalCase.narrative)
        patient = clinicalCase.patient.map(PatientSummary.init)
        metrics = CaseMetrics(clinicalCase)
        extracted = ExtractedSummary(clinicalCase)
        syncHistory = clinicalCase.syncHistory
            .sorted { $0.attemptedAt < $1.attemptedAt }
            .map(SyncAttemptSummary.init)
    }

    private static func sha256(_ value: String) -> String {
        let digest = SHA256.hash(data: Data(value.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

struct PatientSummary: Codable {
    let gender: String
    let birthYear: Int?
    let postalCodePrefix: String?
    let facilityName: String?

    init(_ patient: Patient) {
        gender = patient.genderRaw
        birthYear = patient.birthDate.map { Calendar(identifier: .gregorian).component(.year, from: $0) }
        postalCodePrefix = patient.postalCode.map { String($0.prefix(3)) }
        facilityName = patient.facilityName
    }
}

struct CaseMetrics: Codable {
    let elapsedSeconds: Double
    let tokensGenerated: Int
    let tokensPerSecond: Double

    init(_ clinicalCase: ClinicalCase) {
        elapsedSeconds = clinicalCase.elapsedSeconds
        tokensGenerated = clinicalCase.tokensGenerated
        tokensPerSecond = clinicalCase.tokensPerSecond
    }
}

struct ExtractedSummary: Codable {
    let conditions: [CodeSummary]
    let labs: [LabSummary]
    let medications: [CodeSummary]
    let vitals: VitalsSummary?
    let acceptedEntityCount: Int

    init(_ clinicalCase: ClinicalCase) {
        conditions = clinicalCase.conditions
            .sorted { $0.code < $1.code }
            .map(CodeSummary.init)
        labs = clinicalCase.labs
            .sorted { $0.code < $1.code }
            .map(LabSummary.init)
        medications = clinicalCase.medications
            .sorted { $0.code < $1.code }
            .map(CodeSummary.init)
        if let v = clinicalCase.vitals, !v.isEmpty {
            vitals = VitalsSummary(v)
        } else {
            vitals = nil
        }
        acceptedEntityCount = clinicalCase.acceptedCount
    }
}

struct CodeSummary: Codable {
    let code: String
    let system: String
    let display: String
    let reviewState: String

    init(_ condition: ExtractedCondition) {
        code = condition.code
        system = condition.system
        display = condition.displayName
        reviewState = condition.reviewState.rawValue
    }

    init(_ medication: ExtractedMedication) {
        code = medication.code
        system = medication.system
        display = medication.displayName
        reviewState = medication.reviewState.rawValue
    }
}

struct LabSummary: Codable {
    let code: String
    let system: String
    let display: String
    let interpretation: String?
    let value: Double?
    let unit: String?
    let reviewState: String

    init(_ lab: ExtractedLab) {
        code = lab.code
        system = lab.system
        display = lab.displayName
        interpretation = lab.interpretation
        value = lab.value
        unit = lab.unit
        reviewState = lab.reviewState.rawValue
    }
}

struct VitalsSummary: Codable {
    let tempC: Double?
    let heartRate: Int?
    let respRate: Int?
    let spo2: Int?
    let bpSystolic: Int?
    let bpDiastolic: Int?

    init(_ vitals: Vitals) {
        tempC = vitals.tempC
        heartRate = vitals.heartRate
        respRate = vitals.respRate
        spo2 = vitals.spo2
        bpSystolic = vitals.bpSystolic
        bpDiastolic = vitals.bpDiastolic
    }
}

struct SyncAttemptSummary: Codable {
    let attemptedAt: Date
    let succeeded: Bool
    let endpoint: String
    let message: String?

    init(_ record: SyncRecord) {
        attemptedAt = record.attemptedAt
        succeeded = record.succeeded
        endpoint = record.endpoint
        message = record.message
    }
}
