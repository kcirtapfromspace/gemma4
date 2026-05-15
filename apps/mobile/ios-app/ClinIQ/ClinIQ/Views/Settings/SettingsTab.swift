// SettingsTab.swift
// Demo toggles and environment info. Intentionally narrow — this is a PoC,
// not a production settings screen.

import SwiftUI
import SwiftData

struct SettingsTab: View {
    @EnvironmentObject private var monitor: NetworkMonitor
    @EnvironmentObject private var sync: SyncService
    @AppStorage("ClinIQ.MockSyncSucceeds") private var mockSucceeds: Bool = true
    @AppStorage("ClinIQ.UseLocalEndpoint") private var useLocal: Bool = false
    @AppStorage(InferenceBackend.appStorageKey) private var backendRaw: String = InferenceBackend.default.rawValue
    @AppStorage(LLMReviewMode.appStorageKey) private var llmReviewModeRaw: String = LLMReviewMode.default.rawValue
    // Bridge to the shared ExtractionService so flipping the backend
    // picker invalidates the cached engine on the next extract call.
    @EnvironmentObject private var extractionService: ExtractionService
    @Environment(\.modelContext) private var modelContext
    @Query(sort: [SortDescriptor(\ClinicalCase.createdAt, order: .reverse)])
    private var allCases: [ClinicalCase]

    @State private var showResetConfirm = false
    @State private var lastResetAt: Date? = nil
    @State private var performanceExportURL: URL?
    @State private var performanceExportError: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Backend", selection: $backendRaw) {
                        ForEach(InferenceBackend.allCases) { b in
                            Text(b.displayName).tag(b.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: backendRaw) { _, _ in
                        extractionService.reloadEngine()
                    }
                    LabeledContent("Currently serving") {
                        Text(extractionService.activeBackendLabel)
                            .foregroundStyle(.secondary)
                    }
                    Picker("LLM usage", selection: $llmReviewModeRaw) {
                        ForEach(LLMReviewMode.allCases) { mode in
                            Text(mode.displayName).tag(mode.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Inference backend")
                } footer: {
                    Text("llama.cpp runs Gemma 4 via the vendored GGUF. Always audit forces a model-backed review even when deterministic extraction finds codes, so physical-device benchmarks prove real LLM invocation. Auto shortcuts preserves the faster deterministic/RAG path.")
                }

                // Demo controls live near the top of Settings so the
                // presenter can hand the phone off and re-seed without
                // scrolling. Pairs with the Connectivity → Simulate
                // offline toggle.
                Section {
                    Button(role: .destructive) {
                        showResetConfirm = true
                    } label: {
                        Label("Reset demo cases", systemImage: "arrow.counterclockwise.circle")
                    }
                    if let when = lastResetAt {
                        LabeledContent("Last reset") {
                            Text(when.formatted(date: .omitted, time: .shortened))
                                .foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("Demo")
                } footer: {
                    Text("Wipes every case and re-seeds the four demo patients (Maria, Daniel, Michael, Aisha). Use between demo handoffs so the next reviewer sees the same starting state.")
                }

                Section {
                    Button {
                        preparePerformanceExport()
                    } label: {
                        Label("Prepare benchmark JSON", systemImage: "doc.badge.gearshape")
                    }
                    if let url = performanceExportURL {
                        ShareLink(item: url) {
                            Label("Share latest JSON", systemImage: "square.and.arrow.up")
                        }
                        LabeledContent("Latest file") {
                            Text(url.lastPathComponent)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .multilineTextAlignment(.trailing)
                        }
                    }
                    if let error = performanceExportError {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(Color(red: 0.66, green: 0.15, blue: 0.12))
                    }
                } header: {
                    Text("Performance export")
                } footer: {
                    Text("Writes a PHI-light JSON artifact for physical-device benchmarking: app/build metadata, device profile, live model metrics, per-case latency, tokens/sec, extracted codes, and sync attempts. Narratives are hashed, not exported.")
                }

                Section("Connectivity") {
                    LabeledContent("Network status") {
                        Text(monitor.isOnline ? "Online (\(monitor.interfaceDescription))" : "Offline")
                            .foregroundStyle(monitor.isOnline ? Color(red: 0.13, green: 0.36, blue: 0.22) : Color(red: 0.56, green: 0.10, blue: 0.06))
                    }
                    Toggle("Simulate offline for demo",
                           isOn: Binding(get: { monitor.simulateOffline },
                                         set: { monitor.simulateOffline = $0 }))
                        .tint(ClinIQTheme.accent)
                }

                Section {
                    Toggle("Mock sync succeeds", isOn: $mockSucceeds)
                        .tint(ClinIQTheme.accent)
                    Toggle("Attempt real POST (localhost:8080)", isOn: $useLocal)
                        .tint(ClinIQTheme.accent)
                } header: {
                    Text("Sync behaviour")
                } footer: {
                    Text("In demo mode sync is fully simulated — useful when there's no backend running. Toggle the real POST only if you've started the mock endpoint locally.")
                }

                Section("Endpoint") {
                    LabeledContent("Reports URL") {
                        Text(SyncConfig.currentEndpoint.absoluteString)
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    LabeledContent("Last sync") {
                        Text(sync.lastDrainedAt.map { $0.formatted(.dateTime) } ?? "—")
                            .foregroundStyle(.secondary)
                    }
                }

                Section("About") {
                    LabeledContent("Build") {
                        Text("ClinIQ PoC · team C13")
                            .foregroundStyle(.secondary)
                    }
                    LabeledContent("Inference") {
                        Text(inferenceLabel)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .confirmationDialog("Reset demo cases?",
                                isPresented: $showResetConfirm,
                                titleVisibility: .visible) {
                Button("Reset and re-seed", role: .destructive) {
                    PersistenceController.resetDemo(container: modelContext.container)
                    lastResetAt = Date()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("All current cases — including any queued reports — will be deleted, then the four demo patients will be re-inserted. This action cannot be undone.")
            }
        }
    }

    private var inferenceLabel: String {
        let ggufPresent = LlamaCppInferenceEngine.resolveModelPath() != nil
        let litertlmPresent = LiteRtLmInferenceEngine.resolveModelPath() != nil
        switch (ggufPresent, litertlmPresent) {
        case (true, true): return "GGUF + .litertlm both present"
        case (true, false): return "GGUF only (llama.cpp path)"
        case (false, true): return ".litertlm only (LiteRT-LM path)"
        case (false, false): return "Rule-based fallback (no model)"
        }
    }

    private func preparePerformanceExport() {
        do {
            performanceExportURL = try PerformanceExportBuilder.write(
                cases: allCases,
                activeBackendLabel: extractionService.activeBackendLabel
            )
            performanceExportError = nil
        } catch {
            performanceExportError = "Export failed: \(error.localizedDescription)"
        }
    }
}
