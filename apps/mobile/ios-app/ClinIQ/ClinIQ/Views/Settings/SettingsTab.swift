// SettingsTab.swift
// Demo toggles and environment info. Intentionally narrow — this is a PoC,
// not a production settings screen.

import SwiftUI

struct SettingsTab: View {
    @EnvironmentObject private var monitor: NetworkMonitor
    @EnvironmentObject private var sync: SyncService
    @AppStorage("ClinIQ.MockSyncSucceeds") private var mockSucceeds: Bool = true
    @AppStorage("ClinIQ.UseLocalEndpoint") private var useLocal: Bool = false
    @AppStorage(InferenceBackend.appStorageKey) private var backendRaw: String = InferenceBackend.default.rawValue
    // Bridge to the shared ExtractionService so flipping the backend
    // picker invalidates the cached engine on the next extract call.
    @EnvironmentObject private var extractionService: ExtractionService

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
                } header: {
                    Text("Inference backend")
                } footer: {
                    Text("llama.cpp runs our fine-tuned Gemma 4 via the vendored GGUF. LiteRT-LM runs the stock base model from Google. Switching reloads the model on the next extraction (takes a few seconds on first use).")
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
}
