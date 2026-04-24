// SettingsTab.swift
// Demo toggles and environment info. Intentionally narrow — this is a PoC,
// not a production settings screen.

import SwiftUI

struct SettingsTab: View {
    @EnvironmentObject private var monitor: NetworkMonitor
    @EnvironmentObject private var sync: SyncService
    @AppStorage("ClinIQ.MockSyncSucceeds") private var mockSucceeds: Bool = true
    @AppStorage("ClinIQ.UseLocalEndpoint") private var useLocal: Bool = false

    var body: some View {
        NavigationStack {
            Form {
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
        if LlamaCppInferenceEngine.resolveModelPath() != nil {
            return "llama.cpp · GGUF found"
        }
        return "Rule-based fallback"
    }
}
