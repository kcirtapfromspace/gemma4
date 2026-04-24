// ClinIQApp.swift
// ClinIQ — offline clinician field case-reporting PoC.
//
// Team C13 — 2026-04-23.
//
// Wires the SwiftData container, NetworkMonitor, and SyncService into the
// SwiftUI environment so every screen can observe network state and access
// the ObjectContainer without constructing its own. ContentView is kept in
// the project as the legacy developer testbench — see LEGACY.md.

import SwiftUI
import SwiftData

@main
struct ClinIQApp: App {
    @StateObject private var monitor = NetworkMonitor()
    @StateObject private var sync = SyncService()
    private let container: ModelContainer

    @MainActor
    init() {
        self.container = PersistenceController.makeContainer()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(monitor)
                .environmentObject(sync)
                .modelContainer(container)
                .task {
                    monitor.start()
                    sync.configure(container: container, monitor: monitor)
                    // On cold launch, honour demo env vars so the
                    // screenshot harness can put the app in a particular
                    // state without touching the UI.
                    let env = ProcessInfo.processInfo.environment
                    if env["CLINIQ_SIMULATE_OFFLINE"] == "1" {
                        monitor.simulateOffline = true
                    }
                }
        }
    }
}
