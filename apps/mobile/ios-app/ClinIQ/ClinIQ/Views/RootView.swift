// RootView.swift
// Tab-based shell: Cases | Outbox | History | Settings.
// An offline banner is inset at the top of the window.

import SwiftUI
import SwiftData

struct RootView: View {
    @EnvironmentObject private var monitor: NetworkMonitor
    @EnvironmentObject private var sync: SyncService
    @State private var selectedTab: AppTab = .cases

    enum AppTab: Hashable {
        case cases, outbox, history, settings
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            CasesTab()
                .tabItem { Label("Cases", systemImage: "folder") }
                .tag(AppTab.cases)

            OutboxTab()
                .tabItem { Label("Outbox", systemImage: "tray.and.arrow.up") }
                .tag(AppTab.outbox)
                .badge(OutboxBadge.count)

            HistoryTab()
                .tabItem { Label("History", systemImage: "clock") }
                .tag(AppTab.history)

            SettingsTab()
                .tabItem { Label("Settings", systemImage: "gearshape") }
                .tag(AppTab.settings)
        }
        .tint(ClinIQTheme.accent)
        .safeAreaInset(edge: .top, spacing: 0) {
            OfflineBanner()
        }
        .onAppear {
            // Demo-mode env vars so the screenshot harness can jump directly
            // to a given tab without chasing tap coordinates.
            let env = ProcessInfo.processInfo.environment
            switch env["CLINIQ_TAB"] {
            case "cases": selectedTab = .cases
            case "outbox": selectedTab = .outbox
            case "history": selectedTab = .history
            case "settings": selectedTab = .settings
            default: break
            }
        }
    }
}

/// Small helper that lets the Outbox tab show a red badge count. Backed by a
/// simple observable singleton refreshed whenever the Outbox view fetches.
enum OutboxBadge {
    static var count: Int {
        OutboxCounter.shared.pending
    }
}

final class OutboxCounter: ObservableObject {
    static let shared = OutboxCounter()
    @Published var pending: Int = 0
}
