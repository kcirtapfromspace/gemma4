// OfflineBanner.swift
// Persistent banner inset at the top of the app when NWPathMonitor reports
// we can't reach the network. Also renders a subtle "Online" pill when we
// return so the demo-watcher sees the transition.

import SwiftUI

struct OfflineBanner: View {
    @EnvironmentObject private var monitor: NetworkMonitor

    var body: some View {
        Group {
            if monitor.isOnline {
                HStack(spacing: 8) {
                    Image(systemName: "wifi")
                    Text("Online — \(monitor.interfaceDescription)")
                        .font(.footnote.weight(.medium))
                    Spacer()
                }
                .foregroundStyle(Color(red: 0.13, green: 0.36, blue: 0.22))
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
                .background(ClinIQTheme.statusSubmitted.opacity(0.6))
            } else {
                HStack(spacing: 8) {
                    Image(systemName: "wifi.slash")
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Offline — cases held on device")
                            .font(.footnote.weight(.semibold))
                        Text("Outbox will auto-sync when network returns.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .foregroundStyle(Color(red: 0.45, green: 0.30, blue: 0.00))
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(ClinIQTheme.statusPending.opacity(0.9))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: monitor.isOnline)
    }
}
