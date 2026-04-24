// StatusBadge.swift
// Capsule badge for a `CaseStatus`. Used in list rows, detail header, and
// the Outbox tab.

import SwiftUI

struct StatusBadge: View {
    let status: CaseStatus

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.caption2)
            Text(label)
                .font(.caption.weight(.semibold))
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, 9)
        .padding(.vertical, 4)
        .background(background, in: Capsule())
    }

    private var label: String {
        switch status {
        case .draft: return "Draft"
        case .pending: return "Queued"
        case .syncing: return "Syncing"
        case .submitted: return "Submitted"
        case .failed: return "Failed"
        }
    }

    private var icon: String {
        switch status {
        case .draft: return "pencil.circle"
        case .pending: return "tray.and.arrow.up"
        case .syncing: return "arrow.triangle.2.circlepath"
        case .submitted: return "checkmark.seal"
        case .failed: return "exclamationmark.triangle"
        }
    }

    private var background: Color {
        switch status {
        case .draft: return ClinIQTheme.statusDraft
        case .pending: return ClinIQTheme.statusPending
        case .syncing: return ClinIQTheme.statusSyncing
        case .submitted: return ClinIQTheme.statusSubmitted
        case .failed: return ClinIQTheme.statusFailed
        }
    }

    private var foreground: Color {
        switch status {
        case .submitted: return Color(red: 0.13, green: 0.36, blue: 0.22)
        case .failed: return Color(red: 0.56, green: 0.10, blue: 0.06)
        case .pending: return Color(red: 0.45, green: 0.30, blue: 0.00)
        case .syncing: return Color(red: 0.10, green: 0.30, blue: 0.55)
        case .draft: return Color(red: 0.25, green: 0.28, blue: 0.32)
        }
    }
}
