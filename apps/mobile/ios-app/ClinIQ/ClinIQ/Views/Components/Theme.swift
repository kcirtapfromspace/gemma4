// Theme.swift
// Central palette + spacing tokens. Keeping this separate from the per-view
// modifiers so later UX passes can iterate in one place.

import SwiftUI

enum ClinIQTheme {
    // A teal/green accent reads as "health", stays legible in light+dark,
    // and avoids the overused iOS blue so the app has a recognisable hue.
    static let accent = Color(red: 0.10, green: 0.58, blue: 0.60)

    // Status chip tints. Kept muted and accessible; text on top is always
    // darkened per-chip in the component.
    static let statusDraft = Color(red: 0.78, green: 0.80, blue: 0.84)
    static let statusPending = Color(red: 1.00, green: 0.82, blue: 0.45)
    static let statusSyncing = Color(red: 0.70, green: 0.85, blue: 1.00)
    static let statusSubmitted = Color(red: 0.72, green: 0.87, blue: 0.73)
    static let statusFailed = Color(red: 0.98, green: 0.73, blue: 0.71)

    static let cardBackground = Color(.secondarySystemGroupedBackground)
    static let pageBackground = Color(.systemGroupedBackground)

    static let auditMuted = Color.secondary.opacity(0.78)
}
