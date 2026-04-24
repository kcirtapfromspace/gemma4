// EntityRow.swift
// Shared row renderer for conditions / labs / medications shown across the
// case detail + review screens. Clinician-friendly display on top, muted
// audit codes underneath. No raw JSON surfaced here.

import SwiftUI

struct ConditionRow: View {
    let condition: ExtractedCondition
    var showsReviewChip: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(condition.displayName)
                    .font(.body.weight(.medium))
                    .foregroundStyle(.primary)
                if showsReviewChip {
                    ReviewStateChip(state: condition.reviewState)
                }
                Spacer()
            }
            Text("\(condition.system) · \(condition.code)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(ClinIQTheme.auditMuted)
        }
        .padding(.vertical, 4)
    }
}

struct LabRow: View {
    let lab: ExtractedLab
    var showsReviewChip: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(lab.displayName)
                    .font(.body.weight(.medium))
                    .foregroundStyle(.primary)
                Spacer()
                Text(lab.resultSummary)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(resultColor)
            }
            HStack(spacing: 8) {
                Text("\(lab.system) · \(lab.code)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(ClinIQTheme.auditMuted)
                if showsReviewChip {
                    ReviewStateChip(state: lab.reviewState)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var resultColor: Color {
        let text = (lab.interpretation ?? "").lowercased()
        if text.contains("not detected") || text.contains("negative") {
            return Color(red: 0.13, green: 0.36, blue: 0.22)
        }
        if text.contains("detected") || text.contains("positive") {
            return Color(red: 0.66, green: 0.15, blue: 0.12)
        }
        return .primary
    }
}

struct MedicationRow: View {
    let medication: ExtractedMedication
    var showsReviewChip: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(medication.displayName)
                    .font(.body.weight(.medium))
                    .foregroundStyle(.primary)
                if showsReviewChip {
                    ReviewStateChip(state: medication.reviewState)
                }
                Spacer()
            }
            Text("\(medication.system) · \(medication.code)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(ClinIQTheme.auditMuted)
        }
        .padding(.vertical, 4)
    }
}

struct ReviewStateChip: View {
    let state: ReviewState

    var body: some View {
        switch state {
        case .needsReview:
            Label("Needs review", systemImage: "circle.dashed")
                .labelStyle(.titleAndIcon)
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 7).padding(.vertical, 2)
                .foregroundStyle(Color(red: 0.45, green: 0.30, blue: 0.00))
                .background(ClinIQTheme.statusPending.opacity(0.7), in: Capsule())
        case .confirmed:
            Label("Confirmed", systemImage: "checkmark.circle.fill")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 7).padding(.vertical, 2)
                .foregroundStyle(Color(red: 0.13, green: 0.36, blue: 0.22))
                .background(ClinIQTheme.statusSubmitted.opacity(0.75), in: Capsule())
        case .edited:
            Label("Edited", systemImage: "pencil.circle")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 7).padding(.vertical, 2)
                .foregroundStyle(Color(red: 0.10, green: 0.30, blue: 0.55))
                .background(ClinIQTheme.statusSyncing.opacity(0.7), in: Capsule())
        case .rejected:
            Label("Removed", systemImage: "minus.circle")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 7).padding(.vertical, 2)
                .foregroundStyle(Color(red: 0.56, green: 0.10, blue: 0.06))
                .background(ClinIQTheme.statusFailed.opacity(0.7), in: Capsule())
        }
    }
}
