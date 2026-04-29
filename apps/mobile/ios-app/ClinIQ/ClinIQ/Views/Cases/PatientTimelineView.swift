// PatientTimelineView.swift
// Longitudinal timeline for one patient — list of cases ordered by date,
// each row showing a chip strip of axis-tagged additions / removals /
// unchanged from the previous case in the series.
//
// Mirrors CDC EZeCR's "what's new vs prior" CSV at the edge: every visit
// the clinician sees on this device contributes to the patient's
// longitudinal record. Tap a row to open the existing CaseDetailView for
// that case.

import SwiftUI
import SwiftData

struct PatientTimelineView: View {
    let patientIdentityHash: String
    /// Pretty title — usually the patient's full name. Optional because
    /// some entry points may have only the hash.
    let patientName: String?

    @Environment(\.modelContext) private var context

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if cases.isEmpty {
                    ContentUnavailableView {
                        Label("No history yet", systemImage: "clock.arrow.circlepath")
                    } description: {
                        Text("This is the first eCR for the patient on this device.")
                    }
                    .padding(.top, 40)
                } else {
                    summaryCard
                    ForEach(Array(cases.enumerated()), id: \.element.id) { idx, c in
                        TimelineRow(
                            clinicalCase: c,
                            priorCase: idx == 0 ? nil : cases[idx - 1],
                            isFirst: idx == 0,
                            isLast: idx == cases.count - 1
                        )
                    }
                }
            }
            .padding(16)
        }
        .background(ClinIQTheme.pageBackground)
        .navigationTitle(patientName ?? "Patient timeline")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var cases: [ClinicalCase] {
        PersistenceController.allCases(for: patientIdentityHash, in: context)
    }

    private var summaryCard: some View {
        let n = cases.count
        let span: String = {
            guard let first = cases.first?.createdAt,
                  let last = cases.last?.createdAt,
                  n >= 2 else { return "" }
            let days = Calendar(identifier: .gregorian)
                .dateComponents([.day], from: first, to: last).day ?? 0
            return days <= 0 ? "" : " across \(days) days"
        }()
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "person.text.rectangle")
                    .foregroundStyle(ClinIQTheme.accent)
                Text("\(n) eCR\(n == 1 ? "" : "s")\(span)")
                    .font(.subheadline.weight(.semibold))
                Spacer()
            }
            Text("On-device longitudinal record. Diff is computed locally — \(CaseDiffBuilder.snomedSystem.contains("snomed") ? "no patient data leaves the phone" : "edge-only").")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Row

struct TimelineRow: View {
    let clinicalCase: ClinicalCase
    let priorCase: ClinicalCase?
    let isFirst: Bool
    let isLast: Bool

    var body: some View {
        NavigationLink {
            CaseDetailView(clinicalCase: clinicalCase)
        } label: {
            HStack(alignment: .top, spacing: 12) {
                spineColumn
                contentColumn
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.plain)
    }

    private var spineColumn: some View {
        VStack(spacing: 0) {
            // Top half of the spine
            Rectangle()
                .fill(isFirst ? Color.clear : ClinIQTheme.accent.opacity(0.4))
                .frame(width: 2, height: 12)
            Circle()
                .fill(ClinIQTheme.accent)
                .frame(width: 12, height: 12)
                .overlay(
                    Circle()
                        .stroke(Color.white, lineWidth: 2)
                )
            // Bottom half of the spine
            Rectangle()
                .fill(isLast ? Color.clear : ClinIQTheme.accent.opacity(0.4))
                .frame(width: 2)
                .frame(maxHeight: .infinity)
        }
        .frame(width: 14)
    }

    private var contentColumn: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(clinicalCase.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.callout.weight(.semibold))
                Spacer()
                if isFirst {
                    Text("INITIAL eCR")
                        .font(.caption2.weight(.bold))
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(ClinIQTheme.accent.opacity(0.18), in: Capsule())
                        .foregroundStyle(ClinIQTheme.accent)
                } else {
                    StatusBadge(status: clinicalCase.status)
                }
            }
            Text(clinicalCase.primaryConditionDisplay)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if let prior = priorCase {
                let diff = CaseDiffBuilder.diff(prior: prior, current: clinicalCase)
                ChipStrip(diff: diff)
                Text(diff.summary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Text("First eCR for this patient on this device.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 12))
        .padding(.bottom, isLast ? 0 : 6)
    }
}

// MARK: - Chips

/// Compact green / red / gray chip strip — one per diff entry. Capped to
/// keep the row readable; overflow shows a "+N more" pill.
struct ChipStrip: View {
    let diff: CaseDiff
    let maxChips: Int = 6

    var body: some View {
        let entries: [(DiffEntry, ChipKind)] =
            diff.added.map { ($0, .added) }
            + diff.removed.map { ($0, .removed) }
            + diff.unchanged.map { ($0, .unchanged) }
        let visible = Array(entries.prefix(maxChips))
        let overflow = entries.count - visible.count
        return FlexHStack(spacing: 6) {
            ForEach(0..<visible.count, id: \.self) { idx in
                let (entry, kind) = visible[idx]
                DiffChip(entry: entry, kind: kind)
            }
            if overflow > 0 {
                Text("+\(overflow) more")
                    .font(.caption2.weight(.medium))
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Color(.tertiarySystemGroupedBackground), in: Capsule())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

enum ChipKind {
    case added, removed, unchanged

    var color: Color {
        switch self {
        case .added: return Color(red: 0.13, green: 0.36, blue: 0.22)
        case .removed: return Color(red: 0.66, green: 0.15, blue: 0.12)
        case .unchanged: return Color(.tertiaryLabel)
        }
    }

    var background: Color {
        switch self {
        case .added: return Color(red: 0.72, green: 0.87, blue: 0.73)
        case .removed: return Color(red: 0.98, green: 0.73, blue: 0.71)
        case .unchanged: return Color(.tertiarySystemGroupedBackground)
        }
    }

    var glyph: String {
        switch self {
        case .added: return "plus"
        case .removed: return "minus"
        case .unchanged: return "equal"
        }
    }
}

struct DiffChip: View {
    let entry: DiffEntry
    let kind: ChipKind

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: kind.glyph)
                .font(.caption2.weight(.bold))
            Text(shortLabel)
                .font(.caption2.weight(.medium))
                .lineLimit(1)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(kind.background, in: Capsule())
        .foregroundStyle(kind.color)
    }

    /// Trim long display strings so the chip stays compact. Falls back to
    /// the code for very long names.
    private var shortLabel: String {
        let d = entry.display
        if d.count <= 18 { return d }
        return String(d.prefix(15)) + "..."
    }
}

// MARK: - FlexHStack
//
// A minimal flow-layout for the chip strip — wraps to a new row when the
// available width is exceeded. iOS 16+ has Layout but the project targets
// iOS 17 already so we can use Layout safely.

struct FlexHStack: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var rowWidth: CGFloat = 0
        var rowHeight: CGFloat = 0
        var totalHeight: CGFloat = 0
        var totalWidth: CGFloat = 0
        for sv in subviews {
            let size = sv.sizeThatFits(.unspecified)
            if rowWidth + size.width > maxWidth, rowWidth > 0 {
                totalHeight += rowHeight + spacing
                totalWidth = max(totalWidth, rowWidth - spacing)
                rowWidth = 0
                rowHeight = 0
            }
            rowWidth += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        totalHeight += rowHeight
        totalWidth = max(totalWidth, rowWidth - spacing)
        return CGSize(width: max(0, totalWidth), height: totalHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let maxWidth = bounds.width
        var x: CGFloat = bounds.minX
        var y: CGFloat = bounds.minY
        var rowHeight: CGFloat = 0
        for sv in subviews {
            let size = sv.sizeThatFits(.unspecified)
            if x - bounds.minX + size.width > maxWidth, x > bounds.minX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            sv.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
