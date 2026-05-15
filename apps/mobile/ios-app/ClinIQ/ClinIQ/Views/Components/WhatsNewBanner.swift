// WhatsNewBanner.swift
// "X new findings since last eCR (N days ago)" banner shown at the top of
// the Review screen when the current case has at least one prior for the
// same patientIdentityHash. Tapping the banner expands it to show the
// added / removed / unchanged sections, collapsible.
//
// Hidden entirely on first eCR for a new patient (no prior).

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct WhatsNewBanner: View {
    let diff: CaseDiff
    @State private var expanded: Bool = false
    @State private var copied: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    expanded.toggle()
                }
            } label: {
                summaryRow
            }
            .buttonStyle(.plain)

            if expanded {
                Divider().padding(.vertical, 6)
                expandedBody
                exportButton
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(banding, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(ClinIQTheme.accent.opacity(0.35), lineWidth: 1)
        )
    }

    private var summaryRow: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: "sparkle")
                .foregroundStyle(ClinIQTheme.accent)
            VStack(alignment: .leading, spacing: 2) {
                Text(diff.summary)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.primary)
                Text("Tap to see what changed")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: expanded ? "chevron.up" : "chevron.down")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
        }
    }

    private var expandedBody: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !diff.added.isEmpty {
                section(title: "Added", entries: diff.added, kind: .added)
            }
            if !diff.removed.isEmpty {
                section(title: "Resolved / discontinued", entries: diff.removed, kind: .removed)
            }
            if !diff.unchanged.isEmpty {
                section(title: "Carried forward", entries: diff.unchanged, kind: .unchanged)
            }
        }
    }

    private var exportButton: some View {
        Button {
            copyExport()
        } label: {
            Label(copied ? "Copied ClinIQ diff JSON" : "Copy ClinIQ diff JSON",
                  systemImage: copied ? "checkmark.circle.fill" : "doc.on.doc")
                .font(.caption.weight(.semibold))
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .padding(.top, 8)
    }

    @ViewBuilder
    private func section(title: String, entries: [DiffEntry], kind: ChipKind) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            FlexHStack(spacing: 6) {
                ForEach(entries) { entry in
                    DiffChip(entry: entry, kind: kind)
                }
            }
        }
    }

    /// Subtle accent banding so the banner reads as "informational" rather
    /// than alarmed (red) or success (green) — matches the overall ClinIQ
    /// teal accent.
    private var banding: Color {
        ClinIQTheme.accent.opacity(0.10)
    }

    private func copyExport() {
        let text = CaseDiffExport.jsonString(from: diff)
        #if canImport(UIKit)
        UIPasteboard.general.string = text
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            copied = false
        }
        #else
        copied = false
        #endif
    }
}
