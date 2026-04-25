// ProvenanceLegend.swift
// One-line legend explaining the four provenance tier colors that appear on
// extracted entity rows. Default state is a single chip-row so the colors
// are recognizable in a glance; tap to expand into one-sentence
// descriptions per tier so a first-time viewer (judge) knows what they
// mean within ~5 seconds. Mirrors the tier color mapping used by
// ProvenanceBadge — kept in sync by hand because both reference the same
// CodeProvenance.Tier cases.

import SwiftUI

struct ProvenanceLegend: View {
    @State private var expanded: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("How each code was extracted")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 4)
                tierChip("INLINE", color: tierColor(.inline))
                tierChip("CDA", color: tierColor(.cda))
                tierChip("LOOKUP", color: tierColor(.lookup))
                tierChip("RAG", color: tierColor(.rag))
                Image(systemName: expanded ? "chevron.up" : "chevron.down")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            if expanded {
                VStack(alignment: .leading, spacing: 6) {
                    legendRow(.inline, "INLINE",
                              "Code parenthesized in the narrative — explicit clinician assertion, ~99% conf.")
                    legendRow(.cda, "CDA",
                              "Code from a structured CDA / eICR XML element — same confidence as inline.")
                    legendRow(.lookup, "LOOKUP",
                              "Display-name match against the curated reportable-conditions table, ~85% conf.")
                    legendRow(.rag, "RAG",
                              "Retrieved against on-device CDC NNDSS / WHO IDSR — open semantic match, variable conf.")
                }
                .padding(.top, 2)
            }
        }
        .padding(10)
        .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 10))
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.2)) { expanded.toggle() }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Provenance legend. INLINE green, CDA teal, LOOKUP amber, RAG purple. Tap to expand.")
    }

    private func tierChip(_ label: String, color: Color) -> some View {
        Text(label)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(color.opacity(0.16), in: Capsule())
            .foregroundStyle(color)
    }

    private func legendRow(_ tier: CodeProvenance.Tier,
                           _ name: String,
                           _ description: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            tierChip(name, color: tierColor(tier))
                .frame(width: 72, alignment: .leading)
            Text(description)
                .font(.caption2)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    /// Mirror of ProvenanceBadge.tierColor — kept identical so the legend
    /// chip and the row chip read as the same color across the screen.
    private func tierColor(_ tier: CodeProvenance.Tier) -> Color {
        switch tier {
        case .inline: return Color(red: 0.13, green: 0.50, blue: 0.27)
        case .cda:    return Color(red: 0.10, green: 0.46, blue: 0.55)
        case .lookup: return Color(red: 0.74, green: 0.51, blue: 0.10)
        case .rag:    return Color(red: 0.50, green: 0.30, blue: 0.65)
        }
    }
}
