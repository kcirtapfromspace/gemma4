// ProvenanceBadge.swift
// Renders the per-code provenance produced by EicrPreparser /
// AgentRunner: which tier matched, the confidence score, and a
// tap-revealed expander showing the literal source span and (for RAG
// hits) a tap-through URL to the authoritative source.
//
// This is the "I extracted X because line N says Y" affordance that
// MedGemma's Tracer winner used to surface clinical-decision rationale
// in its review UI. Cheap, narratively powerful, judges can see the
// model's work.

import SwiftUI

struct ProvenanceBadge: View {
    let provenance: CodeProvenance
    @State private var expanded: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                tierChip
                Text(confidenceLabel)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(ClinIQTheme.auditMuted)
                Spacer()
                Button(action: { expanded.toggle() }) {
                    Image(systemName: expanded ? "chevron.up" : "info.circle")
                        .font(.caption)
                        .foregroundStyle(ClinIQTheme.auditMuted)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(expanded ? "Hide provenance details" : "Show provenance details")
            }
            if expanded {
                expandedDetail
            }
        }
    }

    private var tierChip: some View {
        Text(tierLabel)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tierColor.opacity(0.16), in: Capsule())
            .foregroundStyle(tierColor)
    }

    private var expandedDetail: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let alias = provenance.alias, alias != provenance.sourceText {
                detailRow(label: "alias", value: alias)
            }
            detailRow(label: "source", value: shortenSource(provenance.sourceText))
            if let url = provenance.sourceURL, !url.isEmpty,
               let parsed = URL(string: url) {
                Link(destination: parsed) {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.up.right.square")
                        Text(provenance.tier == .rag ? "Authoritative source" : "Reference")
                    }
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(tierColor)
                }
            }
        }
        .padding(8)
        .background(ClinIQTheme.auditMuted.opacity(0.06), in: RoundedRectangle(cornerRadius: 6))
    }

    private func detailRow(label: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text(label)
                .font(.caption2.weight(.medium))
                .foregroundStyle(ClinIQTheme.auditMuted)
                .frame(width: 50, alignment: .leading)
            Text(value)
                .font(.caption2.monospaced())
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var tierLabel: String {
        switch provenance.tier {
        case .inline: return "INLINE"
        case .cda:    return "CDA"
        case .lookup: return "LOOKUP"
        case .rag:    return "RAG"
        }
    }

    private var tierColor: Color {
        // Color encodes confidence-by-construction: parenthesized inline
        // codes and CDA XML are explicit assertions (green/teal), lookup
        // is a curated displayName match (amber), RAG is open-search
        // retrieval over CDC NNDSS / WHO IDSR (purple).
        switch provenance.tier {
        case .inline: return Color(red: 0.13, green: 0.50, blue: 0.27)
        case .cda:    return Color(red: 0.10, green: 0.46, blue: 0.55)
        case .lookup: return Color(red: 0.74, green: 0.51, blue: 0.10)
        case .rag:    return Color(red: 0.50, green: 0.30, blue: 0.65)
        }
    }

    private var confidenceLabel: String {
        let pct = Int((provenance.confidence * 100).rounded())
        return "conf \(pct)%"
    }

    /// Trim very long source spans (full CDA tags can be 100+ chars).
    private func shortenSource(_ s: String) -> String {
        if s.count <= 80 { return s }
        let prefix = s.prefix(36)
        let suffix = s.suffix(36)
        return "\(prefix)…\(suffix)"
    }
}
