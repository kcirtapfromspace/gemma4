// InferenceStatusBar.swift
// Persistent pill at the bottom of the app window (mirroring the
// `OfflineBanner` at the top) showing what the on-device LLM is doing,
// live. Collapsed: one line. Tap to expand into a "nerd-stats" panel
// with TTFT, tokens in/out, avg/peak tok/s, resident memory, threads,
// and a simple inline sparkline — styled like the debug overlay that
// tvOS / streaming apps show during playback.

import SwiftUI

struct InferenceStatusBar: View {
    @ObservedObject private var metrics = InferenceMetrics.shared
    @State private var expanded = false

    var body: some View {
        VStack(spacing: 0) {
            if expanded {
                statsPanel
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            pill
        }
        .animation(.easeInOut(duration: 0.2), value: expanded)
        .animation(.easeInOut(duration: 0.2), value: metrics.isActive)
    }

    // MARK: Collapsed pill

    private var pill: some View {
        HStack(spacing: 10) {
            indicator

            Text(metrics.compactLabel)
                .font(.footnote.weight(.medium))
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.85)

            Spacer(minLength: 4)

            if metrics.isActive, metrics.maxTokens > 0 {
                ProgressView(value: Double(metrics.outputTokens),
                             total: Double(metrics.maxTokens))
                    .progressViewStyle(.linear)
                    .frame(width: 60)
                    .tint(ClinIQTheme.accent)
            }

            Image(systemName: expanded ? "chevron.down" : "chevron.up")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .background(backgroundColor)
        .foregroundStyle(foregroundColor)
        .contentShape(Rectangle())
        .onTapGesture { expanded.toggle() }
    }

    // MARK: Indicator dot / spinner

    @ViewBuilder
    private var indicator: some View {
        if metrics.isActive {
            ZStack {
                Circle()
                    .fill(ClinIQTheme.accent.opacity(0.2))
                    .frame(width: 14, height: 14)
                Circle()
                    .fill(ClinIQTheme.accent)
                    .frame(width: 8, height: 8)
                    .scaleEffect(metrics.phase == .decoding ? 1.0 : 0.6)
                    .opacity(metrics.phase == .decoding ? 1.0 : 0.7)
                    .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true),
                               value: metrics.phase)
            }
        } else if metrics.phase == .error {
            Circle()
                .fill(ClinIQTheme.statusFailed)
                .frame(width: 10, height: 10)
        } else {
            Circle()
                .stroke(Color.secondary.opacity(0.6), lineWidth: 1)
                .frame(width: 10, height: 10)
        }
    }

    // MARK: Expanded stats panel

    private var statsPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Sparkline
            if metrics.tpsHistory.count > 1 {
                TokensPerSecondSparkline(values: metrics.tpsHistory)
                    .frame(height: 28)
                    .padding(.bottom, 2)
            }

            // 2-column grid of stats
            VStack(alignment: .leading, spacing: 5) {
                HStack(alignment: .top, spacing: 14) {
                    statCell(label: "Backend", value: metrics.backend)
                    statCell(label: "Model", value: metrics.modelName)
                }
                HStack(alignment: .top, spacing: 14) {
                    statCell(label: "Phase", value: metrics.phase.rawValue)
                    statCell(label: "TTFT",
                             value: metrics.firstTokenLatencySeconds
                                .map { String(format: "%.2f s", $0) } ?? "—")
                }
                HStack(alignment: .top, spacing: 14) {
                    statCell(label: "Prompt",
                             value: "≈\(metrics.promptTokensApprox) tok · \(metrics.promptChars) ch")
                    statCell(label: "Output",
                             value: "\(metrics.outputTokens) / \(metrics.maxTokens) tok")
                }
                HStack(alignment: .top, spacing: 14) {
                    statCell(label: "tok/s (now)",
                             value: String(format: "%.2f", metrics.instantTokensPerSecond))
                    statCell(label: "tok/s (avg)",
                             value: String(format: "%.2f", metrics.avgTokensPerSecond))
                }
                HStack(alignment: .top, spacing: 14) {
                    statCell(label: "tok/s (peak)",
                             value: String(format: "%.2f", metrics.peakTokensPerSecond))
                    statCell(label: "Elapsed",
                             value: String(format: "%.1f s", metrics.elapsedSeconds))
                }
                HStack(alignment: .top, spacing: 14) {
                    statCell(label: "Resident",
                             value: String(format: "%.0f MB", metrics.residentMemoryMB))
                    statCell(label: "Threads",
                             value: "\(metrics.threadCount)")
                }
                if let err = metrics.lastError {
                    Text(err)
                        .font(.caption2.monospaced())
                        .foregroundStyle(ClinIQTheme.statusFailed)
                        .lineLimit(2)
                        .padding(.top, 2)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color(.secondarySystemBackground))
    }

    private func statCell(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label.uppercased())
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .tracking(0.5)
            Text(value)
                .font(.caption.monospaced())
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Colors

    private var backgroundColor: Color {
        switch metrics.phase {
        case .idle:       return Color(.tertiarySystemBackground)
        case .loading:    return ClinIQTheme.statusSyncing.opacity(0.6)
        case .prefilling: return ClinIQTheme.statusSyncing.opacity(0.7)
        case .decoding:   return ClinIQTheme.accent.opacity(0.18)
        case .finalizing: return ClinIQTheme.statusSyncing.opacity(0.5)
        case .error:      return ClinIQTheme.statusFailed.opacity(0.6)
        }
    }

    private var foregroundColor: Color {
        switch metrics.phase {
        case .error: return Color(red: 0.55, green: 0.16, blue: 0.14)
        default:     return Color.primary
        }
    }
}

// MARK: - Sparkline

private struct TokensPerSecondSparkline: View {
    let values: [Double]

    var body: some View {
        GeometryReader { geo in
            let maxV = max(values.max() ?? 1, 0.5)
            let minV = 0.0
            let range = max(maxV - minV, 0.5)
            let step = geo.size.width / CGFloat(max(values.count - 1, 1))

            Path { path in
                for (i, v) in values.enumerated() {
                    let x = CGFloat(i) * step
                    let norm = (v - minV) / range
                    let y = geo.size.height * (1.0 - CGFloat(norm))
                    if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
                    else      { path.addLine(to: CGPoint(x: x, y: y)) }
                }
            }
            .stroke(ClinIQTheme.accent, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))

            // Baseline
            Path { p in
                p.move(to: CGPoint(x: 0, y: geo.size.height - 0.5))
                p.addLine(to: CGPoint(x: geo.size.width, y: geo.size.height - 0.5))
            }
            .stroke(Color.secondary.opacity(0.25), lineWidth: 0.5)
        }
        .accessibilityHidden(true)
    }
}
