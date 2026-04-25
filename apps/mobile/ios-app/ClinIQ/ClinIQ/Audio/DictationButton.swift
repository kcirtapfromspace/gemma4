// DictationButton.swift
// SwiftUI control that wraps SpeechDictationService for the NewCaseView
// narrative editor. Tap to start dictation, tap again to stop. Live partial
// transcript flows directly into the bound narrative; final transcript
// commits on stop.
//
// Visual: a primary mic button + (when recording) an inline RMS-level
// waveform strip + the live partial transcript. The button color flips
// red while recording so the demo video reads it from across the room.

import SwiftUI

struct DictationButton: View {
    @Binding var narrative: String
    @StateObject private var dictation = SpeechDictationService()
    @State private var startSnapshot: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Button(action: toggle) {
                    Image(systemName: dictation.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(dictation.isRecording
                                         ? Color(red: 0.85, green: 0.20, blue: 0.20)
                                         : Color(red: 0.16, green: 0.42, blue: 0.62))
                }
                .accessibilityLabel(dictation.isRecording ? "Stop dictation" : "Start dictation")
                .disabled(!dictation.isSupported)

                if dictation.isRecording {
                    LevelMeter(level: dictation.audioLevel)
                        .frame(height: 18)
                        .frame(maxWidth: .infinity)
                    Text(timestamp)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                } else if !dictation.isSupported {
                    Text("On-device speech not available on this device")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Tap to dictate the encounter")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if dictation.isRecording, !dictation.liveTranscript.isEmpty {
                Text(dictation.liveTranscript)
                    .font(.callout.italic())
                    .foregroundStyle(.primary.opacity(0.75))
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
            }

            if let err = dictation.errorMessage {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(Color(red: 0.85, green: 0.20, blue: 0.20))
            }
        }
        .onChange(of: dictation.liveTranscript) { _, newValue in
            // Append the live transcript to whatever the user had typed
            // before they tapped the mic. Don't clobber typed input.
            if dictation.isRecording {
                narrative = stitched(prefix: startSnapshot, addition: newValue)
            }
        }
        .task {
            _ = await dictation.requestAuthorization()
        }
    }

    private func toggle() {
        if dictation.isRecording {
            Task { _ = await dictation.stopRecording() }
        } else {
            startSnapshot = narrative
            Task { try? await dictation.startRecording() }
        }
    }

    private var timestamp: String {
        // Simple seconds counter; we don't yet stream a duration through
        // the service so this is a placeholder once we add it.
        return ""
    }

    private func stitched(prefix: String, addition: String) -> String {
        let trimmedPrefix = prefix.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedAddition = addition.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedPrefix.isEmpty { return trimmedAddition }
        if trimmedAddition.isEmpty { return prefix }
        let needsNewline = !prefix.hasSuffix("\n") && !prefix.hasSuffix(" ")
        return prefix + (needsNewline ? "\n" : "") + trimmedAddition
    }
}

/// Tiny RMS-level meter — animated capsule whose width tracks the live
/// audio energy. Uses the system green→red gradient to make the demo
/// reading visceral.
private struct LevelMeter: View {
    var level: Float
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.gray.opacity(0.18))
                Capsule()
                    .fill(LinearGradient(
                        colors: [Color.green, Color.yellow, Color.red],
                        startPoint: .leading, endPoint: .trailing
                    ))
                    .frame(width: max(4, geo.size.width * CGFloat(level)))
                    .animation(.linear(duration: 0.05), value: level)
            }
        }
    }
}
