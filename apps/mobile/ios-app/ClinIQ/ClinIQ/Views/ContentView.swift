// ContentView.swift
// Single-screen SwiftUI view for ClinIQ eICR extraction.

import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ExtractionViewModel()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Paste an eICR summary; tap Extract to receive minified JSON.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)

                // Input editor
                GroupBox(label: Label("Input eICR", systemImage: "doc.text")) {
                    TextEditor(text: $vm.inputEicr)
                        .font(.system(.footnote, design: .monospaced))
                        .frame(minHeight: 220, maxHeight: 300)
                        .scrollContentBackground(.hidden)
                        .overlay(
                            RoundedRectangle(cornerRadius: 4)
                                .stroke(Color.secondary.opacity(0.25), lineWidth: 0.5)
                        )
                }
                .padding(.horizontal)

                // Action row
                HStack(spacing: 12) {
                    Button(action: { Task { await vm.extract() } }) {
                        if vm.isExtracting {
                            ProgressView()
                                .controlSize(.small)
                                .padding(.trailing, 4)
                            Text("Extracting\u{2026}")
                        } else {
                            Image(systemName: "wand.and.stars")
                            Text("Extract")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.isExtracting || vm.inputEicr.isEmpty)

                    if vm.isExtracting {
                        Text(String(format: "%.1f tok/s", vm.tokensPerSecond))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    } else if vm.lastTokensGenerated > 0 {
                        Text("\(vm.lastTokensGenerated) tok / \(String(format: "%.1fs", vm.lastElapsedSeconds))")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Menu {
                        ForEach(TestCase.bundled) { tc in
                            Button(tc.caseId) { vm.loadCase(tc) }
                        }
                    } label: {
                        Label("Sample", systemImage: "tray.and.arrow.down")
                    }
                    .font(.caption)
                }
                .padding(.horizontal)

                // Output editor
                GroupBox(label: Label("JSON Output", systemImage: "curlybraces")) {
                    TextEditor(text: .constant(vm.output))
                        .font(.system(.footnote, design: .monospaced))
                        .frame(minHeight: 200, maxHeight: .infinity)
                        .scrollContentBackground(.hidden)
                        .overlay(
                            RoundedRectangle(cornerRadius: 4)
                                .stroke(
                                    vm.errorMessage != nil
                                        ? Color.red.opacity(0.5)
                                        : Color.secondary.opacity(0.25),
                                    lineWidth: 0.5)
                        )
                }
                .padding(.horizontal)

                if let err = vm.errorMessage {
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }

                Spacer(minLength: 0)
            }
            .padding(.vertical, 8)
            .navigationTitle("ClinIQ eICR Extractor")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                // When the simulator launches us with CLINIQ_AUTO_EXTRACT=1,
                // kick off extraction automatically so screenshot-driving
                // tools (xcrun simctl io ... screenshot) can observe output
                // without a UI tap.
                if ProcessInfo.processInfo.environment["CLINIQ_AUTO_EXTRACT"] == "1" {
                    Task { await vm.extract() }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
