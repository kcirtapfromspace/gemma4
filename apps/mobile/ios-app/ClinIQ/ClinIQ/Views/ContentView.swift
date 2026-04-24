// ContentView.swift
// LEGACY developer testbench (C10/C12). Kept in the source tree so prior
// diagnostic screens are preserved under version control, but NOT wired
// into the app shell — the current app root is `RootView.swift`.
//
// Leaving this file in the build keeps ExtractionViewModel (and the
// headless env-var auto-extract path in BUILD.md) accessible for
// validation runs. See LEGACY.md for the transition notes.

import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ExtractionViewModel()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Legacy JSON-dumper testbench (developer use).\nThe clinician PoC shell lives in RootView.swift.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)

                GroupBox(label: Label("Narrative", systemImage: "doc.text")) {
                    TextEditor(text: $vm.inputEicr)
                        .font(.system(.footnote, design: .monospaced))
                        .frame(minHeight: 180, maxHeight: 240)
                        .scrollContentBackground(.hidden)
                }
                .padding(.horizontal)

                HStack(spacing: 12) {
                    Button(action: { Task { await vm.extract() } }) {
                        if vm.isExtracting {
                            ProgressView().controlSize(.small)
                            Text("Extracting...")
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
                    }
                    Spacer()
                }
                .padding(.horizontal)

                GroupBox(label: Label("JSON Output", systemImage: "curlybraces")) {
                    TextEditor(text: .constant(vm.output))
                        .font(.system(.footnote, design: .monospaced))
                        .frame(minHeight: 150, maxHeight: .infinity)
                        .scrollContentBackground(.hidden)
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
            .navigationTitle("ClinIQ testbench")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                let env = ProcessInfo.processInfo.environment
                if let caseID = env["CLINIQ_CASE"],
                   let tc = TestCase.bundled.first(where: { $0.caseId == caseID })
                {
                    vm.loadCase(tc)
                }
                if env["CLINIQ_AUTO_EXTRACT"] == "1" {
                    Task { await vm.extract() }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
