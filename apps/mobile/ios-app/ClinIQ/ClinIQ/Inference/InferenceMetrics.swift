// InferenceMetrics.swift
// Live telemetry for the on-device LLM. Feeds the InferenceStatusBar
// overlay so a clinician (or a judge) can see exactly what the model is
// doing — tokens/sec, elapsed, TTFT, resident memory, backend.
//
// Single `@MainActor` observable shared by the whole app. The owner of a
// run (`ExtractionService`) calls `begin(...)`, `record(chunk:)` on each
// streamed token chunk, and `end(error:)` when the stream finishes.
//
// Lightweight on purpose: no Combine pipelines, no sampling threads.
// `@Published` updates from the main actor drive SwiftUI directly.

import Foundation
import SwiftUI
import Darwin

@MainActor
final class InferenceMetrics: ObservableObject {
    static let shared = InferenceMetrics()

    enum Phase: String {
        case idle = "Ready"
        case loading = "Loading model"
        case prefilling = "Prefilling"
        case decoding = "Decoding"
        case finalizing = "Finalizing"
        case error = "Error"
    }

    // Published state -------------------------------------------------

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var backend: String = "—"
    @Published private(set) var modelName: String = "—"
    @Published private(set) var promptChars: Int = 0
    @Published private(set) var promptTokensApprox: Int = 0
    @Published private(set) var outputTokens: Int = 0
    @Published private(set) var maxTokens: Int = 0
    @Published private(set) var elapsedSeconds: Double = 0
    @Published private(set) var firstTokenLatencySeconds: Double? = nil
    @Published private(set) var instantTokensPerSecond: Double = 0
    @Published private(set) var avgTokensPerSecond: Double = 0
    @Published private(set) var peakTokensPerSecond: Double = 0
    @Published private(set) var residentMemoryMB: Double = 0
    @Published private(set) var threadCount: Int = 0
    @Published private(set) var lastError: String? = nil

    /// Sparkline points — one sample per chunk, capped at 60 entries.
    @Published private(set) var tpsHistory: [Double] = []

    /// Short, user-readable summary used by the collapsed pill.
    var compactLabel: String {
        switch phase {
        case .idle:
            return "Ready · \(backend)"
        case .loading, .prefilling, .finalizing:
            return phase.rawValue
        case .decoding:
            return String(format: "%@ · %.1f tok/s · %d tok",
                          phase.rawValue, instantTokensPerSecond, outputTokens)
        case .error:
            return "Error"
        }
    }

    var isActive: Bool {
        switch phase {
        case .loading, .prefilling, .decoding, .finalizing: return true
        case .idle, .error: return false
        }
    }

    // Run tracking ----------------------------------------------------

    private var runStart: Date?
    private var lastChunkAt: Date?
    private var recentWindow: [(at: Date, tokens: Int)] = []

    /// Called by `ExtractionService` right before calling `engine.generate(...)`.
    func begin(backend: String, model: String, promptChars: Int, maxTokens: Int) {
        self.phase = .prefilling
        self.backend = backend
        self.modelName = model
        self.promptChars = promptChars
        // Gemma 4 tokenizer averages ~3.6 chars/token on English clinical
        // text — close enough for a live UI estimate.
        self.promptTokensApprox = max(1, promptChars / 4)
        self.outputTokens = 0
        self.maxTokens = maxTokens
        self.elapsedSeconds = 0
        self.firstTokenLatencySeconds = nil
        self.instantTokensPerSecond = 0
        self.avgTokensPerSecond = 0
        self.peakTokensPerSecond = 0
        self.tpsHistory.removeAll(keepingCapacity: true)
        self.lastError = nil
        self.runStart = Date()
        self.lastChunkAt = nil
        self.recentWindow.removeAll(keepingCapacity: true)
        self.sampleHostStats()
    }

    /// Called for every streamed chunk. `chunkTokens` is the token count in
    /// just this chunk (1 for most engines, sometimes higher with batching).
    func record(chunkTokens: Int) {
        guard let runStart else { return }
        let now = Date()

        if firstTokenLatencySeconds == nil {
            firstTokenLatencySeconds = now.timeIntervalSince(runStart)
            phase = .decoding
        }

        outputTokens += max(1, chunkTokens)
        elapsedSeconds = now.timeIntervalSince(runStart)

        // Instant rate over the last ~1 s window
        recentWindow.append((at: now, tokens: max(1, chunkTokens)))
        let cutoff = now.addingTimeInterval(-1.0)
        while let first = recentWindow.first, first.at < cutoff {
            recentWindow.removeFirst()
        }
        let windowTokens = recentWindow.reduce(0) { $0 + $1.tokens }
        let windowSpan = max(0.05, (recentWindow.last?.at.timeIntervalSince(recentWindow.first?.at ?? now)) ?? 0.05)
        instantTokensPerSecond = windowSpan > 0 ? Double(windowTokens) / windowSpan : 0
        peakTokensPerSecond = max(peakTokensPerSecond, instantTokensPerSecond)

        avgTokensPerSecond = elapsedSeconds > 0 ? Double(outputTokens) / elapsedSeconds : 0

        tpsHistory.append(instantTokensPerSecond)
        if tpsHistory.count > 60 { tpsHistory.removeFirst(tpsHistory.count - 60) }

        lastChunkAt = now
        sampleHostStats()
    }

    func finalize() {
        phase = .finalizing
        if let runStart {
            elapsedSeconds = Date().timeIntervalSince(runStart)
        }
        sampleHostStats()
    }

    func end(error: String? = nil) {
        if let error {
            phase = .error
            lastError = error
        } else {
            phase = .idle
        }
        if let runStart {
            elapsedSeconds = Date().timeIntervalSince(runStart)
        }
    }

    // Host-level stats (resident memory, thread count) ----------------

    private func sampleHostStats() {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            residentMemoryMB = Double(info.resident_size) / (1024 * 1024)
        }

        // Thread count via task_threads
        var threads: thread_act_array_t?
        var threadCount_: mach_msg_type_number_t = 0
        let tkr = task_threads(mach_task_self_, &threads, &threadCount_)
        if tkr == KERN_SUCCESS {
            threadCount = Int(threadCount_)
            if let threads {
                let size = vm_size_t(Int(threadCount_) * MemoryLayout<thread_t>.size)
                vm_deallocate(mach_task_self_, vm_address_t(UInt(bitPattern: UnsafeMutablePointer(threads))), size)
            }
        }
    }
}
