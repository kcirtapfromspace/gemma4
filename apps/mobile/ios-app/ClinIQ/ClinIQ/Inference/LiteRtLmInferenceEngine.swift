// LiteRtLmInferenceEngine.swift
// On-device inference via the LiteRT-LM Swift Package (vendored under
// `../../litertlm-swift/`, consumed via SPM). This is the second backend
// alongside `LlamaCppInferenceEngine`, selectable at runtime via the
// `ClinIQ.Backend` @AppStorage key.
//
// Model format is `.litertlm` (NOT GGUF). The stock distribution is
// Google's 2.58 GB `gemma-4-E2B-it.litertlm` from the `litert-community`
// HF repo; see `apps/mobile/litertlm-swift/scripts/fetch_model.sh`.
//
// Why this file exists:
//   - C11 delivered the Swift wrapper; C14 landed streaming decode.
//   - C15 (this file) wires it into the ClinIQ UI surface behind the
//     existing `InferenceEngine` protocol so we can A/B the LoRA-tuned
//     GGUF against the stock base at demo time.
//
// Linker note: the LiteRtLmCore xcframework bundles static archives with
// file-scope engine-registration initializers (LITERT_LM_REGISTER_ENGINE).
// The Package.swift for LiteRtLm already passes `-Xlinker -all_load`, but
// when Xcode's SwiftPM integration merges the final app link, that flag
// doesn't always propagate. If you see `EngineFactory::CreateDefault`
// returning `NotFound` at runtime, add `-Xlinker -all_load` to the ClinIQ
// target's OTHER_LDFLAGS. Tracked in LITERTLM_BACKEND.md.

import Foundation
#if canImport(LiteRtLm)
import LiteRtLm

/// `InferenceEngine` implementation backed by the LiteRT-LM Swift Package.
/// Loads lazily on first `generate(...)` call. Matches the actor pattern
/// of `LlamaCppInferenceEngine` so downstream UI code is uniform.
///
/// The underlying `LiteRtLmEngine` / `LiteRtLmSession` are reference-typed
/// with their own thread-safety contract (see LiteRtLm.swift). We still
/// serialize access through a Swift `actor` to match the llama.cpp shape
/// and to make the Ensure-Loaded pattern easy to reason about.
final class LiteRtLmInferenceEngine: InferenceEngine {
    /// Candidate `.litertlm` filenames, in preference order. Stock base
    /// model first; our fine-tune artifact name is reserved here for the
    /// eventual LoRA re-bake (C6/C8 re-spin).
    static let candidateModelNames: [String] = [
        "cliniq-gemma4-e2b",
        "gemma-4-E2B-it",
    ]

    private let backend: LiteRtLmBackend
    private var loaded: LoadedModel?
    private var loadError: Error?

    /// Minimal holder for the engine + its current session. Separate type
    /// so `ensureLoaded()` can return it without exposing internals.
    private struct LoadedModel {
        let engine: LiteRtLmEngine
    }

    init(backend: LiteRtLmBackend = .cpu) {
        // Default to CPU for simulator compatibility (no Metal on the
        // simulator). Physical iPhone demo should flip to `.gpu`.
        #if targetEnvironment(simulator)
        self.backend = .cpu
        #else
        self.backend = backend
        #endif
    }

    /// Locate a `.litertlm` model in (1) the app bundle, (2) Documents/,
    /// (3) tmp/. Parallel to `LlamaCppInferenceEngine.resolveModelPath()`.
    static func resolveModelPath() -> String? {
        let fm = FileManager.default
        // (1) bundled resource
        for name in candidateModelNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "litertlm") {
                return url.path
            }
        }
        // (2) app Documents dir
        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            for name in candidateModelNames {
                let candidate = docs.appendingPathComponent("\(name).litertlm")
                if fm.fileExists(atPath: candidate.path) {
                    return candidate.path
                }
            }
        }
        // (3) tmp dir (useful when seeded via simctl)
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        for name in candidateModelNames {
            let candidate = tmp.appendingPathComponent("\(name).litertlm")
            if fm.fileExists(atPath: candidate.path) {
                return candidate.path
            }
        }
        return nil
    }

    private func ensureLoaded() throws -> LoadedModel {
        if let loaded = loaded { return loaded }
        if let err = loadError { throw err }
        guard let path = Self.resolveModelPath() else {
            let err = InferenceError.modelNotFound(
                "No .litertlm found in bundle / Documents / tmp. Tried: " +
                Self.candidateModelNames.map { "\($0).litertlm" }.joined(separator: ", ")
            )
            loadError = err
            throw err
        }
        do {
            let engine = try LiteRtLmEngine(modelPath: URL(fileURLWithPath: path), backend: backend)
            let loaded = LoadedModel(engine: engine)
            self.loaded = loaded
            return loaded
        } catch {
            loadError = error
            throw error
        }
    }

    func generate(prompt: String, maxTokens: Int)
        async throws -> AsyncThrowingStream<InferenceChunk, Error>
    {
        // Load + make fresh session per call so each extraction is
        // independent (mirror of LlamaCpp's `ctx.reset()`).
        let loaded = try ensureLoaded()
        let session = try loaded.engine.makeSession()
        try session.prefill(prompt)

        // Fan LiteRtLm's String-chunk stream into our InferenceChunk
        // surface. We count one "token" per chunk — the C++ session
        // calls decode_step once per emitted piece, so this matches
        // the granularity used by `LlamaCppInferenceEngine`.
        let raw = session.decode()
        return AsyncThrowingStream { continuation in
            let task = Task {
                var emitted = 0
                do {
                    for try await piece in raw {
                        if piece.isEmpty { continue }
                        continuation.yield(InferenceChunk(text: piece, tokenCount: 1))
                        emitted += 1
                        if emitted >= maxTokens { break }
                        // Gemma-4 turn-close sentinel. Both our fine-tune
                        // and the stock template emit this; whichever
                        // arrives first, we stop.
                        if piece.contains("<turn|>") { break }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

#else

// Compile-time fallback for builds that haven't yet pulled the LiteRtLm
// package (e.g. CI running `swift build` on the ClinIQ sources alone).
// Keeps the file buildable; runtime selection falls back to llama.cpp.
//
// TODO(C15): Once the SPM dep is wired into the ClinIQ target on every
// CI lane, delete this shim.
final class LiteRtLmInferenceEngine: InferenceEngine {
    init(backend: Int = 0) {}
    static func resolveModelPath() -> String? { nil }
    func generate(prompt: String, maxTokens: Int)
        async throws -> AsyncThrowingStream<InferenceChunk, Error>
    {
        throw InferenceError.backendUnavailable(
            "LiteRtLm Swift package not linked into this build. See LITERTLM_BACKEND.md."
        )
    }
}

#endif
