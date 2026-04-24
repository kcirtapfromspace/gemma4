// LlamaCppInferenceEngine.swift
// Real on-device inference via llama.cpp's iOS xcframework (vendored under
// `ClinIQ/Frameworks/llama.xcframework/`). Drop-in replacement for
// `StubInferenceEngine` — conforms to the same `InferenceEngine` protocol.
//
// Model discovery order:
//   1. Bundle.main.url(forResource: "<name>", withExtension: "gguf")
//   2. <DocumentDirectory>/<name>.gguf  (pre-seeded by `simctl` in dev)
//   3. <DocumentDirectory>/<fallback>.gguf
//
// Simulator: CPU backend (n_gpu_layers = 0). Physical iPhone: Metal via the
// framework's default offload. See BUILD.md § "Inference backend".
//
// References:
//   - llama.cpp b8913 xcframework:
//     https://github.com/ggml-org/llama.cpp/releases/tag/b8913
//   - Reference Swift app (upstream):
//     https://github.com/ggml-org/llama.cpp/tree/master/examples/llama.swiftui

import Foundation
import llama

// MARK: - Batch helpers (copied from upstream llama.swiftui reference)

private func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

private func llama_batch_add(
    _ batch: inout llama_batch,
    _ id: llama_token,
    _ pos: llama_pos,
    _ seq_ids: [llama_seq_id],
    _ logits: Bool
) {
    batch.token[Int(batch.n_tokens)] = id
    batch.pos[Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits[Int(batch.n_tokens)] = logits ? 1 : 0
    batch.n_tokens += 1
}

// MARK: - Actor wrapping the llama.cpp context

/// Owns the model + context + sampler. All llama.cpp calls are serialized
/// through the actor so we can safely hand off tokens to the main thread.
private actor LlamaContext {
    private let model: OpaquePointer
    private let context: OpaquePointer
    private let vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var tokensList: [llama_token] = []
    private var temporaryInvalidCChars: [CChar] = []

    /// Max tokens to decode per call (can be capped lower by the caller).
    var nLen: Int32 = 512
    /// Current KV cache cursor.
    var nCur: Int32 = 0
    /// Number of tokens we've decoded this session.
    var nDecode: Int32 = 0
    /// Set true once we hit EOG or `maxTokens`.
    var isDone: Bool = false

    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        // 1024-token batch is enough to prefill our ~700-token prompts without
        // splitting. Adjust if the system prompt grows.
        self.batch = llama_batch_init(1024, 0, 1)
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        // Greedy-leaning sampler: low temp + deterministic final draw so the
        // scoring script gets reproducible outputs. JSON extraction doesn't
        // need creativity.
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.2))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        self.vocab = llama_model_get_vocab(model)
    }

    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }

    static func create(path: String, nCtx: UInt32 = 2048) throws -> LlamaContext {
        llama_backend_init()
        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        // Simulator Metal is unreliable; force CPU.
        modelParams.n_gpu_layers = 0
        #endif
        guard let model = llama_model_load_from_file(path, modelParams) else {
            throw InferenceError.modelNotFound(path)
        }
        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = nCtx
        ctxParams.n_threads = Int32(nThreads)
        ctxParams.n_threads_batch = Int32(nThreads)
        guard let context = llama_init_from_model(model, ctxParams) else {
            llama_model_free(model)
            throw InferenceError.backendUnavailable("llama_init_from_model failed")
        }
        return LlamaContext(model: model, context: context)
    }

    // MARK: - Prefill

    /// Tokenize `text`, push into the batch, and run a single prefill `llama_decode`.
    /// Returns the number of tokens in the prompt.
    func prefill(text: String, maxTokens: Int) throws -> Int {
        nLen = Int32(maxTokens)
        isDone = false
        nDecode = 0
        temporaryInvalidCChars = []

        tokensList = try tokenize(text: text, addBOS: true)
        guard !tokensList.isEmpty else {
            throw InferenceError.runtimeFailure("empty token list after tokenize")
        }
        let nCtx = Int(llama_n_ctx(context))
        if tokensList.count + Int(nLen) > nCtx {
            // Clip maxTokens to what fits.
            nLen = Int32(max(0, nCtx - tokensList.count - 4))
        }

        llama_batch_clear(&batch)
        for (i, tok) in tokensList.enumerated() {
            llama_batch_add(&batch, tok, Int32(i), [0], false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1

        if llama_decode(context, batch) != 0 {
            throw InferenceError.runtimeFailure("llama_decode failed during prefill")
        }
        nCur = batch.n_tokens
        return tokensList.count
    }

    // MARK: - Decode step

    /// Sample one token and decode it. Returns the (possibly-empty) piece of
    /// text to push to the UI. Sets `isDone=true` on EOG or `nLen` reached.
    func decodeStep() throws -> String {
        let newTokenID = llama_sampler_sample(sampling, context, batch.n_tokens - 1)
        if llama_vocab_is_eog(vocab, newTokenID) || nDecode >= nLen {
            isDone = true
            // Flush any buffered invalid-utf8 bytes as a last string.
            let flush = String(cString: temporaryInvalidCChars + [0])
            temporaryInvalidCChars.removeAll()
            return flush
        }

        let piece = tokenToPiece(token: newTokenID)
        temporaryInvalidCChars.append(contentsOf: piece)
        let tokenStr: String
        if let s = String(validatingUTF8: temporaryInvalidCChars + [0]) {
            temporaryInvalidCChars.removeAll()
            tokenStr = s
        } else if (0..<temporaryInvalidCChars.count).contains(where: {
            $0 != 0 && String(validatingUTF8: Array(temporaryInvalidCChars.suffix($0)) + [0]) != nil
        }) {
            let s = String(cString: temporaryInvalidCChars + [0])
            temporaryInvalidCChars.removeAll()
            tokenStr = s
        } else {
            tokenStr = ""
        }

        llama_batch_clear(&batch)
        llama_batch_add(&batch, newTokenID, nCur, [0], true)
        nDecode += 1
        nCur += 1
        if llama_decode(context, batch) != 0 {
            throw InferenceError.runtimeFailure("llama_decode failed during step")
        }
        return tokenStr
    }

    func reset() {
        tokensList.removeAll()
        temporaryInvalidCChars.removeAll()
        llama_memory_clear(llama_get_memory(context), true)
        nCur = 0
        nDecode = 0
        isDone = false
    }

    // MARK: - Internal helpers

    private func tokenize(text: String, addBOS: Bool) throws -> [llama_token] {
        let utf8Count = text.utf8.count
        let capacity = utf8Count + (addBOS ? 1 : 0) + 1
        let buf = UnsafeMutablePointer<llama_token>.allocate(capacity: capacity)
        defer { buf.deallocate() }
        let n = llama_tokenize(vocab, text, Int32(utf8Count), buf, Int32(capacity), addBOS, false)
        if n < 0 {
            throw InferenceError.runtimeFailure("llama_tokenize returned \(n)")
        }
        var out: [llama_token] = []
        out.reserveCapacity(Int(n))
        for i in 0..<Int(n) {
            out.append(buf[i])
        }
        return out
    }

    private func tokenToPiece(token: llama_token) -> [CChar] {
        var buf = UnsafeMutablePointer<Int8>.allocate(capacity: 16)
        buf.initialize(repeating: 0, count: 16)
        defer { buf.deallocate() }
        let n = llama_token_to_piece(vocab, token, buf, 16, 0, false)
        if n >= 0 {
            return Array(UnsafeBufferPointer(start: buf, count: Int(n)))
        }
        // Required buffer is -n bytes
        let needed = Int(-n)
        let big = UnsafeMutablePointer<Int8>.allocate(capacity: needed)
        big.initialize(repeating: 0, count: needed)
        defer { big.deallocate() }
        let n2 = llama_token_to_piece(vocab, token, big, Int32(needed), 0, false)
        if n2 < 0 { return [] }
        return Array(UnsafeBufferPointer(start: big, count: Int(n2)))
    }
}

// MARK: - Engine

/// `InferenceEngine` implementation backed by the vendored llama.cpp
/// xcframework. Loads lazily on first `generate(...)` call.
final class LlamaCppInferenceEngine: InferenceEngine {
    // Candidate GGUF filenames, in preference order. The fine-tune is
    // preferred; falls back to the stock base if the former isn't found.
    static let candidateModelNames: [String] = [
        // v2 LoRA (Kaggle kernel 26, 2026-04-24). r=32, target_modules
        // drops the 40 KV-shared k/v_proj no-ops, +50 diversified
        // training examples targeting code-elision + neg-lab degen.
        "cliniq-gemma4-e2b-v2-Q3_K_M",
        "cliniq-gemma4-e2b-Q3_K_M",
        "gemma-4-E2B-it-Q3_K_M",
        "cliniq-gemma4-e2b-Q2_K",
    ]

    private var ctx: LlamaContext?
    private var modelLoadError: Error?

    init() {}

    /// Locate a GGUF model in (1) the app bundle, (2) Documents/, (3) tmp/.
    static func resolveModelPath() -> String? {
        let fm = FileManager.default
        // (1) bundled resource
        for name in candidateModelNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "gguf") {
                return url.path
            }
        }
        // (2) app Documents dir
        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            for name in candidateModelNames {
                let candidate = docs.appendingPathComponent("\(name).gguf")
                if fm.fileExists(atPath: candidate.path) {
                    return candidate.path
                }
            }
        }
        // (3) tmp dir (useful when seeded via simctl)
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        for name in candidateModelNames {
            let candidate = tmp.appendingPathComponent("\(name).gguf")
            if fm.fileExists(atPath: candidate.path) {
                return candidate.path
            }
        }
        return nil
    }

    private func ensureLoaded() async throws -> LlamaContext {
        if let ctx = ctx { return ctx }
        if let err = modelLoadError { throw err }
        guard let path = Self.resolveModelPath() else {
            let err = InferenceError.modelNotFound(
                "No GGUF found in bundle / Documents / tmp. Tried: " +
                Self.candidateModelNames.map { "\($0).gguf" }.joined(separator: ", ")
            )
            modelLoadError = err
            throw err
        }
        do {
            let ctx = try LlamaContext.create(path: path)
            self.ctx = ctx
            return ctx
        } catch {
            modelLoadError = error
            throw error
        }
    }

    func generate(prompt: String, maxTokens: Int)
        async throws -> AsyncThrowingStream<InferenceChunk, Error>
    {
        let ctx = try await ensureLoaded()
        // Reset previous KV state so each call is independent. For a
        // multi-turn chat we'd preserve state and only append.
        await ctx.reset()
        let promptTokens = try await ctx.prefill(text: prompt, maxTokens: maxTokens)
        _ = promptTokens

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    while await !ctx.isDone {
                        let piece = try await ctx.decodeStep()
                        if !piece.isEmpty {
                            continuation.yield(InferenceChunk(text: piece, tokenCount: 1))
                        }
                        // Early-exit on the gemma-4 turn-close delimiter.
                        // (EOG also catches this, but chat templates sometimes
                        // emit `<turn|>` as a multi-token sequence.)
                        if piece.contains("<turn|>") {
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
