// High-level Swift wrapper around LiteRT-LM's C-ABI shim.
//
// Usage:
//     let engine = try LiteRtLmEngine(modelPath: url, backend: .gpu)
//     let session = try engine.makeSession()
//     try session.prefill("Summarize this eICR: …")
//     for try await chunk in session.decode() {
//         print(chunk, terminator: "")
//     }
//
// Thread safety: mirror of the C++ API — one Engine can spawn multiple
// Sessions, but each Session is single-writer. We defer to the caller;
// the wrapper does NOT introduce its own locks.

import Foundation
@_implementationOnly import LiteRtLmCShim

public enum LiteRtLmBackend: Int32 {
    case cpu = 0
    case gpu = 1
    case npu = 2
}

public enum LiteRtLmError: Error, CustomStringConvertible {
    case engineCreateFailed(path: String)
    case sessionCreateFailed
    case prefillFailed(code: Int32)
    case decodeFailed(code: Int32)

    public var description: String {
        switch self {
        case .engineCreateFailed(let path):
            return "LiteRtLmEngine: failed to load model at \(path)"
        case .sessionCreateFailed:
            return "LiteRtLmEngine: failed to create session"
        case .prefillFailed(let c):
            return "LiteRtLmSession: prefill failed (code \(c))"
        case .decodeFailed(let c):
            return "LiteRtLmSession: decode failed (code \(c))"
        }
    }
}

/// Returns the version string exposed by the C shim. Useful for smoke
/// tests that only need to confirm the xcframework is linked.
public func liteRtLmShimVersion() -> String {
    return String(cString: litertlm_shim_version())
}

/// An on-device LLM engine backed by LiteRT-LM. Wraps a single model file;
/// thread-safe to share across sessions.
public final class LiteRtLmEngine {
    fileprivate let raw: OpaquePointer

    public init(modelPath: URL, backend: LiteRtLmBackend = .gpu) throws {
        let path = modelPath.path
        guard let p = path.withCString({ cpath in
            litertlm_engine_create(cpath, backend.rawValue)
        }) else {
            throw LiteRtLmError.engineCreateFailed(path: path)
        }
        self.raw = OpaquePointer(p)
    }

    deinit {
        litertlm_engine_destroy(UnsafeMutablePointer(raw))
    }

    public func makeSession() throws -> LiteRtLmSession {
        guard let p = litertlm_session_create(UnsafeMutablePointer(raw)) else {
            throw LiteRtLmError.sessionCreateFailed
        }
        return LiteRtLmSession(raw: OpaquePointer(p), engine: self)
    }
}

/// A single conversational turn with the engine. Keeps a strong reference
/// to the owning `LiteRtLmEngine` to prevent the native pointer from being
/// torn down while this session is alive.
public final class LiteRtLmSession {
    fileprivate let raw: OpaquePointer
    private let engine: LiteRtLmEngine

    fileprivate init(raw: OpaquePointer, engine: LiteRtLmEngine) {
        self.raw = raw
        self.engine = engine
    }

    deinit {
        litertlm_session_destroy(UnsafeMutablePointer(raw))
    }

    public func prefill(_ prompt: String) throws {
        let rc = prompt.withCString { cstr in
            litertlm_session_prefill(UnsafeMutablePointer(raw), cstr)
        }
        if rc != 0 { throw LiteRtLmError.prefillFailed(code: rc) }
    }

    /// Streaming decode. Returns an `AsyncThrowingStream` that yields UTF-8
    /// substrings as the model emits them. Calls `decode_step` on a
    /// background task; closes the stream on EOS or error.
    public func decode(bufferSize: Int = 512) -> AsyncThrowingStream<String, Error> {
        let rawPointer = raw
        return AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                var scratch = [CChar](repeating: 0, count: bufferSize)
                while !Task.isCancelled {
                    let written = scratch.withUnsafeMutableBufferPointer { buf -> Int32 in
                        litertlm_session_decode_step(
                            UnsafeMutablePointer(rawPointer),
                            buf.baseAddress,
                            Int32(bufferSize)
                        )
                    }
                    if written < 0 {
                        continuation.finish(throwing: LiteRtLmError.decodeFailed(code: written))
                        return
                    }
                    if written == 0 {
                        continuation.finish()
                        return
                    }
                    let data = Data(bytes: scratch, count: Int(written))
                    if let s = String(data: data, encoding: .utf8) {
                        continuation.yield(s)
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
