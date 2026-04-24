// LiteRtLmCli — minimal command-line demo of the LiteRtLm Swift package.
//
// Invocation:
//   LITERTLM_MODEL_PATH=/path/to/gemma-4-E2B-it.litertlm \
//   swift run LiteRtLmCli "What is the capital of France?"
//
// Or, from the iOS Simulator via xcodebuild:
//   xcodebuild -scheme LiteRtLmCli \
//     -destination 'platform=iOS Simulator,id=<uuid>' run
//
// On an actual Mac host with `swift run`, the LiteRtLmCore xcframework's
// linker search path isn't automatically wired up — see STATUS.md for
// the `xcodebuild` workaround. On the iOS Simulator the whole package
// links cleanly because SwiftPM + Xcode's build system cooperate.
//
// Environment variables:
//   LITERTLM_MODEL_PATH  (required) — absolute path to a .litertlm file.
//   LITERTLM_BACKEND     (optional) — cpu | gpu. Default cpu on simulator.
//   LITERTLM_MAX_TOKENS  (optional) — decode cap. Default 64.

import Foundation
import LiteRtLm

@MainActor
func runCli() async -> Int32 {
    // CommandLine.arguments[0] is the executable path; [1...] is the
    // user prompt. Join remaining args so quoting is forgiving.
    let args = CommandLine.arguments.dropFirst()
    let prompt: String
    if args.isEmpty {
        prompt = "Hello, who is the president of France?"
        FileHandle.standardError.write(Data(
            "LiteRtLmCli: no prompt on argv, using default: \"\(prompt)\"\n".utf8
        ))
    } else {
        prompt = args.joined(separator: " ")
    }

    let env = ProcessInfo.processInfo.environment
    guard let modelPath = env["LITERTLM_MODEL_PATH"] else {
        FileHandle.standardError.write(Data(
            "LiteRtLmCli: LITERTLM_MODEL_PATH is not set.\n".utf8
        ))
        return 2
    }
    let backend: LiteRtLmBackend = {
        switch env["LITERTLM_BACKEND"]?.lowercased() {
        case "gpu": return .gpu
        case "npu": return .npu
        default:    return .cpu
        }
    }()
    let maxTokens = Int(env["LITERTLM_MAX_TOKENS"] ?? "") ?? 64

    let url = URL(fileURLWithPath: modelPath)
    guard FileManager.default.fileExists(atPath: url.path) else {
        FileHandle.standardError.write(Data(
            "LiteRtLmCli: model file not found at \(url.path)\n".utf8
        ))
        return 3
    }

    // Use `FileHandle.standardOutput.write` so the header lines
    // interleave deterministically with the streamed decode output —
    // Swift's `print` is line-buffered and shows up out of order on
    // iOS Simulator stdout. Trailing newlines are explicit here.
    let stdout = FileHandle.standardOutput
    func outln(_ s: String) { stdout.write(Data((s + "\n").utf8)) }

    outln("LiteRtLmCli v\(liteRtLmShimVersion())")
    outln("  model:    \(url.path)")
    outln("  backend:  \(backend)")
    outln("  prompt:   \(prompt)")
    outln("  maxTok:   \(maxTokens)")
    outln("---")

    do {
        let tLoad0 = Date()
        let engine = try LiteRtLmEngine(modelPath: url, backend: backend)
        let session = try engine.makeSession()
        session.setMaxDecodeTokens(maxTokens)
        let tLoad = Date().timeIntervalSince(tLoad0)

        let tPrefill0 = Date()
        try session.prefill(prompt)
        let tPrefill = Date().timeIntervalSince(tPrefill0)

        let tDecode0 = Date()
        var fullText = ""
        for try await chunk in session.decode(bufferSize: 256) {
            // Mirror cat-style live printing even though the current
            // shim buffers the full decode — when RunDecodeAsync is
            // wired in we'll get genuine token-at-a-time streaming here
            // with no change to the CLI.
            FileHandle.standardOutput.write(Data(chunk.utf8))
            fullText += chunk
        }
        let tDecode = Date().timeIntervalSince(tDecode0)

        let tokens = session.lastTokenCount
        let tokPerSec = Double(tokens) / max(tDecode, 0.001)

        outln("")
        outln("---")
        outln(String(format: "load_s=%.2f prefill_s=%.2f decode_s=%.2f",
                     tLoad, tPrefill, tDecode))
        outln(String(format: "tokens=%d tok_per_sec=%.2f",
                     tokens, tokPerSec))
        return 0
    } catch {
        FileHandle.standardError.write(Data(
            "LiteRtLmCli: \(error)\n".utf8
        ))
        return 1
    }
}

// Bridge an async main to the synchronous process-entry signature.
let exitCode = await runCli()
exit(exitCode)
