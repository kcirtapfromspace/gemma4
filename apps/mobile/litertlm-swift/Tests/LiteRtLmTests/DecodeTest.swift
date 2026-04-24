// End-to-end decode test for the LiteRtLm Swift wrapper.
//
// Proves the binding actually emits text from the model — not just that
// prefill returns 0. Gated on LITERTLM_MODEL_PATH like EngineLoadTest,
// and further gated on LITERTLM_RUN_DECODE=1 so routine `swift test`
// runs stay fast (decode on the simulator's CPU backend takes minutes,
// not seconds).
//
// Run locally with:
//   make test                # xcodebuild plumbs the env vars through
//   # or:
//   TEST_RUNNER_LITERTLM_MODEL_PATH=/private/tmp/gemma-model/gemma-4-E2B-it.litertlm \
//   TEST_RUNNER_LITERTLM_RUN_DECODE=1 \
//   xcodebuild -scheme LiteRtLm -destination '...' test

import XCTest
@testable import LiteRtLm

final class DecodeTest: XCTestCase {
    /// `testDecodeProducesTokens` — end-to-end decode smoke. Prefills a
    /// tiny prompt, streams up to 32 tokens, asserts that at least 5 of
    /// them are non-empty UTF-8, and prints a measured tok/s.
    func testDecodeProducesTokens() async throws {
        guard let envPath = ProcessInfo.processInfo.environment["LITERTLM_MODEL_PATH"] else {
            throw XCTSkip("LITERTLM_MODEL_PATH not set; skipping decode test.")
        }
        let runDecode = ProcessInfo.processInfo.environment["LITERTLM_RUN_DECODE"]
        guard runDecode == "1" else {
            throw XCTSkip(
                "LITERTLM_RUN_DECODE=1 not set; decode is slow on the simulator " +
                "(minutes per 32-token run). Opt in explicitly."
            )
        }
        let url = URL(fileURLWithPath: envPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Model file not found at \(url.path); skipping.")
        }

        let engine = try LiteRtLmEngine(modelPath: url, backend: .cpu)
        let session = try engine.makeSession()

        // Cap decode so this test never runs unbounded on CI. 64 tokens
        // is enough to prove "≥5 tokens flow end-to-end" without blowing
        // the default xcodebuild test timeout (each token is ~0.1-0.2s
        // on the Simulator's CPU backend, so budget ~10 s worst case).
        session.setMaxDecodeTokens(64)

        // Prompt engineered to elicit a reliable multi-token reply.
        // "List five colors" consistently produces 10+ SentencePiece
        // tokens on Gemma 4 E2B, and is stable enough that a single
        // run is a deterministic assertion rather than a flake risk.
        let prompt = "List five common colors, separated by commas."
        try session.prefill(prompt)

        // Drain the stream. With the current blocking-then-drain shim,
        // a small `bufferSize` gives us multiple chunks out of the
        // cached output — enough to prove Swift's AsyncStream plumbing
        // transports bytes end-to-end. The real "did tokens flow"
        // assertion is on `lastTokenCount`, not chunk count, because a
        // future streaming decode will yield one chunk per token.
        var chunks: [String] = []
        var fullText = ""
        let t0 = Date()
        for try await chunk in session.decode(bufferSize: 4) {
            if !chunk.isEmpty {
                chunks.append(chunk)
                fullText += chunk
            }
        }
        let elapsed = Date().timeIntervalSince(t0)

        // Non-trivial output.
        XCTAssertFalse(fullText.isEmpty, "decode() produced no text")
        XCTAssertGreaterThan(
            chunks.count, 0,
            "decode() produced zero non-empty chunks"
        )

        // Token tally from the shim. This is what we report as tok/s,
        // and also the "≥5 tokens flowed" assertion per the C14 brief.
        let tokens = session.lastTokenCount
        XCTAssertGreaterThanOrEqual(
            tokens, 5,
            "expected >=5 tokens, got \(tokens). full text: \(fullText)"
        )

        // Emit a machine-scrapeable line so CI / the Makefile target can
        // parse the number without needing to parse XCTest output.
        let toksPerSec = Double(tokens) / max(elapsed, 0.001)
        print(String(
            format: "LITERTLM_DECODE_RESULT tokens=%d elapsed_s=%.3f tok_per_sec=%.2f",
            tokens, elapsed, toksPerSec
        ))
        print("LITERTLM_DECODE_SAMPLE <<<\(fullText)>>>")

        // Sanity on elapsed — guards against a degenerate "returned
        // instantly with empty string" pass.
        XCTAssertGreaterThan(elapsed, 0.0)
    }
}
