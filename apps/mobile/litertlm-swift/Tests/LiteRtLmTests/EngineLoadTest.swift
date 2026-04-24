// Smoke tests for the LiteRtLm Swift wrapper.
//
// These tests are gated on the LITERTLM_MODEL_PATH environment variable
// which should point at a `.litertlm` file on disk (e.g. Google's stock
// `gemma-4-E2B-it.litertlm` downloaded to /tmp). Without that env var,
// the model-bearing tests are skipped, so `swift test` still succeeds on
// developer machines that don't have a 2.5 GB blob handy.

import XCTest
@testable import LiteRtLm

final class EngineLoadTest: XCTestCase {
    func testShimVersionIsReachable() {
        // Cheapest possible smoke: if the xcframework linked, the C shim
        // symbol resolves and we get back a non-empty version string.
        let v = liteRtLmShimVersion()
        XCTAssertFalse(v.isEmpty)
        XCTAssertTrue(v.contains("team-c11"))
    }

    func testEngineLoadsModel() throws {
        guard let envPath = ProcessInfo.processInfo.environment["LITERTLM_MODEL_PATH"] else {
            throw XCTSkip("LITERTLM_MODEL_PATH not set; skipping model load test.")
        }
        let url = URL(fileURLWithPath: envPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Model file not found at \(url.path); skipping.")
        }
        // CPU backend is the only one guaranteed to work on the Mac Studio
        // host; GPU path is iOS-device-only (Metal delegate).
        let engine = try LiteRtLmEngine(modelPath: url, backend: .cpu)
        let session = try engine.makeSession()
        // Prefill a tiny prompt to prove end-to-end wiring. We don't assert
        // on decode output — decode speed on macOS is not the goal here.
        try session.prefill("Hello")
    }
}
