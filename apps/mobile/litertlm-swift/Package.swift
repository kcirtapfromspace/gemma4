// swift-tools-version:5.9
// Swift Package wrapping the LiteRT-LM C++ runtime for iOS / macOS via a
// small C-ABI shim. The .xcframework is produced by scripts/build_xcframework.sh
// from an adjacent Bazel checkout of google-ai-edge/LiteRT-LM. See STATUS.md.

import PackageDescription

let package = Package(
    name: "LiteRtLm",
    platforms: [
        .iOS(.v15),
        .macOS(.v13),
    ],
    products: [
        .library(name: "LiteRtLm", targets: ["LiteRtLm"]),
    ],
    dependencies: [],
    targets: [
        // Binary xcframework produced by scripts/build_xcframework.sh.
        // Commit the .zip (or fetch over HTTPS) once the Bazel build lands.
        .binaryTarget(
            name: "LiteRtLmCore",
            // TODO(c11): Flip this to `url:` + `checksum:` once a release
            // artifact is published. For now consumers run the build script
            // to produce a local LiteRtLmCore.xcframework.zip and point
            // here at `path: "build/LiteRtLmCore.xcframework"`.
            path: "build/LiteRtLmCore.xcframework"
        ),
        // Swift-visible C module for the shim header.
        .target(
            name: "LiteRtLmCShim",
            dependencies: ["LiteRtLmCore"],
            path: "Sources/LiteRtLmCShim",
            publicHeadersPath: "include"
        ),
        // High-level Swift wrapper.
        .target(
            name: "LiteRtLm",
            dependencies: ["LiteRtLmCShim"],
            path: "Sources/LiteRtLm"
        ),
        .testTarget(
            name: "LiteRtLmTests",
            dependencies: ["LiteRtLm"],
            path: "Tests/LiteRtLmTests"
        ),
    ]
)
