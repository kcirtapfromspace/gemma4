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
        // Tiny demo CLI. On the Mac host `swift run` struggles to
        // resolve the xcframework search paths; the reliable invocation
        // is `xcodebuild -scheme LiteRtLmCli -destination 'platform=iOS
        // Simulator,id=...' run`. See DEMO.md.
        .executable(name: "LiteRtLmCli", targets: ["LiteRtLmCli"]),
    ],
    dependencies: [],
    targets: [
        // Binary xcframework produced by scripts/build_xcframework.sh.
        // The xcframework ships its own umbrella header + modulemap so
        // Swift can `import LiteRtLmCore` without a separate C target.
        .binaryTarget(
            name: "LiteRtLmCore",
            // TODO(c11): Flip this to `url:` + `checksum:` once a release
            // artifact is published. For now consumers run the build script
            // to produce a local LiteRtLmCore.xcframework.
            path: "build/LiteRtLmCore.xcframework"
        ),
        // High-level Swift wrapper. Imports the C symbols directly from
        // the binary target's generated framework module.
        //
        // The `-ObjC` + `-all_load` linker flags force the Apple linker to
        // pull in EVERY .o file from LiteRtLmCore's static archive, even
        // ones that have no directly-referenced external symbols. This is
        // required because LiteRT-LM uses file-scope static initializers
        // (e.g. `LITERT_LM_REGISTER_ENGINE` in runtime/core/engine_impl.cc)
        // that would otherwise be stripped by the linker's dead-strip
        // pass. Bazel's `alwayslink=1` is preserved inside the archive,
        // but the CONSUMER's link must opt in explicitly.
        .target(
            name: "LiteRtLm",
            dependencies: ["LiteRtLmCore"],
            path: "Sources/LiteRtLm",
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-all_load"]),
            ]
        ),
        // Command-line demo target. Must carry the same -all_load
        // linker flag as the library target so the upstream registerer
        // symbols survive dead-strip.
        .executableTarget(
            name: "LiteRtLmCli",
            dependencies: ["LiteRtLm"],
            path: "Sources/LiteRtLmCli",
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-all_load"]),
            ]
        ),
        .testTarget(
            name: "LiteRtLmTests",
            dependencies: ["LiteRtLm"],
            path: "Tests/LiteRtLmTests"
        ),
    ]
)
