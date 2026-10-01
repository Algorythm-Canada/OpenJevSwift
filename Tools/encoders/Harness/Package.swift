// swift-tools-version: 6.0
// The spike #56 harness: Verdict and Laya through Core ML, measured on macOS and iOS. It is a
// separate package so that the main package's graph stays untouched; nothing depends on it.

import PackageDescription

let package = Package(
    name: "EncoderHarness",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "EncoderHarness", targets: ["EncoderHarness"]),
        .executable(name: "encoder-harness", targets: ["encoder-harness"]),
        .executable(name: "encoder-capacity", targets: ["encoder-capacity"]),
    ],
    dependencies: [
        // The version the main package resolves (Package.resolved at the repository root).
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.3.4")
    ],
    targets: [
        .target(
            name: "EncoderHarness",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "encoder-harness",
            dependencies: ["EncoderHarness"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // What each loaded function of a multifunction package costs in memory and load time on a
        // Mac (D-042, docs/spikes/encoder-function-capacity.md).
        .executableTarget(
            name: "encoder-capacity",
            dependencies: ["EncoderHarness"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // Runs on macOS and the iOS Simulator, which read the repository and the caches on the Mac.
        // An iPhone runs the measurements through Tools/encoders/HarnessApp.swiftpm instead: Xcode
        // cannot host a package's test bundle on a device.
        .testTarget(
            name: "EncoderHarnessTests",
            dependencies: ["EncoderHarness"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
