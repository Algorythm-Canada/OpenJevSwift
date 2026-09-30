// swift-tools-version: 6.2
// Tools 6.2 is the minimum that mlx-swift-lm accepts. docs/development.md lists the toolchain each
// platform needs.

import PackageDescription

/// Every target compiles in the Swift 6 language mode, which turns on complete strict concurrency
/// checking.
let swiftSettings: [SwiftSetting] = [
    .swiftLanguageMode(.v6)
]

var products: [Product] = [
    .library(name: "OpenJevCore", targets: ["OpenJevCore"]),
    .executable(name: "openjev", targets: ["openjev"]),
]

var dependencies: [Package.Dependency] = [
    .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.23.0"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.8.0"),
]

var targets: [Target] = [
    // Foundation only, so that it builds on macOS, iOS and Linux.
    .target(
        name: "OpenJevCore",
        swiftSettings: swiftSettings
    ),
    // The HTTP server. It is not a library product, so library consumers never link Hummingbird.
    // The server runs on macOS and Linux, and the platform condition keeps Hummingbird out of iOS
    // builds.
    .target(
        name: "OpenJevServer",
        dependencies: [
            "OpenJevCore",
            .product(
                name: "Hummingbird",
                package: "hummingbird",
                condition: .when(platforms: [.macOS, .linux])
            ),
        ],
        swiftSettings: swiftSettings
    ),
    .executableTarget(
        name: "openjev",
        dependencies: [
            "OpenJevCore",
            "OpenJevServer",
            .product(name: "ArgumentParser", package: "swift-argument-parser"),
        ],
        swiftSettings: swiftSettings
    ),
    .testTarget(
        name: "OpenJevCoreTests",
        dependencies: ["OpenJevCore"],
        swiftSettings: swiftSettings
    ),
    .testTarget(
        name: "OpenJevServerTests",
        dependencies: ["OpenJevServer"],
        swiftSettings: swiftSettings
    ),
]

// OpenJevDiffusionGemma runs on MLX, which needs Apple silicon. SwiftPM evaluates this manifest on
// the host, and every Apple platform build runs on a macOS host, so the MLX packages and the
// targets that use them are declared only there. A Linux build never loads their manifests or
// builds them.
#if os(macOS)
    products.append(
        .library(name: "OpenJevDiffusionGemma", targets: ["OpenJevDiffusionGemma"])
    )
    dependencies += [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.2"),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "c043fb3b1ccf00f54ef8882a1e8da45c6e32e6f8"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            .upToNextMinor(from: "1.3.0")
        ),
        // The same range swift-transformers declares, so one version of swift-jinja is resolved.
        // OpenJevDiffusionGemma renders the chat template to text with it (decision D-008).
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.4.2"),
    ]
    targets += [
        .target(
            name: "OpenJevDiffusionGemma",
            dependencies: [
                "OpenJevCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Jinja", package: "swift-jinja"),
            ],
            swiftSettings: swiftSettings
        ),
        // The tests also load the tokenizer through mlx-swift-lm's MLXHuggingFace macros, to
        // confirm that path gives the same results as the direct one the module uses.
        .testTarget(
            name: "OpenJevDiffusionGemmaTests",
            dependencies: [
                "OpenJevDiffusionGemma",
                "OpenJevCore",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: swiftSettings
        ),
    ]
#endif

let package = Package(
    name: "OpenJevSwift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: products,
    dependencies: dependencies,
    targets: targets
)
