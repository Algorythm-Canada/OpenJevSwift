// swift-tools-version: 6.2
// Scratch package for spike #22 on the stack the port will use: ml-explore/mlx-swift 0.32.2,
// ml-explore/mlx-swift-lm c043fb3 and the main package by path. Not part of OpenJevSwift and not
// built by its CI. Four executables, run from the repository root:
//
//     swift run --package-path Tools/oracle/UpstreamProbe -c release TokenizerCheck
//     swift run --package-path Tools/oracle/UpstreamProbe -c release Transliteration
//     swift run --package-path Tools/oracle/UpstreamProbe -c release Gemma4Load
//     swift run --package-path Tools/oracle/UpstreamProbe -c release ItemReads
//
// TokenizerCheck renders every oracle prompt with the main package's SwiftTransformersTokenizer.
// Transliteration is mlx-vlm's DiffusionGemma read path written out in Swift on upstream
// primitives, compared bit for bit with Fixtures/oracle/reads.json. Gemma4Load loads
// mlx-community/gemma-4-26B-A4B-it-4bit with upstream's LLMModelFactory (risk R20). ItemReads
// records the main package's own engine and model reading JevBench and TypeSafe items, for
// Tools/oracle/item_reads.py to compare with upstream's reads (issue #62).

import Foundation
import PackageDescription

// A path dependency's identity is its directory name, which differs between the main checkout
// and a worktree, so it is derived from where this manifest sits.
let mainPackage = URL(fileURLWithPath: Context.packageDirectory)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
let mainIdentity = mainPackage.lastPathComponent.lowercased()

let package = Package(
    name: "UpstreamProbe",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../../.."),
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.2"),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "c043fb3b1ccf00f54ef8882a1e8da45c6e32e6f8"),
    ],
    targets: [
        .executableTarget(
            name: "TokenizerCheck",
            dependencies: [
                .product(name: "OpenJevCore", package: mainIdentity),
                .product(name: "OpenJevDiffusionGemma", package: mainIdentity),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "Transliteration",
            dependencies: [
                .product(name: "OpenJevCore", package: mainIdentity),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "Gemma4Load",
            dependencies: [
                .product(name: "OpenJevCore", package: mainIdentity),
                .product(name: "OpenJevDiffusionGemma", package: mainIdentity),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "ItemReads",
            dependencies: [
                .product(name: "OpenJevCore", package: mainIdentity),
                .product(name: "OpenJevDiffusionGemma", package: mainIdentity),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
