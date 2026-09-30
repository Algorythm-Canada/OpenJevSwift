// swift-tools-version: 6.2
// Scratch package for spike #22: the read-only pass through the Layr-Labs fork of mlx-swift-lm,
// compared with the mlx-vlm oracle in Fixtures/oracle/reads.json. Not part of OpenJevSwift and
// not built by its CI. Run from the repository root:
//
//     swift run --package-path Tools/oracle/Probe -c release Probe
//
// It cannot depend on the main package by path. The main manifest declares ml-explore/mlx-swift-lm
// at c043fb3 and ml-explore/mlx-swift 0.32.2; the fork has the same package identities
// (mlx-swift-lm, and Layr-Labs/mlx-swift as mlx-swift) at other revisions, pins swift-jinja to
// exactly 2.3.6 where the main package needs 2.4.2 or later, and defines the same module names
// (MLXLMCommon, MLXLLM, MLXVLM). SwiftPM refuses the combination:
//
//     error: mlx-swift-lm is required using two different revision-based requirements
//     (eeba2afaf059a153ff909c9e01aa5e65b7bcad67 and c043fb3b1ccf00f54ef8882a1e8da45c6e32e6f8),
//     which is not supported
//
// So OpenJevCore, which has no dependencies, is compiled here from the main package's own
// sources through the symlink Sources/OpenJevCore, and the tokenizer check that needs
// OpenJevDiffusionGemma is the separate package in TokenizerCheck/.

import PackageDescription

let package = Package(
    name: "OracleProbe",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(
            url: "https://github.com/Layr-Labs/mlx-swift-lm.git",
            revision: "eeba2afaf059a153ff909c9e01aa5e65b7bcad67")
    ],
    targets: [
        // ../../../../Sources/OpenJevCore, unchanged.
        .target(
            name: "OpenJevCore",
            path: "Sources/OpenJevCore",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "Probe",
            dependencies: [
                "OpenJevCore",
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            // Top-level MLX code in main.swift; the scratch probe stays in the Swift 5 mode.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
