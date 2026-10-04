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
    // Hummingbird's own dependency; the server names header fields with it directly.
    .package(url: "https://github.com/apple/swift-http-types.git", from: "1.8.0"),
    // Hummingbird's own dependencies too: the server logs with swift-log and watches each
    // connection with a swift-nio channel handler, the server tests capture log lines and hand
    // requests to the router over swift-nio's testing channel, and the CLI runs the server in a
    // swift-service-lifecycle service group that shuts it down on SIGINT and SIGTERM.
    .package(url: "https://github.com/apple/swift-log.git", from: "1.15.1"),
    .package(url: "https://github.com/apple/swift-nio.git", from: "2.103.0"),
    .package(url: "https://github.com/swift-server/swift-service-lifecycle.git", from: "2.12.0"),
    // Hummingbird's own dependency too, through HummingbirdTesting: the server forwards a request
    // for a routed model (OPENJEV_MODEL_ROUTES) to the server that serves it with this client,
    // which runs on swift-nio on macOS and Linux alike (decision D-040).
    .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.36.2"),
    // The `swift package generate-documentation` and `preview-documentation` commands, which
    // build the DocC catalogs of the library targets (issue #64, docs/development.md). A command
    // plugin only: no target links it.
    .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.5.0"),
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
            // For the HTTP/1 channel's configuration, where the connection watch is installed.
            .product(
                name: "HummingbirdCore",
                package: "hummingbird",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "HTTPTypes",
                package: "swift-http-types",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "Logging",
                package: "swift-log",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "NIOCore",
                package: "swift-nio",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "ServiceLifecycle",
                package: "swift-service-lifecycle",
                condition: .when(platforms: [.macOS, .linux])
            ),
            // The model routes: the client that forwards a request, on swift-nio's own event loops,
            // and the HTTP/1 headers it takes.
            .product(
                name: "AsyncHTTPClient",
                package: "async-http-client",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "NIOPosix",
                package: "swift-nio",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "NIOHTTP1",
                package: "swift-nio",
                condition: .when(platforms: [.macOS, .linux])
            ),
        ],
        swiftSettings: swiftSettings
    ),
    // The command line tool: serve, decide and models. On macOS it also links the encoder
    // backends; the block at the end appends that dependency, since OpenJevEncoders exists only
    // there, and the code that uses it is behind `#if canImport(OpenJevEncoders)`.
    .executableTarget(
        name: "openjev",
        dependencies: [
            "OpenJevCore",
            "OpenJevServer",
            .product(name: "ArgumentParser", package: "swift-argument-parser"),
            .product(name: "Logging", package: "swift-log"),
            .product(name: "ServiceLifecycle", package: "swift-service-lifecycle"),
            .product(name: "UnixSignals", package: "swift-service-lifecycle"),
        ],
        swiftSettings: swiftSettings
    ),
    // What the test targets share: the fixture loaders, the fixture-replaying tokenizer and the
    // stub backends. Test targets cannot import each other, so this is a library target, not a
    // product. Foundation only, so it builds wherever OpenJevCore does, iOS included.
    .target(
        name: "OpenJevTestSupport",
        dependencies: ["OpenJevCore"],
        swiftSettings: swiftSettings
    ),
    // The stub-backed server the SDK compatibility suite runs (Tools/sdk-compat, decision D-040):
    // the real application over OpenJevTestSupport's stub backends, so it runs on Linux, where Core
    // ML does not exist. It is not a product and never ships; `openjev` has no stub backend.
    .executableTarget(
        name: "openjev-stub-server",
        dependencies: [
            "OpenJevCore",
            "OpenJevServer",
            "OpenJevTestSupport",
            .product(name: "Logging", package: "swift-log"),
            .product(name: "ServiceLifecycle", package: "swift-service-lifecycle"),
            .product(name: "UnixSignals", package: "swift-service-lifecycle"),
        ],
        swiftSettings: swiftSettings
    ),
    .testTarget(
        name: "OpenJevCoreTests",
        dependencies: ["OpenJevCore", "OpenJevTestSupport"],
        swiftSettings: swiftSettings
    ),
    // The CLI's tests import the executable's module, and run the built `openjev` binary.
    .testTarget(
        name: "OpenJevCLITests",
        dependencies: [
            "openjev",
            "OpenJevServer",
            "OpenJevCore",
            "OpenJevTestSupport",
            .product(name: "ArgumentParser", package: "swift-argument-parser"),
            .product(name: "HTTPTypes", package: "swift-http-types"),
            .product(name: "Hummingbird", package: "hummingbird"),
            .product(name: "HummingbirdTesting", package: "hummingbird"),
            .product(name: "Logging", package: "swift-log"),
        ],
        swiftSettings: swiftSettings
    ),
    .testTarget(
        name: "OpenJevServerTests",
        dependencies: [
            "OpenJevServer",
            "OpenJevCore",
            "OpenJevTestSupport",
            .product(
                name: "HummingbirdTesting",
                package: "hummingbird",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "HTTPTypes",
                package: "swift-http-types",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "Logging",
                package: "swift-log",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "NIOEmbedded",
                package: "swift-nio",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "NIOCore",
                package: "swift-nio",
                condition: .when(platforms: [.macOS, .linux])
            ),
            .product(
                name: "ServiceLifecycle",
                package: "swift-service-lifecycle",
                condition: .when(platforms: [.macOS, .linux])
            ),
            // The forwarding client's errors, which the route tests name.
            .product(
                name: "AsyncHTTPClient",
                package: "async-http-client",
                condition: .when(platforms: [.macOS, .linux])
            ),
        ],
        swiftSettings: swiftSettings
    ),
    // The live end-to-end suite (issue #41), a port of upstream's tests/test_live.py: plain HTTP
    // through Foundation's URLSession (FoundationNetworking on Linux) to whatever server
    // OPENJEV_LIVE_URL names, so it links no server and no backend and runs on macOS and Linux
    // alike. Without OPENJEV_LIVE_URL the live tests skip; the tests of its settings and client,
    // which read upstream's recorded listings through OpenJevTestSupport, run everywhere. It is
    // not in the iOS scheme.
    .testTarget(
        name: "OpenJevLiveTests",
        dependencies: ["OpenJevCore", "OpenJevTestSupport"],
        swiftSettings: swiftSettings
    ),
]

// OpenJevDiffusionGemma and OpenJevLetterReadout run on MLX, which needs Apple silicon, and
// OpenJevEncoders runs on Core ML, which exists only on Apple platforms. SwiftPM evaluates this
// manifest on the host, and every Apple platform build runs on a macOS host, so the MLX and
// swift-transformers packages and the targets that use them are declared only there. A Linux
// build never loads their manifests or builds them.
#if os(macOS)
    products += [
        .library(name: "OpenJevDiffusionGemma", targets: ["OpenJevDiffusionGemma"]),
        .library(name: "OpenJevEncoders", targets: ["OpenJevEncoders"]),
        .library(name: "OpenJevLetterReadout", targets: ["OpenJevLetterReadout"]),
    ]
    dependencies += [
        // Exact versions: the read parity tests (D-014, D-048) and the regression file hold for
        // these two releases only. A version, unlike a revision, lets another package depend
        // on this one by version (#65).
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.32.3"),
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
            // The licences of the code Vision/ ports (Pillow, libjpeg-turbo), kept with it.
            exclude: ["Vision/ThirdPartyLicenses"],
            swiftSettings: swiftSettings
        ),
        // The tests also load the tokenizer through mlx-swift-lm's MLXHuggingFace macros, to
        // confirm that path gives the same results as the direct one the module uses, and run
        // MLXVLM's Gemma 4 image processor on the vision fixtures to compare it with Pillow's. The
        // live runtime tests take their requests from the wire fixtures through OpenJevTestSupport.
        .testTarget(
            name: "OpenJevDiffusionGemmaTests",
            dependencies: [
                "OpenJevDiffusionGemma",
                "OpenJevCore",
                "OpenJevTestSupport",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: swiftSettings
        ),
        // Verdict and Laya on Core ML (decision D-011): the prompts, the tokenizers, the
        // calibrations, the Core ML runners and the package store. No MLX. Its Core ML types need
        // macOS 15 and iOS 18, which the packages require; the package keeps its macOS 14 and
        // iOS 17 floors.
        .target(
            name: "OpenJevEncoders",
            dependencies: [
                "OpenJevCore",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "OpenJevEncodersTests",
            dependencies: ["OpenJevEncoders", "OpenJevCore", "OpenJevTestSupport"],
            swiftSettings: swiftSettings
        ),
        // JevK5 (`jevk5-0.2`, issue #55): the jevk5 package's prompt and many-option readout,
        // and the letter logits of mlx-swift-lm's Qwen3.5 text model. The checkpoint comes
        // through OpenJevDiffusionGemma's ModelResolver, the downloader that shares the Hugging
        // Face cache with Python (D-052).
        .target(
            name: "OpenJevLetterReadout",
            dependencies: [
                "OpenJevCore",
                "OpenJevDiffusionGemma",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: swiftSettings
        ),
        // The model-free tests (the prompt and readout against the jevk5 package's own output,
        // the backend over a stub model, through the server as upstream's tests run it) and,
        // behind OPENJEV_JEVK5_MODEL, the converted checkpoint's reads.
        .testTarget(
            name: "OpenJevLetterReadoutTests",
            dependencies: [
                "OpenJevLetterReadout", "OpenJevCore", "OpenJevTestSupport",
                "OpenJevDiffusionGemma", "OpenJevServer",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdTesting", package: "hummingbird"),
                .product(name: "HTTPTypes", package: "swift-http-types"),
                // GPU.metallib, which the live tests set before the first MLX call (issue #8).
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: swiftSettings
        ),
        // The performance and memory baseline of DiffusionGemma reads (issue #32,
        // docs/benchmarks.md): the engine in process, or any /v1/systemone server over HTTP. Not
        // a product; it never ships. Its tests cover the model-free statistics through
        // `@testable import`, as OpenJevCLITests does for `openjev`.
        .executableTarget(
            name: "openjev-bench",
            dependencies: [
                "OpenJevCore",
                "OpenJevDiffusionGemma",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "OpenJevBenchTests",
            dependencies: [
                "openjev-bench",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: swiftSettings
        ),
    ]
    // `openjev serve` with OPENJEV_BACKEND=verdict, laya, jevk5 or mlx, and its opt-in smoke
    // test, which finds the converted package the way the store does.
    for target in targets where ["openjev", "OpenJevCLITests"].contains(target.name) {
        target.dependencies.append("OpenJevEncoders")
        target.dependencies.append("OpenJevDiffusionGemma")
        target.dependencies.append("OpenJevLetterReadout")
    }
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
