// swift-tools-version: 6.0
// The JevSwiftSDK driver of the SDK compatibility suite (Tools/sdk-compat, issue #39). A package of
// its own, as the encoder harness is: the main package never builds it, and run.py builds it when
// asked with --swift-sdk.

import PackageDescription

let package = Package(
    name: "JevSwiftSDKChecks",
    platforms: [.macOS(.v13)],
    dependencies: [
        // NSStudent/JevSwiftSDK at its 0.1.0 tag, by commit, so the tag cannot move under us.
        .package(
            url: "https://github.com/NSStudent/JevSwiftSDK.git",
            revision: "ce35d203bbadbc78308e7605668ab0c8bbd446f6")
    ],
    targets: [
        .executableTarget(
            name: "jev-swift-sdk-checks",
            dependencies: [.product(name: "JevSwiftSDK", package: "JevSwiftSDK")],
            swiftSettings: [.swiftLanguageMode(.v6)])
    ]
)
