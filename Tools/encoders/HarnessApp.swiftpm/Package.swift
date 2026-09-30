// swift-tools-version: 5.9
// An iPhone app around Tools/encoders/Harness for spike #56. Xcode cannot host a package's test
// bundle on a device, so the device measurements run in this app instead: one launch measures the
// configurations named in its arguments, prints each result and exits. Tools/encoders/run_ios.sh
// builds, installs and launches it; set the signing team with DEVELOPMENT_TEAM.

import AppleProductTypes
import PackageDescription

let package = Package(
    name: "EncoderHarnessApp",
    platforms: [.iOS("17.0")],
    products: [
        .iOSApplication(
            name: "EncoderHarnessApp",
            targets: ["AppModule"],
            bundleIdentifier: "org.openjevswift.encoderharness",
            displayVersion: "1.0",
            bundleVersion: "1",
            supportedDeviceFamilies: [.phone, .pad],
            supportedInterfaceOrientations: [
                .portrait, .landscapeLeft, .landscapeRight,
                .portraitUpsideDown(.when(deviceFamilies: [.pad])),
            ]
        )
    ],
    dependencies: [
        .package(path: "../Harness")
    ],
    targets: [
        .executableTarget(
            name: "AppModule",
            dependencies: [.product(name: "EncoderHarness", package: "Harness")],
            path: "App",
            // Tools/encoders/stage_harness.sh fills it; git keeps only the README.
            resources: [.copy("Staged")]
        )
    ]
)
