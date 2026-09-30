import EncoderHarness
import Foundation
import XCTest

/// Where the tests find their inputs: the repository, the Hugging Face cache and the
/// converted-model cache on the Mac. The iOS Simulator runs on the Mac and reads them too.
func harnessLocations() -> HarnessLocations? {
    #if os(macOS) || targetEnvironment(simulator)
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // EncoderHarnessTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // Harness
            .deletingLastPathComponent()  // encoders
            .deletingLastPathComponent()  // Tools
            .deletingLastPathComponent()  // repository root
        return HarnessLocations.repository(
            root: root, environment: ProcessInfo.processInfo.environment)
    #else
        return nil
    #endif
}

func requireFixtures() throws -> HarnessLocations {
    guard let locations = harnessLocations(), locations.hasFixtures else {
        throw XCTSkip("Fixtures/encoders is not reachable from this destination")
    }
    return locations
}

func requireTokenizers() throws -> HarnessLocations {
    let locations = try requireFixtures()
    guard locations.hasTokenizers else {
        throw XCTSkip(
            "the Verdict and Laya tokenizers are not in the Hugging Face cache; run Tools/encoders/reference.py once"
        )
    }
    return locations
}
