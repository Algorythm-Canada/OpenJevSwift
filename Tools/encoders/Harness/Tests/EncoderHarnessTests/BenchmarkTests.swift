import EncoderHarness
import Foundation
import XCTest

/// The measurements, one test per package and compute-unit setting, so that a run can select them
/// one at a time (`-only-testing`) and a Core ML crash ends only its own test. On macOS,
/// Tools/encoders/run_macos.sh gives cleaner numbers (one process per configuration); these tests
/// are for the iOS Simulator, where only functional parity means anything.
///
/// They run only when ENCODER_HARNESS_BENCHMARK is 1 (pass TEST_RUNNER_ENCODER_HARNESS_BENCHMARK=1 to
/// xcodebuild) and the package has been converted; otherwise they skip. Each prints its result as
/// one line starting with HARNESS_RESULT and attaches the same JSON to the test.
final class BenchmarkTests: XCTestCase {
    /// The largest probability difference from PyTorch float32 that a float16 package may show:
    /// the bound docs/spikes/encoder-runtime.md proposes for #57 and #58. Every package these tests
    /// measure stayed under 0.015.
    static let maxProbabilityDifference = 0.02

    func measure(_ package: String, _ units: ComputeUnitsName) async throws {
        guard ProcessInfo.processInfo.environment["ENCODER_HARNESS_BENCHMARK"] == "1" else {
            throw XCTSkip("set ENCODER_HARNESS_BENCHMARK=1 to measure")
        }
        let locations = try requireTokenizers()
        let spec = try XCTUnwrap(PackageSpec.named(package))
        guard locations.hasPackage(named: package) else {
            throw XCTSkip("\(package).mlpackage has not been converted")
        }
        let result = try await runBenchmark(spec: spec, units: units, locations: locations)
        let json = try encodeResult(result)
        let attachment = XCTAttachment(data: json, uniformTypeIdentifier: "public.json")
        attachment.name = "\(package)-\(units.rawValue).json"
        attachment.lifetime = .keepAlways
        add(attachment)
        let line = try JSONEncoder().encode(result)
        print("HARNESS_RESULT " + String(decoding: line, as: UTF8.self))
        XCTAssertEqual(result.tokenizationMatches, "200/200")
        for (batch, parity) in [
            ("batch 1", result.parityBatch1), ("batch 16", result.parityBatch16),
        ] {
            guard let parity else { continue }
            XCTAssertEqual(parity.nonFinite, 0, "\(batch): non-finite probabilities")
            XCTAssertLessThanOrEqual(
                parity.maxAbsProbabilityDifference, Self.maxProbabilityDifference,
                "\(batch): probabilities too far from PyTorch float32")
        }
    }

    func testVerdictE17FP16CPUOnly() async throws {
        try await measure("verdict-e17-fp16", .cpuOnly)
    }
    func testVerdictE17FP16CPUAndGPU() async throws {
        try await measure("verdict-e17-fp16", .cpuAndGPU)
    }
    func testVerdictE17FP16CPUAndNeuralEngine() async throws {
        try await measure("verdict-e17-fp16", .cpuAndNeuralEngine)
    }
    func testVerdictE17FP16All() async throws { try await measure("verdict-e17-fp16", .all) }

    func testVerdictE17FP32CPUOnly() async throws {
        try await measure("verdict-e17-fp32", .cpuOnly)
    }
    func testVerdictE17FP32CPUAndGPU() async throws {
        try await measure("verdict-e17-fp32", .cpuAndGPU)
    }
    func testVerdictE17FP32CPUAndNeuralEngine() async throws {
        try await measure("verdict-e17-fp32", .cpuAndNeuralEngine)
    }
    func testVerdictE17FP32All() async throws { try await measure("verdict-e17-fp32", .all) }

    func testVerdictM18FP16CPUOnly() async throws {
        try await measure("verdict-m18-fp16", .cpuOnly)
    }
    func testVerdictM18FP16CPUAndGPU() async throws {
        try await measure("verdict-m18-fp16", .cpuAndGPU)
    }
    func testVerdictM18FP16CPUAndNeuralEngine() async throws {
        try await measure("verdict-m18-fp16", .cpuAndNeuralEngine)
    }
    func testVerdictM18FP16All() async throws { try await measure("verdict-m18-fp16", .all) }

    func testLayaE17FP16CPUOnly() async throws { try await measure("laya-e17-fp16", .cpuOnly) }
    func testLayaE17FP16CPUAndGPU() async throws { try await measure("laya-e17-fp16", .cpuAndGPU) }
    func testLayaE17FP16CPUAndNeuralEngine() async throws {
        try await measure("laya-e17-fp16", .cpuAndNeuralEngine)
    }
    func testLayaE17FP16All() async throws { try await measure("laya-e17-fp16", .all) }

    func testLayaM18FP16CPUOnly() async throws { try await measure("laya-m18-fp16", .cpuOnly) }
    func testLayaM18FP16CPUAndGPU() async throws { try await measure("laya-m18-fp16", .cpuAndGPU) }
    func testLayaM18FP16CPUAndNeuralEngine() async throws {
        try await measure("laya-m18-fp16", .cpuAndNeuralEngine)
    }
    func testLayaM18FP16All() async throws { try await measure("laya-m18-fp16", .all) }
}
