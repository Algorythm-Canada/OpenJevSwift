import Foundation
import OpenJevTestSupport
import Testing

/// The built `openjev` binary run as a child process, which is how launchd and scripts see it:
/// its exit statuses and what it prints. The opt-in smoke test serves Verdict
/// (``LiveSmokeTests``).
@Suite("The openjev binary")
struct BinaryTests {
    @Test("The binary was built beside the tests")
    func exists() {
        #expect(BuiltBinary.url != nil, Comment(rawValue: BuiltBinary.missingMessage))
    }

    @Test("--version prints the package version and exits 0")
    func version() async throws {
        let outcome = try await BuiltBinary.run(
            ["--version"], environment: BuiltBinary.environment())
        #expect(outcome.status == 0)
        #expect(outcome.output == "0.1.0-dev\n")
    }

    @Test("Invalid settings exit 2 with upstream's message naming the variable")
    func invalidSettings() async throws {
        let port = try await BuiltBinary.run(
            ["serve"], environment: BuiltBinary.environment(["OPENJEV_PORT": "eighty"]))
        #expect(port.status == 2)
        #expect(port.errors == "openjev: OPENJEV_PORT='eighty' is not a int\n")
        #expect(port.standardOutput.isEmpty)

        let backend = try await BuiltBinary.run(
            ["decide"], environment: BuiltBinary.environment(["OPENJEV_BACKEND": "vllm"]))
        #expect(backend.status == 2)
        #expect(
            backend.errors
                == "openjev: unknown backend 'vllm'; use one of mlx, laya, verdict "
                + "(OPENJEV_BACKEND)\n")

        let usage = try await BuiltBinary.run(
            ["serve", "--prot", "8080"], environment: BuiltBinary.environment())
        #expect(usage.status == 2)
        #expect(usage.errors.contains("Unknown option '--prot'"))
    }

    @Test(
        "mlx over a non-checkpoint directory exits 3 naming what it lacks; on Linux mlx is unavailable"
    )
    func mlxNotACheckpoint() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-not-a-checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let environment = BuiltBinary.environment(["OPENJEV_MLX_MODEL": directory.path])
        let outcome = try await BuiltBinary.run(
            ["serve", "--backend", "mlx"], environment: environment)
        #if canImport(OpenJevDiffusionGemma)
            let expected = "config.json"
        #else
            // Linux: MLX does not exist, so mlx is known and unavailable.
            let expected = "Apple silicon"
        #endif
        #expect(outcome.status == 3)
        #expect(outcome.errors.contains("OPENJEV_BACKEND=mlx"))
        #expect(outcome.errors.contains(expected))
        // mlx is the default backend, as upstream's settings have it in this port.
        let byDefault = try await BuiltBinary.run(["serve"], environment: environment)
        #expect(byDefault.status == 3)
        #expect(byDefault.errors.contains(expected))
    }

    @Test(
        "models prints the recorded listing without loading a model",
        .enabled(if: WireFixtures.exists("models.json"), WireFixtures.missingMessage))
    func models() async throws {
        for backend in ["verdict", "laya", "mlx"] {
            let outcome = try await BuiltBinary.run(
                ["models", "--backend", backend], environment: BuiltBinary.environment())
            #expect(outcome.status == 0, "\(backend): \(outcome.errors)")
            #expect(
                outcome.output
                    == (try WireFixtures.listing(forBackend: backend)["body_text"]?.stringValue),
                "\(backend)")
        }
    }
}
