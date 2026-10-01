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

    @Test("A backend this build lacks exits 3 naming its issue")
    func backendNotBuilt() async throws {
        let outcome = try await BuiltBinary.run(
            ["serve", "--backend", "laya"], environment: BuiltBinary.environment())
        #expect(outcome.status == 3)
        #expect(outcome.errors.contains("OPENJEV_BACKEND=laya"))
        #expect(outcome.errors.contains("issue #58"))
        let mlx = try await BuiltBinary.run(["serve"], environment: BuiltBinary.environment())
        #expect(mlx.status == 3)
        #expect(mlx.errors.contains("issue #29"))
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
