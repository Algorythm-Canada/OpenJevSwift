import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import OpenJevCore
import OpenJevServer
import OpenJevTestSupport
import Testing

@testable import openjev

/// The three subcommands run in the test process, with stub backends registered through the
/// command context (issue #40): their exit statuses, their messages, `decide`'s bytes against the
/// server's, `models` without a load, and `serve` from the settings line to a clean shutdown.
@Suite(
    "Commands",
    .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage),
    .enabled(if: WireFixtures.exists("models.json"), WireFixtures.missingMessage))
struct CommandTests {
    /// Jev's quickstart request as the policy recording sent it.
    private func quickstart() throws -> [UInt8] {
        try WireEncoder().bytes(
            json: try #require(PolicyFixtures.policyCase(named: "plain")["request"]))
    }

    // MARK: Invalid settings and backends

    @Test(
        "An invalid OPENJEV_PORT exits 2 with upstream's message, whatever the subcommand",
        arguments: ["serve", "decide", "models"])
    func invalidPort(subcommand: String) async {
        let outcome = await CommandHarness.run(
            [subcommand], environment: ["OPENJEV_PORT": "eighty", "OPENJEV_BACKEND": "verdict"])
        #expect(outcome.status == 2)
        #expect(outcome.errors == "openjev: OPENJEV_PORT='eighty' is not a int\n")
        #expect(outcome.standardOutput.isEmpty)
    }

    @Test("--port is checked as OPENJEV_PORT is")
    func invalidPortFlag() async {
        let outcome = await CommandHarness.run(["serve", "--port", "80 80"])
        #expect(outcome.status == 2)
        #expect(outcome.errors == "openjev: OPENJEV_PORT='80 80' is not a int\n")
    }

    @Test(
        "An unknown OPENJEV_BACKEND exits 2 with upstream's message and this port's backends",
        arguments: ["serve", "decide", "models"])
    func unknownBackend(subcommand: String) async {
        for backend in ["vllm", "clm", "jevk5", "Verdict", ""] {
            let outcome = await CommandHarness.run(
                [subcommand], environment: ["OPENJEV_BACKEND": backend])
            #expect(outcome.status == 2, "\(backend)")
            #expect(
                outcome.errors
                    == "openjev: unknown backend '\(backend)'; use one of mlx, laya, verdict "
                    + "(OPENJEV_BACKEND)\n", "\(backend)")
        }
        let flagged = await CommandHarness.run([subcommand, "--backend", "nope"])
        #expect(flagged.status == 2)
        #expect(flagged.errors.contains("unknown backend 'nope'"))
    }

    @Test("Other invalid settings exit 2 with the message upstream's checks give")
    func otherInvalidSettings() async {
        let cases: [([String: String], String)] = [
            (
                ["OPENJEV_MAX_QUEUE": "-1"],
                "max_queue must not be negative, got -1 (OPENJEV_MAX_QUEUE)"
            ),
            (
                ["OPENJEV_CANVAS_STEP": "0"],
                "canvas_step must be at least 1, got 0 (OPENJEV_CANVAS_STEP)"
            ),
            (["OPENJEV_LOG_LEVEL": "loud"], "OPENJEV_LOG_LEVEL='loud' is not a log level"),
            (["OPENJEV_MODEL_ROUTES": "x"], "OPENJEV_MODEL_ROUTES: 'x' is not name=url"),
            (
                ["OPENJEV_ENCODER_FUNCTIONS": "0"],
                "OPENJEV_ENCODER_FUNCTIONS=0 is below the minimum of 1"
            ),
        ]
        for (environment, message) in cases {
            let outcome = await CommandHarness.run(
                ["serve", "--backend", "verdict"], environment: environment)
            #expect(outcome.status == 2, "\(environment)")
            #expect(outcome.errors.hasPrefix("openjev: " + message), "\(outcome.errors)")
        }
    }

    @Test(
        "mlx over a directory that is not a checkpoint exits 3; on Linux mlx is unavailable",
        arguments: ["serve", "decide"])
    func mlxNotACheckpoint(subcommand: String) async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-not-a-checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let environment = ["OPENJEV_MLX_MODEL": directory.path]
        let mlx = await CommandHarness.run(
            [subcommand, "--backend", "mlx"], environment: environment)
        #expect(mlx.status == 3)
        if canImportDiffusionGemma {
            #expect(
                mlx.errors.hasPrefix("openjev: openjev-0.1 failed to load (OPENJEV_BACKEND=mlx): "))
            #expect(mlx.errors.contains("config.json"))
        } else {
            // Linux: MLX does not exist, so mlx is known and unavailable.
            #expect(mlx.errors.hasPrefix("openjev: OPENJEV_BACKEND=mlx: "))
            #expect(mlx.errors.contains("Apple silicon"))
        }
        // The default backend is upstream's, mlx.
        #expect(await CommandHarness.run([subcommand], environment: environment).status == 3)
    }

    /// Checked without a load: Laya's package is 1.7 GB and Verdict's 306 MB.
    @Test("verdict and laya are this build's encoder backends, mlx its diffusion backend")
    func standardBackends() throws {
        let registry = BackendRegistry.standard
        #expect(registry.backends.map(\.name) == ["mlx", "laya", "verdict"])
        for name in ["laya", "verdict"] {
            let backend = try registry.backend(named: name)
            #expect(backend.kind == .encoder)
            #expect(
                backend.servedModels?.version == KnownEncoderModels.named(backend.modelName)?.name)
            switch backend.availability {
            case .available:
                #expect(canImportEncoders, "\(name) is available without Core ML")
            case .unavailable(let message):
                #expect(!canImportEncoders, "\(name): \(message)")
                #expect(message.hasPrefix("OPENJEV_BACKEND=\(name): "))
                #expect(message.contains("Core ML"))
            }
        }
        let mlx = try registry.backend(named: "mlx")
        #expect(mlx.kind == .diffusion)
        #expect(mlx.servedModels == .diffusionGemma)
        switch mlx.availability {
        case .available:
            #expect(canImportDiffusionGemma, "mlx is available without MLX")
        case .unavailable(let message):
            #expect(!canImportDiffusionGemma, "mlx: \(message)")
            #expect(message.hasPrefix("OPENJEV_BACKEND=mlx: "))
        }
    }

    /// Whether this build has the DiffusionGemma backend.
    private var canImportDiffusionGemma: Bool {
        #if canImport(OpenJevDiffusionGemma)
            return true
        #else
            return false
        #endif
    }

    /// Whether this build has the encoder backends.
    private var canImportEncoders: Bool {
        #if canImport(OpenJevEncoders)
            return true
        #else
            return false
        #endif
    }

    #if !canImport(OpenJevEncoders)
        @Test("verdict and laya exit 3 where Core ML does not exist")
        func encodersWithoutCoreML() async {
            for backend in ["verdict", "laya"] {
                let outcome = await CommandHarness.run(["decide", "--backend", backend])
                #expect(outcome.status == 3, "\(backend)")
                #expect(outcome.errors.contains("Core ML"), "\(backend)")
            }
        }
    #endif

    @Test("A backend that fails to load exits 3 with the error", arguments: ["serve", "decide"])
    func loadFailure(subcommand: String) async {
        let outcome = await CommandHarness.run(
            [subcommand, "--backend", "broken"], input: Array("{}".utf8),
            backends: CommandHarness.backends())
        #expect(outcome.status == 3)
        #expect(
            outcome.errors
                == "openjev: broken-1 failed to load (OPENJEV_BACKEND=broken): the weights are "
                + "missing\n")
        #expect(!outcome.logLines.contains { $0.contains("serving on") })
    }

    // MARK: decide

    @Test("decide prints the bytes the server sends for the same request")
    func decideMatchesTheServer() async throws {
        let body = try quickstart()
        let recorded = try #require(
            PolicyFixtures.policyCase(named: "plain")["response"]?["body_text"]?.stringValue)
        let stub = StubBackend()
        let outcome = await CommandHarness.run(
            ["decide", "--backend", "stub"], input: body,
            backends: CommandHarness.backends(diffusion: stub))
        #expect(outcome.status == 0, "\(outcome.errors)")
        #expect(outcome.output == recorded)
        #expect(outcome.standardError.isEmpty)
        #expect(stub.closeCount == 1)

        // The server's answer through its router, for the same request and settings.
        let service = try await DecisionBackendProvider { _ in StubBackend() }
            .makeService(settings: ServerSettings())
        let served = try await Application(
            router: OpenJevApplication.router(settings: ServerSettings(), service: service)
        ).test(.router) { client in
            try await client.execute(
                uri: "/v1/systemone", method: .post, headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: body))
        }
        #expect(served.status == .ok)
        #expect(Array(served.body.readableBytesView) == outcome.standardOutput)
    }

    @Test("decide reads --request, and - is standard input")
    func decideFromAFile() async throws {
        let body = try quickstart()
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-decide-\(UUID().uuidString).json")
        try Data(body).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let backends = CommandHarness.backends()
        let fromFile = await CommandHarness.run(
            ["decide", "--backend", "stub", "--request", file.path], backends: backends)
        let fromInput = await CommandHarness.run(
            ["decide", "--backend", "stub", "--request", "-"], input: body, backends: backends)
        #expect(fromFile.status == 0 && fromInput.status == 0)
        #expect(fromFile.standardOutput == fromInput.standardOutput)
        #expect(!fromFile.standardOutput.isEmpty)

        let missing = await CommandHarness.run(
            ["decide", "--backend", "stub", "--request", file.path + ".missing"],
            backends: backends)
        #expect(missing.status == 1)
        #expect(missing.errors.hasPrefix("openjev: cannot read \(file.path).missing: "))
    }

    @Test("A refused request prints the server's error body to standard error and exits 4")
    func decideRefusals() async throws {
        let cases: [(input: String, environment: [String: String], body: String)] = [
            (
                #"{"state":"x","model":"gpt-4","questions":{"a":{"type":"noul"}}}"#, [:],
                #"{"detail":{"error_type":"api_usage_error","message":"Unknown model: gpt-4"}}"#
            ),
            (
                #"{"state":"#, [:],
                #"{"detail":[{"type":"json_invalid","loc":["body",9],"msg":"JSON decode error","#
                    + #""input":{},"ctx":{"error":"Expecting value"}}]}"#
            ),
            (
                #"{"state":"x","model":"jev-latest","questions":{"a":{"type":"noul"},"b":{"type":"noul"}}}"#,
                ["OPENJEV_MAX_QUESTIONS": "1"],
                #"{"detail":"at most 1 questions per request"}"#
            ),
            (
                #"{"state":"x","model":"jev-latest","questions":{"a":{"type":"noul"}}}"#,
                ["OPENJEV_MAX_QUEUE": "0"],
                #"{"detail":{"error_type":"overloaded_error","message":"OpenJev is at capacity. "#
                    + #"Retry shortly."}}"#
            ),
            (
                #"{"state":"x","model":"jev-latest","questions":{"a":{"type":"noul"}}}"#,
                ["OPENJEV_MAX_BODY_BYTES": "16"],
                #"{"detail":{"error_type":"api_usage_error","message":"request body is larger "#
                    + #"than 16 bytes"}}"#
            ),
        ]
        let backends = CommandHarness.backends()
        for (input, environment, body) in cases {
            let outcome = await CommandHarness.run(
                ["decide", "--backend", "stub"], environment: environment,
                input: Array(input.utf8), backends: backends)
            #expect(outcome.status == 4, "\(input)")
            #expect(outcome.errors == body, "\(input)")
            #expect(outcome.standardOutput.isEmpty, "\(input)")
        }
        // An empty body is no body: pydantic's missing body, a refusal too.
        let empty = await CommandHarness.run(
            ["decide", "--backend", "stub"], input: [], backends: backends)
        #expect(empty.status == 4)
        #expect(empty.errors.hasPrefix(#"{"detail":[{"type":"missing","loc":["body"]"#))
    }

    @Test("A backend that fails during the decision prints the 503 body and exits 1")
    func decideBackendFailure() async throws {
        struct ConnectError: Error {}
        let stub = StubBackend(failure: ConnectError())
        let outcome = await CommandHarness.run(
            ["decide", "--backend", "stub"], input: try quickstart(),
            backends: CommandHarness.backends(diffusion: stub))
        #expect(outcome.status == 1)
        #expect(
            outcome.errors
                == #"{"detail":{"error_type":"api_error","message":"inference backend "#
                + #"unavailable: ConnectError"}}"#)
        #expect(stub.closeCount == 1)
    }

    @Test("decide skips the warm-up read")
    func decideWithoutWarmUp() async throws {
        let encoder = StubQuestionReadBackend()
        let outcome = await CommandHarness.run(
            ["decide", "--backend", "stub-encoder"], environment: ["OPENJEV_WARMUP": "1"],
            input: try quickstart(), backends: CommandHarness.backends(encoder: encoder))
        #expect(outcome.status == 0, "\(outcome.errors)")
        #expect(encoder.calls.count == 1)
        #expect(encoder.calls.first?.state != EncoderDecisionEngine.warmUpState)
    }

    // MARK: models

    @Test("models prints the recorded listing of every backend without loading a model")
    func modelsListings() async throws {
        let loads = LoadCounter()
        let backends = CommandHarness.backends(loads: loads)
        for backend in ["mlx", "laya", "verdict"] {
            let outcome = await CommandHarness.run(
                ["models", "--backend", backend], backends: backends)
            #expect(outcome.status == 0, "\(backend): \(outcome.errors)")
            #expect(
                outcome.output
                    == (try WireFixtures.listing(forBackend: backend)["body_text"]?.stringValue),
                "\(backend)")
        }
        let stub = await CommandHarness.run(["models", "--backend", "stub"], backends: backends)
        #expect(
            stub.output == (try WireFixtures.listing(forBackend: "mlx")["body_text"]?.stringValue))
        #expect(loads.count == 0)
    }

    @Test("models lists the routed models after the backend's, as the server does")
    func modelsWithRoutes() async throws {
        let listings = try #require(WireFixtures.load("models.json")["listings"]?.arrayValue)
        let routed = try #require(listings.first { $0["model_routes"] != nil })
        let routes = (routed["model_routes"]?.objectValue ?? [:]).map { name, url in
            "\(name)=\(url.stringValue ?? "")"
        }
        #expect(routes == ["verdict-1.4=http://verdict:8000", "custom-model=http://custom:8000"])
        let loads = LoadCounter()
        // The recording's backend is vllm, whose listing is mlx's; the routed hosts are not asked.
        let outcome = await CommandHarness.run(
            ["models", "--backend", "mlx"],
            environment: ["OPENJEV_MODEL_ROUTES": routes.joined(separator: ",")],
            backends: CommandHarness.backends(loads: loads))
        #expect(outcome.status == 0, "\(outcome.errors)")
        #expect(outcome.output == routed["body_text"]?.stringValue)
        #expect(loads.count == 0)
    }

    @Test("models loads a backend whose listing only the loaded model knows")
    func modelsWithALoad() async throws {
        let loads = LoadCounter()
        let encoder = StubQuestionReadBackend()
        let outcome = await CommandHarness.run(
            ["models", "--backend", "stub-encoder"],
            backends: CommandHarness.backends(encoder: encoder, loads: loads))
        #expect(outcome.status == 0)
        #expect(
            outcome.output
                == (try WireFixtures.listing(forBackend: "laya")["body_text"]?.stringValue))
        #expect(loads.count == 1)
        #expect(encoder.calls.isEmpty)
        #expect(encoder.closeCount == 1)
    }

    // MARK: serve

    @Test("serve prints its phases, logs each request and exits 0 after a graceful shutdown")
    func serve() async throws {
        let encoder = StubQuestionReadBackend()
        let started = Handoff<(port: Int, stop: @Sendable () async -> Void)>()
        let body = try quickstart()
        async let ran = CommandHarness.run(
            ["serve", "--backend", "stub-encoder", "--port", "0"],
            environment: [
                "OPENJEV_API_KEY": "sk-SECRET-KEY", "OPENJEV_ORIGIN_SECRET": "ORIGIN-SECRET",
                "OPENJEV_MODEL_ROUTES": "verdict-1.4=http://user:PASSWORD@127.0.0.1:9",
            ],
            backends: CommandHarness.backends(encoder: encoder),
            onServing: { port, stop in started.resolve((port, stop)) })
        let (port, stop) = await started.value
        let client = TestClient(host: "127.0.0.1", port: port)
        client.connect()
        let response = try await client.execute(
            TestClient.Request(
                "/v1/systemone", method: .post, authority: "localhost",
                headers: [
                    .contentType: "application/json", .authorization: "Bearer sk-SECRET-KEY",
                    HTTPField.Name("x-origin-secret")!: "ORIGIN-SECRET",
                ],
                body: ByteBuffer(bytes: body)))
        #expect(response.status == .ok)
        try await client.shutdown()
        await stop()
        let outcome = await ran
        #expect(outcome.status == 0, "\(outcome.errors)")
        #expect(outcome.standardError.isEmpty)
        let lines = outcome.logLines
        let expected = [
            "info settings: host=127.0.0.1 port=0 backend=stub-encoder log_level=info "
                + "warmup=on max_queue=512 max_questions=256 max_body_bytes=67108864 "
                + "encoder_batch=16 encoder_functions=all encoder_models=downloads api_key=set "
                + "origin_secret=set model_routes=verdict-1.4",
            "info forwarding verdict-1.4 to http://127.0.0.1:9",
            "info loading laya-1.0 (OPENJEV_BACKEND=stub-encoder)",
            "info warming up",
            "info serving on 127.0.0.1:\(port)",
        ]
        #expect(Array(lines.filter { !$0.contains("Server started") }.prefix(5)) == expected)
        #expect(lines.contains { $0.hasPrefix("info POST /v1/systemone 200 ") })
        #expect(lines.contains { $0.hasPrefix("info shutting down") })
        #expect(lines.contains("info released laya-1.0"))
        #expect(lines.last == "info stopped")
        for secret in ["SECRET-KEY", "ORIGIN-SECRET", "PASSWORD"] {
            #expect(!lines.contains { $0.contains(secret) }, "\(secret)")
        }
        // The warm-up read, then the request's.
        #expect(encoder.calls.count == 2)
        #expect(encoder.closeCount == 1)
    }

    @Test("The settings line gives an encoder's OPENJEV_ENCODER_FUNCTIONS, all when unset")
    func settingsLineEncoderFunctions() throws {
        let capped = try ServerSettings(environment: ["OPENJEV_ENCODER_FUNCTIONS": "3"])
        let encoder = SettingsSummary.line(capped, environment: [:], kind: .encoder)
        #expect(encoder.contains(" encoder_batch=16 encoder_functions=3 encoder_models="))
        let unset = SettingsSummary.line(try ServerSettings(), environment: [:], kind: .encoder)
        #expect(unset.contains(" encoder_functions=all "))
        let diffusion = SettingsSummary.line(capped, environment: [:], kind: .diffusion)
        #expect(!diffusion.contains("encoder_functions"))
    }

    @Test("serve logs a diffusion backend's MLX settings and re-read policy, at their defaults")
    func serveDiffusionSettingsLine() async throws {
        let started = Handoff<@Sendable () async -> Void>()
        async let ran = CommandHarness.run(
            ["serve", "--backend", "stub", "--port", "0"], backends: CommandHarness.backends(),
            onServing: { _, stop in started.resolve(stop) })
        let stop = await started.value
        await stop()
        let outcome = await ran
        #expect(outcome.status == 0, "\(outcome.errors)")
        let expected: String =
            "info settings: host=127.0.0.1 port=0 backend=stub log_level=info warmup=on "
            + "max_queue=512 max_questions=256 max_body_bytes=67108864 max_inflight=64 "
            + "canvas=64 mlx_model=mlx-community/diffusiongemma-26B-A4B-it-4bit "
            + "mlx_cache_limit_gb=unset mlx_prompt_cache=12 mlx_max_prompt=32768 "
            + "auto_threshold=0.1 auto_max=4 api_key=unset origin_secret=unset "
            + "model_routes=none"
        #expect(outcome.logLines.first == expected)
    }

    @Test(
        "The settings line gives OPENJEV_MLX_CACHE_LIMIT_GB in GB, 0 included, and unset if empty",
        arguments: [("4", "4"), ("0", "0"), ("2.5", "2.5"), ("16.0", "16"), ("", "unset")])
    func settingsLineCacheLimit(_ value: String, _ shown: String) throws {
        let settings = try ServerSettings(environment: ["OPENJEV_MLX_CACHE_LIMIT_GB": value])
        let line = SettingsSummary.line(settings, environment: [:], kind: .diffusion)
        #expect(line.contains(" mlx_cache_limit_gb=\(shown) mlx_prompt_cache=12 "))
        #expect(!SettingsSummary.line(settings, environment: [:], kind: .encoder).contains("mlx_"))
    }

    @Test("The settings line gives the prompt settings and the re-read policy as they are set")
    func settingsLinePromptsAndRereads() throws {
        let settings = try ServerSettings(environment: [
            "OPENJEV_MLX_PROMPT_CACHE": "0", "OPENJEV_MLX_MAX_PROMPT": "4096",
            "OPENJEV_AUTO_THRESHOLD": "0.25", "OPENJEV_AUTO_MAX": "1",
        ])
        let line = SettingsSummary.line(settings, environment: [:], kind: .diffusion)
        #expect(
            line.contains(
                " mlx_cache_limit_gb=unset mlx_prompt_cache=0 mlx_max_prompt=4096 "
                    + "auto_threshold=0.25 auto_max=1 api_key=unset "))
        let whole = try ServerSettings(environment: ["OPENJEV_AUTO_THRESHOLD": "2.0"])
        #expect(
            SettingsSummary.line(whole, environment: [:], kind: .diffusion)
                .contains(" auto_threshold=2 auto_max=4 "))
        #expect(!SettingsSummary.line(settings, environment: [:], kind: .encoder).contains("auto_"))
    }

    @Test("serve --no-warmup says so and skips the read")
    func serveWithoutWarmUp() async throws {
        let encoder = StubQuestionReadBackend()
        let started = Handoff<@Sendable () async -> Void>()
        async let ran = CommandHarness.run(
            ["serve", "--backend", "stub-encoder", "--port", "0", "--no-warmup"],
            backends: CommandHarness.backends(encoder: encoder),
            onServing: { _, stop in started.resolve(stop) })
        let stop = await started.value
        await stop()
        let outcome = await ran
        #expect(outcome.status == 0)
        #expect(outcome.logLines.contains("info warm-up skipped (OPENJEV_WARMUP=0)"))
        #expect(!outcome.logLines.contains("info warming up"))
        #expect(encoder.calls.isEmpty)
    }

    @Test(
        "serve exits 0 when nothing is in flight, even with a shutdown timeout of 0",
        arguments: ["0", "0.001"])
    func serveZeroTimeout(timeout: String) async throws {
        let encoder = StubQuestionReadBackend()
        let started = Handoff<@Sendable () async -> Void>()
        async let ran = CommandHarness.run(
            [
                "serve", "--backend", "stub-encoder", "--port", "0", "--no-warmup",
                "--shutdown-timeout", timeout,
            ],
            backends: CommandHarness.backends(encoder: encoder),
            onServing: { _, stop in started.resolve(stop) })
        let stop = await started.value
        await stop()
        let outcome = await ran
        #expect(outcome.status == 0, "\(outcome.errors)")
        #expect(outcome.standardError.isEmpty)
        #expect(encoder.closeCount == 1)
    }

    @Test("serve exits 1 when the shutdown timeout cuts a request short, after releasing")
    func serveShutdownTimeout() async throws {
        let gate = ReadGate()
        let encoder = StubQuestionReadBackend(gate: gate)
        let started = Handoff<(port: Int, stop: @Sendable () async -> Void)>()
        async let ran = CommandHarness.run(
            [
                "serve", "--backend", "stub-encoder", "--port", "0", "--no-warmup",
                "--shutdown-timeout", "0.2",
            ],
            backends: CommandHarness.backends(encoder: encoder),
            onServing: { port, stop in started.resolve((port, stop)) })
        let (port, stop) = await started.value
        let client = TestClient(host: "127.0.0.1", port: port)
        client.connect()
        try await client.executeAndDontWaitForResponse(
            TestClient.Request(
                "/v1/systemone", method: .post, authority: "localhost",
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: try quickstart())))
        await gate.waitForArrivals(1)
        await stop()
        let outcome = await ran
        #expect(outcome.status == 1)
        #expect(
            outcome.errors
                == "openjev: the requests in flight did not finish within the shutdown timeout "
                + "(0.2 s) and were cancelled\n")
        #expect(gate.cancellations == 1)
        #expect(encoder.closeCount == 1)
        try? await client.shutdown()
    }

    @Test("serve exits 1 when the address is taken")
    func serveAddressInUse() async throws {
        let first = Handoff<(port: Int, stop: @Sendable () async -> Void)>()
        let backends = CommandHarness.backends()
        async let running = CommandHarness.run(
            ["serve", "--backend", "stub-encoder", "--port", "0", "--no-warmup"],
            backends: backends, onServing: { port, stop in first.resolve((port, stop)) })
        let (port, stop) = await first.value
        let second = await CommandHarness.run(
            ["serve", "--backend", "stub-encoder", "--port", String(port), "--no-warmup"],
            backends: backends)
        #expect(second.status == 1)
        #expect(second.errors.hasPrefix("openjev: the server stopped: "))
        await stop()
        #expect(await running.status == 0)
    }
}
