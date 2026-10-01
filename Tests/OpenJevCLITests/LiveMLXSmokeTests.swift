#if canImport(OpenJevDiffusionGemma)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import OpenJevCore
    import OpenJevDiffusionGemma
    import OpenJevTestSupport
    import Testing

    /// `openjev serve` with `OPENJEV_BACKEND=mlx` as a child process over the pinned checkpoint's
    /// directory: the binary finds MLX's Metal library, loads and warms the runtime, answers the
    /// quickstart with three answers, and exits 0 on SIGTERM (issue #29).
    ///
    /// Opt-in: it needs the 16 GB checkpoint, which `OPENJEV_TEST_MODEL` names, else the Hugging
    /// Face cache snapshot. Without it it skips with a comment naming that variable.
    @Suite(
        "Live smoke test, DiffusionGemma",
        .serialized,
        .enabled(if: MLXCheckpoint.available, MLXCheckpoint.missingMessage),
        .enabled(if: WireFixtures.exists("cases.json"), WireFixtures.missingMessage))
    struct LiveMLXSmokeTests {
        @Test("serve --backend mlx answers the quickstart and SIGTERM exits 0")
        func serveMLX() async throws {
            let environment = BuiltBinary.environment([
                "OPENJEV_BACKEND": "mlx", "OPENJEV_HOST": "127.0.0.1", "OPENJEV_PORT": "0",
                "OPENJEV_MLX_MODEL": MLXCheckpoint.directory.path,
            ])
            let child = try BuiltBinary.Child(arguments: ["serve"], environment: environment)
            defer { child.terminateIfRunning() }
            let clock = ContinuousClock()
            let started = clock.now
            let deadline = started + .seconds(600)
            var port: Int?
            while port == nil, child.process.isRunning, clock.now < deadline {
                port = LiveSmokeTests.port(in: child.errorsSoFar)
                try await Task.sleep(for: .milliseconds(100))
            }
            let serving = try #require(port, "no serving line: \(child.errorsSoFar)")
            print("openjev serve --backend mlx was serving after \(clock.now - started)")

            let row = try WireFixtures.recordedCase(named: "quickstart")
            let request = try #require(row["request"])
            let body = try #require(try WireFixtures.bodyBytes(of: request))
            let client = TestClient(host: "127.0.0.1", port: serving)
            client.connect()
            let response = try await client.execute(
                TestClient.Request(
                    "/v1/systemone", method: .post, authority: "localhost",
                    headers: [.contentType: "application/json"],
                    body: ByteBuffer(bytes: body)))
            try await client.shutdown()
            #expect(response.status == .ok)
            let answer = try JSONParser().parse(
                Array(try #require(response.body).readableBytesView))
            #expect(answer["model"]?.stringValue == "openjev-0.1")
            #expect(
                answer["answers"]?.objectValue?.keys == ["department", "frustration", "is_urgent"])
            #expect(answer["answers"]?["department"]?["choice"]?.stringValue == "technical")

            child.terminate()
            let ended = try #require(await child.wait(upTo: .seconds(60)), "no exit after SIGTERM")
            #expect(ended.status == 0, "\(ended.errors)")
            print("openjev serve --backend mlx, standard error:\n\(ended.errors)")
            for phase in [
                "backend=mlx", "loading openjev-0.1", "warming up",
                "serving on 127.0.0.1:\(serving)", "POST /v1/systemone 200 ", "stopped",
            ] {
                #expect(ended.errors.contains(phase), "\(phase)")
            }
        }
    }

    /// The pinned checkpoint, found as the DiffusionGemma tests find it.
    enum MLXCheckpoint {
        /// `OPENJEV_TEST_MODEL`, else the Hugging Face cache snapshot of the pinned revision.
        static let directory: URL = {
            if let path = ProcessInfo.processInfo.environment["OPENJEV_TEST_MODEL"], !path.isEmpty {
                return URL(fileURLWithPath: path, isDirectory: true)
            }
            guard case .hub(let repository, let revision?) = ModelSource.fourBit else {
                return URL(fileURLWithPath: "/nonexistent")
            }
            return HubCacheLocation(environment: [:]).snapshotDirectory(
                repository, commit: revision)
        }()

        /// Whether the directory holds the configuration, the index and every shard.
        static var available: Bool {
            ModelResolver.missingFiles(in: directory).isEmpty
        }

        /// The skip comment, which names the variable CI's log check accepts.
        static let missingMessage = Comment(
            rawValue: "OPENJEV_TEST_MODEL is unset and \(directory.path) lacks the checkpoint; "
                + "set OPENJEV_TEST_MODEL to the diffusiongemma-26B-A4B-it-4bit directory")
    }
#endif
