#if canImport(OpenJevEncoders)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import OpenJevCore
    import OpenJevEncoders
    import OpenJevTestSupport
    import Testing

    /// `openjev serve` with `OPENJEV_BACKEND=verdict` as a child process, on an ephemeral port:
    /// Jev's quickstart request is answered with three answers and the server's headers, `decide`
    /// prints the same body, and SIGTERM ends the process cleanly (issue #40).
    ///
    /// Opt-in: it needs Verdict's converted package and tokenizer, which `OPENJEV_ENCODER_MODELS`
    /// points to, else ~/Library/Caches/OpenJevSwift/encoders and the Hugging Face cache, as the
    /// encoder tests find them. Without them it skips with a comment naming that variable.
    @Suite(
        "Live smoke test",
        .serialized,
        .enabled(if: VerdictFiles.available, VerdictFiles.missingMessage),
        .enabled(if: WireFixtures.exists("cases.json"), WireFixtures.missingMessage))
    struct LiveSmokeTests {
        /// The quickstart request of Fixtures/wire/cases.json.
        private func quickstart() throws -> Data {
            let row = try WireFixtures.recordedCase(named: "quickstart")
            let request = try #require(row["request"])
            return Data(try #require(try WireFixtures.bodyBytes(of: request)))
        }

        @Test("serve answers the quickstart, decide prints the same body, and SIGTERM exits 0")
        func serveVerdict() async throws {
            let environment = BuiltBinary.environment([
                "OPENJEV_BACKEND": "verdict", "OPENJEV_HOST": "127.0.0.1", "OPENJEV_PORT": "0",
                EncoderPackageStore.localModelsVariable: VerdictFiles.modelsDirectory.path,
            ])
            let child = try BuiltBinary.Child(arguments: ["serve"], environment: environment)
            let clock = ContinuousClock()
            let deadline = clock.now + .seconds(120)
            var port: Int?
            while port == nil, child.process.isRunning, clock.now < deadline {
                port = Self.port(in: child.errorsSoFar)
                try await Task.sleep(for: .milliseconds(50))
            }
            let serving = try #require(port, "no serving line: \(child.errorsSoFar)")

            let client = TestClient(host: "127.0.0.1", port: serving)
            client.connect()
            let body = try quickstart()
            let response = try await client.execute(
                TestClient.Request(
                    "/v1/systemone", method: .post, authority: "localhost",
                    headers: [.contentType: "application/json"],
                    body: ByteBuffer(bytes: Array(body))))
            let health = try await client.get("/health")
            try await client.shutdown()
            #expect(response.status == .ok)
            #expect(health.status == .ok)
            let bytes = Array(try #require(response.body).readableBytesView)
            let answer = try JSONParser().parse(bytes)
            #expect(answer["model"]?.stringValue == "verdict-1.4")
            #expect(
                answer["answers"]?.objectValue?.keys == ["department", "frustration", "is_urgent"])
            let timing = try #require(response.headers[HTTPField.Name("server-timing")!])
            #expect(timing.hasPrefix("model;dur="))
            let id = try #require(response.headers[HTTPField.Name("x-request-id")!])
            #expect(response.headers[HTTPField.Name("x-typesafe-request-id")!] == id)
            #expect(id.hasPrefix("req_") && id.count == 36)

            child.terminate()
            let ended = try #require(await child.wait(upTo: .seconds(30)), "no exit after SIGTERM")
            #expect(ended.status == 0, "\(ended.errors)")
            #expect(child.process.terminationReason == .exit)
            let lines = ended.errors
            print("openjev serve, standard error:\n\(lines)")
            for phase in [
                "settings: host=127.0.0.1 port=0 backend=verdict", "loading verdict-1.4",
                "warming up", "serving on 127.0.0.1:\(serving)",
                "POST /v1/systemone 200 ", "GET /health 200 ", "shutting down",
                "released verdict-1.4", "stopped",
            ] {
                #expect(lines.contains(phase), "\(phase)")
            }
            #expect(lines.contains(id))

            // decide gives the bytes the server sent.
            let decided = try await BuiltBinary.run(
                ["decide"], environment: environment, input: body, upTo: .seconds(120))
            #expect(decided.status == 0, "\(decided.errors)")
            #expect(Array(decided.standardOutput) == bytes)
        }

        /// The port of the `serving on 127.0.0.1:{port}` line, once it is there.
        static func port(in text: String) -> Int? {
            guard let range = text.range(of: "serving on 127.0.0.1:") else { return nil }
            return Int(text[range.upperBound...].prefix { $0.isNumber })
        }
    }

    /// Verdict's converted package and tokenizer, found as the encoder tests find them.
    enum VerdictFiles {
        /// The process environment.
        static let environment = ProcessInfo.processInfo.environment

        /// `OPENJEV_ENCODER_MODELS`, else the converters' cache.
        static let modelsDirectory: URL = {
            if let path = environment[EncoderPackageStore.localModelsVariable], !path.isEmpty {
                return URL(fileURLWithPath: path, isDirectory: true)
            }
            return URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library/Caches/OpenJevSwift/encoders", isDirectory: true)
        }()

        /// Whether the package and the tokenizer files are there.
        static var available: Bool {
            let manifest = EncoderPackageManifest.verdict
            let package = modelsDirectory.appendingPathComponent(
                manifest.package + ".mlpackage", isDirectory: true)
            let names = (manifest.tokenizerFiles + [manifest.calibrator]).map(\.path)
            let tokenizers = [
                modelsDirectory.appendingPathComponent(manifest.package, isDirectory: true)
                    .appendingPathComponent("tokenizer", isDirectory: true),
                manifest.checkpoint.snapshot(
                    in: EncoderPackageStore.huggingFaceHubDirectory(environment: environment)),
            ]
            return FileManager.default.fileExists(atPath: package.path)
                && tokenizers.contains { folder in
                    names.allSatisfy {
                        FileManager.default.fileExists(
                            atPath: folder.appendingPathComponent($0).path)
                    }
                }
        }

        /// The skip comment, which names the variable CI's log check accepts.
        static let missingMessage = Comment(
            rawValue: "\(modelsDirectory.path) (OPENJEV_ENCODER_MODELS, else the converters' "
                + "cache) has no verdict-m18-fp16.mlpackage, or Verdict's tokenizer is missing; "
                + "set OPENJEV_ENCODER_MODELS to a folder of converted packages")
    }
#endif
