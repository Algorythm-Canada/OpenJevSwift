#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// The routes and headers of ``OpenJevApplication`` through Hummingbird's test client,
    /// against upstream's recorded exchanges in Fixtures/wire and Fixtures/policies.
    @Suite(
        "Application",
        .enabled(if: WireFixtures.exists("cases.json"), WireFixtures.missingMessage),
        .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    struct ApplicationTests {
        @Test("GET /health returns upstream's body and headers")
        func health() async throws {
            let recorded = try WireFixtures.recordedCase(named: "get_health")
            let service = try await ServerHarness.diffusionService(ServerSettings())
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.send(client, .get, "/health")
                try Self.expectRecorded(response, recorded, "get_health")
            }
        }

        @Test("GET /v1/models lists the DiffusionGemma models as upstream does")
        func models() async throws {
            let recorded = try WireFixtures.recordedCase(named: "get_v1_models")
            let service = try await ServerHarness.diffusionService(ServerSettings())
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.send(client, .get, "/v1/models")
                try Self.expectRecorded(response, recorded, "get_v1_models")
            }
        }

        @Test("GET /v1/models lists an encoder backend's model as upstream does")
        func encoderModels() async throws {
            let listing = try WireFixtures.listing(forBackend: "laya")
            let service = try await ServerHarness.encoderService()
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.send(client, .get, "/v1/models")
                #expect(response.status == .ok)
                #expect(ServerHarness.text(response) == listing["body_text"]?.stringValue)
                ServerHarness.expectServerHeaders(response, "laya models")
            }
        }

        /// The policy recording's `plain` case is the README quickstart.
        @Test("The quickstart request returns the recorded response byte for byte")
        func quickstart() async throws {
            let row = try PolicyFixtures.policyCase(named: "plain")
            let request = try #require(row["request"])
            let expected = try #require(row["response"]?["body_text"]?.stringValue)
            let service = try await ServerHarness.diffusionService(ServerSettings())
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(client, request)
                #expect(response.status == .ok)
                #expect(ServerHarness.text(response) == expected)
                #expect(ServerHarness.header(response, "content-type") == "application/json")
                ServerHarness.expectServerHeaders(response, "quickstart")
            }
        }

        @Test("Every policy case served over HTTP returns its recorded body")
        func policyCases() async throws {
            for row in try UpstreamFixtures.cases(PolicyFixtures.policies) {
                let name = row["name"]?.stringValue ?? "?"
                let configuration = try PolicyFixtures.configuration(row["settings"])
                let settings = try ServerSettings(
                    canvas: configuration.geometry.canvas,
                    canvasStep: configuration.geometry.step)
                let service = try await ServerHarness.diffusionService(settings)
                let request = try #require(row["request"])
                let expected = row["response"]?["body_text"]?.stringValue
                try await ServerHarness.withClient(settings: settings, service: service) { client in
                    let response = try await ServerHarness.post(client, request)
                    #expect(response.status == .ok, "\(name): status")
                    #expect(ServerHarness.text(response) == expected, "\(name): body")
                }
            }
        }

        /// The recorded 400s and 422s that end before the engine reads, or that the engine
        /// refuses, with upstream's default settings. Malformed and non-JSON bodies and the body
        /// cap are issue #35's; authentication is issue #36's.
        @Test("Recorded refusals come back with upstream's status, body and headers")
        func recordedRefusals() async throws {
            let cases = try #require(WireFixtures.load("cases.json")["cases"]?.arrayValue)
            let refusals = cases.filter { row in
                let name = row["name"]?.stringValue ?? ""
                let status = row["response"]?["status"]?.intValue ?? 0
                let settings = row["settings"]?.objectValue?.keys.map { $0 } ?? []
                return [400, 422].contains(status) && settings.allSatisfy { $0 == "upstream" }
                    && !name.hasPrefix("body_") && !name.hasPrefix("auth_")
            }
            #expect(refusals.count > 100)
            let service = try await ServerHarness.diffusionService(ServerSettings())
            try await ServerHarness.withClient(service: service) { client in
                for row in refusals {
                    let name = row["name"]?.stringValue ?? "?"
                    let request = try #require(row["request"])
                    let response = try await ServerHarness.send(
                        client, .post, try #require(request["path"]?.stringValue),
                        headers: Self.headers(request["headers"]),
                        body: try WireFixtures.bodyBytes(of: request))
                    try Self.expectRecorded(response, row, name)
                }
            }
        }

        @Test("An empty body is FastAPI's missing-body 422, with the headers")
        func emptyBody() async throws {
            let recorded = try WireFixtures.recordedCase(named: "body_empty")
            let service = try await ServerHarness.diffusionService(ServerSettings())
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.send(
                    client, .post, "/v1/systemone",
                    headers: ["content-type": "application/json"], body: [])
                try Self.expectRecorded(response, recorded, "body_empty")
            }
        }

        @Test("A full queue is the 529 with retry-after 1 and the request id")
        func overloaded() async throws {
            let settings = try ServerSettings(maxQueue: 0)
            let service = try await ServerHarness.diffusionService(settings)
            let request = try #require(PolicyFixtures.policyCase(named: "plain")["request"])
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                let response = try await ServerHarness.post(client, request)
                #expect(response.status.code == 529)
                #expect(
                    ServerHarness.text(response)
                        == #"{"detail":{"error_type":"overloaded_error","message":"#
                        + #""OpenJev is at capacity. Retry shortly."}}"#)
                #expect(ServerHarness.header(response, "retry-after") == "1")
                ServerHarness.expectServerHeaders(response, "529")
            }
        }

        /// Upstream answers the 413 from its middleware without `server-timing`; this server
        /// still sets it, which issue #35 settles along with the rest of the body cap.
        @Test("A body over the cap is the 413, with the request ids")
        func bodyCap() async throws {
            let settings = try ServerSettings(maxBodyBytes: 16)
            let service = try await ServerHarness.diffusionService(settings)
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                let response = try await ServerHarness.send(
                    client, .post, "/v1/systemone",
                    headers: ["content-type": "application/json"],
                    body: Array(repeating: UInt8(ascii: " "), count: 17))
                #expect(response.status == .contentTooLarge)
                #expect(
                    ServerHarness.text(response)
                        == #"{"detail":{"error_type":"api_usage_error","message":"#
                        + #""request body is larger than 16 bytes"}}"#)
                let id = try #require(ServerHarness.header(response, "x-request-id"))
                #expect(ServerHarness.isRequestID(id))
                #expect(ServerHarness.header(response, "x-typesafe-request-id") == id)
            }
        }

        @Test("An unknown route is FastAPI's 404, with the headers")
        func notFound() async throws {
            let service = try await ServerHarness.diffusionService(ServerSettings())
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.send(client, .get, "/v1/nope")
                #expect(response.status == .notFound)
                #expect(ServerHarness.text(response) == #"{"detail":"Not Found"}"#)
                ServerHarness.expectServerHeaders(response, "404")
            }
        }

        @Test("Each response gets its own request id")
        func distinctRequestIDs() async throws {
            let service = try await ServerHarness.diffusionService(ServerSettings())
            try await ServerHarness.withClient(service: service) { client in
                var ids: Set<String> = []
                for _ in 0..<20 {
                    let response = try await ServerHarness.send(client, .get, "/health")
                    ids.insert(try #require(ServerHarness.header(response, "x-request-id")))
                }
                #expect(ids.count == 20)
            }
        }

        @Test("server-timing reports the engine's model time")
        func serverTimingCountsTheRead() async throws {
            let backend = StubBackend(delay: .milliseconds(30))
            let service = try await ServerHarness.diffusionService(
                ServerSettings(), backend: backend)
            let request = try #require(PolicyFixtures.policyCase(named: "plain")["request"])
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(client, request)
                #expect(response.status == .ok)
                let timing = try #require(
                    ServerHarness.header(response, "server-timing").flatMap(ServerTimingValue.init))
                #expect(timing.model >= 30)
                #expect(timing.total >= 30)
                #expect(timing.server >= 0)
            }
        }

        @Test("server-timing formats with one decimal and never reports negative server time")
        func serverTimingFormat() {
            #expect(
                ServerTiming.header(model: .microseconds(12_340), total: .microseconds(20_060))
                    == "model;dur=12.3, server;dur=7.7, total;dur=20.1")
            #expect(
                ServerTiming.header(model: .milliseconds(5), total: .milliseconds(3))
                    == "model;dur=5.0, server;dur=0.0, total;dur=3.0")
            #expect(
                ServerTiming.header(model: .zero, total: .zero)
                    == "model;dur=0.0, server;dur=0.0, total;dur=0.0")
        }

        @Test("Request ids are req_ and 32 lowercase hex characters")
        func requestIDFormat() {
            for _ in 0..<100 {
                #expect(ServerHarness.isRequestID(RequestIdentifier.make()))
            }
        }

        /// Checks status, body bytes, `content-type`, `retry-after` and the per-response headers
        /// against a recorded case.
        private static func expectRecorded(
            _ response: TestResponse, _ recorded: JSONValue, _ label: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            let expected = try #require(recorded["response"])
            #expect(
                Int(response.status.code) == expected["status"]?.intValue, "\(label): status",
                sourceLocation: sourceLocation)
            #expect(
                ServerHarness.text(response) == expected["body_text"]?.stringValue,
                "\(label): body", sourceLocation: sourceLocation)
            for name in ["content-type", "retry-after"] {
                #expect(
                    ServerHarness.header(response, name)
                        == expected["headers"]?[name]?.stringValue,
                    "\(label): \(name)", sourceLocation: sourceLocation)
            }
            ServerHarness.expectServerHeaders(response, label, sourceLocation: sourceLocation)
        }

        /// A recorded request's headers.
        private static func headers(_ value: JSONValue?) -> [String: String] {
            var headers: [String: String] = [:]
            for (name, value) in value?.objectValue ?? [:] {
                headers[name] = value.stringValue
            }
            return headers
        }
    }
#endif
