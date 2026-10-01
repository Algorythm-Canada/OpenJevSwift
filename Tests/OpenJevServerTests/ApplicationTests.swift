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
                try ServerHarness.expectRecorded(response, recorded, "get_health")
            }
        }

        @Test("GET /v1/models lists the DiffusionGemma models as upstream does")
        func models() async throws {
            let recorded = try WireFixtures.recordedCase(named: "get_v1_models")
            let service = try await ServerHarness.diffusionService(ServerSettings())
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.send(client, .get, "/v1/models")
                try ServerHarness.expectRecorded(response, recorded, "get_v1_models")
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

        /// Upstream's `test_typesafe_sdk_default_model_is_accepted`: the SDK names `jev-latest`
        /// unless told otherwise, and an encoder backend answers it as its own model.
        @Test("An encoder backend answers the SDK's default model, jev-latest")
        func encoderAcceptsTheSDKDefault() async throws {
            let service = try await ServerHarness.encoderService()
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(
                    client, ModelRouteTests.asking("jev-latest"))
                #expect(response.status == .ok)
                let answer = try SystemOneResponse(
                    json: JSONParser().parse(Array(response.body.readableBytesView)))
                #expect(answer.model == "laya-1.0")
                guard case .choice(let choice, _, _) = answer.answers["team"] else {
                    Issue.record("no choice answer for team: \(answer.answers)")
                    return
                }
                #expect(choice == "billing")
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
        /// refuses, with upstream's default settings. The `body_*` and `auth_*` rows, which need
        /// other settings or other ways of sending, are ``ErrorContractTests``'.
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
                        headers: ServerHarness.headers(request["headers"]),
                        body: try WireFixtures.bodyBytes(of: request))
                    try ServerHarness.expectRecorded(response, row, name)
                }
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
    }
#endif
