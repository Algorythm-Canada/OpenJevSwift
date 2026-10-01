#if canImport(HummingbirdTesting)
    import AsyncHTTPClient
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import Logging
    import NIOCore
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// `OPENJEV_MODEL_ROUTES` (issue #38) against upstream's `forward`, its recordings in
    /// Fixtures/errors and Fixtures/wire/models.json, and `test_routes_forward_other_models` in
    /// `tests/test_encoders.py`, with an in-process server standing in for the routed one
    /// (``RoutedTarget``).
    @Suite(
        "Model routes",
        .enabled(if: WireFixtures.exists("models.json"), WireFixtures.missingMessage),
        .enabled(if: UpstreamFixtures.exists("errors/cases.json"), UpstreamFixtures.missingMessage)
    )
    struct ModelRouteTests {
        /// The request of upstream's `tests/test_encoders.py`, which asks Laya.
        static let encoderRequest: JSONValue = [
            "state": "I was charged twice this month.",
            "model": "laya-1.0",
            "questions": [
                "team": [
                    "type": "choice", "instructions": "Which team should handle it?",
                    "criteria": [
                        "outage": "service down", "billing": "charges, refunds", "feature": .null,
                    ],
                ],
                "tone": [
                    "type": "score", "instructions": "How upset is the customer?",
                    "criteria": ["calm", "annoyed", "furious"],
                ],
                "urgent": [
                    "type": "noul",
                    "instructions": "Does the customer need a reply within the hour?",
                ],
            ],
        ]

        /// The answer upstream's test has the routed server send.
        static let routedAnswer = Array(#"{"model":"verdict-1.4","answers":{},"usage":{}}"#.utf8)

        /// `request` asking `model`.
        static func asking(_ model: String, _ request: JSONValue = encoderRequest) -> JSONValue {
            var object = request.objectValue ?? [:]
            object.updateValue(.string(model), forKey: "model")
            return .object(object)
        }

        /// A target that answers every request with upstream's routed answer, after `delay`.
        static func answering(delay: Duration = .zero) -> RoutedTarget.Behaviour {
            .answer(
                status: 200,
                headers: [RoutedTarget.Header(name: "content-type", value: "application/json")],
                body: routedAnswer, delay: delay)
        }

        /// The `server-timing` of a response, parsed.
        static func timing(_ response: some CheckedResponse) throws -> ServerTimingValue {
            let header = try #require(ServerHarness.header(response, "server-timing"))
            return try #require(ServerTimingValue(header), "\(header)")
        }

        // MARK: Upstream's recordings

        @Test("Every recorded route_* exchange comes back byte for byte")
        func recordedRows() async throws {
            let rows = try UpstreamFixtures.cases("errors/cases.json").filter {
                $0["name"]?.stringValue?.hasPrefix("route_") == true
            }
            #expect(
                rows.compactMap { $0["name"]?.stringValue } == [
                    "route_down", "route_read_timeout", "route_passes_429_through",
                    "route_passes_500_text_through",
                ])
            for row in rows {
                let name = row["name"]?.stringValue ?? "?"
                guard let route = row["stub"]?["route"] else {
                    // route_down: nothing listens on the recorded URL, 127.0.0.1:9.
                    let settings = try RecordedSettings.serverSettings(row["settings"])
                    #expect(settings.modelRoutes["remote-1.0"] == "http://127.0.0.1:9")
                    let service = try await ServerHarness.diffusionService(settings)
                    let response = try await ErrorContractTests.replay(
                        row, settings: settings, service: service)
                    try ServerHarness.expectRecorded(response, row, name)
                    continue
                }
                let behaviour = try RoutedTarget.behaviour(stubbing: route)
                try await RoutedTarget.run(behaviour) { target in
                    // A read timeout of one second stands in for the recorded ReadTimeout.
                    let timesOut = route["raise"] != nil
                    let settings = try RecordedSettings.serverSettings(
                        row["settings"], routeURLs: ["http://remote": target.url],
                        forwardTimeout: timesOut ? 1 : 300)
                    let service = try await ServerHarness.diffusionService(settings)
                    let response = try await ErrorContractTests.replay(
                        row, settings: settings, service: service)
                    try ServerHarness.expectRecorded(response, row, name)
                    #expect(target.received.count == 1, "\(name)")
                    // Only content-type and retry-after come back.
                    let names = response.headers.map(\.name.canonicalName)
                    #expect(
                        Set(names).subtracting(["content-length", "date", "server"])
                            == Set(
                                [
                                    "content-type", "x-typesafe-request-id", "x-request-id",
                                    "server-timing",
                                ]
                                    + (row["response"]?["headers"]?["retry-after"] == nil
                                        ? [] : ["retry-after"])),
                        "\(name): \(names)")
                    if timesOut {
                        // The wait for the routed server is model time.
                        #expect(try Self.timing(response).model >= 1000, "\(name)")
                    }
                }
            }
        }

        @Test("The recorded listing with routes comes back byte for byte, routed servers down")
        func recordedListing() async throws {
            let listings = try #require(WireFixtures.load("models.json")["listings"]?.arrayValue)
            let routed = try #require(listings.first { $0["model_routes"] != nil })
            var routes = OrderedMap<String>()
            for (name, url) in routed["model_routes"]?.objectValue ?? [:] {
                routes.updateValue(try #require(url.stringValue), forKey: name)
            }
            #expect(Array(routes.keys) == ["verdict-1.4", "custom-model"])
            // The recorded URLs name hosts that do not resolve here: nothing is asked.
            let settings = try ServerSettings(modelRoutes: routes)
            let service = try await ServerHarness.diffusionService(settings)
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                let response = try await ServerHarness.send(client, .get, "/v1/models")
                #expect(response.status == .ok)
                #expect(ServerHarness.text(response) == routed["body_text"]?.stringValue)
                ServerHarness.expectServerHeaders(response, "listing with routes")
            }
        }

        // MARK: test_routes_forward_other_models

        /// Upstream's test, with its routes, its origin secret and its answer.
        @Test("Other models are forwarded, this server's own are answered here and listed once")
        func forwardsOtherModels() async throws {
            try await RoutedTarget.run(Self.answering(delay: .milliseconds(2))) { target in
                let routes = try ServerSettings.parseRoutes(
                    "verdict-1.4=\(target.url)/, laya-1.0=http://laya:8080")
                let settings = try ServerSettings(
                    originSecret: "s", warmup: false, modelRoutes: routes)
                let service = try await ServerHarness.encoderService(settings: settings)
                let secret = ["x-origin-secret": "s"]
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    let listing = try await ServerHarness.send(
                        client, .get, "/v1/models", headers: secret)
                    let models = try ModelsResponse(
                        json: JSONParser().parse(Array(listing.body.readableBytesView)))
                    #expect(models.models.map(\.name) == ["laya-1.0", "verdict-1.4"])

                    let forwarded = try await ServerHarness.send(
                        client, .post, "/v1/systemone",
                        headers: secret.merging(["content-type": "application/json"]) { a, _ in a },
                        body: try WireEncoder().bytes(json: Self.asking("verdict-1.4")))
                    #expect(forwarded.status == .ok)
                    #expect(Array(forwarded.body.readableBytesView) == Self.routedAnswer)
                    #expect(try Self.timing(forwarded).model >= 2)
                    let seen = try #require(target.received.first)
                    #expect(seen.path == "/v1/systemone")
                    #expect(seen.header("x-origin-secret") == "s")

                    // Its own model is answered here, not forwarded.
                    let own = try await ServerHarness.send(
                        client, .post, "/v1/systemone",
                        headers: secret.merging(["content-type": "application/json"]) { a, _ in a },
                        body: try WireEncoder().bytes(json: Self.encoderRequest))
                    #expect(own.status == .ok)
                    let answer = try SystemOneResponse(
                        json: JSONParser().parse(Array(own.body.readableBytesView)))
                    #expect(answer.model == "laya-1.0")
                    #expect(target.received.count == 1)
                }
            }
        }

        @Test("The SDK aliases and the served names are answered here even when routed")
        func servedNamesStayHere() async throws {
            try await RoutedTarget.run(Self.answering()) { target in
                let routes = try ServerSettings.parseRoutes(
                    "jev-latest=\(target.url), openjev-0.1=\(target.url), remote-1.0=\(target.url)")
                let settings = try ServerSettings(modelRoutes: routes)
                let service = try await ServerHarness.diffusionService(settings)
                let quickstart = try #require(PolicyFixtures.policyCase(named: "plain")["request"])
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    for model in ["jev-latest", "openjev-0.1"] {
                        let response = try await ServerHarness.post(
                            client, Self.asking(model, quickstart))
                        #expect(response.status == .ok, "\(model)")
                        #expect(
                            ServerHarness.text(response).hasPrefix(#"{"model":"openjev-0.1","#))
                    }
                    #expect(target.received.isEmpty)
                    let listing = try await ServerHarness.send(client, .get, "/v1/models")
                    let names = try ModelsResponse(
                        json: JSONParser().parse(Array(listing.body.readableBytesView))
                    ).models.map(\.name)
                    #expect(
                        names == [
                            "openjev-latest", "openjev-0.1", "diffusiongemma-26b", "remote-1.0",
                        ])
                }
            }
        }

        // MARK: What goes out and what comes back

        @Test("A forwarded request carries the body's bytes and three headers, the first of each")
        func forwardedRequest() async throws {
            try await RoutedTarget.run(Self.answering()) { target in
                let settings = try ServerSettings(
                    apiKey: "sk-x", originSecret: "s",
                    modelRoutes: ["remote-1.0": target.url])
                let service = try await ServerHarness.diffusionService(settings)
                // Spacing and escapes a re-encoding would change.
                let body = Array(
                    #"{ "model" : "remote-1.0", "questions":{"q":{"type":"noul"}}, "state":"café" }"#
                        .utf8)
                var fields: HTTPFields = [
                    .authorization: "Bearer sk-x",
                    HTTPField.Name("x-origin-secret")!: "s",
                    .contentType: "application/json; charset=utf-8",
                    HTTPField.Name("x-custom")!: "1", .cookie: "a=b", .userAgent: "client/1.0",
                    .accept: "*/*", .acceptEncoding: "gzip", .acceptLanguage: "fr",
                    HTTPField.Name("x-request-id")!: "req_client",
                    HTTPField.Name("x-typesafe-request-id")!: "req_client",
                    HTTPField.Name("x-typesafe-retry-count")!: "1",
                ]
                // Starlette reads the first field of a name; so does the forwarded request.
                fields.append(HTTPField(name: HTTPField.Name("x-origin-secret")!, value: "second"))
                let headers = fields
                let response = try await ServerHarness.withClient(
                    settings: settings, service: service
                ) { client in
                    try await client.execute(
                        uri: "/v1/systemone", method: .post, headers: headers,
                        body: ByteBuffer(bytes: body))
                }
                #expect(response.status == .ok)
                let seen = try #require(target.received.first)
                #expect(seen.path == "/v1/systemone")
                #expect(seen.body == body)
                // The transport adds the host and the length; nothing else of the client's goes.
                #expect(
                    seen.headerNames
                        == ["authorization", "x-origin-secret", "content-type", "content-length"])
                #expect(seen.authority == "127.0.0.1:\(target.port)")
                #expect(seen.header("authorization") == "Bearer sk-x")
                #expect(seen.header("x-origin-secret") == "s")
                #expect(seen.header("content-type") == "application/json; charset=utf-8")
                #expect(seen.header("content-length") == String(body.count))
            }
        }

        @Test("A request without the optional headers is forwarded with what it has")
        func forwardedWithoutAuthorization() async throws {
            try await RoutedTarget.run(Self.answering()) { target in
                let settings = try ServerSettings(modelRoutes: ["remote-1.0": target.url])
                let service = try await ServerHarness.diffusionService(settings)
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    let response = try await ServerHarness.post(
                        client, Self.asking("remote-1.0"))
                    #expect(response.status == .ok)
                }
                let seen = try #require(target.received.first)
                #expect(seen.headerNames == ["content-type", "content-length"])
            }
        }

        @Test("A route's credentials become Basic authorization in place of the client's")
        func credentials() async throws {
            try await RoutedTarget.run(Self.answering()) { target in
                let url = "http://us%40er:p%3Ass@127.0.0.1:\(target.port)"
                let settings = try ServerSettings(
                    apiKey: "sk-x", modelRoutes: ["remote-1.0": url])
                let service = try await ServerHarness.diffusionService(settings)
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    let response = try await ServerHarness.send(
                        client, .post, "/v1/systemone",
                        headers: [
                            "authorization": "Bearer sk-x", "content-type": "application/json",
                        ],
                        body: try WireEncoder().bytes(json: Self.asking("remote-1.0")))
                    #expect(response.status == .ok)
                }
                // httpx sends Basic base64("us@er:p:ss") for this URL, as recorded with httpx
                // 0.28.1; the host header holds no credentials.
                let seen = try #require(target.received.first)
                #expect(seen.header("authorization") == "Basic dXNAZXI6cDpzcw==")
                #expect(seen.authority == "127.0.0.1:\(target.port)")
                #expect(seen.headerNames.filter { $0 == "authorization" }.count == 1)
            }
        }

        @Test("A forwarded 200 comes back unchanged, with its content-type alone")
        func forwardedAnswer() async throws {
            let headers = [
                ("content-type", "application/json"), ("x-request-id", "req_routed"),
                ("x-typesafe-request-id", "req_routed"),
                ("server-timing", "model;dur=9.0, server;dur=1.0, total;dur=10.0"),
                ("set-cookie", "a=b"), ("x-extra", "yes"), ("cache-control", "no-store"),
            ].map { RoutedTarget.Header(name: $0.0, value: $0.1) }
            let behaviour = RoutedTarget.Behaviour.answer(
                status: 200, headers: headers, body: Self.routedAnswer)
            try await RoutedTarget.run(behaviour) { target in
                let settings = try ServerSettings(modelRoutes: ["verdict-1.4": target.url])
                let service = try await ServerHarness.diffusionService(settings)
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    let response = try await ServerHarness.post(
                        client, Self.asking("verdict-1.4"))
                    #expect(response.status == .ok)
                    #expect(Array(response.body.readableBytesView) == Self.routedAnswer)
                    #expect(ServerHarness.header(response, "content-type") == "application/json")
                    for name in ["set-cookie", "x-extra", "cache-control", "retry-after"] {
                        #expect(ServerHarness.header(response, name) == nil, "\(name)")
                    }
                    // This server's own request ids and timing, not the routed server's.
                    ServerHarness.expectServerHeaders(response, "forwarded 200")
                    #expect(ServerHarness.header(response, "x-request-id") != "req_routed")
                    #expect(try Self.timing(response).model != 9.0)
                }
            }
        }

        @Test("Several retry-after fields come back joined, as httpx joins them")
        func joinedRetryAfter() async throws {
            let headers = [
                ("content-type", "application/json"), ("retry-after", "7"), ("retry-after", "8"),
            ].map { RoutedTarget.Header(name: $0.0, value: $0.1) }
            let behaviour = RoutedTarget.Behaviour.answer(
                status: 503, headers: headers, body: Array("{}".utf8))
            try await RoutedTarget.run(behaviour) { target in
                let settings = try ServerSettings(modelRoutes: ["remote-1.0": target.url])
                let service = try await ServerHarness.diffusionService(settings)
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    let response = try await ServerHarness.post(client, Self.asking("remote-1.0"))
                    #expect(response.status == .serviceUnavailable)
                    #expect(response.headers[values: HTTPField.Name("retry-after")!] == ["7, 8"])
                    #expect(ServerHarness.text(response) == "{}")
                }
            }
        }

        @Test("An answer without a content-type comes back without one")
        func answerWithoutContentType() async throws {
            let behaviour = RoutedTarget.Behaviour.answer(
                status: 202, headers: [], body: Array("accepted".utf8))
            try await RoutedTarget.run(behaviour) { target in
                let settings = try ServerSettings(modelRoutes: ["remote-1.0": target.url])
                let service = try await ServerHarness.diffusionService(settings)
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    let response = try await ServerHarness.post(client, Self.asking("remote-1.0"))
                    #expect(response.status.code == 202)
                    #expect(ServerHarness.header(response, "content-type") == nil)
                    #expect(ServerHarness.text(response) == "accepted")
                }
            }
        }

        @Test("The routed server's time, network included, is server-timing's model")
        func modelTime() async throws {
            try await RoutedTarget.run(Self.answering(delay: .milliseconds(80))) { target in
                let settings = try ServerSettings(modelRoutes: ["remote-1.0": target.url])
                let service = try await ServerHarness.diffusionService(settings)
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    let response = try await ServerHarness.post(client, Self.asking("remote-1.0"))
                    #expect(response.status == .ok)
                    let timing = try Self.timing(response)
                    #expect(timing.model >= 80, "\(timing)")
                    #expect(timing.total >= timing.model, "\(timing)")
                }
            }
        }

        // MARK: Where the forwarding happens

        @Test("A routed request is forwarded after the shape is checked and before the cap")
        func forwardedAfterValidation() async throws {
            try await RoutedTarget.run(Self.answering()) { target in
                let settings = try ServerSettings(
                    maxQuestions: 1, modelRoutes: ["remote-1.0": target.url])
                let service = try await ServerHarness.diffusionService(settings)
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    // A body pydantic refuses is refused here, never forwarded.
                    let invalid = try await ServerHarness.post(
                        client, ["model": "remote-1.0", "questions": ["a": ["type": "noul"]]])
                    #expect(invalid.status.code == 422)
                    let tagless = try await ServerHarness.post(
                        client,
                        ["state": "x", "model": "remote-1.0", "questions": ["a": ["type": "x"]]])
                    #expect(
                        ServerHarness.text(tagless)
                            == #"{"detail":{"error_type":"api_usage_error","message":"Invalid request."}}"#
                    )
                    #expect(target.received.isEmpty)
                    // The questions cap is the routed server's to apply, as upstream forwards
                    // before it.
                    let two = try await ServerHarness.post(
                        client, Self.asking("remote-1.0", Self.encoderRequest))
                    #expect(two.status == .ok)
                    #expect(target.received.count == 1)
                    // A local model is still capped.
                    let capped = try await ServerHarness.post(
                        client, Self.asking("jev-latest", Self.encoderRequest))
                    #expect(
                        ServerHarness.text(capped)
                            == #"{"detail":"at most 1 questions per request"}"#)
                }
            }
        }

        // MARK: Failures

        @Test("A routed server that is down is the 503, logged with the model, never the URL")
        func downIsLogged() async throws {
            let recorder = LogRecorder()
            let settings = try ServerSettings(
                modelRoutes: ["remote-1.0": "http://user:SECRET@127.0.0.1:9"])
            let service = try await ServerHarness.diffusionService(settings)
            try await ServerHarness.withClient(
                settings: settings, service: service, logger: recorder.logger
            ) { client in
                let response = try await ServerHarness.post(client, Self.asking("remote-1.0"))
                #expect(response.status == .serviceUnavailable)
                #expect(
                    ServerHarness.text(response)
                        == #"{"detail":{"error_type":"api_error","message":"inference backend unavailable: ConnectError"}}"#
                )
                #expect(ServerHarness.header(response, "retry-after") == "2")
                let id = try #require(ServerHarness.header(response, "x-request-id"))
                let errors = recorder.lines.filter { $0.level == .error }
                #expect(
                    errors.map(\.message) == [
                        "503 \(id) inference backend unavailable: ConnectError (forwarding remote-1.0)"
                    ])
            }
            #expect(!recorder.lines.contains { $0.message.contains("SECRET") })
            #expect(!recorder.lines.contains { $0.message.contains("127.0.0.1") })
        }

        @Test(
            "A route httpx cannot send to is the 503 naming httpx's error",
            arguments: [
                ("ftp://x", "UnsupportedProtocol"), ("x", "UnsupportedProtocol"),
                ("http://", "UnsupportedProtocol"), ("http://[::1", "InvalidURL"),
            ])
        func unusableRoute(url: String, name: String) async throws {
            let settings = try ServerSettings(modelRoutes: ["remote-1.0": url])
            let service = try await ServerHarness.diffusionService(settings)
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                let response = try await ServerHarness.post(client, Self.asking("remote-1.0"))
                #expect(response.status == .serviceUnavailable)
                #expect(
                    ServerHarness.text(response)
                        == #"{"detail":{"error_type":"api_error","message":"inference backend unavailable: \#(name)"}}"#
                )
            }
        }

        @Test("A client that goes away cancels its forwarded request")
        func clientDisconnect() async throws {
            try await RoutedTarget.run(.silent) { target in
                let settings = try ServerSettings(
                    host: "127.0.0.1", port: 0, modelRoutes: ["remote-1.0": target.url])
                let service = try await ServerHarness.diffusionService(settings)
                let recorder = LogRecorder()
                try await LiveServer.run(
                    service: service, settings: settings, logger: recorder.logger
                ) { server in
                    let client = server.client()
                    try await client.executeAndDontWaitForResponse(
                        LiveServer.post(Self.asking("remote-1.0")))
                    #expect(try await eventually { !target.received.isEmpty })
                    try await client.shutdown()
                    #expect(
                        try await eventually {
                            recorder.lines.contains {
                                $0.level == .info && $0.message.hasPrefix("POST /v1/systemone 499 ")
                            }
                        })
                }
                // A client that went away is not a routed server's failure.
                #expect(!recorder.lines.contains { $0.level >= .warning })
            }
        }

        // MARK: The router's parts

        @Test("Transport errors are named after httpx's exceptions")
        func failureNames() {
            let cases: [(any Error, ForwardingFailure?)] = [
                (HTTPClientError.connectTimeout, .connectTimeout),
                (HTTPClientError.tlsHandshakeTimeout, .connectTimeout),
                (HTTPClientError.readTimeout, .readTimeout),
                (HTTPClientError.deadlineExceeded, .readTimeout),
                (HTTPClientError.writeTimeout, .writeTimeout),
                (HTTPClientError.remoteConnectionClosed, .remoteProtocolError),
                (HTTPClientError.invalidURL, .invalidURL),
                (HTTPClientError.emptyScheme, .unsupportedProtocol),
                (HTTPClientError.emptyHost, .unsupportedProtocol),
                (HTTPClientError.unsupportedScheme("ftp"), .unsupportedProtocol),
                (HTTPClientError.alreadyShutdown, nil),
                (IOError(errnoCode: ECONNREFUSED, reason: "connect"), .connectError),
                (
                    IOError(errnoCode: ECONNRESET, reason: "read(descriptor:pointer:size:)"),
                    .readError
                ),
                (IOError(errnoCode: EPIPE, reason: "write(descriptor:pointer:size:)"), .writeError),
                (ChannelError.connectTimeout(.seconds(5)), .connectTimeout),
                (ChannelError.eof, .remoteProtocolError),
                (CancellationError(), nil),
            ]
            for (error, expected) in cases {
                #expect(ForwardingFailure(error) == expected, "\(error)")
            }
            #expect(ForwardingFailure.allCases.map(\.name).allSatisfy { !$0.isEmpty })
            #expect(ModelRouter.failure(for: HTTPClientError.cancelled) is CancellationError)
            #expect(ModelRouter.failure(for: CancellationError()) is CancellationError)
        }

        @Test("The client waits 5 seconds to connect and the forward timeout to read and write")
        func clientConfiguration() throws {
            let router = ModelRouter(settings: try ServerSettings(forwardTimeout: 2.5))
            let configuration = router.configuration
            #expect(configuration.timeout.connect == .seconds(5))
            #expect(configuration.timeout.read == .milliseconds(2500))
            #expect(configuration.timeout.write == .milliseconds(2500))
            #expect(configuration.connectionPool.retryConnectionEstablishment == false)
            #expect(configuration.httpVersion == .http1Only)
            #expect(ModelRouter.timeAmount(seconds: .infinity) == nil)
            #expect(ModelRouter.timeAmount(seconds: 300) == .seconds(300))
        }

        @Test("Only a URL with credentials gets Basic authorization")
        func basicAuthorization() {
            #expect(ModelRouter.basicAuthorization("http://127.0.0.1:9/v1/systemone") == nil)
            #expect(
                ModelRouter.basicAuthorization("http://a:b@h/v1/systemone")
                    == "Basic " + Data("a:b".utf8).base64EncodedString())
            #expect(
                ModelRouter.basicAuthorization("http://a@h/v1/systemone")
                    == "Basic " + Data("a:".utf8).base64EncodedString())
        }
    }
#endif
