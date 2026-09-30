#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import Logging
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// Stands in for httpx's `ConnectError`. Upstream's recordings point the engine at a vLLM
    /// port where nothing listens, so every request that passes validation ends in upstream's
    /// 503 `inference backend unavailable: ConnectError`; a stub backend that throws this type
    /// ends the same way here, since the 503 names the error's type.
    struct ConnectError: Error {}

    /// The `settings` of a recorded case, upstream's `Settings` field names, as `ServerSettings`.
    enum RecordedSettings {
        /// The settings, from upstream's defaults.
        ///
        /// - Throws: ``FixtureError`` for a field this map does not know, so that a new kind of
        ///   recording cannot pass by being ignored.
        static func serverSettings(_ value: JSONValue?) throws -> ServerSettings {
            let fields = value?.objectValue ?? [:]
            func integer(_ name: String, _ defaultValue: Int) throws -> Int {
                guard let field = fields[name] else { return defaultValue }
                return try #require(field.intValue, "settings.\(name)")
            }
            for name in fields.keys {
                let known = [
                    "upstream", "api_key", "origin_secret", "max_body_bytes", "canvas",
                    "max_image_bytes", "max_images", "max_queue", "mlx_max_prompt", "backend",
                    "warmup",
                ]
                if !known.contains(name) {
                    throw FixtureError("settings.\(name) has no ServerSettings counterpart here")
                }
            }
            // `upstream` is the vLLM URL, which this port does not have.
            return try ServerSettings(
                backend: fields["backend"]?.stringValue ?? "mlx",
                mlxMaxPrompt: try integer("mlx_max_prompt", 32768),
                canvas: try integer("canvas", 64),
                maxQueue: try integer("max_queue", 512),
                maxBodyBytes: try integer("max_body_bytes", 64 * 1024 * 1024),
                apiKey: fields["api_key"]?.stringValue ?? "",
                originSecret: fields["origin_secret"]?.stringValue ?? "",
                maxImages: try integer("max_images", 8),
                maxImageBytes: try integer("max_image_bytes", 5 * 1024 * 1024),
                warmup: fields["warmup"]?.boolValue ?? true)
        }
    }

    /// The error contract of issue #35 against upstream's recordings: every `body_*` and
    /// `auth_*` row of Fixtures/wire/cases.json and every in-scope row of
    /// Fixtures/errors/cases.json, byte for byte, plus nesting, logging, the refusal cut and the
    /// `Content-Length` check.
    ///
    /// Every response is also checked for `server-timing`. Upstream's middleware answers the
    /// 401, 403 and 413 before it sets that header (`server_timing_present` is false for those
    /// rows); this server sets it on every response (decisions D-030 and D-031).
    @Suite(
        "Error contract",
        .enabled(if: WireFixtures.exists("cases.json"), WireFixtures.missingMessage),
        .enabled(if: UpstreamFixtures.exists("errors/cases.json"), UpstreamFixtures.missingMessage)
    )
    struct ErrorContractTests {
        /// A stub whose every read throws `failure`, over a tokenizer that also renders the
        /// prompts the fixtures never recorded (``AnyPromptTokenizer``).
        static func failingBackend(_ failure: any Error) -> StubBackend {
            StubBackend(tokenizer: AnyPromptTokenizer(), failure: failure)
        }

        /// A stub whose every read fails as upstream's did with nothing listening.
        static func unreachableBackend() -> StubBackend {
            failingBackend(ConnectError())
        }

        /// Sends a recorded request, streamed without `Content-Length` when it was recorded so.
        static func replay(
            _ row: JSONValue, settings: ServerSettings, service: any SystemOneService
        ) async throws -> any CheckedResponse {
            let request = try #require(row["request"])
            let method = try #require(
                request["method"]?.stringValue.flatMap(HTTPRequest.Method.init(rawValue:)))
            let path = try #require(request["path"]?.stringValue)
            let headers = ServerHarness.headers(request["headers"])
            let body = try WireFixtures.bodyBytes(of: request)
            if request["streamed_without_content_length"]?.boolValue == true {
                return try await ServerHarness.respond(
                    settings: settings, service: service, method: method, path: path,
                    headers: headers, body: ChunkedBody(body ?? [], chunkSize: 64))
            }
            return try await ServerHarness.withClient(settings: settings, service: service) {
                client in
                try await ServerHarness.send(client, method, path, headers: headers, body: body)
            }
        }

        @Test("Every recorded body_* and auth_* exchange comes back byte for byte")
        func wireBodyAndAuthRows() async throws {
            let cases = try #require(WireFixtures.load("cases.json")["cases"]?.arrayValue)
            let rows = cases.filter { row in
                let name = row["name"]?.stringValue ?? ""
                return name.hasPrefix("body_") || name.hasPrefix("auth_")
            }
            let names = rows.compactMap { $0["name"]?.stringValue }
            #expect(names.filter { $0.hasPrefix("body_") }.count == 24)
            #expect(names.filter { $0.hasPrefix("auth_") }.count == 18)
            for row in rows {
                let name = row["name"]?.stringValue ?? "?"
                let settings = try RecordedSettings.serverSettings(row["settings"])
                let service = try await ServerHarness.diffusionService(
                    settings, backend: Self.unreachableBackend())
                let response = try await Self.replay(row, settings: settings, service: service)
                try ServerHarness.expectRecorded(response, row, name)
                // The only responses upstream sends without server-timing are the ones its
                // middleware answers before routing.
                if row["server_timing_present"]?.boolValue == false {
                    let status = row["response"]?["status"]?.intValue ?? 0
                    #expect([401, 403, 413].contains(status), "\(name): no server-timing")
                }
            }
        }

        /// The service a recorded error case needs, from its `stub`: none (an engine error, or
        /// the unreachable backend once validation passes), a stub MLX runtime (the prompt
        /// limit) or an encoder with nothing loaded.
        static func service(
            for row: JSONValue, settings: ServerSettings
        ) async throws -> any SystemOneService {
            let stub = row["stub"]
            if stub == nil || stub == .null {
                return try await ServerHarness.diffusionService(
                    settings, backend: unreachableBackend())
            }
            if stub?["mlx_runtime"] != nil {
                return try await ServerHarness.diffusionService(
                    settings, backend: StubBackend(maxPromptTokens: settings.mlxMaxPrompt))
            }
            if stub?["encoder"] != nil {
                let models: [String: (ModelInfo, Int)] = [
                    "laya": (KnownEncoderModels.laya, 255),
                    "verdict": (KnownEncoderModels.verdict, 24),
                    "clm": (KnownEncoderModels.clm, 255),
                    "jevk5": (KnownEncoderModels.jevk5, 255),
                ]
                let (modelInfo, maxChoices) = try #require(models[settings.backend])
                return try await ServerHarness.encoderService(
                    StubQuestionReadBackend(modelInfo: modelInfo, maxChoices: maxChoices),
                    settings: settings)
            }
            throw FixtureError("no stub for \(String(describing: stub))")
        }

        /// Out of scope, and so not replayed:
        ///
        /// - `backend_*`: upstream's vLLM backend answered by a mock transport. This port has no
        ///   vLLM backend, so how a vLLM response becomes a refusal or a failure is not ported;
        ///   the 400 `the model rejected this request` a backend's ``BackendRefusal`` gets is
        ///   checked by ``modelRejection()`` with the three recorded refusals whose reason is
        ///   `error.message`.
        /// - `route_*`: `OPENJEV_MODEL_ROUTES` forwarding, issue #38.
        ///
        /// `canvas_9_single_noul_fits` and `image_decoded_at_limit` pass validation and end in
        /// upstream's 503 like the others, which the unreachable stub reproduces.
        @Test("Every in-scope recorded error comes back byte for byte")
        func errorRows() async throws {
            let rows = try UpstreamFixtures.cases("errors/cases.json")
            let outOfScope = rows.filter { row in
                let name = row["name"]?.stringValue ?? ""
                return name.hasPrefix("backend_") || name.hasPrefix("route_")
            }
            #expect(rows.count == 54)
            #expect(outOfScope.count == 17)
            for row in rows where !outOfScope.contains(row) {
                let name = row["name"]?.stringValue ?? "?"
                let settings = try RecordedSettings.serverSettings(row["settings"])
                let service = try await Self.service(for: row, settings: settings)
                let response = try await Self.replay(row, settings: settings, service: service)
                try ServerHarness.expectRecorded(response, row, name)
            }
        }

        /// A refusal is cut at 500 characters, counted as Python counts them, and a backend's
        /// other errors are the 503 naming their type.
        @Test("A backend refusal is the 400 with its reason cut at 500 characters")
        func modelRejection() async throws {
            // The vLLM 4xx answers whose reason upstream took from `error.message`.
            let recorded = try UpstreamFixtures.cases("errors/cases.json").filter { row in
                let backend = row["stub"]?["backend"]
                let status = backend?["status"]?.intValue ?? 0
                return (400..<500).contains(status)
                    && backend?["json"]?["error"]?["message"]?.stringValue != nil
            }
            #expect(
                recorded.compactMap { $0["name"]?.stringValue } == [
                    "backend_400_error_message", "backend_400_long_message",
                    "backend_400_non_ascii_message",
                ])
            let settings = try ServerSettings()
            for row in recorded {
                let name = row["name"]?.stringValue ?? "?"
                let reason = try #require(
                    row["stub"]?["backend"]?["json"]?["error"]?["message"]?.stringValue)
                let service = try await ServerHarness.diffusionService(
                    settings, backend: Self.failingBackend(BackendRefusal(reason: reason)))
                let response = try await Self.replay(row, settings: settings, service: service)
                try ServerHarness.expectRecorded(response, row, name)
            }
            // The cut counts scalars: an emoji straddling the 500th character is kept whole.
            let reason = String(repeating: "x", count: 499) + "😀😀"
            let service = try await ServerHarness.diffusionService(
                settings, backend: Self.failingBackend(BackendRefusal(reason: reason)))
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(client, Self.smallRequest)
                #expect(response.status == .badRequest)
                let expected = PlainDetailBody(
                    detail: "the model rejected this request: "
                        + String(repeating: "x", count: 499) + "😀")
                #expect(ServerHarness.text(response) == (try WireEncoder().string(expected)))
            }
        }

        /// The request every small case sends: one noul question about `x`.
        static let smallRequest: JSONValue = [
            "state": "x", "model": "jev-latest", "questions": ["a": ["type": "noul"]],
        ]

        @Test("A backend's own error is the 503 naming its type, with retry-after 2")
        func backendFailure() async throws {
            struct TensorAllocationFailure: Error {}
            let service = try await ServerHarness.diffusionService(
                ServerSettings(), backend: Self.failingBackend(TensorAllocationFailure()))
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(client, Self.smallRequest)
                #expect(response.status == .serviceUnavailable)
                #expect(
                    ServerHarness.text(response)
                        == #"{"detail":{"error_type":"api_error","message":"#
                        + #""inference backend unavailable: TensorAllocationFailure"}}"#)
                #expect(ServerHarness.header(response, "retry-after") == "2")
                ServerHarness.expectServerHeaders(response, "503")
            }
        }

        // MARK: Deep nesting

        /// `n` levels of `{"a": ...}` around `null`.
        static func nested(_ depth: Int) -> String {
            String(repeating: #"{"a":"#, count: depth) + "null"
                + String(repeating: "}", count: depth)
        }

        /// The quickstart request with a question replaced by raw JSON text.
        static func quickstart(question: String) -> [UInt8] {
            let state =
                "Hi, I've been trying to connect my Stripe account but keep getting a 403 error."
            let text =
                #"{"state":"\#(state)","model":"jev-latest","questions":{"q":\#(question)}}"#
            return Array(text.utf8)
        }

        /// Upstream's `test_deeply_nested_body_is_rejected_not_crashed`, and past the parser's
        /// depth. Upstream answers the 1,000-level question with this exact body.
        @Test("Deep nesting is refused with a 422 or a 400 and never crashes")
        func deepNesting() async throws {
            let service = try await ServerHarness.diffusionService(ServerSettings())
            let json = ["content-type": "application/json"]
            try await ServerHarness.withClient(service: service) { client in
                let shallow = try await ServerHarness.send(
                    client, .post, "/v1/systemone", headers: json,
                    body: Self.quickstart(question: Self.nested(1000)))
                #expect(shallow.status.code == 422)
                let untagged =
                    #"{"detail":[{"type":"union_tag_not_found","loc":["body","questions","q"],"#
                    + #""msg":"Unable to extract tag using discriminator 'type'","#
                    + #""input":{"a":{"a":{"a":{"a":"..."}}}},"#
                    + #""ctx":{"discriminator":"'type'"}}]}"#
                #expect(ServerHarness.text(shallow) == untagged)
                ServerHarness.expectServerHeaders(shallow, "1,000 levels")

                // Deeper than the parser follows: upstream parses it (CPython 3.14 stops only
                // when its stack runs out, with the same 400); here it is the 400 at once.
                let deep = try await ServerHarness.send(
                    client, .post, "/v1/systemone", headers: json,
                    body: Self.quickstart(question: Self.nested(5000)))
                #expect(deep.status == .badRequest)
                let unparsable = #"{"detail":"There was an error parsing the body"}"#
                #expect(ServerHarness.text(deep) == unparsable)

                // A syntax error past the parser's depth is found where CPython finds it.
                let truncated =
                    Array(repeating: UInt8(ascii: "["), count: 5000)
                    + Array(repeating: UInt8(ascii: "]"), count: 4999)
                let broken = try await ServerHarness.send(
                    client, .post, "/v1/systemone", headers: json, body: truncated)
                #expect(
                    ServerHarness.text(broken)
                        == #"{"detail":[{"type":"json_invalid","loc":["body",9999],"#
                        + #""msg":"JSON decode error","input":{},"#
                        + #""ctx":{"error":"Expecting ',' delimiter"}}]}"#)

                // A 1,000-level array body: upstream's model_attributes_type, input trimmed.
                let array =
                    Array(repeating: UInt8(ascii: "["), count: 1000)
                    + Array(repeating: UInt8(ascii: "]"), count: 1000)
                let listBody = try await ServerHarness.send(
                    client, .post, "/v1/systemone", headers: json, body: array)
                #expect(
                    ServerHarness.text(listBody)
                        == #"{"detail":[{"type":"model_attributes_type","loc":["body"],"#
                        + #""msg":"Input should be a valid dictionary or object to extract fields from","#
                        + #""input":[[[["..."]]]]}]}"#)
            }
        }

        @Test("A 1,000-level state passes validation and is read")
        func deepState() async throws {
            let service = try await ServerHarness.encoderService()
            try await ServerHarness.withClient(service: service) { client in
                let body = Array(
                    #"{"state":\#(Self.nested(1000)),"model":"laya-1.0","questions":{"a":{"type":"noul"}}}"#
                        .utf8)
                let response = try await ServerHarness.send(
                    client, .post, "/v1/systemone", headers: ["content-type": "application/json"],
                    body: body)
                #expect(response.status == .ok)
            }
        }

        // MARK: Logging

        /// Upstream's `test_invalid_requests_are_logged_without_their_body`, and the other lines
        /// upstream writes, with the texts it wrote for the same requests.
        @Test("Refusals are logged with where and why, never the body")
        func refusalLogging() async throws {
            let recorder = LogRecorder()
            let settings = try ServerSettings()
            let service = try await ServerHarness.diffusionService(
                settings, backend: Self.unreachableBackend())
            let secret = "SECRET STATE"
            let instructions = "SECRET INSTRUCTIONS"
            try await ServerHarness.withClient(
                settings: settings, service: service, logger: recorder.logger
            ) { client in
                func post(_ value: JSONValue) async throws -> TestResponse {
                    try await ServerHarness.post(client, value)
                }
                let unknownType = try await post([
                    "state": .string(secret), "model": "jev-latest",
                    "questions": ["q": ["type": "nope", "instructions": .string(instructions)]],
                ])
                let mistyped = try await post([
                    "state": .string(secret), "model": .null,
                    "questions": ["q": ["type": "noul", "instructions": .string(instructions)]],
                ])
                let levels = try await post([
                    "state": .string(secret), "model": "jev-latest",
                    "questions": [
                        "q": [
                            "type": "score", "instructions": .string(instructions),
                            "criteria": .array((0..<11).map { .string("l\($0)") }),
                        ]
                    ],
                ])
                let malformed = try await ServerHarness.send(
                    client, .post, "/v1/systemone",
                    headers: ["content-type": "application/json"],
                    body: Array(#"{"state": "\#(secret)""#.utf8))
                let unavailable = try await post(Self.smallRequest)
                let unknownModel = try await post([
                    "state": .string(secret), "model": "gpt-4",
                    "questions": ["a": ["type": "noul"]],
                ])
                let lines = recorder.lines.filter { $0.level >= .warning }
                func line(_ response: TestResponse, _ text: String) -> LogRecorder.Line {
                    let id = ServerHarness.header(response, "x-request-id") ?? "?"
                    let message = text.replacingOccurrences(of: "{id}", with: id)
                    return LogRecorder.Line(level: .warning, message: message)
                }
                var expected = [
                    line(unknownType, "400 {id} body.questions.q: union_tag_invalid"),
                    line(mistyped, "422 {id} body.model: string_type"),
                    line(
                        levels,
                        "400 {id} body.questions.q.criteria: "
                            + "Too many score levels. Must have at most 10 levels."),
                    line(malformed, "422 {id} body.24: json_invalid"),
                ]
                var failure = line(
                    unavailable, "503 {id} inference backend unavailable: ConnectError")
                failure.level = .error
                expected.append(failure)
                #expect(lines == expected)
                #expect(unknownModel.status == .badRequest)
                // upstream's own check: the location and the tag, never the state
                #expect(
                    lines.contains {
                        $0.message.contains("questions.q") && $0.message.contains("tag")
                    })
                #expect(lines.contains { $0.message.contains("at most 10 levels") })
                #expect(
                    !recorder.lines.contains {
                        $0.message.contains("SECRET") || $0.message.contains("Stripe")
                    })
            }
        }

        @Test("The questions cap and a backend refusal log their location and reason")
        func semanticLogging() async throws {
            let recorder = LogRecorder()
            let settings = try ServerSettings(maxQuestions: 1)
            let service = try await ServerHarness.diffusionService(
                settings, backend: Self.failingBackend(BackendRefusal(reason: "prompt too long")))
            try await ServerHarness.withClient(
                settings: settings, service: service, logger: recorder.logger
            ) { client in
                let capped = try await ServerHarness.post(
                    client,
                    [
                        "state": "x", "model": "jev-latest",
                        "questions": ["a": ["type": "noul"], "b": ["type": "noul"]],
                    ])
                let refused = try await ServerHarness.post(client, Self.smallRequest)
                let ids = [capped, refused].map { ServerHarness.header($0, "x-request-id") ?? "?" }
                #expect(
                    recorder.lines.map(\.message) == [
                        "400 \(ids[0]) body.questions: at most 1 questions per request",
                        "400 \(ids[1]) body: the model rejected this request: prompt too long",
                    ])
                #expect(recorder.lines.allSatisfy { $0.level == .warning })
            }
        }

        @Test("Authentication, the body cap and a full queue are not logged, as upstream")
        func quietRefusals() async throws {
            let recorder = LogRecorder()
            let settings = try ServerSettings(maxQueue: 0, maxBodyBytes: 16, apiKey: "sk-test")
            let service = try await ServerHarness.diffusionService(settings)
            try await ServerHarness.withClient(
                settings: settings, service: service, logger: recorder.logger
            ) { client in
                let denied = try await ServerHarness.send(client, .get, "/v1/models")
                #expect(denied.status == .forbidden)
                let bearer = [
                    "authorization": "Bearer sk-test", "content-type": "application/json",
                ]
                let tooLarge = try await ServerHarness.send(
                    client, .post, "/v1/systemone", headers: bearer,
                    body: Array(repeating: 32, count: 17))
                #expect(tooLarge.status == .contentTooLarge)
            }
            let full = try ServerSettings(maxQueue: 0)
            let fullService = try await ServerHarness.diffusionService(full)
            try await ServerHarness.withClient(
                settings: full, service: fullService, logger: recorder.logger
            ) { client in
                let response = try await ServerHarness.post(client, Self.smallRequest)
                #expect(response.status.code == 529)
            }
            let logged = recorder.lines.filter { $0.level >= .warning }
            #expect(logged.isEmpty, "\(logged)")
        }

        // MARK: The body cap

        @Test("A declared Content-Length over the cap is refused before a byte is read")
        func declaredLengthPrecheck() async throws {
            let settings = try ServerSettings(maxBodyBytes: 512)
            let service = try await ServerHarness.diffusionService(settings)
            let body = UnreadableBody()
            let response = try await ServerHarness.respond(
                settings: settings, service: service, method: .post, path: "/v1/systemone",
                headers: ["content-type": "application/json", "content-length": "10485760"],
                body: body)
            #expect(response.status == .contentTooLarge)
            #expect(
                ServerHarness.text(response)
                    == #"{"detail":{"error_type":"api_usage_error","message":"#
                    + #""request body is larger than 512 bytes"}}"#)
            ServerHarness.expectServerHeaders(response, "precheck")
            #expect(body.reads.count == 0)
        }

        @Test("A body without Content-Length is refused at the first chunk past the cap")
        func streamedBodyStopsAtTheCap() async throws {
            let settings = try ServerSettings(maxBodyBytes: 512)
            let service = try await ServerHarness.diffusionService(settings)
            let body = ChunkedBody(Array(repeating: UInt8(ascii: " "), count: 2048), chunkSize: 64)
            let response = try await ServerHarness.respond(
                settings: settings, service: service, method: .post, path: "/v1/systemone",
                headers: ["content-type": "application/json"], body: body)
            #expect(response.status == .contentTooLarge)
            // 8 chunks make 512 bytes; the ninth passes the cap, and nothing more is read.
            #expect(body.reads.count == 9)
        }

        /// `int()` reads `Content-Length`: whitespace, a sign and `_` between digits are fine,
        /// a value that does not parse is ignored and the body is counted instead.
        @Test("Content-Length is read as Python's int() reads it")
        func contentLengthParsing() {
            let exceeds = { (text: String) in
                ContentLength.exceeds(HeaderText(bytes: Array(text.utf8)), limit: 512)
            }
            #expect(exceeds("513") == true)
            #expect(exceeds("512") == false)
            #expect(exceeds(" +6_00 ") == true)
            #expect(exceeds("-5") == false)
            #expect(exceeds("99999999999999999999999") == true)
            #expect(exceeds("-99999999999999999999999") == false)
            #expect(exceeds("abc") == nil)
            #expect(exceeds("5, 5") == nil)
            #expect(exceeds("") == nil)
        }

        @Test("A Content-Length that does not parse is ignored and the body counted")
        func unparsableContentLength() async throws {
            let settings = try ServerSettings(maxBodyBytes: 512)
            let service = try await ServerHarness.diffusionService(
                settings, backend: Self.unreachableBackend())
            let small = try WireEncoder().bytes(json: Self.smallRequest)
            let response = try await ServerHarness.respond(
                settings: settings, service: service, method: .post, path: "/v1/systemone",
                headers: ["content-type": "application/json", "content-length": "abc"],
                body: ChunkedBody(small, chunkSize: 16))
            #expect(response.status == .serviceUnavailable)
        }

        @Test("A GET is never capped and its body never read")
        func getIsNotCapped() async throws {
            let settings = try ServerSettings(maxBodyBytes: 512)
            let service = try await ServerHarness.diffusionService(settings)
            let body = UnreadableBody()
            let response = try await ServerHarness.respond(
                settings: settings, service: service, method: .get, path: "/v1/models",
                headers: ["content-length": "10485760"], body: body)
            #expect(response.status == .ok)
            #expect(body.reads.count == 0)
        }

        // MARK: Reading the body

        /// Checked against upstream's app with FastAPI 0.142.1: these content types are parsed
        /// as JSON and the others are handed to pydantic as bytes.
        @Test("The content type decides whether the body is JSON, as FastAPI decides")
        func contentTypes() {
            let json = [
                "application/json", "application/JSON", "APPLICATION/JSON",
                "application/json; charset=latin-1", "application/json;charset=utf-8",
                " application/json ", "application/ld+json", "application/+json",
                "application/vnd.api+JSON", #"application/json; boundary="x;y""#,
                "application/json\t", "\tapplication/json", "Application/Json;",
            ]
            let notJSON = [
                "application/ json", "application /json", "application/json/x", "application",
                "json", "", "text/json", "application/x-json", "application/json5",
                "application/jsonp", "multipart/form-data", "application/x-www-form-urlencoded",
                ";application/json",
            ]
            for text in json {
                #expect(RequestBodyReader.isJSON(HeaderText(bytes: Array(text.utf8))), "\(text)")
            }
            for text in notJSON {
                #expect(!RequestBodyReader.isJSON(HeaderText(bytes: Array(text.utf8))), "\(text)")
            }
            #expect(!RequestBodyReader.isJSON(nil))
            // A Latin-1 no-break space is whitespace to Python's str.strip().
            let spaced = HeaderText(bytes: Array("application/json".utf8) + [0xA0])
            #expect(RequestBodyReader.isJSON(spaced))
        }

        /// Each body is sent with a JSON content type; the answers are upstream's for the same
        /// bytes (FastAPI 0.142.1, CPython 3.14.7), except where the comment names decision
        /// D-031.
        @Test("Bodies json.loads reads differently from RFC 8259 are answered as upstream does")
        func bodyEdgeCases() async throws {
            let service = try await ServerHarness.diffusionService(
                ServerSettings(), backend: Self.unreachableBackend())
            let small = try WireEncoder().bytes(json: Self.smallRequest)
            let unavailable =
                #"{"detail":{"error_type":"api_error","message":"inference backend unavailable: ConnectError"}}"#
            let unparsable = #"{"detail":"There was an error parsing the body"}"#
            func invalid(_ message: String, _ position: Int) -> String {
                #"{"detail":[{"type":"json_invalid","loc":["body",\#(position)],"msg":"JSON decode error","#
                    + #""input":{},"ctx":{"error":"\#(message)"}}]}"#
            }
            let digits = String(repeating: "1", count: 4301)
            let mark = PythonJSONLoads.byteOrderMark
            let cases: [(String, [UInt8], String)] = [
                ("byte order mark", mark + small, unavailable),
                (
                    "byte order mark, then an error", mark + Array("{,}".utf8),
                    invalid("Expecting property name enclosed in double quotes", 1)
                ),
                ("whitespace only", Array("   ".utf8), invalid("Expecting value", 3)),
                ("not UTF-8 after an error", Array(#"{,""#.utf8) + [0xFF, 0x22, 0x7D], unparsable),
                ("an integer of 4,301 digits", Array(#"{"state":\#(digits)}"#.utf8), unparsable),
                (
                    "an error before a long integer",
                    Array(#"{"state" "x","model":\#(digits)}"#.utf8),
                    invalid("Expecting ':' delimiter", 9)
                ),
                // D-031: CPython reads these; the parser refuses them at the value.
                ("NaN", Array(#"{"state":NaN}"#.utf8), invalid("Expecting value", 9)),
                (
                    "-Infinity", Array(#"{"state":[1,-Infinity]}"#.utf8),
                    invalid("Expecting value", 12)
                ),
                (
                    "a float beyond Double", Array(#"{"state":1e400}"#.utf8),
                    invalid("Expecting value", 9)
                ),
                (
                    "a lone surrogate", Array(#"{"state":"\ud83d"}"#.utf8),
                    invalid(#"Invalid \\uXXXX escape"#, 11)
                ),
            ]
            try await ServerHarness.withClient(service: service) { client in
                for (label, body, expected) in cases {
                    let response = try await ServerHarness.send(
                        client, .post, "/v1/systemone",
                        headers: ["content-type": "application/json"], body: body)
                    #expect(ServerHarness.text(response) == expected, "\(label)")
                    ServerHarness.expectServerHeaders(response, label)
                }
            }
        }

        @Test("A body that is not JSON is echoed as Python's bytes repr, cut at 500 characters")
        func bytesBody() async throws {
            let service = try await ServerHarness.diffusionService(ServerSettings())
            let message = "Input should be a valid dictionary or object to extract fields from"
            try await ServerHarness.withClient(service: service) { client in
                let quoted = try await ServerHarness.send(
                    client, .post, "/v1/systemone", headers: ["content-type": "text/plain"],
                    body: Array("it's \u{00E9}\n".utf8) + [0xFF])
                #expect(
                    ServerHarness.text(quoted)
                        == #"{"detail":[{"type":"model_attributes_type","loc":["body"],"msg":"\#(message)","#
                        + #""input":"b\"it's \\xc3\\xa9\\n\\xff\""}]}"#)
                let long = try await ServerHarness.send(
                    client, .post, "/v1/systemone",
                    body: Array(repeating: UInt8(ascii: "a"), count: 600))
                #expect(
                    ServerHarness.text(long)
                        == #"{"detail":[{"type":"model_attributes_type","loc":["body"],"msg":"\#(message)","#
                        + #""input":"b'\#(String(repeating: "a", count: 498))"}]}"#)
            }
        }
    }

    extension UpstreamFixtures {
        /// ``missingMessageText`` as a test comment.
        static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
    }
#endif
