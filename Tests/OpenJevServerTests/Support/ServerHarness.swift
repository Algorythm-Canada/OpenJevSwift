#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// Builds the OpenJev application over stub backends and sends it requests through
    /// Hummingbird's in-process test client.
    enum ServerHarness {
        /// A DiffusionGemma engine over ``StubBackend`` and the fixture tokenizer, configured
        /// from the settings, as ``DecisionBackendProvider`` builds it.
        static func diffusionService(
            _ settings: ServerSettings, backend: StubBackend = StubBackend()
        ) async throws -> any SystemOneService {
            try await DecisionBackendProvider { _ in backend }.makeService(settings: settings)
        }

        /// An encoder engine over ``StubQuestionReadBackend``, without the warm-up read.
        static func encoderService(
            _ backend: StubQuestionReadBackend = StubQuestionReadBackend()
        ) async throws -> any SystemOneService {
            try await QuestionReadBackendProvider { _ in backend }
                .makeService(settings: ServerSettings(warmup: false))
        }

        /// Runs `body` with a test client of the application over `service`; `nil` settings are
        /// upstream's defaults.
        static func withClient<Value: Sendable>(
            settings: ServerSettings? = nil,
            service: any SystemOneService,
            _ body: @Sendable (any TestClientProtocol) async throws -> Value
        ) async throws -> Value {
            let app = Application(
                router: OpenJevApplication.router(
                    settings: try settings ?? ServerSettings(), service: service))
            return try await app.test(.router, body)
        }

        /// A request with string headers and a body.
        static func send(
            _ client: any TestClientProtocol, _ method: HTTPRequest.Method, _ path: String,
            headers: [String: String] = [:], body: [UInt8]? = nil
        ) async throws -> TestResponse {
            var fields = HTTPFields()
            for (name, value) in headers.sorted(by: { $0.key < $1.key }) {
                fields[try #require(HTTPField.Name(name))] = value
            }
            return try await client.execute(
                uri: path, method: method, headers: fields, body: body.map { ByteBuffer(bytes: $0) }
            )
        }

        /// A `POST /v1/systemone` of a wire request, rendered as upstream's fixtures render it.
        static func post(
            _ client: any TestClientProtocol, _ request: JSONValue
        ) async throws -> TestResponse {
            try await send(
                client, .post, "/v1/systemone", headers: ["content-type": "application/json"],
                body: try WireEncoder().bytes(json: request))
        }

        /// A response header by name.
        static func header(_ response: TestResponse, _ name: String) -> String? {
            HTTPField.Name(name).flatMap { response.headers[$0] }
        }

        /// The body as text.
        static func text(_ response: TestResponse) -> String {
            String(buffer: response.body)
        }

        /// Checks upstream's per-response headers: equal request ids of the recorded format and a
        /// `server-timing` value with its three parts.
        static func expectServerHeaders(
            _ response: TestResponse, _ label: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let id = header(response, "x-request-id")
            #expect(
                id.map(isRequestID) == true, "\(label): x-request-id \(String(describing: id))",
                sourceLocation: sourceLocation)
            #expect(
                header(response, "x-typesafe-request-id") == id,
                "\(label): x-typesafe-request-id", sourceLocation: sourceLocation)
            let timing = header(response, "server-timing")
            #expect(
                timing.flatMap(ServerTimingValue.init) != nil,
                "\(label): server-timing \(String(describing: timing))",
                sourceLocation: sourceLocation)
        }

        /// `req_` followed by 32 lowercase hex characters.
        static func isRequestID(_ text: String) -> Bool {
            guard text.hasPrefix("req_") else { return false }
            let hex = text.dropFirst(4)
            return hex.count == 32 && hex.allSatisfy { "0123456789abcdef".contains($0) }
        }
    }

    /// A parsed `server-timing` value, `model;dur=A, server;dur=B, total;dur=C`, each with one
    /// decimal.
    struct ServerTimingValue: Equatable {
        var model: Double
        var server: Double
        var total: Double

        init?(_ text: String) {
            let parts = text.components(separatedBy: ", ")
            let names = ["model", "server", "total"]
            guard parts.count == 3 else { return nil }
            var values: [Double] = []
            for (part, name) in zip(parts, names) {
                let prefix = "\(name);dur="
                guard part.hasPrefix(prefix) else { return nil }
                let number = part.dropFirst(prefix.count)
                let pieces = number.split(separator: ".", omittingEmptySubsequences: false)
                guard pieces.count == 2, pieces[1].count == 1, let value = Double(number) else {
                    return nil
                }
                values.append(value)
            }
            (model, server, total) = (values[0], values[1], values[2])
        }
    }

    extension WireFixtures {
        /// ``missingMessageText`` as a test comment.
        static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
    }

    extension PolicyFixtures {
        /// ``missingMessageText`` as a test comment.
        static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
    }
#endif
