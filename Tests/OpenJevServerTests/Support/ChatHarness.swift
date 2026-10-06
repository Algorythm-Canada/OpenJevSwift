#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import Logging
    @testable import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// Builds the application with `POST /v1/chat/completions` over a stub generator, as a
    /// ``/OpenJevCore/DecisionEngine`` over ``StubGeneratingBackend`` serves it.
    enum ChatHarness {
        /// The id and creation time the recordings were made with, which every reply carries here.
        static let identity = ChatCompletionIdentity(
            id: "chatcmpl-0123456789abcdef01234567", created: 1_790_000_000)

        /// Upstream's `CHAT` request.
        static let chat: JSONValue = [
            "model": "diffusiongemma-26b",
            "messages": [["role": "user", "content": "Which city?"]],
        ]

        /// `chat` with more fields.
        static func chat(_ fields: JSONObject) -> JSONValue {
            var body = chat.objectValue ?? JSONObject()
            for (key, value) in fields {
                body.updateValue(value, forKey: key)
            }
            return .object(body)
        }

        /// A DiffusionGemma engine over the stub reads and `generator`, from the settings.
        static func service(
            _ settings: ServerSettings, generator: StubTextGenerator
        ) async throws -> any SystemOneService {
            try await DecisionBackendProvider { _ in
                StubGeneratingBackend(generation: generator)
            }.makeService(settings: settings)
        }

        /// The chat service the routes use: the settings' capacity and token bound, and
        /// ``identity``.
        static func chatService(
            _ settings: ServerSettings, generator: StubTextGenerator
        ) -> ChatCompletions {
            ChatCompletions(
                generator: generator, configuration: ChatCompletionsConfiguration(settings),
                identity: { identity })
        }

        /// Runs `body` with a test client of the routes over `generator`, and the chat service
        /// they use, whose capacity a test can inspect or occupy.
        static func withClient<Value: Sendable>(
            settings: ServerSettings? = nil, generator: StubTextGenerator,
            logger: Logger? = nil,
            _ body: @Sendable (any TestClientProtocol, ChatCompletions) async throws -> Value
        ) async throws -> Value {
            let settings = try settings ?? ServerSettings()
            let service = try await service(settings, generator: generator)
            let chat = chatService(settings, generator: generator)
            let app = Application(
                router: OpenJevApplication.router(
                    settings: settings, service: service, connections: nil, chat: chat),
                logger: logger)
            return try await app.test(.router) { client in try await body(client, chat) }
        }

        /// A `POST /v1/chat/completions` of `body` as compact JSON.
        static func post(
            _ client: any TestClientProtocol, _ body: JSONValue,
            headers: [String: String] = ["content-type": "application/json"]
        ) async throws -> TestResponse {
            try await ServerHarness.send(
                client, .post, "/v1/chat/completions", headers: headers,
                body: try WireEncoder().bytes(json: body))
        }

        /// The JSON payloads of a streamed body's events, `[DONE]` left out, after checking that
        /// every event is `data: ...` and a blank line.
        static func events(
            _ response: TestResponse, sourceLocation: SourceLocation = #_sourceLocation
        ) throws -> [JSONValue] {
            let text = ServerHarness.text(response)
            #expect(text.hasSuffix("\n\n"), sourceLocation: sourceLocation)
            var events: [JSONValue] = []
            for event in text.components(separatedBy: "\n\n") where !event.isEmpty {
                #expect(event.hasPrefix("data: "), "\(event)", sourceLocation: sourceLocation)
                let payload = String(event.dropFirst(6))
                if payload != "[DONE]" {
                    events.append(try JSONParser().parse(payload))
                }
            }
            return events
        }

        /// The reply's text in streamed events: every delta's content, joined.
        static func streamedText(_ events: [JSONValue]) -> String {
            events.compactMap { $0["choices"]?[0]?["delta"]?["content"]?.stringValue }.joined()
        }

        /// The body of an error answer, in OpenAI's shape.
        static func error(_ response: TestResponse) throws -> JSONValue {
            try #require(try JSONParser().parse(ServerHarness.text(response))["error"])
        }

        /// Counts `count` requests against the bound, as upstream's tests set `running`, and
        /// returns a function that counts them out again.
        static func occupy(_ chat: ChatCompletions, _ count: Int) throws -> @Sendable () -> Void {
            for _ in 0..<count {
                try chat.capacity.admit()
            }
            return {
                for _ in 0..<count {
                    chat.capacity.leave()
                }
            }
        }
    }
#endif
