#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    @testable import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// Upstream's chat tests (`tests/test_api.py` and `tests/test_mlx_backend.py`) over HTTP, with
    /// a stub generator in place of the model, as upstream's run with `StubRuntime` (issue #53).
    /// The tests that need the model are in `ChatCompletionsLiveTests` of the DiffusionGemma
    /// tests and in the live suite, disabled until the model is wired.
    @Suite("Chat completions over HTTP")
    struct ChatCompletionsRouteTests {
        // MARK: tests/test_api.py

        /// jev-ultrafast's text request: JSON mode, a reasoning switch, a temperature and a seed.
        /// Only the passthrough fields reach the generation, JSON mode becomes an instruction and
        /// the reply is reduced to its JSON object.
        @Test("test_chat_normalizes_jev_ultrafast_request")
        func normalizesJevUltrafastRequest() async throws {
            let generator = StubTextGenerator(
                generate: StubTextGenerator.upstreamReply(
                    reply: ["Sure!\n```json\n{\"text\": \"Zurich\"}\n```"], tail: ""))
            let body: JSONValue = [
                "model": "diffusiongemma-26b", "max_tokens": 1024,
                "response_format": ["type": "json_object"], "reasoning": ["enabled": false],
                "temperature": 0.2, "seed": 1,
                "messages": [
                    ["role": "system", "content": "Return {\"text\": ...}"],
                    ["role": "user", "content": "{}"],
                ],
            ]
            try await ChatHarness.withClient(generator: generator) { client, _ in
                let response = try await ChatHarness.post(client, body)
                #expect(response.status == .ok, "\(ServerHarness.text(response))")
                let reply = try JSONParser().parse(ServerHarness.text(response))
                #expect(reply["model"] == "diffusiongemma-26b")
                let content = try #require(
                    reply["choices"]?[0]?["message"]?["content"]?.stringValue)
                #expect(try JSONParser().parse(content) == ["text": "Zurich"])
                let wrong = try await ChatHarness.post(
                    client, ChatHarness.chat(["model": "gpt-4"]))
                #expect(wrong.status == .notFound)
            }
            let call = try #require(generator.calls.first)
            #expect(call.maxTokens == 1024)
            let (messages, thinking) = try #require(generator.renderedPrompts.first)
            #expect(!thinking)
            #expect(messages[0]["content"]?.stringValue?.contains("JSON object") == true)
            #expect(messages.count == 2)
        }

        /// `int()` crashed the route on a string, and `True` and `3.9` were silently read as 1 and
        /// 3; a non-integer limit is a 400.
        @Test("test_chat_max_tokens_must_be_an_integer")
        func maxTokensMustBeAnInteger() async throws {
            let generator = StubTextGenerator()
            try await ChatHarness.withClient(generator: generator) { client, _ in
                let bad: [JSONValue] = ["abc", true, 3.9, 0, -5]
                for value in bad {
                    let response = try await ChatHarness.post(
                        client, ChatHarness.chat(["max_tokens": value]))
                    #expect(response.status == .badRequest, "\(value)")
                    let message = try ChatHarness.error(response)["message"]?.stringValue
                    #expect(message?.contains("max_tokens") == true, "\(value)")
                }
                let string = try await ChatHarness.post(
                    client, ChatHarness.chat(["max_completion_tokens": "50"]))
                #expect(string.status == .badRequest)
                let good = try await ChatHarness.post(
                    client, ChatHarness.chat(["max_completion_tokens": 50]))
                #expect(good.status == .ok)
            }
            #expect(generator.calls.map(\.maxTokens) == [50])
        }

        @Test("test_chat_model_is_required")
        func modelIsRequired() async throws {
            try await ChatHarness.withClient(generator: StubTextGenerator()) { client, _ in
                let messages: JSONValue = [["role": "user", "content": "hi"]]
                let missing = try await ChatHarness.post(client, ["messages": messages])
                #expect(missing.status == .badRequest)
                let number = try await ChatHarness.post(
                    client, ["messages": messages, "model": 42])
                #expect(number.status == .badRequest)
                #expect(
                    try ChatHarness.error(number)["message"]
                        == "model is required and must be a string.")
                let unknown = try await ChatHarness.post(
                    client, ["messages": messages, "model": "gpt-4"])
                #expect(unknown.status == .notFound)
            }
        }

        // MARK: tests/test_mlx_backend.py

        @Test("test_chat_rejects_an_unknown_model")
        func rejectsAnUnknownModel() async throws {
            try await ChatHarness.withClient(generator: StubTextGenerator()) { client, _ in
                let response = try await ChatHarness.post(
                    client, ChatHarness.chat(["model": "gpt-4"]))
                #expect(response.status == .notFound)
                #expect(
                    ServerHarness.text(response)
                        == #"{"error":{"message":"Model 'gpt-4' not found. Available: "#
                        + #"diffusiongemma-26b.","type":"invalid_request_error","#
                        + #""code":"model_not_found"}}"#)
                ServerHarness.expectServerHeaders(response, "404")
            }
        }

        /// The limit is checked before the answer starts, so a stream gets the 400 too.
        @Test("test_chat_refuses_an_over_long_prompt")
        func refusesAnOverLongPrompt() async throws {
            let generator = StubTextGenerator(maxPromptTokens: 8)
            let long = ChatHarness.chat([
                "messages": [
                    ["role": "user", "content": .string(String(repeating: "word ", count: 200))]
                ]
            ])
            try await ChatHarness.withClient(generator: generator) { client, _ in
                for stream: JSONValue in [false, true] {
                    var body = try #require(long.objectValue)
                    body.updateValue(stream, forKey: "stream")
                    let response = try await ChatHarness.post(client, .object(body))
                    #expect(response.status == .badRequest)
                    #expect(ServerHarness.header(response, "content-type") == "application/json")
                    let message = try ChatHarness.error(response)["message"]?.stringValue
                    #expect(message?.hasSuffix("limit is 8") == true, "\(message ?? "")")
                }
            }
            #expect(generator.calls.isEmpty)
        }

        @Test("test_chat_capacity_is_refused")
        func capacityIsRefused() async throws {
            let generator = StubTextGenerator()
            let settings = try ServerSettings()
            try await ChatHarness.withClient(settings: settings, generator: generator) {
                client, chat in
                let release = try ChatHarness.occupy(
                    chat, settings.genMaxInflight + settings.genMaxQueue)
                defer { release() }
                let response = try await ChatHarness.post(client, ChatHarness.chat)
                #expect(response.status.code == 529)
                #expect(try ChatHarness.error(response)["type"] == "overloaded_error")
                #expect(ServerHarness.header(response, "retry-after") == "2")
            }
            #expect(generator.calls.isEmpty)
        }

        /// The model opens a thought channel of its own accord on some replies, and the markers
        /// reach the detokenizer fused into a later segment: the reply is kept clean by asking
        /// the generator to skip those ids, not by matching them in the text.
        @Test("test_the_thought_channel_never_reaches_a_chat_client")
        func thoughtChannelNeverReachesAChatClient() async throws {
            let generator = StubTextGenerator(generate: StubTextGenerator.replay())
            try await ChatHarness.withClient(generator: generator) { client, _ in
                let response = try await ChatHarness.post(client, ChatHarness.chat)
                #expect(response.status == .ok)
                let text = try JSONParser().parse(ServerHarness.text(response))["choices"]?[0]?[
                    "message"]?["content"]?.stringValue
                #expect(text == "six seven eight nine ten")
            }
        }

        @Test("test_the_thought_channel_never_reaches_a_streaming_client")
        func thoughtChannelNeverReachesAStreamingClient() async throws {
            let generator = StubTextGenerator(generate: StubTextGenerator.replay())
            try await ChatHarness.withClient(generator: generator) { client, _ in
                let response = try await ChatHarness.post(
                    client, ChatHarness.chat(["stream": true]))
                #expect(response.status == .ok)
                let text = ChatHarness.streamedText(try ChatHarness.events(response))
                #expect(!text.contains("channel"))
                #expect(text == "six seven eight nine ten")
            }
        }

        /// Chat wants the thought gone, so the generator is told to drop the markers. (Upstream's
        /// test also checks that `think` keeps them; `think` reaches the model through the
        /// decision backend's own call, issue #52, not through this one.)
        @Test("test_chat_skips_the_marker_tokens_and_think_does_not")
        func skipsTheMarkerTokens() async throws {
            let generator = StubTextGenerator()
            try await ChatHarness.withClient(generator: generator) { client, _ in
                let response = try await ChatHarness.post(client, ChatHarness.chat)
                #expect(response.status == .ok)
            }
            #expect(generator.calls.map(\.skipSpecialTokenIDs) == [[100, 45518, 107, 101]])
        }

        /// The shortest replies exposed a race upstream: the generation finished while its last
        /// piece was still on its way to the stream, and the stream ended empty. It was a race,
        /// so once proves nothing.
        @Test("test_a_one_token_reply_still_streams")
        func oneTokenReplyStillStreams() async throws {
            let generator = StubTextGenerator(generate: StubTextGenerator.oneToken())
            try await ChatHarness.withClient(generator: generator) { client, _ in
                for _ in 0..<20 {
                    let response = try await ChatHarness.post(
                        client, ChatHarness.chat(["stream": true]))
                    #expect(response.status == .ok)
                    #expect(ChatHarness.streamedText(try ChatHarness.events(response)) == "7")
                }
            }
        }

        /// `normalize` behaves as it does on vLLM: JSON mode adds the instruction and extracts
        /// the object, and `max_tokens` is clamped.
        @Test("test_chat_json_mode_and_max_tokens")
        func jsonModeAndMaxTokens() async throws {
            let generator = StubTextGenerator()
            let body = ChatHarness.chat([
                "response_format": ["type": "json_object"], "temperature": 0.7, "seed": 3,
                "max_tokens": 99999,
            ])
            try await ChatHarness.withClient(generator: generator) { client, _ in
                let response = try await ChatHarness.post(client, body)
                #expect(response.status == .ok, "\(ServerHarness.text(response))")
                let content = try JSONParser().parse(ServerHarness.text(response))["choices"]?[0]?[
                    "message"]?["content"]?.stringValue
                #expect(try JSONParser().parse(try #require(content)) == ["city": "Zurich"])
            }
            let call = try #require(generator.calls.first)
            #expect(call.maxTokens == (try ServerSettings()).genMaxTokens)
            // The JSON instruction went into the prompt the generator was handed.
            let (messages, _) = try #require(generator.renderedPrompts.first)
            #expect(
                messages[0] == [
                    "role": "system", "content": .string(ChatCompletionRequest.jsonInstruction),
                ])
        }

        @Test("test_chat_stream_on_mlx")
        func streamOnTheStub() async throws {
            let body = ChatHarness.chat(["stream": true, "stream_options": ["include_usage": true]])
            try await ChatHarness.withClient(generator: StubTextGenerator()) { client, _ in
                let response = try await ChatHarness.post(client, body)
                #expect(response.status == .ok)
                #expect(
                    ServerHarness.header(response, "content-type")?.hasPrefix("text/event-stream")
                        == true)
                #expect(ServerHarness.text(response).hasSuffix("data: [DONE]\n\n"))
                let events = try ChatHarness.events(response)
                #expect(events[0]["choices"]?[0]?["delta"] == ["role": "assistant", "content": ""])
                #expect(
                    ChatHarness.streamedText(events)
                        == StubTextGenerator.reply.joined() + StubTextGenerator.tail)
                #expect(events.contains { $0["choices"]?[0]?["finish_reason"] == "stop" })
                let usage = events.compactMap { $0["usage"] }
                #expect(usage.count == 1)
                #expect(usage.first?["completion_tokens"] == 4)
                #expect(events.allSatisfy { $0["model"] == "diffusiongemma-26b" })
                ServerHarness.expectServerHeaders(response, "stream")
            }
        }

        @Test("test_generation_model_is_listed")
        func generationModelIsListed() async throws {
            try await ChatHarness.withClient(generator: StubTextGenerator()) { client, _ in
                let response = try await ServerHarness.send(client, .get, "/v1/models")
                let names = try JSONParser().parse(ServerHarness.text(response))["models"]?
                    .arrayValue?.compactMap { $0["name"]?.stringValue }
                #expect(names?.contains("diffusiongemma-26b") == true)
            }
        }

        // MARK: This port's

        /// Upstream adds the chat routes for its DiffusionGemma backends and not for the encoder
        /// models; here they exist when the service's model generates text.
        @Test("The chat route exists only for a model that generates text")
        func routeOnlyForAGenerator() async throws {
            let settings = try ServerSettings()
            let services: [(any SystemOneService, Bool)] = [
                (try await ServerHarness.diffusionService(settings), false),
                (try await ServerHarness.encoderService(), false),
                (try await ChatHarness.service(settings, generator: StubTextGenerator()), true),
            ]
            for (service, serves) in services {
                #expect((service.textGenerator != nil) == serves)
                try await ServerHarness.withClient(settings: settings, service: service) {
                    client in
                    let response = try await ChatHarness.post(client, ChatHarness.chat)
                    #expect(response.status == (serves ? .ok : .notFound))
                }
            }
        }

        @Test("A generator that fails is a 503 naming its error's type, logged as a failure")
        func failingGenerator() async throws {
            struct ModelFault: Error {}
            let generator = StubTextGenerator(generate: { _, _ in throw ModelFault() })
            let recorder = LogRecorder()
            try await ChatHarness.withClient(generator: generator, logger: recorder.logger) {
                client, chat in
                let response = try await ChatHarness.post(client, ChatHarness.chat)
                #expect(response.status == .serviceUnavailable)
                #expect(
                    ServerHarness.text(response)
                        == #"{"error":{"message":"inference backend unavailable: ModelFault","#
                        + #""type":"api_error","code":null}}"#)
                #expect(ServerHarness.header(response, "retry-after") == "2")
                #expect(chat.running == 0 && chat.freeSlots == 8)
            }
            #expect(
                recorder.lines.contains {
                    $0.level == .error
                        && $0.message.hasSuffix("inference backend unavailable: ModelFault")
                })
        }

        @Test("A body that is not JSON is the 400, whatever its content type")
        func notJSON() async throws {
            try await ChatHarness.withClient(generator: StubTextGenerator()) { client, _ in
                for body in ["", "{", "[1", "{\"a\": NaN}"] {
                    let response = try await ServerHarness.send(
                        client, .post, "/v1/chat/completions", headers: [:],
                        body: Array(body.utf8))
                    #expect(response.status == .badRequest, "\(body)")
                    #expect(
                        try ChatHarness.error(response)["message"]
                            == "The request body is not valid JSON.")
                }
                let plain = try await ServerHarness.send(
                    client, .post, "/v1/chat/completions",
                    headers: ["content-type": "text/plain"],
                    body: try WireEncoder().bytes(json: ChatHarness.chat))
                #expect(plain.status == .ok)
            }
        }
    }
#endif
