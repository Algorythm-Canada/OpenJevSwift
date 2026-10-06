#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    @testable import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// Every exchange Tools/fixtures/chat_tables.py recorded with upstream's chat route on its MLX
    /// backend, replayed against this server over the same stub runtimes: the status, the body
    /// byte for byte (streamed events included), `content-type` and `retry-after`, what the model
    /// was asked to generate, and which prompts were rendered (issue #53).
    @Suite(
        "Chat route against upstream's recorded exchanges",
        .enabled(
            if: ChatFixtures.exists("routes.json"),
            Comment(rawValue: ChatFixtures.missingMessageText)))
    struct ChatRouteFixtureTests {
        /// The exchanges this server answers differently, with its answer: upstream's route raised
        /// on these and Starlette answered a bare 500, except `json_mode_string_message`, which
        /// upstream answered with `dict()`'s message, and `completion_nan_in_body`, which
        /// `json.loads` reads and this port's parser refuses (D-016, D-058).
        static let departures: [String: String] = [
            "message_without_role": "messages[0] must be an object with a string role.",
            "message_not_an_object": "messages[0] must be an object with a string role.",
            "message_role_not_a_string": "messages[0] must be an object with a string role.",
            "json_mode_string_message": "messages[0] must be an object with a string role.",
            "response_format_string": "response_format must be an object.",
            "stop_number": "stop must be a string or an array of strings.",
            "chat_template_kwargs_list": "chat_template_kwargs must be an object.",
            "completion_nan_in_body": "The request body is not valid JSON.",
        ]

        /// The server settings a recording's settings ask for.
        static func settings(_ recorded: JSONValue?) throws -> ServerSettings {
            let defaults = try ServerSettings()
            let apiKey: String = recorded?["api_key"]?.stringValue ?? defaults.apiKey
            let maxBodyBytes: Int = recorded?["max_body_bytes"]?.intValue ?? defaults.maxBodyBytes
            let genMaxTokens: Int = recorded?["gen_max_tokens"]?.intValue ?? defaults.genMaxTokens
            return try ServerSettings(
                maxBodyBytes: maxBodyBytes, apiKey: apiKey, genMaxTokens: genMaxTokens)
        }

        @Test("Every recorded exchange is answered as upstream answered it")
        func exchanges() async throws {
            let cases = try ChatFixtures.cases("routes.json")
            #expect(cases.count == 69)
            #expect(try ChatFixtures.recordedIdentity() == ChatHarness.identity)
            var departures = 0
            for recorded in cases {
                let name = try #require(recorded["name"]?.stringValue)
                let settings = try Self.settings(recorded["settings"])
                let maxPrompt = recorded["settings"]?["mlx_max_prompt"]?.intValue ?? 32768
                let generator = try ChatFixtures.replayGenerator(
                    for: recorded, maxPromptTokens: maxPrompt)
                let request = try #require(recorded["request"])
                let body = try WireFixtures.bodyBytes(of: request) ?? []
                let headers = ServerHarness.headers(request["headers"])
                let running = recorded["running"]?.intValue ?? 0
                let response = try await ChatHarness.withClient(
                    settings: settings, generator: generator
                ) { client, chat in
                    let release = try ChatHarness.occupy(chat, running)
                    defer { release() }
                    return try await ServerHarness.send(
                        client, .post, "/v1/chat/completions", headers: headers, body: body)
                }
                if let message = Self.departures[name] {
                    departures += 1
                    #expect(response.status.code == 400, "\(name)")
                    let error = try ChatHarness.error(response)
                    #expect(error["message"]?.stringValue == message, "\(name)")
                    #expect(error["type"] == "invalid_request_error", "\(name)")
                    #expect(generator.calls.isEmpty, "\(name)")
                    continue
                }
                try ServerHarness.expectRecorded(response, recorded, name)
                // What the model was asked: the same prompt, budget, stop ids and skipped markers.
                let generations = recorded["generations"]?.arrayValue ?? []
                #expect(generator.calls.count == generations.count, "\(name)")
                for (call, generation) in zip(generator.calls, generations) {
                    #expect(
                        call.prompt == generation["prompt"]?.arrayValue?.compactMap(\.intValue),
                        "\(name)")
                    #expect(call.maxTokens == generation["max_tokens"]?.intValue, "\(name)")
                    #expect(
                        call.stopIDs == generation["stop_ids"]?.arrayValue?.compactMap(\.intValue),
                        "\(name)")
                    #expect(
                        call.skipSpecialTokenIDs
                            == generation["skip_special"]?.arrayValue?.compactMap(\.intValue),
                        "\(name)")
                }
                // Every prompt upstream rendered was rendered here, from the same messages.
                let prompts = recorded["prompts"]?.arrayValue ?? []
                #expect(generator.renderedPrompts.count == prompts.count, "\(name)")
                for (rendered, prompt) in zip(generator.renderedPrompts, prompts) {
                    #expect(.array(rendered.messages) == prompt["messages"], "\(name)")
                    #expect(rendered.thinking == prompt["thinking"]?.boolValue, "\(name)")
                }
            }
            #expect(departures == Self.departures.count)
        }
    }
#endif
