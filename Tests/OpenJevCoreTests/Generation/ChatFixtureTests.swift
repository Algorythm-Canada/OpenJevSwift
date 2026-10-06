import Foundation
import OpenJevTestSupport
import Testing

@testable import OpenJevCore

/// `Generator.normalize` and `extract_json` against what upstream's own functions made of the
/// same inputs, recorded in Fixtures/chat-completions by Tools/fixtures/chat_tables.py.
@Suite(
    "Chat normalization and JSON extraction against upstream's",
    .enabled(
        if: ChatFixtures.exists("normalize.json", "extract_json.json"),
        Comment(rawValue: ChatFixtures.missingMessageText)))
struct ChatFixtureTests {
    /// The bodies on which upstream's `normalize` raises something other than its `ValueError`,
    /// or `dict()`'s on a message that is not an object, and the port's 400 for each (D-058).
    static let departures: [String: String] = [
        "crash_kwargs_list": "chat_template_kwargs must be an object.",
        "crash_stream_options_string": "stream_options must be an object.",
        "crash_response_format_string": "response_format must be an object.",
        "crash_json_schema_string": "response_format.json_schema must be an object.",
        "json_mode_string_message": "messages[0] must be an object with a string role.",
        "json_mode_number_message": "messages[0] must be an object with a string role.",
    ]

    @Test("Every recorded body normalizes as upstream's Generator.normalize does")
    func normalize() throws {
        let rows = try ChatFixtures.cases("normalize.json")
        #expect(rows.count == 56)
        var departures = 0
        for row in rows {
            let name = try #require(row["name"]?.stringValue)
            let body = try #require(row["body"])
            let cap = row["settings"]?["gen_max_tokens"]?.intValue ?? 8192
            let checked = try ChatCompletionRequest.checked(body)
            do {
                let request = try ChatCompletionRequest(normalizing: checked, maxTokensCap: cap)
                guard var expected = row["upstream"]?.objectValue else {
                    Issue.record("\(name): upstream refused the body, the port took it")
                    continue
                }
                // The vLLM model name normalize adds is not this port's.
                expected.removeValue(forKey: "model")
                #expect(request.normalized == expected, "\(name)")
                #expect(request.jsonMode == row["json_mode"]?.boolValue, "\(name)")
                #expect(request.messages == expected["messages"]?.arrayValue, "\(name)")
                #expect(request.maxTokens == expected["max_tokens"]?.intValue, "\(name)")
                #expect(
                    request.thinking
                        == (expected["chat_template_kwargs"]?["enable_thinking"]?.isPythonTruthy
                            ?? false),
                    "\(name)")
                #expect(request.stream == (expected["stream"]?.isPythonTruthy ?? false), "\(name)")
                #expect(request.includeUsage == request.stream, "\(name)")
            } catch {
                #expect(error.status == 400, "\(name)")
                let recorded = try #require(
                    row["error"], "\(name): the port refused, upstream took it")
                if let departure = Self.departures[name] {
                    departures += 1
                    #expect(error.message == departure, "\(name)")
                } else {
                    #expect(recorded["type"] == "ValueError", "\(name)")
                    #expect(error.message == recorded["message"]?.stringValue, "\(name)")
                }
            }
        }
        #expect(departures == Self.departures.count)
    }

    @Test("Every recorded reply gives the text upstream's extract_json gives")
    func extractJSON() throws {
        let rows = try ChatFixtures.cases("extract_json.json")
        #expect(rows.count == 44)
        for row in rows {
            let name = try #require(row["name"]?.stringValue)
            let text = try #require(row["text"]?.stringValue)
            let expected = try #require(row["result"]?.stringValue)
            #expect(ExtractJSON.extract(text) == expected, "\(name)")
        }
    }

    @Test("A lone surrogate decodes to U+FFFD, where CPython keeps it and cannot answer")
    func loneSurrogate() {
        #expect(ExtractJSON.extract(#"{"a": "\ud800 x"}"#) == "{\"a\": \"\u{FFFD} x\"}")
        #expect(ExtractJSON.extract(#"{"a": "\udc00"}"#) == "{\"a\": \"\u{FFFD}\"}")
        // A high surrogate before an escape that is not a low one: both are read on their own.
        #expect(ExtractJSON.extract(#"["\ud83d\u0041"]"#) == "[\"\u{FFFD}A\"]")
        // Four characters after a high surrogate's \u that are not hexadecimal fail the value.
        #expect(ExtractJSON.extract(#"["\ud83d\uZZZZ"] [1]"#) == "[1]")
        #expect(ExtractJSON.extract(#"["\ud83d\ude00"]"#) == "[\"\u{1F600}\"]")
    }

    @Test("A value nested deeper than the parser's bound gives the reply unchanged")
    func tooDeep() {
        let depth = ExtractJSON.maximumNesting
        let deep =
            String(repeating: "[", count: depth + 1) + String(repeating: "]", count: depth + 1)
        #expect(ExtractJSON.extract(deep + " {\"a\": 1}") == deep + " {\"a\": 1}")
        let allowed = String(repeating: "[", count: depth) + String(repeating: "]", count: depth)
        #expect(ExtractJSON.extract(allowed) == allowed)
    }

    @Test("Python's repr and truth value of the values json.loads gives")
    func pythonViews() {
        #expect(JSONValue.null.pythonRepr == "None")
        #expect(JSONValue.bool(true).pythonRepr == "True")
        #expect(JSONValue.float(3.0).pythonRepr == "3.0")
        #expect(JSONValue.float(1e16).pythonRepr == "1e+16")
        #expect(JSONValue.string("it's").pythonRepr == #""it's""#)
        let list: JSONValue = [1, "a", nil, false, ["k": [2.5]]]
        #expect(list.pythonRepr == "[1, 'a', None, False, {'k': [2.5]}]")
        #expect(JSONValue.object(JSONObject()).pythonRepr == "{}")
        #expect(JSONValue.array([]).pythonRepr == "[]")
        let falsy: [JSONValue] = [nil, false, 0, 0.0, -0.0, "", [], [:]]
        #expect(falsy.allSatisfy { !$0.isPythonTruthy })
        let truthy: [JSONValue] = [true, 1, -1, 0.5, .float(.nan), " ", [0], ["": nil]]
        #expect(truthy.allSatisfy { $0.isPythonTruthy })
    }

    @Test("Messages that nest deeper than the template's bound are refused before rendering")
    func nesting() throws {
        var value: JSONValue = 1
        // messages (1) > message (2) > tool_calls (3) > call (4) > function (5) > arguments (6
        // and on): arguments nesting 59 levels make 64.
        for _ in 0..<(ChatCompletionRequest.maximumNesting - 5) {
            value = ["k": value]
        }
        func body(_ arguments: JSONValue) -> JSONValue {
            [
                "model": "diffusiongemma-26b",
                "messages": [
                    ["role": "user", "content": "hi"],
                    [
                        "role": "assistant", "content": "",
                        "tool_calls": [["function": ["name": "f", "arguments": arguments]]],
                    ],
                ],
            ]
        }
        let allowed = try ChatCompletionRequest.checked(body(value))
        #expect(throws: Never.self) {
            _ = try ChatCompletionRequest(normalizing: allowed, maxTokensCap: 8192)
        }
        let deeper = try ChatCompletionRequest.checked(body(["k": value]))
        #expect(
            throws: ChatCompletionError.invalidRequest(
                "messages nest 65 levels deep; the limit is 64.")
        ) {
            _ = try ChatCompletionRequest(normalizing: deeper, maxTokensCap: 8192)
        }
    }
}
