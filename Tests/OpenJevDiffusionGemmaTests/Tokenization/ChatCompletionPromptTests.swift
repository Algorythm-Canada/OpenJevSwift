import Foundation
import OpenJevCore
import OpenJevTestSupport
import Testing

@testable import OpenJevDiffusionGemma

/// The prompts of `POST /v1/chat/completions` against what upstream's `MlxGenerator.prompt_ids`
/// rendered (issue #53): Fixtures/chat-completions/prompts.json, written by
/// Tools/fixtures/chat_tables.py from request bodies through `Generator.normalize`, and every
/// prompt the recorded route exchanges rendered. Like the other tokenizer parity tests, these need
/// the checkpoint's tokenizer files and are disabled without them.
@Suite(
    "Chat completion prompts against upstream's MlxGenerator.prompt_ids",
    .enabled(
        if: TokenizerFixtures.tokenizerAvailable
            && ChatFixtures.exists("prompts.json", "routes.json"),
        TokenizerFixtures.tokenizerAvailable
            ? Comment(rawValue: ChatFixtures.missingMessageText)
            : TokenizerFixtures.missingTokenizerMessage))
struct ChatCompletionPromptTests {
    @Test("Every recorded conversation renders to upstream's text and ids, scaffold included")
    func prompts() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let file = try ChatFixtures.load("prompts.json")
        let scaffold = try TokenizerFixtures.ints(file["scaffold"])
        #expect(scaffold == [100, 45518, 107, 101])
        let rows = try ChatFixtures.cases("prompts.json")
        #expect(rows.count == 51)
        var textMismatches: [String] = []
        var idMismatches: [String] = []
        for row in rows {
            let name = try #require(row["name"]?.stringValue)
            let messages = try #require(row["messages"]?.arrayValue)
            let thinking = try #require(row["thinking"]?.boolValue as Bool?)
            let expectedText = try #require(row["text"]?.stringValue)
            let expectedIDs = try TokenizerFixtures.ints(row["ids"])
            let text = try await tokenizer.renderChatTemplate(
                chatMessages: messages, thinking: thinking)
            if text != expectedText {
                textMismatches.append(
                    "\(name): "
                        + ChatTemplateParityTests.firstDifference(
                            expected: expectedText, actual: text))
            }
            let ids = try await tokenizer.generationPromptIDs(
                messages: messages, thinking: thinking)
            if ids != expectedIDs {
                idMismatches.append(
                    "\(name): "
                        + ChatTemplateParityTests.firstDifference(
                            expected: expectedIDs, actual: ids))
            }
            #expect(
                try tokenizer.encode(text, addSpecialTokens: false) + scaffold == ids, "\(name)")
        }
        SpikeReport.record(
            "chat-completion-prompts",
            "prompts.json: \(rows.count) conversations; text: "
                + "\(rows.count - textMismatches.count) matched, \(textMismatches.count) "
                + "mismatched; ids: \(rows.count - idMismatches.count) matched, "
                + "\(idMismatches.count) mismatched")
        #expect(textMismatches.isEmpty, "\(textMismatches)")
        #expect(idMismatches.isEmpty, "\(idMismatches)")
    }

    @Test("Every prompt the recorded route exchanges rendered gives upstream's ids")
    func routePrompts() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        var checked = 0
        for recorded in try ChatFixtures.cases("routes.json") {
            let name = try #require(recorded["name"]?.stringValue)
            for prompt in recorded["prompts"]?.arrayValue ?? [] {
                let messages = try #require(prompt["messages"]?.arrayValue)
                let thinking = try #require(prompt["thinking"]?.boolValue as Bool?)
                let ids = try await tokenizer.generationPromptIDs(
                    messages: messages, thinking: thinking)
                #expect(ids == (try TokenizerFixtures.ints(prompt["ids"])), "\(name)")
                checked += 1
            }
        }
        #expect(checked == 33)
    }

    /// Tool call arguments nested `levels` deep in messages, counting the messages array as one.
    static func toolCall(nesting levels: Int) -> [JSONValue] {
        // messages (1) > message (2) > tool_calls (3) > call (4) > function (5) > arguments (6
        // and on).
        var arguments: JSONValue = ["leaf": 1]
        for _ in 0..<(levels - 6) {
            arguments = ["k": arguments]
        }
        return [
            ["role": "user", "content": "hi"],
            [
                "role": "assistant", "content": "",
                "tool_calls": [
                    [
                        "id": "c", "type": "function",
                        "function": ["name": "f", "arguments": arguments],
                    ]
                ],
            ],
        ]
    }

    /// swift-jinja recurses into tool call arguments; a debug build overflowed a task's 512 KB
    /// stack at about 16 levels. The template renders on a thread of its own with room for the
    /// deepest messages the bound allows, and deeper ones are refused before rendering.
    @Test("Messages at the nesting bound render from a task; one level deeper is refused")
    func nestingBound() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let deepest = Self.toolCall(nesting: ChatCompletionRequest.maximumNesting)
        #expect(ChatCompletionRequest.nesting(of: .array(deepest)) == 64)
        let text = try await tokenizer.renderChatTemplate(chatMessages: deepest, thinking: false)
        // The call writes the arguments' own braces; the 58 objects inside them close, then the
        // call does.
        let closing = String(repeating: "}", count: ChatCompletionRequest.maximumNesting - 5)
        #expect(text.hasSuffix("{leaf:1" + closing + "<tool_call|><|tool_response>"))
        let deeper = Self.toolCall(nesting: ChatCompletionRequest.maximumNesting + 1)
        await #expect(throws: TokenizerError.self) {
            _ = try await tokenizer.renderChatTemplate(chatMessages: deeper, thinking: false)
        }
    }
}
