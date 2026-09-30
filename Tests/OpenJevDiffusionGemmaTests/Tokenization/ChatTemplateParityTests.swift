import Foundation
import OpenJevCore
import OpenJevDiffusionGemma
import Testing

/// Compares the chat prompts ``SwiftTransformersTokenizer`` renders with what Python's
/// `apply_chat_template` recorded in Fixtures/chat-prompts/prompts.json (spike #21, decision
/// D-008): the text, through swift-jinja, and the ids, through swift-transformers.
@Suite(
    "Chat template parity with the Python fixtures",
    .enabled(if: TokenizerFixtures.available, TokenizerFixtures.missingMessage))
struct ChatTemplateParityTests {
    /// Where two texts first differ, line by line, for a failure message.
    static func firstDifference(expected: String, actual: String) -> String {
        let expectedLines = expected.split(separator: "\n", omittingEmptySubsequences: false)
        let actualLines = actual.split(separator: "\n", omittingEmptySubsequences: false)
        for (index, (lhs, rhs)) in zip(expectedLines, actualLines).enumerated() where lhs != rhs {
            return "line \(index + 1): expected \(String(lhs).debugDescription), "
                + "got \(String(rhs).debugDescription)"
        }
        if expectedLines.count != actualLines.count {
            let index = min(expectedLines.count, actualLines.count)
            let extra = expectedLines.count > actualLines.count ? expectedLines : actualLines
            let kind = expectedLines.count > actualLines.count ? "missing" : "extra"
            return "line \(index + 1): \(kind) \(String(extra[index]).debugDescription)"
        }
        return "identical"
    }

    /// Where two id sequences first differ, for a failure message.
    static func firstDifference(expected: [Int], actual: [Int]) -> String {
        for (index, (lhs, rhs)) in zip(expected, actual).enumerated() where lhs != rhs {
            return "id \(index): expected \(lhs), got \(rhs)"
        }
        if expected.count != actual.count {
            return "length: expected \(expected.count), got \(actual.count)"
        }
        return "identical"
    }

    @Test("Every recorded prompt renders to the same text and ids, thinking off and on")
    func prompts() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let rows = try TokenizerFixtures.cases("chat-prompts/prompts.json")
        #expect(rows.count == 24)
        var textMismatches: [String] = []
        var idMismatches: [String] = []
        var renderings = 0
        for row in rows {
            let name = try #require(row["name"]?.stringValue)
            let messages = try #require(row["messages"]?.arrayValue)
            let system = try #require(
                messages.first { $0["role"]?.stringValue == "system" }?["content"]?.stringValue)
            let user = try #require(
                messages.first { $0["role"]?.stringValue == "user" }?["content"]?.stringValue)
            for (thinking, key) in [(false, "thinking_off"), (true, "thinking_on")] {
                renderings += 1
                let expectedText = try #require(row[key]?["text"]?.stringValue)
                let expectedIDs = try TokenizerFixtures.ints(row[key]?["ids"])
                let text = try tokenizer.chatPromptText(
                    system: system, user: user, thinking: thinking)
                if text != expectedText {
                    textMismatches.append(
                        "\(name) \(key): "
                            + Self.firstDifference(expected: expectedText, actual: text))
                }
                let ids = try tokenizer.chatPromptIDs(
                    system: system, user: user, thinking: thinking)
                if ids != expectedIDs {
                    idMismatches.append(
                        "\(name) \(key): "
                            + Self.firstDifference(expected: expectedIDs, actual: ids))
                }
                // The two paths agree with each other: the ids are the text, tokenized without
                // special tokens, as the fixture generator checked in Python.
                #expect(
                    try tokenizer.encode(text, addSpecialTokens: false) == ids, "\(name) \(key)")
            }
        }
        SpikeReport.record(
            "chat-template-parity",
            "prompts.json: \(renderings) renderings (\(rows.count) rows, thinking off and on); "
                + "text: \(renderings - textMismatches.count) matched, \(textMismatches.count) "
                + "mismatched; ids: \(renderings - idMismatches.count) matched, "
                + "\(idMismatches.count) mismatched")
        #expect(textMismatches.isEmpty, "\(textMismatches)")
        #expect(idMismatches.isEmpty, "\(idMismatches)")
    }

    @Test("The prompt has the layout the fixtures describe")
    func layout() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let off = try tokenizer.chatPromptText(system: " sys ", user: " state ", thinking: false)
        #expect(off == "<bos><|turn>system\nsys<turn|>\n<|turn>user\nstate<turn|>\n<|turn>model\n")
        let on = try tokenizer.chatPromptText(system: "sys", user: "state", thinking: true)
        #expect(
            on
                == "<bos><|turn>system\n<|think|>\nsys<turn|>\n<|turn>user\nstate<turn|>\n"
                + "<|turn>model\n")
        let ids = try tokenizer.chatPromptIDs(system: "sys", user: "state", thinking: false)
        #expect(ids.first == 2)
        #expect(ids.suffix(3) == [105, 4368, 107])
        #expect(ids.filter { $0 == 2 }.count == 1, "the template writes <bos> once")
    }

    @Test("The image message shape mlx-vlm's processor uses renders through the same path")
    func imageMessages() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let system = "Answer the questions."
        let state = "What is in the picture?"
        for imageCount in [1, 2] {
            let parts: [[String: any Sendable]] =
                Array(repeating: ["type": "image"], count: imageCount)
                + [["type": "text", "text": state]]
            let messages: [[String: any Sendable]] = [
                ["role": "system", "content": system],
                ["role": "user", "content": parts],
            ]
            let text = try tokenizer.renderChatTemplate(
                messages: messages, addGenerationPrompt: true, thinking: false)
            let placeholders = String(repeating: "<|image|>", count: imageCount)
            #expect(
                text
                    == "<bos><|turn>system\n\(system)<turn|>\n<|turn>user\n\(placeholders)\(state)"
                    + "<turn|>\n<|turn>model\n")
            let ids = try tokenizer.applyChatTemplate(messages: messages, thinking: false)
            #expect(try tokenizer.encode(text, addSpecialTokens: false) == ids)
            #expect(ids.filter { $0 == 258_880 }.count == imageCount, "one <|image|> id per image")
            SpikeReport.record(
                "chat-template-images",
                "\(imageCount) image(s):\n\(text.debugDescription)\nids: \(ids)")
        }
    }
}
