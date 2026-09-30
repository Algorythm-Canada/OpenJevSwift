import OpenJevCore
import Testing

/// Checks that ``FixtureTokenizer`` replays the recordings and refuses what was not recorded.
@Suite(
    "Fixture tokenizer", .enabled(if: FixtureTokenizer.exists, FixtureTokenizer.missingMessage))
struct FixtureTokenizerTests {
    private let tokenizer = FixtureTokenizer.shared

    @Test("A recorded text encodes to its ids and decodes back")
    func roundTrip() throws {
        let ids = try tokenizer.encode("q1: A", addSpecialTokens: false)
        #expect(ids == [236809, 236770, 236787, 562])
        #expect(try tokenizer.encode("q1: A", addSpecialTokens: true) == ids)
        #expect(try tokenizer.decode(ids, skipSpecialTokens: false) == "q1: A")
        #expect(try tokenizer.decode(ids, skipSpecialTokens: true) == "q1: A")
    }

    @Test("Every corpus row replays its ids, decodes and special-token ids")
    func corpus() throws {
        let rows = try UpstreamFixtures.cases("tokenizer/corpus.json")
        #expect(rows.count == 917)
        for row in rows {
            let text = try #require(row["text"]?.stringValue)
            let ids = try #require(row["ids"]?.arrayValue).compactMap(\.intValue)
            let withSpecial = try #require(row["ids_with_special_tokens"]?.arrayValue)
                .compactMap(\.intValue)
            #expect(try tokenizer.encode(text, addSpecialTokens: false) == ids)
            #expect(try tokenizer.encode(text, addSpecialTokens: true) == withSpecial)
            #expect(
                try tokenizer.decode(ids, skipSpecialTokens: false)
                    == row["decoded"]?.stringValue)
            #expect(
                try tokenizer.decode(ids, skipSpecialTokens: true)
                    == row["decoded_skip_special_tokens"]?.stringValue)
        }
    }

    @Test("Every engine encoding replays")
    func engineEncodings() throws {
        let pairs = try UpstreamFixtures.cases("tokenizer/engine_encodings.json")
        #expect(pairs.count == 2634)
        for pair in pairs {
            let text = try #require(pair[0]?.stringValue)
            let ids = try #require(pair[1]?.arrayValue).compactMap(\.intValue)
            #expect(try tokenizer.encode(text, addSpecialTokens: false) == ids, "\(text)")
        }
    }

    @Test("An unrecorded text throws with the text in the message")
    func unrecordedText() {
        let text = "a text no fixture recorded 7f3a"
        for special in [false, true] {
            #expect {
                try tokenizer.encode(text, addSpecialTokens: special)
            } throws: { error in
                (error as? TokenizerError)?.message.contains(text) == true
            }
        }
        #expect(throws: TokenizerError.self) {
            try tokenizer.decode([1, 2, 3, 4, 5, 6, 7, 8, 9], skipSpecialTokens: false)
        }
        #expect {
            try tokenizer.chatPromptIDs(system: "system 7f3a", user: text, thinking: false)
        } throws: { error in
            (error as? TokenizerError)?.message.contains(text) == true
        }
    }

    @Test("A curated chat prompt returns the recorded ids with thinking off and on")
    func chatPrompt() throws {
        let rows = try UpstreamFixtures.cases("chat-prompts/prompts.json")
        let row = try #require(rows.first { $0["source"]?.stringValue == "curated" })
        let messages = try #require(row["messages"]?.arrayValue)
        let system = try #require(messages.first?["content"]?.stringValue)
        let user = try #require(messages.last?["content"]?.stringValue)
        let off = try #require(row["thinking_off"]?["ids"]?.arrayValue).compactMap(\.intValue)
        let on = try #require(row["thinking_on"]?["ids"]?.arrayValue).compactMap(\.intValue)
        #expect(!off.isEmpty && off != on)
        #expect(try tokenizer.chatPromptIDs(system: system, user: user, thinking: false) == off)
        #expect(try tokenizer.chatPromptIDs(system: system, user: user, thinking: true) == on)
    }
}
