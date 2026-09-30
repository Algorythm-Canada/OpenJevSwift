import Foundation
import OpenJevCore
import Testing

/// A ``DecisionTokenizer`` that replays the recorded tokenizations, the same tables and rules as
/// `FixtureTokenizer` in OpenJevCoreTests (test targets cannot share sources).
///
/// The real tokenizer must agree with it on every text either one knows: that is the contract
/// that lets the core's tests run against recordings on Linux and the model target against the
/// real tokenizer on macOS. ``texts``, ``idsDecoded`` and ``chatPrompts`` expose everything the
/// tables know so a test can walk them.
struct ReplayTokenizer: DecisionTokenizer {
    /// A text's identity: its UTF-8 bytes, which differ whenever the scalars differ.
    typealias TextKey = [UInt8]

    /// One recorded chat prompt.
    struct ChatPrompt: Sendable {
        var system: String
        var user: String
        var thinkingOff: [Int]
        var thinkingOn: [Int]
    }

    /// The recorded results, indexed for lookup.
    struct Tables: Sendable {
        var engineEncodings: [TextKey: [Int]] = [:]
        var corpusIDs: [TextKey: [Int]] = [:]
        var corpusIDsWithSpecialTokens: [TextKey: [Int]] = [:]
        var decoded: [[Int]: String] = [:]
        var decodedSkippingSpecialTokens: [[Int]: String] = [:]
        var chatPrompts: [[TextKey]: ChatPrompt] = [:]
        /// Every text with a recorded encoding, as a string.
        var texts: [TextKey: String] = [:]
    }

    /// The tokenizer, sharing one copy of the tables.
    static let shared = ReplayTokenizer()

    private static let tables: Result<Tables, any Error> = Result { try loadTables() }

    private static func loadTables() throws -> Tables {
        var tables = Tables()
        for pair in try TokenizerFixtures.cases("tokenizer/engine_encodings.json") {
            let text = try #require(pair[0]?.stringValue)
            tables.engineEncodings[Array(text.utf8)] = try TokenizerFixtures.ints(pair[1])
            tables.texts[Array(text.utf8)] = text
        }
        for row in try TokenizerFixtures.cases("tokenizer/corpus.json") {
            let text = try #require(row["text"]?.stringValue)
            let plain = try TokenizerFixtures.ints(row["ids"])
            tables.corpusIDs[Array(text.utf8)] = plain
            tables.corpusIDsWithSpecialTokens[Array(text.utf8)] = try TokenizerFixtures.ints(
                row["ids_with_special_tokens"])
            tables.decoded[plain] = try #require(row["decoded"]?.stringValue)
            tables.decodedSkippingSpecialTokens[plain] = try #require(
                row["decoded_skip_special_tokens"]?.stringValue)
            tables.texts[Array(text.utf8)] = text
        }
        for row in try TokenizerFixtures.cases("chat-prompts/prompts.json") {
            let messages = try #require(row["messages"]?.arrayValue)
            let system = try #require(
                messages.first { $0["role"]?.stringValue == "system" }?["content"]?.stringValue)
            let user = try #require(
                messages.first { $0["role"]?.stringValue == "user" }?["content"]?.stringValue)
            tables.chatPrompts[[Array(system.utf8), Array(user.utf8)]] = ChatPrompt(
                system: system, user: user,
                thinkingOff: try TokenizerFixtures.ints(row["thinking_off"]?["ids"]),
                thinkingOn: try TokenizerFixtures.ints(row["thinking_on"]?["ids"]))
        }
        return tables
    }

    /// Every text with a recorded encoding.
    var texts: [String] {
        get throws { try Array(Self.tables.get().texts.values) }
    }

    /// Every text with a recorded encoding with special tokens (the corpus rows).
    var textsWithSpecialTokens: [String] {
        get throws {
            let tables = try Self.tables.get()
            return tables.corpusIDsWithSpecialTokens.keys.compactMap { tables.texts[$0] }
        }
    }

    /// Every id sequence with a recorded decode.
    var idsDecoded: [[Int]] {
        get throws { try Array(Self.tables.get().decoded.keys) }
    }

    /// Every recorded chat prompt.
    var chatPrompts: [ChatPrompt] {
        get throws { try Array(Self.tables.get().chatPrompts.values) }
    }

    func encode(_ text: String, addSpecialTokens: Bool) throws -> [Int] {
        let tables = try Self.tables.get()
        let key = Array(text.utf8)
        if addSpecialTokens {
            if let ids = tables.corpusIDsWithSpecialTokens[key] {
                return ids
            }
            throw TokenizerError(
                "no recorded encoding with special tokens for \(text.debugDescription)")
        }
        if let ids = tables.engineEncodings[key] ?? tables.corpusIDs[key] {
            return ids
        }
        throw TokenizerError("no recorded encoding for \(text.debugDescription)")
    }

    func decode(_ ids: [Int], skipSpecialTokens: Bool) throws -> String {
        let tables = try Self.tables.get()
        let table = skipSpecialTokens ? tables.decodedSkippingSpecialTokens : tables.decoded
        if let text = table[ids] {
            return text
        }
        throw TokenizerError("no recorded decoding for the ids \(ids)")
    }

    func chatPromptIDs(system: String, user: String, thinking: Bool) throws -> [Int] {
        let tables = try Self.tables.get()
        if let prompt = tables.chatPrompts[[Array(system.utf8), Array(user.utf8)]] {
            return thinking ? prompt.thinkingOn : prompt.thinkingOff
        }
        throw TokenizerError(
            "no recorded chat prompt for system \(system.debugDescription) and user "
                + "\(user.debugDescription)")
    }
}
