import Foundation
import OpenJevCore
import Testing

/// A ``DecisionTokenizer`` that replays the recorded tokenizations of the pinned DiffusionGemma
/// tokenizer, so the core's tests need no real tokenizer and run on Linux too.
///
/// - `encode(_, addSpecialTokens: false)` looks the text up in
///   Fixtures/tokenizer/engine_encodings.json (`cases`, `[text, ids]` pairs), then in
///   Fixtures/tokenizer/corpus.json (`cases[].text` and `ids`).
/// - `encode(_, addSpecialTokens: true)` uses the corpus's `ids_with_special_tokens`.
/// - `decode` uses the corpus's `decoded` or `decoded_skip_special_tokens` for the row whose `ids`
///   are the given ids.
/// - `chatPromptIDs` looks the system and user texts up in Fixtures/chat-prompts/prompts.json
///   (`cases[].messages`) and returns `thinking_off.ids` or `thinking_on.ids`.
///
/// Texts are compared by Unicode scalars, as Python compares them, so an NFC and an NFD spelling
/// are different entries. An input that was not recorded throws a ``TokenizerError`` whose message
/// holds the input. The files are read once per process.
struct FixtureTokenizer: DecisionTokenizer {
    /// The paths, relative to Fixtures/, of the files the tokenizer replays.
    static let files = [
        "tokenizer/engine_encodings.json", "tokenizer/corpus.json", "chat-prompts/prompts.json",
    ]

    /// The message shown when a file is missing.
    static let missingMessage: Comment =
        "A tokenizer fixture is missing; run make upstream, make fixtures-venv and make fixtures"

    /// True when every file the tokenizer replays exists.
    static var exists: Bool {
        files.allSatisfy { UpstreamFixtures.exists($0) }
    }

    /// The tokenizer, sharing one copy of the tables.
    static let shared = FixtureTokenizer()

    /// A text's identity: its UTF-8 bytes, which differ whenever the scalars differ.
    private typealias TextKey = [UInt8]

    /// The recorded results, indexed for lookup.
    private struct Tables: Sendable {
        var engineEncodings: [TextKey: [Int]] = [:]
        var corpusIDs: [TextKey: [Int]] = [:]
        var corpusIDsWithSpecialTokens: [TextKey: [Int]] = [:]
        var decoded: [[Int]: String] = [:]
        var decodedSkippingSpecialTokens: [[Int]: String] = [:]
        var chatPrompts: [[TextKey]: (thinkingOff: [Int], thinkingOn: [Int])] = [:]
    }

    /// The tables, or the error reading them gave.
    private static let tables: Result<Tables, any Error> = Result { try loadTables() }

    private static func loadTables() throws -> Tables {
        var tables = Tables()
        for pair in try UpstreamFixtures.cases("tokenizer/engine_encodings.json") {
            let text = try #require(pair[0]?.stringValue)
            tables.engineEncodings[Array(text.utf8)] = try ids(pair[1])
        }
        for row in try UpstreamFixtures.cases("tokenizer/corpus.json") {
            let text = try #require(row["text"]?.stringValue)
            let plain = try ids(row["ids"])
            tables.corpusIDs[Array(text.utf8)] = plain
            tables.corpusIDsWithSpecialTokens[Array(text.utf8)] = try ids(
                row["ids_with_special_tokens"])
            tables.decoded[plain] = try #require(row["decoded"]?.stringValue)
            tables.decodedSkippingSpecialTokens[plain] = try #require(
                row["decoded_skip_special_tokens"]?.stringValue)
        }
        for row in try UpstreamFixtures.cases("chat-prompts/prompts.json") {
            let messages = try #require(row["messages"]?.arrayValue)
            let system = try #require(messages.first { $0["role"]?.stringValue == "system" })
            let user = try #require(messages.first { $0["role"]?.stringValue == "user" })
            let key = [
                Array(try #require(system["content"]?.stringValue).utf8),
                Array(try #require(user["content"]?.stringValue).utf8),
            ]
            tables.chatPrompts[key] = (
                thinkingOff: try ids(row["thinking_off"]?["ids"]),
                thinkingOn: try ids(row["thinking_on"]?["ids"])
            )
        }
        return tables
    }

    private static func ids(_ value: JSONValue?) throws -> [Int] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.intValue) }
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
