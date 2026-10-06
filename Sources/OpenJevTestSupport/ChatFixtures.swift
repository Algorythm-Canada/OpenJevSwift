import Foundation
import OpenJevCore

/// Loads the chat completion recordings in Fixtures/chat-completions, which
/// Tools/fixtures/chat_tables.py writes from upstream's `openjev/chat.py` and the pinned tokenizer.
///
/// `routes.json` holds HTTP exchanges with every prompt and stop string the tokenizer handled for
/// them, so ``replayGenerator(for:maxPromptTokens:)`` can answer each exchange as upstream's
/// stub runtime did, without the tokenizer, on any platform.
public enum ChatFixtures {
    /// Fixtures/chat-completions.
    public static let directory = UpstreamFixtures.directory.appendingPathComponent(
        "chat-completions")

    /// The message shown when a file is missing.
    public static let missingMessageText =
        "Fixtures/chat-completions is missing; run make upstream, make fixtures-venv and make "
        + "fixtures"

    /// True when the named files exist.
    public static func exists(_ names: String...) -> Bool {
        names.allSatisfy {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    /// The named file, parsed.
    public static func load(_ name: String) throws -> JSONValue {
        try JSONParser().parse(Data(contentsOf: directory.appendingPathComponent(name)))
    }

    /// The `cases` array of the named file.
    public static func cases(_ name: String) throws -> [JSONValue] {
        try unwrap(load(name)["cases"]?.arrayValue, "\(name) has no cases array")
    }

    /// The recorded route exchange named `name`.
    public static func route(named name: String) throws -> JSONValue {
        try unwrap(
            cases("routes.json").first { $0["name"]?.stringValue == name },
            "routes.json has no case \(name)")
    }

    /// The completion id and creation time every recorded reply carries.
    public static func recordedIdentity() throws -> ChatCompletionIdentity {
        let file = try load("routes.json")
        return ChatCompletionIdentity(
            id: try unwrap(file["completion_id"]?.stringValue, "routes.json: completion_id"),
            created: try unwrap(file["created"]?.intValue, "routes.json: created"))
    }

    /// A generator that answers a recorded exchange as upstream did: the prompts and stop strings
    /// the exchange rendered and encoded, the markers the file records, and the stub runtime the
    /// exchange names (`stub`, `replay` or `one_token`). A prompt or a text the exchange did not
    /// record throws a ``FixtureError``.
    public static func replayGenerator(
        for recorded: JSONValue, maxPromptTokens: Int
    ) throws -> StubTextGenerator {
        let name = recorded["name"]?.stringValue ?? "?"
        let prompts = try (recorded["prompts"]?.arrayValue ?? []).map { row in
            (
                messages: try unwrap(row["messages"]?.arrayValue, "\(name): prompt messages"),
                thinking: try unwrap(row["thinking"]?.boolValue, "\(name): prompt thinking"),
                ids: try ints(row["ids"], "\(name): prompt ids")
            )
        }
        var encodings: [String: [Int]] = [:]
        for row in recorded["encodings"]?.arrayValue ?? [] {
            let text = try unwrap(row["text"]?.stringValue, "\(name): encoding text")
            encodings[text] = try ints(row["ids"], "\(name): encoding ids")
        }
        let markers = try ints(load("routes.json")["markers"], "routes.json: markers")
        let body: StubTextGenerator.Body
        switch recorded["runtime"]?.stringValue {
        case "stub":
            body = StubTextGenerator.upstreamReply()
        case "replay":
            body = StubTextGenerator.replay()
        case "one_token":
            body = StubTextGenerator.oneToken()
        case let other:
            throw FixtureError("\(name): unknown runtime \(String(describing: other))")
        }
        let frozenEncodings = encodings
        return StubTextGenerator(
            maxPromptTokens: maxPromptTokens, markers: markers,
            prompt: { messages, thinking in
                guard
                    let row = prompts.first(where: {
                        $0.messages == messages && $0.thinking == thinking
                    })
                else {
                    throw FixtureError("\(name): no recorded prompt for these messages")
                }
                return row.ids
            },
            encode: { text in
                try unwrap(frozenEncodings[text], "\(name): no recorded encoding of \(text)")
            },
            generate: body)
    }

    /// The integers of a JSON array.
    static func ints(_ value: JSONValue?, _ what: String) throws -> [Int] {
        try unwrap(value?.arrayValue, "\(what) is not an array").map {
            try unwrap($0.intValue, "\(what) holds a value that is not an integer")
        }
    }
}
