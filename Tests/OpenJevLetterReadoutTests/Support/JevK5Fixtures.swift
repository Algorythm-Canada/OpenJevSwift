import CryptoKit
import Foundation
import OpenJevCore
import OpenJevLetterReadout
import Testing

/// Fixtures/jevk5/reads.json, which `Tools/jevk5/reference.py` records through upstream's
/// `JevK5Engine.read_question` and the `jevk5` package, with the 4-bit conversion on MLX in place of
/// vLLM.
enum JevK5Fixtures {
    /// The repository root, found relative to this source file.
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Support
        .deletingLastPathComponent()  // OpenJevLetterReadoutTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root

    /// The fixture file.
    static let file = root.appendingPathComponent("Fixtures/jevk5/reads.json")

    /// Whether the fixture is there; it is committed, so a test that needs it fails without it
    /// rather than skipping.
    static var available: Bool { FileManager.default.fileExists(atPath: file.path) }

    /// One pass of a question: the option texts read, the prompt's digest, length and (when it is
    /// short) text, its token count and ids digest, and the letters' logits.
    struct Pass: Sendable {
        var texts: [String]
        var promptSHA256: String
        var characters: Int
        var tokens: Int
        var idsSHA256: String
        var logits: [Float]
        var prompt: String?
    }

    /// One read question: its options, its passes, and upstream's distribution and token count.
    struct Read: Sendable {
        var request: String
        var key: String
        var type: String
        var options: [JevK5Option]
        var passes: [Pass]
        var probabilities: [Double]
        var tokens: Int
    }

    /// One request of the corpus, decoded as the API decodes a body, with upstream's billing.
    struct Request: Sendable {
        var name: String
        var request: SystemOneRequest
        var inputTokens: Int
    }

    /// One `spread` case over generated logits.
    struct Spread: Sendable {
        var title: String
        var method: JevK5Readout.Method
        var texts: [String]
        var passes: [(texts: [String], logits: [Double])]
        var probabilities: [Double]
    }

    /// Everything the file holds.
    struct Reference: Sendable {
        var temperature: Double
        var letterIDs: [Int]
        var maxCharactersPerToken: Int
        var requests: [Request]
        var reads: [Read]
        var spreads: [Spread]

        /// Every pass of every read, in file order.
        var passes: [Pass] { reads.flatMap(\.passes) }

        /// The reads of a request, in question order.
        func reads(of request: String) -> [Read] {
            reads.filter { $0.request == request }
        }
    }

    /// The file, parsed once.
    private static let loaded = Result { try parse() }

    /// The parsed file.
    static func reference() throws -> Reference {
        try loaded.get()
    }

    private static func parse() throws -> Reference {
        let root = try JSONParser().parse(Data(contentsOf: file))
        func strings(_ value: JSONValue?) throws -> [String] {
            try (value?.arrayValue ?? []).map { try need($0.stringValue, "a text is not a string") }
        }
        func numbers(_ value: JSONValue?) throws -> [Double] {
            try (value?.arrayValue ?? []).map {
                try need($0.doubleValue, "a value is not a number")
            }
        }
        func text(_ value: JSONValue?, _ field: String) throws -> String {
            try need(value?[field]?.stringValue, "no \(field)")
        }
        func integer(_ value: JSONValue?, _ field: String) throws -> Int {
            try need(value?[field]?.intValue, "no \(field)")
        }
        let temperature = try need(root["temperature"]?.doubleValue, "no temperature")
        let tokenizer = try need(root["tokenizer"], "no tokenizer")
        let corpus = try need(root["corpus"]?.arrayValue, "no corpus")
        let totals = try need(root["requests"]?.arrayValue, "no requests")
        var requests: [Request] = []
        for (entry, total) in zip(corpus, totals) {
            let body: JSONValue = .object([
                "model": "jevk5-0.2", "state": try need(entry["state"], "no state"),
                "questions": try need(entry["questions"], "no questions"),
            ])
            requests.append(
                Request(
                    name: try text(entry, "name"), request: try SystemOneRequest(json: body),
                    inputTokens: try integer(total, "input_tokens")))
        }
        var reads: [Read] = []
        for entry in root["reads"]?.arrayValue ?? [] {
            var options: [JevK5Option] = []
            for pair in entry["options"]?.arrayValue ?? [] {
                options.append(
                    JevK5Option(
                        id: try need(pair[0]?.stringValue, "no option id"),
                        text: try need(pair[1]?.stringValue, "no option text")))
            }
            var passes: [Pass] = []
            for pass in entry["passes"]?.arrayValue ?? [] {
                passes.append(
                    Pass(
                        texts: try strings(pass["texts"]),
                        promptSHA256: try text(pass, "prompt_sha256"),
                        characters: try integer(pass, "chars"),
                        tokens: try integer(pass, "tokens"),
                        idsSHA256: try text(pass, "ids_sha256"),
                        logits: try numbers(pass["logits"]).map { Float($0) },
                        prompt: pass["prompt"]?.stringValue))
            }
            reads.append(
                Read(
                    request: try text(entry, "request"), key: try text(entry, "key"),
                    type: try text(entry, "type"), options: options, passes: passes,
                    probabilities: try numbers(entry["probabilities"]),
                    tokens: try integer(entry, "tokens")))
        }
        var spreads: [Spread] = []
        for entry in root["spreads"]?.arrayValue ?? [] {
            var passes: [(texts: [String], logits: [Double])] = []
            for pass in entry["passes"]?.arrayValue ?? [] {
                passes.append(
                    (texts: try strings(pass["texts"]), logits: try numbers(pass["logits"])))
            }
            spreads.append(
                Spread(
                    title: try text(entry, "title"),
                    method: try need(
                        JevK5Readout.Method(rawValue: try text(entry, "method")),
                        "an unknown method"),
                    texts: try strings(entry["texts"]), passes: passes,
                    probabilities: try numbers(entry["probabilities"])))
        }
        let letterIDs = try (tokenizer["letter_ids"]?.arrayValue ?? []).map {
            try need($0.intValue, "a letter id is not an integer")
        }
        return Reference(
            temperature: temperature, letterIDs: letterIDs,
            maxCharactersPerToken: try integer(tokenizer, "max_chars_per_token"),
            requests: requests, reads: reads, spreads: spreads)
    }

    /// The SHA-256 of a text's UTF-8 bytes, in lowercase hexadecimal.
    static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// The SHA-256 of ids written as decimal numbers joined by commas, as the script digests them.
    static func idsSHA256(_ ids: [Int]) -> String {
        sha256(ids.map(String.init).joined(separator: ","))
    }
}

/// A tokenizer that knows the fixture's prompts by their digests: a letter is its recorded id,
/// and a prompt is as many ids as the fixture recorded, the first of which is the pass's index
/// among all the fixture's passes, so ``ReplayLetterModel`` can answer it. A prompt the fixture
/// does not hold gets the id -1, which the model refuses.
struct ReplayTokenizer: LetterReadoutTokenizing {
    let letterIDs: [Int]
    let maxCharactersPerToken: Int
    private let passes: [String: (index: Int, tokens: Int)]

    init(_ reference: JevK5Fixtures.Reference) {
        letterIDs = reference.letterIDs
        maxCharactersPerToken = reference.maxCharactersPerToken
        var passes: [String: (index: Int, tokens: Int)] = [:]
        for (index, pass) in reference.passes.enumerated() {
            passes[pass.promptSHA256] = (index, pass.tokens)
        }
        self.passes = passes
    }

    func encode(_ text: String) -> [Int] {
        if let letter = JevK5Prompt.letters.firstIndex(of: text) {
            return [letterIDs[letter]]
        }
        guard let pass = passes[JevK5Fixtures.sha256(text)] else {
            return [-1]
        }
        return [pass.index] + [Int](repeating: 0, count: pass.tokens - 1)
    }
}

/// A model that answers the fixture's passes with their recorded logits, found by the index
/// ``ReplayTokenizer`` puts first.
struct ReplayLetterModel: LetterReadoutModel {
    struct UnknownPrompt: Error, CustomStringConvertible {
        var description: String { "the prompt is not one the fixture recorded" }
    }

    let logits: [[Float]]

    init(_ reference: JevK5Fixtures.Reference) {
        logits = reference.passes.map(\.logits)
    }

    func letterLogits(tokens: [Int], letterIDs: [Int]) throws -> [Float] {
        guard let index = tokens.first, logits.indices.contains(index) else {
            throw UnknownPrompt()
        }
        return logits[index]
    }
}

/// The value, or a ``FixtureFieldError`` saying what was missing.
func need<Value>(_ value: Value?, _ message: String) throws -> Value {
    guard let value else { throw FixtureFieldError(message: message) }
    return value
}

/// A field of Fixtures/jevk5/reads.json is missing or of the wrong type.
struct FixtureFieldError: Error, CustomStringConvertible {
    var message: String
    var description: String { "Fixtures/jevk5/reads.json: \(message)" }
}
