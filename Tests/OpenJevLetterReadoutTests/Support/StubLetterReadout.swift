import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import OpenJevCore
import OpenJevLetterReadout
import OpenJevServer

/// Upstream's `StubVllm` in `tests/test_encoders.py`: a model that prefers the letter B. A letter
/// is the id `1000 + its code point`, as the stub's `/tokenize` gives it, and its logprob is A
/// -2.0, B -0.5, C -3.0, every other letter -6.0.
///
/// Every pass is recorded with the letters asked for. ``maxConcurrentPasses`` tells whether two
/// passes ever overlapped; the backend runs them one at a time. ``failures`` makes the passes
/// whose prompt holds a text throw an error instead.
final class StubLetterModel: LetterReadoutModel, @unchecked Sendable {
    /// Upstream's `LETTER_LOGPROBS`.
    static let logprobs: [Character: Float] = ["A": -2.0, "B": -0.5, "C": -3.0]
    /// Every other letter's.
    static let otherLogprob: Float = -6.0

    /// One recorded pass.
    struct Pass: Sendable {
        var tokens: [Int]
        var letterIDs: [Int]
    }

    /// Errors by the prompt id ``StubLetterTokenizer`` gave, thrown instead of answering.
    let failures: [Int: any Error & Sendable]
    /// How long a pass takes, so that concurrent questions would overlap if nothing serialized
    /// them.
    let delay: TimeInterval

    // Guarded by `lock`.
    private let lock = NSLock()
    private var recorded: [Pass] = []
    private var active = 0
    private var mostActive = 0

    init(failures: [Int: any Error & Sendable] = [:], delay: TimeInterval = 0) {
        self.failures = failures
        self.delay = delay
    }

    /// The passes, in the order they ran.
    var passes: [Pass] { lock.withLock { recorded } }
    /// The most passes that ran at once.
    var maxConcurrentPasses: Int { lock.withLock { mostActive } }

    func letterLogits(tokens: [Int], letterIDs: [Int]) throws -> [Float] {
        lock.withLock {
            active += 1
            mostActive = max(mostActive, active)
            recorded.append(Pass(tokens: tokens, letterIDs: letterIDs))
        }
        defer { lock.withLock { active -= 1 } }
        if delay > 0 {
            Thread.sleep(forTimeInterval: delay)
        }
        if let first = tokens.first, let failure = failures[first] {
            throw failure
        }
        return letterIDs.map { id in
            let letter = Character(Unicode.Scalar(UInt32(id - 1000))!)
            return Self.logprobs[letter] ?? Self.otherLogprob
        }
    }
}

/// The tokenizer for ``StubLetterModel``: a letter is `1000 + its code point`; a prompt is
/// ``promptTokens`` ids (upstream's stub bills 100 per pass), the first of which numbers the
/// prompt, 1 for the first prompt seen. Every prompt is recorded.
final class StubLetterTokenizer: LetterReadoutTokenizing, @unchecked Sendable {
    let promptTokens: Int
    let maxCharactersPerToken = 128
    /// A prompt holding this text is tokenized after a pause of `slowSeconds`, so its question
    /// reaches the model after the others.
    let slowText: String?
    let slowSeconds: Double

    // Guarded by `lock`.
    private let lock = NSLock()
    private var recorded: [String] = []

    init(promptTokens: Int = 100, slowText: String? = nil, slowSeconds: Double = 0) {
        self.promptTokens = promptTokens
        self.slowText = slowText
        self.slowSeconds = slowSeconds
    }

    /// The prompts, in the order they were tokenized.
    var prompts: [String] { lock.withLock { recorded } }

    func encode(_ text: String) -> [Int] {
        if JevK5Prompt.letters.contains(text), let scalar = text.unicodeScalars.first {
            return [1000 + Int(scalar.value)]
        }
        if let slowText, text.contains(slowText) {
            Thread.sleep(forTimeInterval: slowSeconds)
        }
        let number = lock.withLock {
            recorded.append(text)
            return recorded.count
        }
        return [number] + [Int](repeating: 0, count: max(0, promptTokens - 1))
    }
}

/// Upstream's `REQUEST` in `tests/test_encoders.py`, asking JevK5.
enum UpstreamRequest {
    static let body: JSONValue = [
        "state": "I was charged twice this month.",
        "model": "jevk5-0.2",
        "questions": [
            "team": [
                "type": "choice", "instructions": "Which team should handle it?",
                "criteria": [
                    "outage": "service down", "billing": "charges, refunds", "feature": .null,
                ],
            ],
            "tone": [
                "type": "score", "instructions": "How upset is the customer?",
                "criteria": ["calm", "annoyed", "furious"],
            ],
            "urgent": [
                "type": "noul", "instructions": "Does the customer need a reply within the hour?",
            ],
        ],
    ]

    /// Upstream's `softmax_t(logprobs, t=2.0)`.
    static func softmax(_ values: [Double], temperature: Double = 2.0) -> [Double] {
        let weights = values.map { exp($0 / temperature) }
        let total = weights.reduce(0, +)
        return weights.map { $0 / total }
    }
}

/// The OpenJev application over a service, driven through Hummingbird's in-process test client as
/// upstream's tests drive FastAPI's `TestClient`.
enum JevK5Server {
    /// The engine over `backend`, as the server builds it for `OPENJEV_BACKEND=jevk5`, without
    /// the warm-up read.
    static func service(_ backend: JevK5Backend) async throws -> any SystemOneService {
        try await QuestionReadBackendProvider { _ in backend }
            .makeService(settings: ServerSettings(backend: "jevk5", warmup: false))
    }

    /// Sends `method` to `path` with an optional JSON body and returns the status and the body
    /// parsed as JSON.
    static func send(
        _ service: any SystemOneService, _ method: HTTPRequest.Method, _ path: String,
        body: JSONValue? = nil
    ) async throws -> (status: Int, body: JSONValue) {
        let settings = try ServerSettings(backend: "jevk5", warmup: false)
        let app = Application(
            router: OpenJevApplication.router(settings: settings, service: service))
        return try await app.test(.router) { client in
            var headers = HTTPFields()
            headers[.contentType] = "application/json"
            let bytes = try body.map { try WireEncoder().bytes(json: $0) }
            let response = try await client.execute(
                uri: path, method: method, headers: headers,
                body: bytes.map { ByteBuffer(bytes: $0) })
            let parsed = try JSONParser().parse(Array(buffer: response.body))
            return (Int(response.status.code), parsed)
        }
    }
}
