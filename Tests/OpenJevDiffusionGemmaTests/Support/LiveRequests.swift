import Foundation
import OpenJevCore
import OpenJevTestSupport
import Testing

/// The oracle's prompts and reads, the parts the runtime tests use.
struct OracleFixture: Decodable {
    struct Prompt: Decodable {
        let system: String
        let user: String
        let ids: [Int]
    }
    struct Slot: Decodable {
        let pos: Int
        let labelIDs: [Int]
        enum CodingKeys: String, CodingKey {
            case pos
            case labelIDs = "label_ids"
        }
    }
    struct Distribution: Decodable {
        let probs: [Double]
        let entropy: Double
    }
    struct Read: Decodable {
        let id: String
        let prompt: String
        let width: Int
        let canvas: [Int]
        let slots: [Slot]
        let steps: Int
        let promptTokens: Int
        let distributions: [Distribution]
        enum CodingKeys: String, CodingKey {
            case id, prompt, width, canvas, slots, steps, distributions
            case promptTokens = "prompt_tokens"
        }
    }
    /// A JevBench item among the requests (D-048); the tests need only its id.
    struct JevBenchItem: Decodable {}
    let prompts: [String: Prompt]
    let reads: [Read]
    /// The JevBench items, by id, which is also their request name.
    let jevbench: [String: JevBenchItem]

    /// The prompt keys (`request/gGROUP`) of the fixture's own requests, without the JevBench
    /// items', sorted.
    var fixtureRequestPrompts: [String] {
        prompts.keys.filter { key in
            !jevbench.keys.contains(String(key.prefix { $0 != "/" }))
        }.sorted()
    }

    static func load() throws -> OracleFixture {
        let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent("oracle/reads.json")
        return try JSONDecoder().decode(OracleFixture.self, from: Data(contentsOf: url))
    }
}

/// A request body as JSON text.
func liveRequest(_ text: String) throws -> SystemOneRequest {
    try SystemOneRequest(json: JSONParser().parse(text))
}

/// Upstream's tests/test_live.py `QUESTIONS`, the README example.
let readmeQuestions = """
    {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
     "team": {"type": "choice", "instructions": "Which team should handle it?",
              "criteria": {"outage": "service down", "billing": "charges, refunds",
                           "feature": "requests, how-to"}},
     "tone": {"type": "score", "instructions": "How upset is the customer?",
              "criteria": ["calm", "annoyed", "furious"]}}
    """

/// The README example over `state`.
func readmeRequest(state: String) throws -> SystemOneRequest {
    let quoted = String(decoding: try JSONEncoder().encode(state), as: UTF8.self)
    return try liveRequest(
        #"{"model": "openjev-latest", "state": \#(quoted), "questions": \#(readmeQuestions)}"#)
}

/// The README quickstart request of Fixtures/wire/cases.json.
func quickstartRequest() throws -> SystemOneRequest {
    let recorded = try WireFixtures.recordedCase(named: "quickstart")
    let body = try #require(recorded["request"]?["body_text"]?.stringValue)
    return try liveRequest(body)
}
