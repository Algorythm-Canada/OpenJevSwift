import Foundation
import OpenJevCore
import Testing

/// Loads Fixtures/policies/, the recordings of every read and thought upstream's engine makes
/// for a request and the response it sends, and helpers the engine tests share.
enum PolicyFixtures {
    /// The recording with the stub's entropy at 0.05, under the re-read threshold.
    static let policies = "policies/policies.json"
    /// The same recording with the entropy at 0.5, so the automatic re-reads run.
    static let autoRereads = "policies/auto_rereads.json"

    /// True when both files, the labels and the tokenizer fixtures exist.
    static var exists: Bool {
        FixtureTokenizer.exists && UpstreamFixtures.exists(policies, autoRereads, "labels.json")
    }

    /// The message shown when a file is missing.
    static let missingMessage: Comment =
        "Fixtures/policies is missing; run make upstream, make fixtures-venv and make fixtures"

    /// The recorded case named `name` in `policies.json`.
    static func policyCase(named name: String) throws -> JSONValue {
        try #require(
            UpstreamFixtures.cases(policies).first { $0["name"]?.stringValue == name },
            "no case \(name)")
    }

    /// The request of the recorded case named `name`, decoded.
    static func request(named name: String) throws -> SystemOneRequest {
        try RequestValidator().validate(policyCase(named: name)["request"])
    }

    /// The configuration a recorded case's `settings` describe; only the canvas and the re-read
    /// settings vary.
    static func configuration(_ settings: JSONValue?) throws -> EngineConfiguration {
        EngineConfiguration(
            geometry: try CanvasGeometry(
                canvas: settings?["canvas"]?.intValue ?? 64,
                step: settings?["step"]?.intValue ?? 16),
            autoThreshold: settings?["auto_threshold"]?.doubleValue ?? 0.1,
            autoMax: settings?["auto_max"]?.intValue ?? 4)
    }

    /// The response body upstream sends for a decision, as the recorded `body_text`.
    static func body(of decision: Decision) throws -> String {
        let usage = Usage(inputTokens: decision.inputTokens, outputTokens: decision.outputTokens)
        return try WireEncoder().string(
            SystemOneResponse(model: "openjev-0.1", answers: decision.answers, usage: usage))
    }

    static func ints(_ value: JSONValue?) throws -> [Int] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.intValue) }
    }

    /// `null` is `nil`; anything else is an integer array.
    static func optionalInts(_ value: JSONValue?) throws -> [Int]? {
        guard let value, value != .null else { return nil }
        return try ints(value)
    }

    static func strings(_ value: JSONValue?) throws -> [String] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.stringValue) }
    }

    static func slots(_ value: JSONValue?) throws -> [ResolvedTemplate.Slot] {
        try #require(value?.arrayValue).map { slot in
            ResolvedTemplate.Slot(
                position: try #require(slot["pos"]?.intValue),
                labelIDs: try ints(slot["label_ids"]))
        }
    }

    /// `[token id, logprob]` pairs in the recorded order.
    static func tops(_ value: JSONValue?) throws -> [(tokenID: Int, logprob: Double)] {
        try #require(value?.arrayValue).map { pair in
            (
                tokenID: try #require(pair[0]?.intValue),
                logprob: try #require(pair[1]?.doubleValue)
            )
        }
    }
}
