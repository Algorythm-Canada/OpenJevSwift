import Foundation
import OpenJevCore

/// Loads Fixtures/policies/, the recordings of every read and thought upstream's engine makes
/// for a request and the response it sends, and helpers the engine tests share.
public enum PolicyFixtures {
    /// The recording with the stub's entropy at 0.05, under the re-read threshold.
    public static let policies = "policies/policies.json"
    /// The same recording with the entropy at 0.5, so the automatic re-reads run.
    public static let autoRereads = "policies/auto_rereads.json"

    /// True when both files, the labels and the tokenizer fixtures exist.
    public static var exists: Bool {
        FixtureTokenizer.exists && UpstreamFixtures.exists(policies, autoRereads, "labels.json")
    }

    /// The message shown when a file is missing.
    public static let missingMessageText =
        "Fixtures/policies is missing; run make upstream, make fixtures-venv and make fixtures"

    /// The recorded case named `name` in `policies.json`.
    public static func policyCase(named name: String) throws -> JSONValue {
        try unwrap(
            UpstreamFixtures.cases(policies).first { $0["name"]?.stringValue == name },
            "no case \(name)")
    }

    /// The request of the recorded case named `name`, decoded.
    ///
    /// Never inlined: Swift 6.4 at -O crashed in LLVM's CoroSplit ("While splitting coroutine")
    /// on async tests that inlined the untyped throw of `policyCase(named:)` next to the typed
    /// throw of `validate`, so `swift build -c release --build-tests` failed. The LLVM fix is
    /// llvm/llvm-project#217372.
    @inline(never)
    public static func request(named name: String) throws -> SystemOneRequest {
        try RequestValidator().validate(policyCase(named: name)["request"])
    }

    /// The configuration a recorded case's `settings` describe; only the canvas and the re-read
    /// settings vary.
    public static func configuration(_ settings: JSONValue?) throws -> EngineConfiguration {
        EngineConfiguration(
            geometry: try CanvasGeometry(
                canvas: settings?["canvas"]?.intValue ?? 64,
                step: settings?["step"]?.intValue ?? 16),
            autoThreshold: settings?["auto_threshold"]?.doubleValue ?? 0.1,
            autoMax: settings?["auto_max"]?.intValue ?? 4)
    }

    /// The response body upstream sends for a decision, as the recorded `body_text`.
    public static func body(of decision: Decision) throws -> String {
        let usage = Usage(inputTokens: decision.inputTokens, outputTokens: decision.outputTokens)
        return try WireEncoder().string(
            SystemOneResponse(model: "openjev-0.1", answers: decision.answers, usage: usage))
    }

    /// An array of integers.
    public static func ints(_ value: JSONValue?) throws -> [Int] {
        let values = try unwrap(value?.arrayValue, "not an array: \(String(describing: value))")
        return try values.map { try unwrap($0.intValue, "not an integer: \($0)") }
    }

    /// `null` is `nil`; anything else is an integer array.
    public static func optionalInts(_ value: JSONValue?) throws -> [Int]? {
        guard let value, value != .null else { return nil }
        return try ints(value)
    }

    /// An array of strings.
    public static func strings(_ value: JSONValue?) throws -> [String] {
        let values = try unwrap(value?.arrayValue, "not an array: \(String(describing: value))")
        return try values.map { try unwrap($0.stringValue, "not a string: \($0)") }
    }

    /// The recorded `{pos, label_ids}` slots.
    public static func slots(_ value: JSONValue?) throws -> [ResolvedTemplate.Slot] {
        try unwrap(value?.arrayValue, "slots are not an array").map { slot in
            ResolvedTemplate.Slot(
                position: try unwrap(slot["pos"]?.intValue, "slot: pos"),
                labelIDs: try ints(slot["label_ids"]))
        }
    }

    /// `[token id, logprob]` pairs in the recorded order.
    public static func tops(_ value: JSONValue?) throws -> [(tokenID: Int, logprob: Double)] {
        try unwrap(value?.arrayValue, "tops are not an array").map { pair in
            (
                tokenID: try unwrap(pair[0]?.intValue, "top: token id"),
                logprob: try unwrap(pair[1]?.doubleValue, "top: logprob")
            )
        }
    }
}
