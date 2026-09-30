import Foundation
import OpenJevCore
import Testing

/// Loads the upstream read tables in Fixtures/distributions/distributions.json, which
/// Tools/fixtures/upstream_tables.py writes (issue #6). Two of its arrays are read here:
/// - `slot_distribution`: `{name, top, label_ids, result: {probs, entropy, ...}}`, the arguments
///   and result of `slot_distribution`, with `top` as `[token id, logprob]` pairs in the order
///   the backend returned them;
/// - `confidence`: `{name, p, result}`, the argument and result of `confidence`.
///
/// Rows that record an upstream exception (`error`) are skipped: an empty map is a
/// precondition failure here, and a single option gives 1.0 where upstream divides by zero
/// (decision D-020). `to_answer` and `read_group` belong to other tests.
///
/// Tests that need the file are disabled with a message when it is missing.
enum DistributionFixtures {
    /// Fixtures/distributions, found relative to this source file.
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Read
        .deletingLastPathComponent()  // OpenJevCoreTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root
        .appendingPathComponent("Fixtures/distributions")

    /// The message shown when the file is missing.
    static let missingMessage: Comment =
        "Fixtures/distributions/distributions.json is missing; run make fixtures"

    /// The fixture file.
    static let file = directory.appendingPathComponent("distributions.json")

    /// True when the file exists.
    static var exists: Bool {
        FileManager.default.fileExists(atPath: file.path)
    }

    /// One recorded `slot_distribution` call.
    struct Slot {
        var name: String
        var top: [(tokenID: Int, logprob: Double)]
        var labelIDs: [Int]
        var probabilities: [Double]
        var entropy: Double
    }

    /// One recorded `confidence` call.
    struct ConfidenceCase {
        var name: String
        var probabilities: [Double]
        var confidence: Double
    }

    /// The rows of the two arrays that have a result. A row with neither a result nor an error
    /// is reported, so a layout change is noticed.
    static func load() throws -> (slots: [Slot], confidences: [ConfidenceCase]) {
        let document = try JSONParser().parse(Data(contentsOf: file))
        var slots: [Slot] = []
        for entry in try #require(document["slot_distribution"]?.arrayValue) {
            let name = entry["name"]?.stringValue ?? "?"
            guard let result = entry["result"] else {
                if entry["error"] == nil { Issue.record("\(name) has no result and no error") }
                continue
            }
            let top = try #require(entry["top"]?.arrayValue, "\(name)").map { pair in
                (
                    tokenID: try #require(pair[0]?.intValue, "\(name)"),
                    logprob: try #require(pair[1]?.doubleValue, "\(name)")
                )
            }
            slots.append(
                Slot(
                    name: name, top: top,
                    labelIDs: try numbers(entry["label_ids"], name).map { Int($0) },
                    probabilities: try numbers(result["probs"], name),
                    entropy: try #require(result["entropy"]?.doubleValue, "\(name)")))
        }
        var confidences: [ConfidenceCase] = []
        for entry in try #require(document["confidence"]?.arrayValue) {
            let name = entry["name"]?.stringValue ?? "?"
            guard let result = entry["result"] else {
                if entry["error"] == nil { Issue.record("\(name) has no result and no error") }
                continue
            }
            confidences.append(
                ConfidenceCase(
                    name: name, probabilities: try numbers(entry["p"], name),
                    confidence: try #require(result.doubleValue, "\(name)")))
        }
        try #require(!slots.isEmpty, "no slot distribution entries")
        try #require(!confidences.isEmpty, "no confidence entries")
        return (slots, confidences)
    }

    /// An array of numbers, or an error naming the entry.
    private static func numbers(_ value: JSONValue?, _ name: String) throws -> [Double] {
        let array = try #require(value?.arrayValue, "\(name)")
        return try array.map { try #require($0.doubleValue, "\(name)") }
    }
}
