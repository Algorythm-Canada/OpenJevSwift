import Foundation
import OpenJevCore
import Testing

/// Loads the upstream read tables in Fixtures/distributions, which issue #6 will write.
///
/// Every `.json` file in the folder is read. A file is either an array of entries or an object
/// holding the array under `entries` or `rows`. Two entry shapes are recognised:
/// - `{top: {tokenId: logprob}, label_ids: [...], probs: [...], entropy: x}`, the arguments and
///   result of `slot_distribution`, with `top` in the order the backend returned it;
/// - `{probabilities: [...], confidence: x}`, the argument and result of `confidence`.
///
/// Tests that need the folder are disabled with a message when it is missing.
enum DistributionFixtures {
    /// Fixtures/distributions, found relative to this source file.
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Read
        .deletingLastPathComponent()  // OpenJevCoreTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root
        .appendingPathComponent("Fixtures/distributions")

    /// The message shown when the folder is missing.
    static let missingMessage: Comment =
        "Fixtures/distributions is missing; issue #6 generates it from upstream"

    /// True when the folder exists.
    static var exists: Bool {
        FileManager.default.fileExists(atPath: directory.path)
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

    /// Every entry of every file, sorted into the two shapes. An entry of neither shape is
    /// reported as an error so a layout change is noticed.
    static func load() throws -> (slots: [Slot], confidences: [ConfidenceCase]) {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "json" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var slots: [Slot] = []
        var confidences: [ConfidenceCase] = []
        for file in files {
            let document = try JSONParser().parse(Data(contentsOf: file))
            let entries = try #require(
                document.arrayValue ?? document["entries"]?.arrayValue
                    ?? document["rows"]?.arrayValue,
                "\(file.lastPathComponent) has no entries")
            for (index, entry) in entries.enumerated() {
                let name = "\(file.lastPathComponent)[\(index)]"
                if let top = entry["top"]?.objectValue {
                    var pairs: [(tokenID: Int, logprob: Double)] = []
                    for (key, value) in top {
                        pairs.append(
                            (try #require(Int(key), "\(name)"), try #require(value.doubleValue)))
                    }
                    slots.append(
                        Slot(
                            name: name, top: pairs,
                            labelIDs: try numbers(entry["label_ids"], name).map { Int($0) },
                            probabilities: try numbers(entry["probs"], name),
                            entropy: try #require(entry["entropy"]?.doubleValue, "\(name)")))
                } else if entry["confidence"] != nil {
                    confidences.append(
                        ConfidenceCase(
                            name: name, probabilities: try numbers(entry["probabilities"], name),
                            confidence: try #require(entry["confidence"]?.doubleValue, "\(name)")))
                } else {
                    Issue.record("\(name) is neither a slot nor a confidence entry")
                }
            }
        }
        return (slots, confidences)
    }

    /// An array of numbers, or an error naming the entry.
    private static func numbers(_ value: JSONValue?, _ name: String) throws -> [Double] {
        let array = try #require(value?.arrayValue, "\(name)")
        return try array.map { try #require($0.doubleValue, "\(name)") }
    }
}
