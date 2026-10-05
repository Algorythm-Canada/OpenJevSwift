import CryptoKit
import Foundation
import OpenJevCore
import OpenJevDiffusionGemma
import Testing

/// Fixtures/vision: the images and what upstream's processor made of them
/// (Tools/fixtures/vision_oracle.py, issue #46).
///
/// The synthetic images are committed, so the tests of them run everywhere. Upstream's hot dog
/// photo is read from the pinned Upstream/openjev checkout (`make upstream`), and the full tensors
/// from Tools/oracle/results/vision (the oracle's run); the tests that need either are opt-in and
/// skip naming `OPENJEV_TEST_MODEL`, as the tokenizer tests do.
enum VisionFixtures {
    /// The largest difference any compared value may have (issue #46's bound).
    static let bound: Float = 1e-3

    /// The repository root, found relative to this source file.
    static let root = TokenizerFixtures.fixturesDirectory.deletingLastPathComponent()

    /// Fixtures/vision.
    static let directory = TokenizerFixtures.fixturesDirectory.appendingPathComponent("vision")

    /// Upstream's tests/data/hotdog.jpg in the pinned checkout.
    static let hotdog = root.appendingPathComponent("Upstream/openjev/tests/data/hotdog.jpg")

    /// The oracle's full tensors.
    static let tensors = root.appendingPathComponent("Tools/oracle/results/vision")

    /// True when the hot dog photo is checked out.
    static var hotdogAvailable: Bool { FileManager.default.fileExists(atPath: hotdog.path) }

    /// True when the oracle's full tensors are present for every image.
    static var tensorsAvailable: Bool {
        (try? preprocessing())?.images.allSatisfy {
            FileManager.default.fileExists(
                atPath: tensors.appendingPathComponent("\($0.name).pixel_values.f32").path)
        } ?? false
    }

    static let hotdogMessage = Comment(
        rawValue:
            "\(TokenizerFixtures.modelVariable) tests: Upstream/openjev/tests/data/hotdog.jpg "
            + "is missing; run make upstream")

    static let tensorsMessage = Comment(
        rawValue: "\(TokenizerFixtures.modelVariable) tests: Tools/oracle/results/vision lacks the "
            + "oracle's tensors; run Tools/fixtures/vision_oracle.py --only preprocessing")

    /// One image's record.
    struct Image {
        var name: String
        var file: String
        var contentType: String
        var bytes: Int
        var sha256: String
        var frames: Int
        var decodedWidth: Int
        var decodedHeight: Int
        var decodedSHA256: String
        var resizedWidth: Int
        var resizedHeight: Int
        var softTokens: Int
        var shape: [Int]
        var valuesSHA256: String
        /// Per channel: mean, std, min, max.
        var channels: [[Double]]
        /// Flat position in the C-order `(1, 3, H, W)` tensor and the value there.
        var samples: [(position: Int, value: Float)]

        /// Where the image's bytes are: the committed file, or upstream's for the hot dog.
        var url: URL {
            name == "hotdog"
                ? VisionFixtures.hotdog : VisionFixtures.directory.appendingPathComponent(file)
        }

        /// The image's bytes, checked against the recorded digest.
        func data() throws -> Data {
            let data = try Data(contentsOf: url)
            #expect(data.count == bytes, "\(file) is \(data.count) bytes, not \(bytes)")
            #expect(VisionFixtures.sha256(data) == sha256, "\(file) is not the recorded image")
            return data
        }
    }

    /// One prompt's record.
    struct Prompt {
        var key: String
        var images: [String]
        var system: String
        var state: String
        var ids: [Int]
        var mmTokenTypeIDs: [Int]
        var softTokens: [Int]
        var stacked: Bool
        var shapes: [[Int]]
    }

    /// One state of issue #124 as the text of a prompt with one image, from `state_prompts`.
    struct StatePrompt {
        var key: String
        var images: [String]
        var system: String
        var state: String
        var ids: [Int]
        var mmTokenTypeIDs: [Int]
        var softTokens: [Int]
    }

    /// One row of the resize rule.
    struct BudgetRow {
        var width: Int
        var height: Int
        /// The resized width and height and the soft tokens, or nil when upstream raised.
        var target: (width: Int, height: Int, softTokens: Int)?
    }

    /// One small GIF for the rest of Pillow's GIF reader, and what upstream decoded it to.
    struct GIFCase {
        var name: String
        var data: Data
        var sha256: String
        /// The decoded width, height and the SHA-256 of the RGB bytes, or nil when upstream raised.
        var decoded: (width: Int, height: Int, sha256: String)?
        /// The exception upstream raised, when it did.
        var error: String?
    }

    struct Preprocessing {
        var images: [Image]
        var prompts: [Prompt]
        var statePrompts: [StatePrompt]
        var budget: [BudgetRow]
        var gifCases: [GIFCase]
        var textPromptIDs: [Int]
        var processor: JSONValue
    }

    private static let loaded = Result { try load() }

    /// preprocessing.json, parsed once.
    static func preprocessing() throws -> Preprocessing { try loaded.get() }

    private static func load() throws -> Preprocessing {
        let root = try JSONParser().parse(
            Data(contentsOf: directory.appendingPathComponent("preprocessing.json")))
        func ints(_ value: JSONValue?) throws -> [Int] { try TokenizerFixtures.ints(value) }
        var images: [Image] = []
        for (name, row) in try #require(root["images"]?.objectValue) {
            let pixels = try #require(row["pixel_values"])
            let channels = try #require(pixels["channels"]?.arrayValue).map { channel in
                try ["mean", "std", "min", "max"].map { try #require(channel[$0]?.doubleValue) }
            }
            let samples = try #require(pixels["samples"]?.arrayValue).map { pair in
                let values = try #require(pair.arrayValue)
                return (
                    position: try #require(values[0].intValue),
                    value: Float(try #require(values[1].doubleValue))
                )
            }
            images.append(
                Image(
                    name: name, file: try #require(row["file"]?.stringValue),
                    contentType: try #require(row["content_type"]?.stringValue),
                    bytes: try #require(row["bytes"]?.intValue),
                    sha256: try #require(row["sha256"]?.stringValue),
                    frames: try #require(row["frames"]?.intValue),
                    decodedWidth: try #require(row["decoded"]?["width"]?.intValue),
                    decodedHeight: try #require(row["decoded"]?["height"]?.intValue),
                    decodedSHA256: try #require(row["decoded"]?["sha256"]?.stringValue),
                    resizedWidth: try #require(row["resized"]?["width"]?.intValue),
                    resizedHeight: try #require(row["resized"]?["height"]?.intValue),
                    softTokens: try #require(row["soft_tokens"]?.intValue),
                    shape: try ints(pixels["shape"]),
                    valuesSHA256: try #require(pixels["sha256"]?.stringValue),
                    channels: channels, samples: samples))
        }
        var prompts: [Prompt] = []
        for (key, row) in try #require(root["prompts"]?.objectValue) {
            let pixels = try #require(row["pixel_values"])
            let stacked = pixels["stacked"]?.boolValue == true
            let shapes =
                stacked
                ? [try ints(pixels["shape"])]
                : try #require(pixels["shapes"]?.arrayValue).map { try ints($0) }
            prompts.append(
                Prompt(
                    key: key, images: try TokenizerFixtures.strings(row["images"]),
                    system: try #require(row["system"]?.stringValue),
                    state: try #require(row["state"]?.stringValue), ids: try ints(row["ids"]),
                    mmTokenTypeIDs: try ints(row["mm_token_type_ids"]),
                    softTokens: try ints(row["soft_tokens"]), stacked: stacked, shapes: shapes))
        }
        var statePrompts: [StatePrompt] = []
        for (key, row) in try #require(root["state_prompts"]?.objectValue) {
            statePrompts.append(
                StatePrompt(
                    key: key, images: try TokenizerFixtures.strings(row["images"]),
                    system: try #require(row["system"]?.stringValue),
                    state: try #require(row["state"]?.stringValue), ids: try ints(row["ids"]),
                    mmTokenTypeIDs: try ints(row["mm_token_type_ids"]),
                    softTokens: try ints(row["soft_tokens"])))
        }
        let budget = try #require(root["budget_rule"]?.arrayValue).map { row in
            let target: (Int, Int, Int)? =
                row["error"] == nil
                ? (
                    try #require(row["target_width"]?.intValue),
                    try #require(row["target_height"]?.intValue),
                    try #require(row["soft_tokens"]?.intValue)
                ) : nil
            return BudgetRow(
                width: try #require(row["width"]?.intValue),
                height: try #require(row["height"]?.intValue), target: target)
        }
        var gifCases: [GIFCase] = []
        for (name, row) in try #require(root["gif_cases"]?.objectValue) {
            let decoded: (Int, Int, String)? =
                row["decoded"] == nil
                ? nil
                : (
                    try #require(row["decoded"]?["width"]?.intValue),
                    try #require(row["decoded"]?["height"]?.intValue),
                    try #require(row["decoded"]?["sha256"]?.stringValue)
                )
            let base64 = try #require(row["base64"]?.stringValue)
            gifCases.append(
                GIFCase(
                    name: name, data: try #require(Data(base64Encoded: base64)),
                    sha256: try #require(row["sha256"]?.stringValue), decoded: decoded,
                    error: row["error"]?.stringValue))
        }
        return Preprocessing(
            images: images, prompts: prompts, statePrompts: statePrompts, budget: budget,
            gifCases: gifCases,
            textPromptIDs: try ints(root["text_prompt"]?["ids"]),
            processor: try #require(root["processor"]))
    }

    /// The hex SHA-256 of `data`.
    static func sha256<D: DataProtocol>(_ data: D) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The hex SHA-256 of float32 values' little-endian bytes, as NumPy's `tobytes()` gives them.
    static func sha256(_ values: [Float]) -> String {
        values.withUnsafeBytes { sha256($0) }
    }

    /// How far processed values are from an image's record.
    struct Comparison: CustomStringConvertible {
        var name: String
        var shapeMatches: Bool
        /// The largest difference at the 4,096 sampled positions.
        var maxSampleDifference: Float
        /// The largest difference of a channel statistic.
        var maxStatisticDifference: Double
        /// Whether the float32 bytes hash to the recorded digest: bit for bit.
        var exact: Bool

        var withinBound: Bool {
            shapeMatches && maxSampleDifference <= VisionFixtures.bound
                && maxStatisticDifference <= Double(VisionFixtures.bound)
        }

        var description: String {
            "\(name): shape \(shapeMatches ? "matches" : "differs"), samples within "
                + "\(maxSampleDifference), statistics within \(maxStatisticDifference)"
                + (exact ? ", bit for bit" : "")
        }
    }

    /// Compares `(3, H, W)` values of a `width` by `height` image with `image`'s record.
    static func compare(_ values: [Float], width: Int, height: Int, with image: Image)
        -> Comparison
    {
        let shapeMatches =
            [1, 3, height, width] == image.shape && values.count == 3 * width * height
        guard shapeMatches else {
            return Comparison(
                name: image.name, shapeMatches: false, maxSampleDifference: .infinity,
                maxStatisticDifference: .infinity, exact: false)
        }
        var maxSample: Float = 0
        for sample in image.samples {
            maxSample = max(maxSample, abs(values[sample.position] - sample.value))
        }
        let plane = width * height
        var maxStatistic = 0.0
        for c in 0..<3 {
            let channel = values[(c * plane)..<((c + 1) * plane)].map(Double.init)
            let mean = channel.reduce(0, +) / Double(plane)
            let variance = channel.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(plane)
            let stats = [mean, variance.squareRoot(), channel.min() ?? 0, channel.max() ?? 0]
            for (mine, theirs) in zip(stats, image.channels[c]) {
                maxStatistic = max(maxStatistic, abs(mine - theirs))
            }
        }
        return Comparison(
            name: image.name, shapeMatches: true, maxSampleDifference: maxSample,
            maxStatisticDifference: maxStatistic, exact: sha256(values) == image.valuesSHA256)
    }

    /// The oracle's full float32 tensor of an image.
    static func fullTensor(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: tensors.appendingPathComponent("\(name).pixel_values.f32"))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// The oracle's decoded RGB bytes of an image.
    static func fullRGB(_ name: String) throws -> [UInt8] {
        [UInt8](try Data(contentsOf: tensors.appendingPathComponent("\(name).rgb.u8")))
    }

    /// The largest difference between two equally long arrays, and how many values differ.
    static func difference(_ a: [Float], _ b: [Float]) -> (max: Float, differing: Int) {
        var largest: Float = 0
        var differing = 0
        for (x, y) in zip(a, b) where x != y {
            largest = max(largest, abs(x - y))
            differing += 1
        }
        return (a.count == b.count ? largest : .infinity, differing)
    }
}
