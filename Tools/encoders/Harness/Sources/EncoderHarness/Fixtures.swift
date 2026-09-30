import Foundation

/// The parts of Fixtures/encoders/verdict.json that the harness reads.
///
/// Tools/encoders/reference.py wrote the file from upstream's VerdictEngine on the CPU in float32.
public struct VerdictReference: Decodable, Sendable {
    public struct Calibrator: Decodable, Sendable {
        public let temperature: Double
        public let perK: [String: Double]

        enum CodingKeys: String, CodingKey {
            case temperature
            case perK = "per_k"
        }
    }

    public struct Read: Decodable, Sendable {
        public let request: String
        public let key: String
        public let type: String
        public let options: Int
        /// The caller's options plus the trailing "insufficient evidence" label.
        public let k: Int
        public let prompt: String
        public let inputIds: [Int]
        public let truncated: Bool
        public let temperature: Double
        public let logits: [Double]
        public let probabilities: [Double]

        enum CodingKeys: String, CodingKey {
            case request, key, type, options, k, prompt, truncated, temperature, logits,
                probabilities
            case inputIds = "input_ids"
        }
    }

    public let maxLength: Int
    public let padTokenId: Int
    public let classTokenIndex: Int
    public let calibrator: Calibrator
    public let reads: [Read]

    enum CodingKeys: String, CodingKey {
        case calibrator, reads
        case maxLength = "max_length"
        case padTokenId = "pad_token_id"
        case classTokenIndex = "class_token_index"
    }
}

/// The parts of Fixtures/encoders/laya.json that the harness reads.
///
/// Tools/encoders/reference.py wrote the file from upstream's LayaEngine on the CPU in float32.
public struct LayaReference: Decodable, Sendable {
    public struct SpecialTokens: Decodable, Sendable {
        public let cls: Int
        public let sep: Int
        public let mask: Int
        public let pad: Int
    }

    public struct Calibration: Decodable, Sendable {
        public let temperatureRaw: [Double]
        public let temperatureByOptionsRaw: [String: Double]
        public let temperature: [Double]
        public let temperatureByOptions: [String: Double]
        public let clamp: [Double]

        enum CodingKeys: String, CodingKey {
            case temperature, clamp
            case temperatureRaw = "temperature_raw"
            case temperatureByOptionsRaw = "temperature_by_options_raw"
            case temperatureByOptions = "temperature_by_options"
        }
    }

    public struct Texts: Decodable, Sendable {
        public let head: String
        public let options: [String]
    }

    public struct Read: Decodable, Sendable {
        public let request: String
        public let key: String
        public let type: String
        public let options: Int
        public let texts: Texts
        public let ids: [Int]
        public let markers: [Int]
        public let qtype: Int
        public let truncated: Bool
        public let bucket: String
        public let temperature: Double
        public let logits: [Double]
        public let probabilitiesUnrounded: [Double]
        public let probabilities: [Double]

        enum CodingKeys: String, CodingKey {
            case request, key, type, options, texts, ids, markers, qtype, truncated, bucket,
                temperature
            case logits, probabilities
            case probabilitiesUnrounded = "probabilities_unrounded"
        }
    }

    public let maxLen: Int
    public let headMaxLen: Int
    public let specialTokens: SpecialTokens
    public let calibration: Calibration
    public let qtypes: [String: Int]
    public let stateTexts: [String: String]
    public let reads: [Read]

    enum CodingKeys: String, CodingKey {
        case calibration, qtypes, reads
        case maxLen = "max_len"
        case headMaxLen = "head_max_len"
        case specialTokens = "special_tokens"
        case stateTexts = "state_texts"
    }
}

/// The generator object every fixture file starts with, for the pins a result records.
public struct FixtureGenerator: Decodable, Sendable {
    public let script: String
    public let version: Int
    public let verdictRevision: String
    public let layaRevision: String

    enum CodingKeys: String, CodingKey {
        case script, version
        case verdictRevision = "verdict_revision"
        case layaRevision = "laya_revision"
    }
}

enum FixtureFile {
    private struct Header: Decodable {
        let generator: FixtureGenerator
    }

    static func decode<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }

    static func generator(of url: URL) throws -> FixtureGenerator {
        try decode(Header.self, from: url).generator
    }
}
