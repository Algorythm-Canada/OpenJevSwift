// The oracle written by Tools/fixtures/mlx_vlm_oracle.py (Fixtures/oracle/reads.json), as far
// as the probe reads it.

import Foundation

struct OracleFile: Decodable {
    let generator: [String: JSONScalar]
    let prompts: [String: OraclePrompt]
    let reads: [OracleRead]
}

/// A generator value: the pins are strings, the version an integer.
enum JSONScalar: Decodable, CustomStringConvertible {
    case string(String)
    case number(Double)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .string(text)
        } else {
            self = .number(try container.decode(Double.self))
        }
    }

    var description: String {
        switch self {
        case .string(let text): text
        case .number(let value): String(value)
        }
    }
}

struct OraclePrompt: Decodable {
    let system: String
    let user: String
    let ids: [Int]
    let tokens: Int
    let cache: [CacheDigest]?
}

struct TensorDigest: Codable {
    let dtype: String
    let shape: [Int]
    let sha256: String
    let sum: Double
    let sumOfSquares: Double
    let maxAbs: Double

    enum CodingKeys: String, CodingKey {
        case dtype, shape, sha256, sum
        case sumOfSquares = "sum_of_squares"
        case maxAbs = "max_abs"
    }
}

struct CacheView: Decodable {
    let positions: [Int]
    let keys: TensorDigest
    let values: TensorDigest
}

struct CacheDigest: Decodable {
    let layer: Int
    let kind: String
    let offset: Int
    let positions: [Int]
    let keys: TensorDigest
    let values: TensorDigest
    let decoderView: CacheView?

    enum CodingKeys: String, CodingKey {
        case layer, kind, offset, positions, keys, values
        case decoderView = "decoder_view"
    }
}

struct OracleSlot: Decodable {
    let pos: Int
    let labelIDs: [Int]

    enum CodingKeys: String, CodingKey {
        case pos
        case labelIDs = "label_ids"
    }
}

struct OracleDistribution: Decodable {
    let probs: [Double]
    let entropy: Double
}

struct OracleRead: Decodable {
    let id: String
    let request: String
    let prompt: String
    let width: Int
    let canvas: [Int]
    let slots: [OracleSlot]
    let steps: Int
    let promptTokens: Int
    let written: [[Int]]
    /// Per slot, `[token id, logprob]` pairs in the order MlxRuntime.read returned them.
    let logprobs: [[[Double]]]
    let distributions: [OracleDistribution]

    enum CodingKeys: String, CodingKey {
        case id, request, prompt, width, canvas, slots, steps, written, logprobs, distributions
        case promptTokens = "prompt_tokens"
    }

    /// The slot maps as `(token id, logprob)` pairs.
    var pairs: [[(tokenID: Int, logprob: Double)]] {
        logprobs.map { slot in slot.map { (tokenID: Int($0[0]), logprob: $0[1]) } }
    }
}
