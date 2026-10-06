import Foundation
import MLX
import Testing

/// Fixtures/generation/generation.json, which Tools/fixtures/generation_oracle.py records through
/// upstream's `MlxRuntime.generate` and `MlxEngine.think` on mlx-vlm 0.6.15.
struct GenerationOracle: Decodable {
    /// A float32 array as its shape and bit patterns.
    struct Floats: Decodable {
        let shape: [Int]
        let float32Bits: [UInt32]
        enum CodingKeys: String, CodingKey {
            case shape
            case float32Bits = "float32_bits"
        }
        var array: MLXArray {
            MLXArray(float32Bits.map { Float(bitPattern: $0) }, shape)
        }
    }
    struct Ints: Decodable {
        let shape: [Int]
        let values: [Int]
        var array: MLXArray { MLXArray(values.map(Int32.init), shape) }
    }
    struct Bools: Decodable {
        let shape: [Int]
        let values: [Bool]
        var array: MLXArray { MLXArray(values, shape) }
    }

    struct Settings: Decodable {
        let eosTokenIDs: [Int]
        let canvasLength: Int
        let minCanvasLength: Int
        let maxDenoisingSteps: Int
        let confidenceThreshold: Double
        let thoughtOpen: [Int]
        let thoughtClose: [Int]
        let scaffold: [Int]
        enum CodingKeys: String, CodingKey {
            case eosTokenIDs = "eos_token_ids"
            case canvasLength = "canvas_length"
            case minCanvasLength = "min_canvas_length"
            case maxDenoisingSteps = "max_denoising_steps"
            case confidenceThreshold = "confidence_threshold"
            case thoughtOpen = "thought_open"
            case thoughtClose = "thought_close"
            case scaffold
        }
    }

    struct Sampler: Decodable {
        struct Draws: Decodable {
            let seed: UInt64
            let vocabSize: Int
            let draws: [Ints]
            enum CodingKeys: String, CodingKey {
                case seed, draws
                case vocabSize = "vocab_size"
            }
        }
        struct Temperature: Decodable {
            let maxDenoisingSteps: Int
            let values: [Double]
            let float32Bits: [UInt32]
            enum CodingKeys: String, CodingKey {
                case values
                case maxDenoisingSteps = "max_denoising_steps"
                case float32Bits = "float32_bits"
            }
        }
        struct SampleCase: Decodable {
            let temperature: Double
            let seed: UInt64?
            let ids: Ints
        }
        struct SampleCanvas: Decodable {
            let logits: Floats
            let cases: [SampleCase]
        }
        struct Probability: Decodable {
            let logits: Floats
            let tokenIDs: Ints
            let probability: Floats
            enum CodingKeys: String, CodingKey {
                case logits, probability
                case tokenIDs = "token_ids"
            }
        }
        struct Entropy: Decodable {
            let logits: Floats
            let entropy: Floats
        }
        struct EntropyMask: Decodable {
            let entropy: Floats
            let entropyBound: Double
            let mask: Bools
            enum CodingKeys: String, CodingKey {
                case entropy, mask
                case entropyBound = "entropy_bound"
            }
        }
        struct ConfidenceCase: Decodable {
            let threshold: Double
            let forceAll: Bool
            let mask: Bools
            enum CodingKeys: String, CodingKey {
                case threshold, mask
                case forceAll = "force_all"
            }
        }
        struct ConfidenceMask: Decodable {
            let confidence: Floats
            let unrevealed: Bools
            let cases: [ConfidenceCase]
        }
        struct StableConfig: Decodable {
            let confidenceThreshold: Double
            let stabilityThreshold: Int
            enum CodingKeys: String, CodingKey {
                case confidenceThreshold = "confidence_threshold"
                case stabilityThreshold = "stability_threshold"
            }
        }
        struct StableCase: Decodable {
            let config: StableConfig?
            let results: [Bool]
        }
        struct Stable: Decodable {
            let logits: [String: Floats]
            let canvases: [String: Ints]
            let sequence: [[String]]
            let cases: [StableCase]
        }
        let initializeCanvas: [Draws]
        let linearTemperature: Temperature
        let sampleCanvas: SampleCanvas
        let tokenProbability: Probability
        let tokenEntropy: [Entropy]
        let entropyTransferMask: [EntropyMask]
        let confidenceTransferMask: ConfidenceMask
        let stableAndConfident: Stable
        enum CodingKeys: String, CodingKey {
            case initializeCanvas = "initialize_canvas"
            case linearTemperature = "linear_temperature"
            case sampleCanvas = "sample_canvas"
            case tokenProbability = "token_probability"
            case tokenEntropy = "token_entropy"
            case entropyTransferMask = "entropy_transfer_mask"
            case confidenceTransferMask = "confidence_transfer_mask"
            case stableAndConfident = "stable_and_confident"
        }
    }

    struct Block: Decodable {
        let canvasLength: Int
        let initialCanvas: [Int]
        let steps: Int
        let canvasDraws: Int
        let ended: String
        let finalCanvas: [Int]
        let committed: [Int]
        enum CodingKeys: String, CodingKey {
            case steps, ended, committed
            case canvasLength = "canvas_length"
            case initialCanvas = "initial_canvas"
            case canvasDraws = "canvas_draws"
            case finalCanvas = "final_canvas"
        }
    }

    /// One `emit(text, token)` call.
    struct Piece: Decodable, Equatable, CustomStringConvertible {
        let text: String
        let token: Int?
        init(text: String, token: Int?) {
            self.text = text
            self.token = token
        }
        init(from decoder: any Decoder) throws {
            var c = try decoder.unkeyedContainer()
            text = try c.decode(String.self)
            token = try c.decodeIfPresent(Int.self)
        }
        var description: String { "(\(text.debugDescription), \(token.map(String.init) ?? "nil"))" }
    }

    struct Generation: Decodable, CustomTestStringConvertible {
        let name: String
        let maxTokens: Int
        let stopIDs: [Int]
        let skipSpecialTokenIDs: [Int]
        let seed: UInt64
        let prompt: [Int]
        let promptTokens: Int
        let finishReason: String
        let stopToken: Int?
        let generated: [Int]
        let text: String
        let pieces: [Piece]
        let blocks: [Block]
        enum CodingKeys: String, CodingKey {
            case name, seed, prompt, generated, text, pieces, blocks
            case maxTokens = "max_tokens"
            case stopIDs = "stop_ids"
            case skipSpecialTokenIDs = "skip_special_token_ids"
            case promptTokens = "prompt_tokens"
            case finishReason = "finish_reason"
            case stopToken = "stop_token"
        }
        var testDescription: String { name }
    }

    struct Thought: Decodable {
        let prompt: [Int]
        let budget: Int
        let stopIDs: [Int]
        let generated: [Int]
        let promptTokens: Int
        let finishReason: String
        let thoughtTokens: Int
        let billed: Int
        let prefix: [Int]
        let thought: [Int]
        let system: String
        let user: String
        enum CodingKeys: String, CodingKey {
            case prompt, budget, generated, billed, prefix, thought, system, user
            case stopIDs = "stop_ids"
            case promptTokens = "prompt_tokens"
            case finishReason = "finish_reason"
            case thoughtTokens = "thought_tokens"
        }
    }

    struct Think: Decodable, CustomTestStringConvertible {
        let name: String
        let seed: UInt64
        let thoughts: [Thought]
        let inputTokens: Int
        let outputTokens: Int
        enum CodingKeys: String, CodingKey {
            case name, seed, thoughts
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
        }
        var testDescription: String { name }
    }

    let settings: Settings
    let sampler: Sampler
    let generations: [Generation]
    let think: [Think]

    static let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent(
        "generation/generation.json")

    static let shared: Result<GenerationOracle, any Error> = Result {
        try JSONDecoder().decode(GenerationOracle.self, from: Data(contentsOf: url))
    }

    static func load() throws -> GenerationOracle {
        try shared.get()
    }

    /// The recorded generations, for parameterized tests; empty when the file does not decode,
    /// which ``load()`` reports.
    static var generationCases: [Generation] {
        (try? load().generations) ?? []
    }
}
