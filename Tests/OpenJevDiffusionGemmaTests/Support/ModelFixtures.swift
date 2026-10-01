import Foundation
import MLX
import MLXLMCommon
import OpenJevDiffusionGemma
import Testing

/// The configurations, checkpoint files and oracle data the model tests use.
enum ModelFixtures {
    /// A tiny text configuration for the synthetic tests: two layers, sliding then full; hidden
    /// 64; 2 heads; 1 KV head; head_dim 16 and global_head_dim 32; intermediate 32;
    /// moe_intermediate_size 16; 8 experts, top 2; vocab 128; sliding_window 8.
    static let tinyTextJSON = """
        {
            "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 2,
            "num_key_value_heads": 1, "num_global_key_value_heads": 1, "head_dim": 16,
            "global_head_dim": 32, "intermediate_size": 32, "moe_intermediate_size": 16,
            "num_experts": 8, "top_k_experts": 2, "vocab_size": 128, "sliding_window": 8,
            "rms_norm_eps": 1e-6
        }
        """

    /// The tiny model's quantization: 4 bits in groups of 32, 8 bits for the embedding, and the
    /// experts' `down_proj` left unquantized because its 16 inputs are fewer than a group.
    static let tinyQuantizationJSON = """
        {
            "group_size": 32, "bits": 4, "mode": "affine",
            "model.decoder.embed_tokens": {"group_size": 32, "bits": 8},
            "model.decoder.layers.0.experts.down_proj": false,
            "model.decoder.layers.1.experts.down_proj": false
        }
        """

    /// The tiny checkpoint's `config.json`.
    static let tinyConfigJSON = """
        {"model_type": "diffusion_gemma", "text_config": \(tinyTextJSON),
         "quantization": \(tinyQuantizationJSON)}
        """

    static func tinyConfiguration() throws -> DiffusionGemmaConfiguration {
        try DiffusionGemmaConfiguration(data: Data(tinyConfigJSON.utf8))
    }

    static func tinyText() throws -> DiffusionGemmaTextConfiguration {
        try tinyConfiguration().text
    }

    /// The tiny configuration's per-layer quantization.
    static func tinyPerLayerQuantization() throws -> BaseConfiguration.PerLayerQuantization {
        try #require(tinyConfiguration().quantization).perLayerQuantization
    }

    /// The pinned checkpoint's configuration, from Fixtures/model/config.json.
    static func checkpointConfiguration() throws -> DiffusionGemmaConfiguration {
        let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent("model/config.json")
        let object = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let config = try JSONSerialization.data(withJSONObject: try #require(object["config"]))
        return try DiffusionGemmaConfiguration(data: config)
    }

    /// Fixtures/model/weight_map.json's `weight_map`: the checkpoint's 1,647 tensor names and
    /// their shards.
    static func checkpointWeightMap() throws -> [String: String] {
        struct File: Decodable {
            let weightMap: [String: String]
            enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
        }
        let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent(
            "model/weight_map.json")
        return try JSONDecoder().decode(File.self, from: Data(contentsOf: url)).weightMap
    }

    // MARK: The live checkpoint

    /// The checkpoint directory: `OPENJEV_TEST_MODEL`, else the Hugging Face cache snapshot.
    static let checkpointDirectory: URL = {
        if let path = ProcessInfo.processInfo.environment[TokenizerFixtures.modelVariable],
            !path.isEmpty
        {
            return URL(fileURLWithPath: path)
        }
        return TokenizerFixtures.cachedSnapshot
    }()

    /// True when the checkpoint directory holds `config.json` and the shard index.
    static var checkpointAvailable: Bool {
        ["config.json", "model.safetensors.index.json"].allSatisfy {
            FileManager.default.fileExists(
                atPath: checkpointDirectory.appendingPathComponent($0).path)
        }
    }

    /// The skip message of the live tests. It names `OPENJEV_TEST_MODEL`, the skip CI accepts.
    static let missingCheckpointMessage = Comment(
        rawValue: "\(TokenizerFixtures.modelVariable) is unset and \(checkpointDirectory.path) "
            + "lacks the checkpoint; set \(TokenizerFixtures.modelVariable) to the "
            + "diffusiongemma-26B-A4B-it-4bit directory")

    // MARK: Stage dumps

    /// One stage dump written by Tools/oracle/stage_dump.py.
    struct StageDump: Sendable, CustomTestStringConvertible {
        let url: URL
        /// The oracle prompt it records, `request/gGROUP`, read from the file name
        /// `stages-REQUEST-gGROUP.safetensors`; nil for another name.
        var promptKey: String? {
            let stem = url.deletingPathExtension().lastPathComponent
            guard stem.hasPrefix("stages-"), let dash = stem.lastIndex(of: "-") else { return nil }
            let request = stem[stem.index(stem.startIndex, offsetBy: 7)..<dash]
            return "\(request)/\(stem[stem.index(after: dash)...])"
        }
        var testDescription: String { url.lastPathComponent }
    }

    /// The repository root.
    static let repositoryRoot = TokenizerFixtures.fixturesDirectory.deletingLastPathComponent()

    /// The environment variable naming the stage dumps, separated by `:`.
    static let stagesVariable = "OPENJEV_TEST_STAGES"

    /// The dumps the parity test compares with: those `OPENJEV_TEST_STAGES` names, else the
    /// two Tools/oracle/results holds after the commands in docs/spikes/backend-validation.md
    /// "Reproducing" (quickstart/g0, 182 tokens; indexed_12_mixed/g0, 1,572, past the window).
    static let stageDumps: [StageDump] = {
        if let paths = ProcessInfo.processInfo.environment[stagesVariable], !paths.isEmpty {
            return paths.split(separator: ":").map {
                StageDump(url: URL(fileURLWithPath: String($0)))
            }
        }
        return ["quickstart-g0", "indexed_12_mixed-g0"].map {
            StageDump(
                url: repositoryRoot.appendingPathComponent(
                    "Tools/oracle/results/stages-\($0).safetensors"))
        }
    }()

    // MARK: The oracle

    struct Oracle: Decodable {
        struct Prompt: Decodable {
            let ids: [Int]
        }
        struct Rope: Decodable {
            let float32Bits: [UInt32]
            enum CodingKeys: String, CodingKey { case float32Bits = "float32_bits" }
        }
        let prompts: [String: Prompt]
        let rope: Rope
    }

    /// Fixtures/oracle/reads.json's prompts and RoPE table.
    static func oracle() throws -> Oracle {
        let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent("oracle/reads.json")
        return try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: url))
    }
}
