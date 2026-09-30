import Foundation
import MLXLMCommon
import OpenJevDiffusionGemma
import Testing

/// Fixtures/model/config.json: the checkpoint's config.json and generation_config.json objects,
/// wrapped with the generator object every fixture starts with.
private struct CheckpointFixture: Decodable {
    let config: DiffusionGemmaConfiguration
    let generationConfig: DiffusionGemmaGenerationConfiguration

    enum CodingKeys: String, CodingKey {
        case config
        case generationConfig = "generation_config"
    }

    static func load() throws -> CheckpointFixture {
        let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent("model/config.json")
        return try JSONDecoder().decode(CheckpointFixture.self, from: Data(contentsOf: url))
    }
}

private typealias LayerType = DiffusionGemmaTextConfiguration.LayerType

private let sliding = LayerType.slidingAttention
private let full = LayerType.fullAttention

/// `[sliding ×5, full] ×5`, the pinned checkpoint's layers.
private let checkpointLayers = Array(
    repeating: Array(repeating: sliding, count: 5) + [full], count: 5
).flatMap { $0 }

/// Decodes a hand-written config.json.
private func decode(_ json: String) throws -> DiffusionGemmaConfiguration {
    try DiffusionGemmaConfiguration(data: Data(json.utf8))
}

/// The error decoding `json` throws, which must be a configuration error.
private func decodingError(_ json: String) throws -> DiffusionGemmaConfigurationError {
    do {
        _ = try decode(json)
    } catch let error as DiffusionGemmaConfigurationError {
        return error
    }
    Issue.record("\(json) decoded")
    throw CancellationError()
}

@Suite("DiffusionGemma configuration")
struct ConfigurationTests {
    @Test("The checkpoint's top-level values")
    func topLevel() throws {
        let config = try CheckpointFixture.load().config
        #expect(config.modelType == "diffusion_gemma")
        #expect(config.architectures == ["DiffusionGemmaForBlockDiffusion"])
        #expect(config.canvasLength == 256)
        #expect(config.boiTokenID == 255_999)
        #expect(config.eoiTokenID == 258_882)
        #expect(config.imageTokenID == 258_880)
        #expect(config.videoTokenID == nil)
        #expect(config.eosTokenIDs == [1, 106, 50])
        #expect(config.dtype == "bfloat16")
        #expect(config.tieWordEmbeddings)
        #expect(config.visionSoftTokensPerImage == 280)
    }

    @Test("The checkpoint's text_config")
    func text() throws {
        let text = try CheckpointFixture.load().config.text
        #expect(text.modelType == "diffusion_gemma_text")
        #expect(text.vocabSize == 262_144)
        #expect(text.hiddenSize == 2816)
        #expect(text.intermediateSize == 2112)
        #expect(text.moeIntermediateSize == 704)
        #expect(text.numHiddenLayers == 30)
        #expect(text.numAttentionHeads == 16)
        #expect(text.numKeyValueHeads == 8)
        #expect(text.numGlobalKeyValueHeads == 2)
        #expect(text.headDim == 256)
        #expect(text.globalHeadDim == 512)
        #expect(text.hiddenActivation == "gelu_pytorch_tanh")
        #expect(text.rmsNormEps == 1e-6)
        #expect(text.maxPositionEmbeddings == 262_144)
        #expect(text.padTokenID == 0)
        #expect(text.eosTokenIDs == [1])
        #expect(text.bosTokenID == 2)
        #expect(text.tieWordEmbeddings)
        #expect(text.slidingWindow == 1024)
        #expect(text.layerTypes == checkpointLayers)
        #expect(text.finalLogitSoftcapping == 30)
        #expect(text.useBidirectionalAttention == "vision")
        #expect(text.numExperts == 128)
        #expect(text.topKExperts == 8)
        #expect(text.attentionBias == false)
        #expect(text.attentionDropout == 0)
        #expect(
            text.ropeParameters == [
                sliding: .init(ropeType: "default", ropeTheta: 10_000),
                full: .init(
                    ropeType: "proportional", ropeTheta: 1_000_000, partialRotaryFactor: 0.25),
            ])
        #expect(text.fullAttentionLayers == [5, 11, 17, 23, 29])
        #expect(text.headDim(for: sliding) == 256)
        #expect(text.headDim(for: full) == 512)
        #expect(text.keyValueHeads(for: sliding) == 8)
        #expect(text.keyValueHeads(for: full) == 2)
        #expect(text.ropeParameters(for: full).partialRotaryFactor == 0.25)
        #expect(text.ropeParameters(for: sliding).ropeTheta == 10_000)
    }

    @Test("The checkpoint's vision_config")
    func vision() throws {
        let vision = try #require(try CheckpointFixture.load().config.vision)
        #expect(vision.modelType == "gemma4_vision")
        #expect(vision.hiddenLayers == 27)
        #expect(vision.hiddenSize == 1152)
        #expect(vision.intermediateSize == 4304)
        #expect(vision.attentionHeads == 16)
        #expect(vision.keyValueHeads == 16)
        #expect(vision.headDim == 72)
        #expect(vision.patchSize == 16)
        #expect(vision.rmsNormEps == 1e-6)
        #expect(vision.defaultOutputLength == 280)
        #expect(vision.positionEmbeddingSize == 10_240)
        #expect(vision.poolingKernelSize == 3)
        #expect(vision.useClippedLinears == false)
        #expect(vision.standardize)
        #expect(vision.ropeParameters["rope_type"] == .string("default"))
        let theta = vision.ropeParameters["rope_theta"]
        #expect(theta == .float(100) || theta == .int(100))
    }

    @Test("The checkpoint's quantization: 4-bit default and 236 8-bit overrides")
    func quantization() throws {
        let quantization = try #require(try CheckpointFixture.load().config.quantization)
        #expect(
            quantization.defaultQuantization
                == .init(groupSize: 64, bits: 4, mode: .affine))
        var expected: Set<String> = ["model.decoder.embed_tokens"]
        for layer in 0..<30 {
            let prefix = "model.decoder.layers.\(layer)"
            for module in [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj",
                "mlp.up_proj", "mlp.down_proj", "router.proj",
            ] {
                expected.insert("\(prefix).\(module)")
            }
            if checkpointLayers[layer] == sliding {
                expected.insert("\(prefix).self_attn.v_proj")
            }
        }
        #expect(expected.count == 236)
        #expect(Set(quantization.overrides.keys) == expected)
        #expect(quantization.overrides.values.allSatisfy { $0 == .init(groupSize: 64, bits: 8) })
        #expect(quantization.unquantizedModules.isEmpty)
        for layer in [5, 11, 17, 23, 29] {
            #expect(quantization.overrides["model.decoder.layers.\(layer).self_attn.v_proj"] == nil)
        }

        let perLayer = quantization.perLayerQuantization
        let cases: [(String, Int)] = [
            ("model.decoder.embed_tokens", 8),
            ("model.decoder.layers.7.self_attn.v_proj", 8),
            ("model.decoder.layers.7.mlp.up_proj", 8),
            ("model.decoder.layers.7.router.proj", 8),
            ("model.decoder.layers.7.experts.gate_up_proj", 4),
            ("model.decoder.layers.5.self_attn.v_proj", 4),
            ("model.decoder.self_conditioning.gate_proj", 4),
            ("model.encoder.embed_vision.embedding_projection", 4),
        ]
        for (module, bits) in cases {
            let own = try #require(quantization.quantization(forModule: module))
            #expect(own.bits == bits, "\(module)")
            #expect(own.groupSize == 64, "\(module)")
            let base = try #require(perLayer.quantization(layer: module))
            #expect(base.bits == bits, "\(module)")
            #expect(base.groupSize == 64, "\(module)")
            #expect(base.mode == .affine, "\(module)")
        }
    }

    @Test("An override without mode is affine, whatever the default's mode")
    func overrideModeIsAffine() throws {
        let config = try decode(
            #"""
            {"text_config": {},
             "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4",
                              "a.plain": {"group_size": 64, "bits": 8},
                              "a.explicit": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}}
            """#)
        let quantization = try #require(config.quantization)
        #expect(quantization.defaultQuantization == .init(groupSize: 32, bits: 4, mode: .mxfp4))
        #expect(quantization.quantization(forModule: "a.plain")?.mode == .affine)
        #expect(quantization.quantization(forModule: "a.explicit")?.mode == .mxfp4)
        #expect(quantization.quantization(forModule: "a.other")?.mode == .mxfp4)
        let perLayer = quantization.perLayerQuantization
        #expect(perLayer.quantization(layer: "a.plain")?.mode == .affine)
        #expect(perLayer.quantization(layer: "a.explicit")?.mode == .mxfp4)
        #expect(perLayer.quantization(layer: "a.other")?.mode == .mxfp4)
    }

    @Test("generation_config in config.json equals generation_config.json")
    func generation() throws {
        let fixture = try CheckpointFixture.load()
        let generation = try #require(fixture.config.generation)
        #expect(generation == fixture.generationConfig)
        #expect(generation.maxDenoisingSteps == 48)
        #expect(generation.sampler?.className == "EntropyBoundSamplerConfig")
        #expect(generation.sampler?.entropyBound == 0.1)
        #expect(generation.confidenceThreshold == 0.005)
        #expect(generation.stabilityThreshold == 1)
        #expect(generation.tMax == 0.8)
        #expect(generation.tMin == 0.4)
        #expect(generation.maxNewTokens == 256)
        #expect(generation.eosTokenIDs == [1, 106, 50])
        #expect(generation.padTokenID == 0)
    }

    @Test("Encoding and decoding again gives the same value")
    func roundTrip() throws {
        let config = try CheckpointFixture.load().config
        let again = try DiffusionGemmaConfiguration(data: JSONEncoder().encode(config))
        #expect(again == config)
    }

    /// The layers config.py derives for each layer count the test tries.
    fileprivate static let derivedLayers: [Int: [LayerType]] = [
        30: checkpointLayers,
        7: [sliding, sliding, sliding, sliding, sliding, full, full],
        1: [full],
        31: checkpointLayers + [full],
    ]

    @Test(
        "Derived layer types: five sliding then one full, repeated, last forced full",
        arguments: [30, 7, 1, 31])
    func derivedLayerTypes(count: Int) throws {
        let expected = try #require(Self.derivedLayers[count])
        let config = try decode(
            #"{"model_type": "diffusion_gemma", "text_config": {"num_hidden_layers": \#(count)}}"#)
        #expect(config.text.layerTypes == expected)
        #expect(DiffusionGemmaTextConfiguration.defaultLayerTypes(count: count) == expected)
    }

    @Test("A minimal config takes mlx-vlm's defaults")
    func defaults() throws {
        let config = try decode(#"{"model_type": "diffusion_gemma", "text_config": {}}"#)
        #expect(config.canvasLength == 256)
        #expect(config.boiTokenID == 255_999)
        #expect(config.eosTokenIDs == [])
        #expect(config.vision == nil)
        #expect(config.quantization == nil)
        #expect(config.generation == nil)
        let text = config.text
        #expect(text.modelType == "diffusion_gemma_text")
        #expect(text.vocabSize == 262_144)
        #expect(text.hiddenSize == 2816)
        #expect(text.numHiddenLayers == 30)
        #expect(text.numKeyValueHeads == 8)
        #expect(text.numGlobalKeyValueHeads == 2)
        #expect(text.globalHeadDim == 512)
        #expect(text.rmsNormEps == 1e-6)
        #expect(text.slidingWindow == 1024)
        #expect(text.finalLogitSoftcapping == 30)
        #expect(text.useBidirectionalAttention == "vision")
        #expect(text.eosTokenIDs == [1])
        #expect(text.bosTokenID == 2)
        #expect(text.topKExperts == 8)
        #expect(text.layerTypes == checkpointLayers)
        #expect(text.ropeParameters(for: sliding) == .init(ropeType: "default", ropeTheta: 10_000))
        #expect(
            text.ropeParameters(for: full)
                == .init(ropeType: "proportional", ropeTheta: 1_000_000, partialRotaryFactor: 0.25))
    }

    @Test("A null num_global_key_value_heads falls back to num_key_value_heads")
    func globalKeyValueHeadsFallBack() throws {
        let config = try decode(
            #"{"text_config": {"num_key_value_heads": 4, "num_global_key_value_heads": null}}"#)
        #expect(config.text.numGlobalKeyValueHeads == nil)
        #expect(config.text.keyValueHeads(for: full) == 4)
    }

    @Test("Unknown keys are ignored at every level")
    func unknownKeys() throws {
        let config = try decode(
            #"""
            {"model_type": "diffusion_gemma", "transformers_version": "5.8.0", "invented": [1],
             "quantization_config": {"group_size": 32, "bits": 2},
             "quantization": {"group_size": 64, "bits": 4, "quant_method": "mlx", "m.x": {"group_size": 32, "bits": 8}},
             "generation_config": {"max_new_tokens": 9, "transformers_version": "5.8.0",
                                   "sampler_config": {"_cls_name": "S", "invented": true}},
             "text_config": {"hidden_size": 8, "initializer_range": 0.02, "invented": {"a": 1},
                             "rope_parameters": {"full_attention": {"rope_theta": 5.0, "invented": 1}}},
             "vision_config": {"hidden_size": 16, "invented": "x"}}
            """#)
        #expect(config.text.hiddenSize == 8)
        #expect(config.quantization?.defaultQuantization == .init(groupSize: 64, bits: 4))
        #expect(config.quantization?.overrides == ["m.x": .init(groupSize: 32, bits: 8)])
        #expect(config.generation?.maxNewTokens == 9)
        #expect(config.generation?.sampler?.className == "S")
        #expect(config.text.ropeParameters(for: full) == .init(ropeType: "default", ropeTheta: 5))
        #expect(config.text.ropeParameters(for: sliding) == .missingEntry)
        #expect(config.vision?.hiddenSize == 16)
    }

    @Test(
        "Errors name the offending key",
        arguments: [
            (#"{"model_type": "diffusion_gemma"}"#, "text_config: required key is missing"),
            (
                #"{"model_type": "gemma4", "text_config": {}}"#,
                "model_type: expected \"diffusion_gemma\""
            ),
            (
                #"{"text_config": {"model_type": "gemma4_text"}}"#,
                "text_config.model_type: expected \"diffusion_gemma_text\""
            ),
            (
                #"""
                {"text_config": {"num_hidden_layers": 2,
                                 "layer_types": ["full_attention", "diagonal_attention"]}}
                """#,
                "text_config.layer_types[1]: unknown layer type \"diagonal_attention\""
            ),
            (
                #"{"text_config": {"layer_types": []}}"#,
                "text_config.layer_types: expected 30 entries, one per layer (num_hidden_layers), "
                    + "found 0"
            ),
            (
                #"{"text_config": {"num_hidden_layers": 3, "layer_types": ["full_attention"]}}"#,
                "text_config.layer_types: expected 3 entries"
            ),
            (
                #"{"text_config": {"rope_parameters": {"diagonal_attention": {}}}}"#,
                "text_config.rope_parameters.diagonal_attention: unknown layer type"
            ),
            (
                #"{"text_config": {"hidden_size": "big"}}"#,
                "text_config.hidden_size: expected a number"
            ),
            (
                #"{"text_config": {"num_hidden_layers": 0}}"#,
                "text_config.num_hidden_layers: expected at least 1"
            ),
            (
                #"{"text_config": {}, "quantization": {"bits": 4}}"#,
                "quantization.group_size: required key is missing"
            ),
            (#"{"text_config": {}, "eos_token_id": "x"}"#, "eos_token_id: expected a list"),
            (#"{"text_config": {"#, "config.json: "),
        ])
    func errors(json: String, message: String) throws {
        let error = try decodingError(json)
        #expect(error.description.hasPrefix(message), "\(error)")
    }

    @Test("load(from:) reports a missing config.json and merges generation_config.json")
    func loadFromDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-configuration-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        #expect(throws: DiffusionGemmaConfigurationError.missingFile(configURL)) {
            try DiffusionGemmaConfiguration.load(from: directory)
        }
        #expect(
            DiffusionGemmaConfigurationError.missingFile(configURL).description.contains(
                "config.json"))

        try Data(#"{"text_config": {}, "eos_token_id": 1}"#.utf8).write(to: configURL)
        #expect(try DiffusionGemmaConfiguration.load(from: directory).generation == nil)

        try Data(#"{"eos_token_id": [1, 106], "max_new_tokens": 12}"#.utf8).write(
            to: directory.appendingPathComponent("generation_config.json"))
        let merged = try DiffusionGemmaConfiguration.load(from: directory)
        #expect(merged.generation?.maxNewTokens == 12)
        #expect(merged.eosTokenIDs == [1, 106])
    }

    /// The pinned checkpoint directory: `OPENJEV_TEST_MODEL` when set, else the Hugging Face
    /// cache snapshot the fixture scripts download.
    static let checkpointDirectory: URL = {
        let path = ProcessInfo.processInfo.environment[TokenizerFixtures.modelVariable] ?? ""
        return path.isEmpty ? TokenizerFixtures.cachedSnapshot : URL(fileURLWithPath: path)
    }()

    static var checkpointAvailable: Bool {
        FileManager.default.fileExists(
            atPath: checkpointDirectory.appendingPathComponent("config.json").path)
    }

    @Test(
        "The checkpoint's own config.json loads and equals the fixture",
        .enabled(
            if: checkpointAvailable,
            Comment(
                rawValue: "\(TokenizerFixtures.modelVariable) is unset and "
                    + "\(checkpointDirectory.path) has no config.json; set "
                    + "\(TokenizerFixtures.modelVariable) to the checkpoint directory or run "
                    + "make fixtures")))
    func liveCheckpoint() throws {
        let url = Self.checkpointDirectory.appendingPathComponent("config.json")
        let digests = try TokenizerFixtures.recordedDigests()
        #expect(try TokenizerFiles.sha256Hex(of: url) == digests["config.json"])
        let loaded = try DiffusionGemmaConfiguration.load(from: Self.checkpointDirectory)
        #expect(loaded == (try CheckpointFixture.load().config))
    }
}
