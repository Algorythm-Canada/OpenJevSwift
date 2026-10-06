// The typed form of a DiffusionGemma checkpoint's config.json, with the defaults of mlx-vlm 0.6.15's
// mlx_vlm/models/diffusion_gemma/config.py (adapted from mlx-vlm, Copyright © 2025 Prince Canuma,
// MIT). Every file under Model/ reads its settings from these types.

import Foundation
import MLX
import MLXLMCommon

/// A DiffusionGemma checkpoint's `config.json`.
///
/// Keys that are absent take mlx-vlm's defaults (`ModelConfig` in config.py), and keys the type
/// does not know are ignored, as mlx-vlm's `_config_kwargs` drops them. `text_config` is
/// required: mlx-vlm leaves it `None` and fails later, so a missing one is reported here, by name.
/// The file's `quantization_config` duplicates `quantization` and is ignored, as mlx-vlm does.
public struct DiffusionGemmaConfiguration: Codable, Equatable, Sendable {
    /// `model_type`, always `diffusion_gemma`.
    public let modelType: String
    /// `architectures`, `["DiffusionGemmaForBlockDiffusion"]` for the pinned checkpoint.
    public let architectures: [String]
    /// `canvas_length`, the number of positions a read denoises at once.
    public let canvasLength: Int
    /// `boi_token_id`, the token that opens an image.
    public let boiTokenID: Int
    /// `eoi_token_id`, the token that closes an image.
    public let eoiTokenID: Int
    /// `image_token_id`, the placeholder each image soft token replaces.
    public let imageTokenID: Int
    /// `video_token_id`, absent from the pinned checkpoint.
    public let videoTokenID: Int?
    /// `eos_token_id`, an integer or a list in the file. Empty when absent.
    public private(set) var eosTokenIDs: [Int]
    /// `dtype`, the unquantized weights' type (`bfloat16`).
    public let dtype: String?
    /// `initializer_range`, from mlx-vlm's `ModelConfig`; not used for inference.
    public let initializerRange: Float
    /// `tie_word_embeddings`: whether the output projection reuses the embedding.
    public let tieWordEmbeddings: Bool
    /// `vision_soft_tokens_per_image`, the soft tokens one image expands to.
    public let visionSoftTokensPerImage: Int
    /// `quantization`, absent for an unquantized checkpoint.
    public let quantization: DiffusionGemmaQuantization?
    /// `generation_config`, the checkpoint's sampling settings.
    public private(set) var generation: DiffusionGemmaGenerationConfiguration?
    /// `text_config`, the language model.
    public let text: DiffusionGemmaTextConfiguration
    /// `vision_config`, the vision tower, when the checkpoint has one.
    public let vision: DiffusionGemmaVisionConfiguration?

    /// The only `model_type` accepted.
    public static let expectedModelType = "diffusion_gemma"

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case canvasLength = "canvas_length"
        case boiTokenID = "boi_token_id"
        case eoiTokenID = "eoi_token_id"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case eosTokenIDs = "eos_token_id"
        case dtype
        case initializerRange = "initializer_range"
        case tieWordEmbeddings = "tie_word_embeddings"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case quantization
        case generation = "generation_config"
        case text = "text_config"
        case vision = "vision_config"
    }

    /// Decodes config.json: the top level with mlx-vlm's `ModelConfig` defaults, the required
    /// `text_config`, and `quantization`, `generation_config` and `vision_config` when present.
    /// Another `model_type` is refused (D-032).
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? Self.expectedModelType
        if modelType != Self.expectedModelType {
            throw DiffusionGemmaConfigurationError.unsupportedModelType(
                modelType, keyPath: keyPath(decoder.codingPath, CodingKeys.modelType),
                expected: Self.expectedModelType)
        }
        guard c.contains(.text), try !c.decodeNil(forKey: .text) else {
            throw DiffusionGemmaConfigurationError.decodingFailed(
                keyPath: keyPath(decoder.codingPath, CodingKeys.text),
                reason: "required key is missing; a DiffusionGemma configuration needs its "
                    + "language model settings")
        }
        text = try c.decode(DiffusionGemmaTextConfiguration.self, forKey: .text)
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        canvasLength = try c.decodeIfPresent(Int.self, forKey: .canvasLength) ?? 256
        boiTokenID = try c.decodeIfPresent(Int.self, forKey: .boiTokenID) ?? 255_999
        eoiTokenID = try c.decodeIfPresent(Int.self, forKey: .eoiTokenID) ?? 258_882
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 258_880
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID)
        eosTokenIDs = try c.decodeTokenIDs(forKey: .eosTokenIDs) ?? []
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype)
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage) ?? 280
        quantization = try c.decodeIfPresent(DiffusionGemmaQuantization.self, forKey: .quantization)
        generation = try c.decodeIfPresent(
            DiffusionGemmaGenerationConfiguration.self, forKey: .generation)
        vision = try c.decodeIfPresent(DiffusionGemmaVisionConfiguration.self, forKey: .vision)
    }

    /// Encodes the configuration under config.json's keys.
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(architectures, forKey: .architectures)
        try c.encode(canvasLength, forKey: .canvasLength)
        try c.encode(boiTokenID, forKey: .boiTokenID)
        try c.encode(eoiTokenID, forKey: .eoiTokenID)
        try c.encode(imageTokenID, forKey: .imageTokenID)
        try c.encodeIfPresent(videoTokenID, forKey: .videoTokenID)
        try c.encode(eosTokenIDs, forKey: .eosTokenIDs)
        try c.encodeIfPresent(dtype, forKey: .dtype)
        try c.encode(initializerRange, forKey: .initializerRange)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(visionSoftTokensPerImage, forKey: .visionSoftTokensPerImage)
        try c.encodeIfPresent(quantization, forKey: .quantization)
        try c.encodeIfPresent(generation, forKey: .generation)
        try c.encode(text, forKey: .text)
        try c.encodeIfPresent(vision, forKey: .vision)
    }

    /// Decodes a `config.json` document.
    ///
    /// - Throws: ``DiffusionGemmaConfigurationError``. A value of the wrong type or a missing
    ///   required key is ``DiffusionGemmaConfigurationError/decodingFailed(keyPath:reason:)``
    ///   naming the key, for example `text_config.hidden_size: expected a number`.
    public init(data: Data) throws {
        self = try Self.decode(Self.self, from: data)
    }

    /// Reads `config.json` from a checkpoint directory.
    ///
    /// When the directory also holds a `generation_config.json` object that is not empty, it
    /// replaces ``generation``, and its `eos_token_id`, when present, replaces ``eosTokenIDs``,
    /// as mlx-vlm's `load_config` merges the two files (utils.py, `_merge_generation_config`).
    /// A `generation_config.json` that is not valid JSON is skipped, as mlx-vlm skips it.
    ///
    /// - Throws: ``DiffusionGemmaConfigurationError/missingFile(_:)`` when `config.json` is
    ///   absent, the error reading it gave, or a decoding error as ``init(data:)`` throws.
    public static func load(from directory: URL) throws -> DiffusionGemmaConfiguration {
        let url = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DiffusionGemmaConfigurationError.missingFile(url)
        }
        var configuration = try DiffusionGemmaConfiguration(data: Data(contentsOf: url))
        let generationURL = directory.appendingPathComponent("generation_config.json")
        if FileManager.default.fileExists(atPath: generationURL.path) {
            let data = try Data(contentsOf: generationURL)
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            if let object, !object.isEmpty {
                let generation = try decode(DiffusionGemmaGenerationConfiguration.self, from: data)
                configuration.generation = generation
                if let eos = generation.eosTokenIDs {
                    configuration.eosTokenIDs = eos
                }
            }
        }
        return configuration
    }

    /// Decodes `type` from JSON, turning a `DecodingError` into one that names the key.
    static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch let error as DecodingError {
            throw DiffusionGemmaConfigurationError(error)
        }
    }
}

/// `text_config`: the language model's settings, with the defaults of mlx-vlm's `TextConfig`.
public struct DiffusionGemmaTextConfiguration: Codable, Equatable, Sendable {
    /// The attention kind of one decoder layer.
    public enum LayerType: String, Codable, Sendable, Hashable, CaseIterable,
        CodingKeyRepresentable
    {
        /// Local attention over ``DiffusionGemmaTextConfiguration/slidingWindow`` positions.
        case slidingAttention = "sliding_attention"
        /// Global attention, with its own head size, key-value heads and RoPE.
        case fullAttention = "full_attention"
    }

    /// One layer type's rotary position embedding settings.
    public struct RoPEParameters: Codable, Equatable, Sendable {
        /// `rope_type`: `default`, or `proportional` for the full-attention layers.
        public let ropeType: String
        /// `rope_theta`, the base frequency.
        public let ropeTheta: Float
        /// `partial_rotary_factor`, the fraction of each head that rotates, when not all of it.
        public let partialRotaryFactor: Float?

        /// Creates one layer type's RoPE settings.
        public init(ropeType: String, ropeTheta: Float, partialRotaryFactor: Float? = nil) {
            self.ropeType = ropeType
            self.ropeTheta = ropeTheta
            self.partialRotaryFactor = partialRotaryFactor
        }

        /// What language.py uses for a layer type the file has no entry for: an empty dict,
        /// which `initialize_rope` reads as the default RoPE with theta 10,000.
        public static let missingEntry = RoPEParameters(ropeType: "default", ropeTheta: 10_000)

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }

        /// Decodes an entry of `rope_parameters`; a missing `rope_type` is `default` and a missing
        /// `rope_theta` is 10,000.
        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            ropeType = try c.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
            partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        }
    }

    /// The only `model_type` accepted when the key is present.
    public static let expectedModelType = "diffusion_gemma_text"

    /// `model_type`.
    public let modelType: String
    /// `vocab_size`.
    public let vocabSize: Int
    /// `hidden_size`.
    public let hiddenSize: Int
    /// `intermediate_size`, the dense MLP's width.
    public let intermediateSize: Int
    /// `moe_intermediate_size`, each expert's width.
    public let moeIntermediateSize: Int
    /// `num_hidden_layers`.
    public let numHiddenLayers: Int
    /// `num_attention_heads`.
    public let numAttentionHeads: Int
    /// `num_key_value_heads`, the sliding layers' key-value heads.
    public let numKeyValueHeads: Int
    /// `num_global_key_value_heads`, the full layers' key-value heads; nil when the file has
    /// `null`, and then the full layers use ``numKeyValueHeads``.
    public let numGlobalKeyValueHeads: Int?
    /// `head_dim`, the sliding layers' head size.
    public let headDim: Int
    /// `global_head_dim`, the full layers' head size.
    public let globalHeadDim: Int
    /// `hidden_activation`.
    public let hiddenActivation: String
    /// `rms_norm_eps`.
    public let rmsNormEps: Float
    /// `max_position_embeddings`.
    public let maxPositionEmbeddings: Int
    /// `pad_token_id`.
    public let padTokenID: Int
    /// `eos_token_id`, an integer or a list in the file; empty when the file has `null`.
    public let eosTokenIDs: [Int]
    /// `bos_token_id`.
    public let bosTokenID: Int?
    /// `tie_word_embeddings`.
    public let tieWordEmbeddings: Bool
    /// `rope_parameters`, keyed by layer type. ``ropeParameters(for:)`` is the value to use.
    public let ropeParameters: [LayerType: RoPEParameters]
    /// `attention_bias`.
    public let attentionBias: Bool
    /// `attention_dropout`.
    public let attentionDropout: Float
    /// `sliding_window`.
    public let slidingWindow: Int
    /// `layer_types`, exactly one per layer. When absent, ``defaultLayerTypes(count:)``.
    public let layerTypes: [LayerType]
    /// `final_logit_softcapping`.
    public let finalLogitSoftcapping: Float
    /// `use_bidirectional_attention`: `vision` makes an image's tokens attend to each other.
    public let useBidirectionalAttention: String?
    /// `num_experts`.
    public let numExperts: Int
    /// `top_k_experts`, the experts each token is routed to.
    public let topKExperts: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case hiddenActivation = "hidden_activation"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case padTokenID = "pad_token_id"
        case eosTokenIDs = "eos_token_id"
        case bosTokenID = "bos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useBidirectionalAttention = "use_bidirectional_attention"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
    }

    /// Decodes `text_config` with mlx-vlm's `TextConfig` defaults for absent keys, deriving
    /// `layer_types` and `rope_parameters` as config.py does when they are absent (D-032).
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let path = decoder.codingPath
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? Self.expectedModelType
        if modelType != Self.expectedModelType {
            throw DiffusionGemmaConfigurationError.unsupportedModelType(
                modelType, keyPath: keyPath(path, CodingKeys.modelType),
                expected: Self.expectedModelType)
        }
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262_144
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 704
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        if numHiddenLayers < 1 {
            throw DiffusionGemmaConfigurationError.decodingFailed(
                keyPath: keyPath(path, CodingKeys.numHiddenLayers),
                reason: "expected at least 1 layer, found \(numHiddenLayers)")
        }
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        numGlobalKeyValueHeads = try c.decodeNullable(
            Int.self, forKey: .numGlobalKeyValueHeads, default: 2)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        hiddenActivation =
            try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "gelu_pytorch_tanh"
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 0
        eosTokenIDs =
            try c.contains(.eosTokenIDs) ? c.decodeTokenIDs(forKey: .eosTokenIDs) ?? [] : [1]
        bosTokenID = try c.decodeNullable(Int.self, forKey: .bosTokenID, default: 2)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30
        useBidirectionalAttention = try c.decodeNullable(
            String.self, forKey: .useBidirectionalAttention, default: "vision")
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 8

        if let names = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            let base = keyPath(path, CodingKeys.layerTypes)
            // language.py builds layer i from layer_types[i], and Transformers refuses a list
            // of another length, so a mismatch is reported here rather than at model building.
            guard names.count == numHiddenLayers else {
                throw DiffusionGemmaConfigurationError.decodingFailed(
                    keyPath: base,
                    reason: "expected \(numHiddenLayers) entries, one per layer "
                        + "(num_hidden_layers), found \(names.count)")
            }
            layerTypes = try names.enumerated().map { index, name in
                try Self.layerType(name, keyPath: "\(base)[\(index)]")
            }
        } else {
            layerTypes = Self.defaultLayerTypes(count: numHiddenLayers)
        }
        if let entries = try c.decodeIfPresent(
            [String: RoPEParameters].self, forKey: .ropeParameters)
        {
            let base = keyPath(path, CodingKeys.ropeParameters)
            var parameters: [LayerType: RoPEParameters] = [:]
            for (name, entry) in entries {
                let type = try Self.layerType(name, keyPath: "\(base).\(name)")
                parameters[type] = entry
            }
            ropeParameters = parameters
        } else {
            ropeParameters = Self.defaultRoPEParameters
        }
    }

    /// The layer types config.py derives when `layer_types` is absent (lines 40 to 46): five
    /// sliding layers then one full layer, repeated and cut to `count`, with the last layer
    /// forced to full attention.
    public static func defaultLayerTypes(count: Int) -> [LayerType] {
        guard count > 0 else { return [] }
        let pattern: [LayerType] = Array(repeating: .slidingAttention, count: 5) + [.fullAttention]
        var types = (0..<count).map { pattern[$0 % pattern.count] }
        types[count - 1] = .fullAttention
        return types
    }

    /// The RoPE settings config.py uses when `rope_parameters` is absent (lines 48 to 59).
    public static let defaultRoPEParameters: [LayerType: RoPEParameters] = [
        .slidingAttention: RoPEParameters(ropeType: "default", ropeTheta: 10_000),
        .fullAttention: RoPEParameters(
            ropeType: "proportional", ropeTheta: 1_000_000, partialRotaryFactor: 0.25),
    ]

    /// The indices of the full-attention layers, `[5, 11, 17, 23, 29]` for the pinned checkpoint.
    public var fullAttentionLayers: [Int] {
        layerTypes.indices.filter { layerTypes[$0] == .fullAttention }
    }

    /// A layer's head size, as language.py's `Attention` picks it: ``globalHeadDim`` for a full
    /// layer unless it is 0, else ``headDim``.
    public func headDim(for layerType: LayerType) -> Int {
        layerType == .fullAttention && globalHeadDim != 0 ? globalHeadDim : headDim
    }

    /// A layer's key-value heads, as language.py lines 148 to 156 pick them:
    /// ``numGlobalKeyValueHeads`` for a full layer when set, else ``numKeyValueHeads``.
    public func keyValueHeads(for layerType: LayerType) -> Int {
        layerType == .fullAttention ? numGlobalKeyValueHeads ?? numKeyValueHeads : numKeyValueHeads
    }

    /// A layer's RoPE settings: the file's entry for its type, else
    /// ``RoPEParameters/missingEntry``, as language.py's `rope_parameters.get(layer_type, {})`.
    public func ropeParameters(for layerType: LayerType) -> RoPEParameters {
        ropeParameters[layerType] ?? .missingEntry
    }

    /// The layer type named `name`, or an error naming the key it was read from.
    private static func layerType(_ name: String, keyPath: String) throws -> LayerType {
        guard let type = LayerType(rawValue: name) else {
            throw DiffusionGemmaConfigurationError.unknownLayerType(name, keyPath: keyPath)
        }
        return type
    }
}

/// `quantization`: the default group-wise quantization and the modules that differ from it.
///
/// The file mixes the two in one object, as MLXLMCommon's `BaseConfiguration` reads it:
/// `group_size`, `bits` and `mode` are the default, and every key whose value is an object is a
/// module path with its own `group_size` and `bits`. An override without `mode` is `affine`, not
/// the default's mode: MLXLMCommon decodes each override on its own, and mlx-vlm passes the
/// object to `to_quantized`, whose `mode` defaults to `affine`. A module path set to `false` is
/// left unquantized, and other keys are ignored.
public struct DiffusionGemmaQuantization: Codable, Equatable, Sendable {
    /// One module's quantization.
    public struct Quantization: Codable, Hashable, Sendable {
        /// `group_size`, the weights that share a scale and bias.
        public let groupSize: Int
        /// `bits`, the width of each quantized weight.
        public let bits: Int
        /// `mode`, `affine` when absent.
        public let mode: QuantizationMode

        /// Creates one module's quantization.
        public init(groupSize: Int, bits: Int, mode: QuantizationMode = .affine) {
            self.groupSize = groupSize
            self.bits = bits
            self.mode = mode
        }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
            case mode
        }

        /// Decodes `group_size` and `bits`, and `mode`, which is `affine` when absent.
        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            groupSize = try c.decode(Int.self, forKey: .groupSize)
            bits = try c.decode(Int.self, forKey: .bits)
            mode = try c.decodeIfPresent(QuantizationMode.self, forKey: .mode) ?? .affine
        }

        /// The same settings as MLXLMCommon's type, for `loadWeights`.
        public var baseQuantization: BaseConfiguration.Quantization {
            let affine = BaseConfiguration.Quantization(groupSize: groupSize, bits: bits)
            guard mode != affine.mode else { return affine }
            // BaseConfiguration.Quantization takes any other mode only when it is decoded, and
            // its keys are the ones this type encodes.
            let decoded = (try? JSONEncoder().encode(self)).flatMap {
                try? JSONDecoder().decode(BaseConfiguration.Quantization.self, from: $0)
            }
            return decoded ?? affine
        }
    }

    /// The default for every module the file does not name.
    public let defaultQuantization: Quantization
    /// Module path (for example `model.decoder.embed_tokens`) to its own quantization.
    public let overrides: [String: Quantization]
    /// Module paths the file sets to `false`: kept unquantized.
    public let unquantizedModules: Set<String>

    /// Creates a quantization map from a default, the per-module overrides and the modules kept
    /// unquantized.
    public init(
        defaultQuantization: Quantization, overrides: [String: Quantization] = [:],
        unquantizedModules: Set<String> = []
    ) {
        self.defaultQuantization = defaultQuantization
        self.overrides = overrides
        self.unquantizedModules = unquantizedModules
    }

    /// The quantization of the module at `path`, or nil when it stays unquantized.
    public func quantization(forModule path: String) -> Quantization? {
        if unquantizedModules.contains(path) {
            return nil
        }
        return overrides[path] ?? defaultQuantization
    }

    /// The same settings in MLXLMCommon's form, for `loadWeights(perLayerQuantization:)`.
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization {
        var perLayer = overrides.mapValues {
            BaseConfiguration.QuantizationOption.quantize($0.baseQuantization)
        }
        for path in unquantizedModules {
            perLayer[path] = .skip
        }
        return BaseConfiguration.PerLayerQuantization(
            quantization: defaultQuantization.baseQuantization, perLayerQuantization: perLayer)
    }

    /// Decodes the `quantization` object as the type describes: the default's scalar keys, an
    /// object for each overridden module and `false` for each module kept unquantized.
    public init(from decoder: any Decoder) throws {
        defaultQuantization = try Quantization(from: decoder)
        let c = try decoder.container(keyedBy: AnyCodingKey.self)
        var overrides: [String: Quantization] = [:]
        var unquantized: Set<String> = []
        for key in c.allKeys {
            if Quantization.CodingKeys(rawValue: key.stringValue) != nil {
                continue
            }
            if let flag = try? c.decode(Bool.self, forKey: key) {
                if !flag {
                    unquantized.insert(key.stringValue)
                }
            } else if (try? c.nestedContainer(keyedBy: AnyCodingKey.self, forKey: key)) != nil {
                overrides[key.stringValue] = try c.decode(Quantization.self, forKey: key)
            }
        }
        self.overrides = overrides
        unquantizedModules = unquantized
    }

    /// Encodes the map in the `quantization` object's form.
    public func encode(to encoder: any Encoder) throws {
        try defaultQuantization.encode(to: encoder)
        var c = encoder.container(keyedBy: AnyCodingKey.self)
        for (path, quantization) in overrides {
            try c.encode(quantization, forKey: AnyCodingKey(path))
        }
        for path in unquantizedModules {
            try c.encode(false, forKey: AnyCodingKey(path))
        }
    }
}

/// `generation_config`: the checkpoint's sampling settings.
///
/// mlx-vlm keeps this object as an untyped dict with no defaults, so every field is nil when
/// the file lacks it.
public struct DiffusionGemmaGenerationConfiguration: Codable, Equatable, Sendable {
    /// `sampler_config`.
    public struct SamplerConfiguration: Codable, Equatable, Sendable {
        /// `_cls_name`, the sampler's configuration class (`EntropyBoundSamplerConfig`).
        public let className: String?
        /// `entropy_bound`.
        public let entropyBound: Double?

        enum CodingKeys: String, CodingKey {
            case className = "_cls_name"
            case entropyBound = "entropy_bound"
        }
    }

    /// `max_denoising_steps`.
    public let maxDenoisingSteps: Int?
    /// `sampler_config`.
    public let sampler: SamplerConfiguration?
    /// `confidence_threshold`.
    public let confidenceThreshold: Double?
    /// `stability_threshold`.
    public let stabilityThreshold: Int?
    /// `t_max`.
    public let tMax: Double?
    /// `t_min`.
    public let tMin: Double?
    /// `max_new_tokens`.
    public let maxNewTokens: Int?
    /// `eos_token_id`, an integer or a list in the file.
    public let eosTokenIDs: [Int]?
    /// `pad_token_id`.
    public let padTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case maxDenoisingSteps = "max_denoising_steps"
        case sampler = "sampler_config"
        case confidenceThreshold = "confidence_threshold"
        case stabilityThreshold = "stability_threshold"
        case tMax = "t_max"
        case tMin = "t_min"
        case maxNewTokens = "max_new_tokens"
        case eosTokenIDs = "eos_token_id"
        case padTokenID = "pad_token_id"
    }

    /// Decodes `generation_config`. Every field is optional, as mlx-vlm keeps the object untyped.
    /// The thresholds and temperatures are Doubles, the Python floats mlx-vlm computes the step
    /// temperature with (``DiffusionSampler/linearTemperature(step:maxSteps:schedule:)``).
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxDenoisingSteps = try c.decodeIfPresent(Int.self, forKey: .maxDenoisingSteps)
        sampler = try c.decodeIfPresent(SamplerConfiguration.self, forKey: .sampler)
        confidenceThreshold = try c.decodeIfPresent(Double.self, forKey: .confidenceThreshold)
        stabilityThreshold = try c.decodeIfPresent(Int.self, forKey: .stabilityThreshold)
        tMax = try c.decodeIfPresent(Double.self, forKey: .tMax)
        tMin = try c.decodeIfPresent(Double.self, forKey: .tMin)
        maxNewTokens = try c.decodeIfPresent(Int.self, forKey: .maxNewTokens)
        eosTokenIDs = try c.decodeTokenIDs(forKey: .eosTokenIDs)
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID)
    }
}

/// Why a ``DiffusionGemmaConfiguration`` could not be read.
public enum DiffusionGemmaConfigurationError: Error, Sendable, Hashable, CustomStringConvertible {
    /// The checkpoint directory has no `config.json`.
    case missingFile(URL)
    /// A `model_type` other than the one expected at that key.
    case unsupportedModelType(String, keyPath: String, expected: String)
    /// A layer type other than `sliding_attention` and `full_attention`.
    case unknownLayerType(String, keyPath: String)
    /// A missing required key or a value of the wrong type, at `keyPath` (for example
    /// `text_config.hidden_size`).
    case decodingFailed(keyPath: String, reason: String)

    /// A `DecodingError` restated with the key path it happened at.
    init(_ error: DecodingError) {
        switch error {
        case .typeMismatch(let type, let context):
            self = .decodingFailed(
                keyPath: keyPath(context.codingPath), reason: "expected \(Self.kind(of: type))")
        case .valueNotFound(let type, let context):
            self = .decodingFailed(
                keyPath: keyPath(context.codingPath),
                reason: "expected \(Self.kind(of: type)), found null")
        case .keyNotFound(let key, let context):
            self = .decodingFailed(
                keyPath: keyPath(context.codingPath, key), reason: "required key is missing")
        case .dataCorrupted(let context):
            self = .decodingFailed(
                keyPath: keyPath(context.codingPath), reason: context.debugDescription)
        @unknown default:
            self = .decodingFailed(keyPath: "", reason: "\(error)")
        }
    }

    /// How a message names the JSON value `type` decodes from.
    private static func kind(of type: Any.Type) -> String {
        switch type {
        case is any BinaryInteger.Type, is any BinaryFloatingPoint.Type: return "a number"
        case is String.Type: return "a string"
        case is Bool.Type: return "a boolean"
        case is [Any].Type: return "a list"
        case is [String: Any].Type: return "an object"
        default: return "\(type)"
        }
    }

    /// A readable account of the failure that starts with the key it concerns.
    public var description: String {
        switch self {
        case .missingFile(let url):
            return "\(url.path): no such file; a DiffusionGemma checkpoint needs its config.json"
        case .unsupportedModelType(let found, let keyPath, let expected):
            return "\(keyPath): expected \"\(expected)\", found \"\(found)\""
        case .unknownLayerType(let found, let keyPath):
            return "\(keyPath): unknown layer type \"\(found)\"; expected \"sliding_attention\" "
                + "or \"full_attention\""
        case .decodingFailed(let keyPath, let reason):
            return "\(keyPath.isEmpty ? "config.json" : keyPath): \(reason)"
        }
    }
}

/// A coding key for the arbitrary module paths of `quantization`.
private struct AnyCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init(_ string: String) {
        stringValue = string
    }

    init?(stringValue: String) {
        self.stringValue = stringValue
    }

    init?(intValue: Int) {
        return nil
    }
}

/// `path` and `key` as a dotted key path with list indices in brackets, for example
/// `text_config.layer_types[3]`.
private func keyPath(_ path: [any CodingKey], _ key: (any CodingKey)? = nil) -> String {
    var text = ""
    for component in path + (key.map { [$0] } ?? []) {
        // JSONDecoder names a list element "Index n" and gives it the integer n.
        if let index = component.intValue, component.stringValue != "\(index)" {
            text += "[\(index)]"
        } else {
            text += text.isEmpty ? component.stringValue : ".\(component.stringValue)"
        }
    }
    return text
}

extension KeyedDecodingContainer {
    /// The token ids at `key`, which the file writes as one integer or a list. Nil when the key
    /// is absent or null.
    fileprivate func decodeTokenIDs(forKey key: Key) throws -> [Int]? {
        guard contains(key), try !decodeNil(forKey: key) else {
            return nil
        }
        if let id = try? decode(Int.self, forKey: key) {
            return [id]
        }
        return try decode([Int].self, forKey: key)
    }

    /// The value at `key`: `value` when the key is absent, nil when it is null, as a Python
    /// dataclass field with an `Optional` type and a default reads it.
    fileprivate func decodeNullable<T: Decodable>(
        _ type: T.Type, forKey key: Key, default value: T
    ) throws -> T? {
        guard contains(key) else {
            return value
        }
        if try decodeNil(forKey: key) {
            return nil
        }
        return try decode(type, forKey: key)
    }
}
