// The module tree of mlx-vlm 0.6.15's DiffusionGemma (diffusion_gemma/language.py and
// diffusion_gemma.py, adapted from mlx-vlm, Copyright © 2025 Prince Canuma, MIT), ported from the
// spike #22 transliteration. Its paths are the checkpoint's tensor names: model.decoder.* holds
// every text weight, model.encoder.language_model.layers.N.layer_scalar the encoder's 30 scalars,
// and model.encoder.vision_tower.* and model.encoder.embed_vision.* the vision tower (#47).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `model.decoder`: the embedding, the layers, the final norm and self-conditioning. The encoder
/// runs these same layers with its own scalars.
public final class DecoderModel: Module {
    /// The token embedding, `embed_tokens`, which is also the tied output head.
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    /// The layers, which the encoder runs too, with its own scalars.
    @ModuleInfo public var layers: [DecoderLayer]
    /// The final norm, `norm`.
    @ModuleInfo public var norm: RMSNorm
    /// The self-conditioning module, `self_conditioning`.
    @ModuleInfo(key: "self_conditioning") public var selfConditioning: SelfConditioning

    /// `sqrt(hidden_size)`, a Float that MLX rounds to the embedding's dtype when it multiplies,
    /// as Python's weak scalar typing does (√2816 rounds to bfloat16).
    public let embedScale: Float

    /// Builds the decoder from the text configuration, with MLXNN's initial values until the
    /// weights load.
    public init(_ config: DiffusionGemmaTextConfiguration) {
        embedScale = Float(pow(Double(config.hiddenSize), 0.5))
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = config.layerTypes.indices.map {
            DecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = rmsNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfConditioning.wrappedValue = SelfConditioning(config)
        super.init()
    }

    /// `embed_tokens(ids) * embed_scale`.
    public func embed(_ ids: MLXArray) -> MLXArray {
        embedTokens(ids) * embedScale
    }
}

/// `model.encoder.language_model.layers.N`: one encoder layer's scalar. mlx-vlm's encoder reaches
/// the decoder's modules through a weak reference; only its scalars are its own.
public final class EncoderLayerScalar: Module {
    /// The encoder's scalar for this layer, `layer_scalar`.
    @ParameterInfo(key: "layer_scalar") public var layerScalar: MLXArray

    /// A scalar of 1 until the weights load.
    public override init() {
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }
}

/// `model.encoder.language_model`.
public final class EncoderLanguageModel: Module {
    /// One scalar per layer, in layer order.
    @ModuleInfo public var layers: [EncoderLayerScalar]

    /// One scalar for each of `layerCount` layers.
    public init(layerCount: Int) {
        _layers.wrappedValue = (0..<layerCount).map { _ in EncoderLayerScalar() }
        super.init()
    }
}

/// `model.encoder`: the encoder's layer scalars, one per layer, and the vision tower and its
/// embedder, `embed_vision`, when the tree reads images. A text-only tree has neither.
public final class EncoderModel: Module {
    /// The encoder's layer scalars, `language_model`.
    @ModuleInfo(key: "language_model") public var languageModel: EncoderLanguageModel
    /// The vision tower, `vision_tower`, or nil for a text-only tree.
    @ModuleInfo(key: "vision_tower") public var visionTower: VisionModel?
    /// The soft tokens' projection to the text width, `embed_vision`, or nil for a text-only
    /// tree.
    @ModuleInfo(key: "embed_vision") public var embedVision: MultimodalEmbedder?

    /// Builds one scalar for each layer of the configuration, and the vision tower and its
    /// embedder when `vision` is given, as language.py's `EncoderModel` does when the checkpoint
    /// has a `vision_config`.
    public init(
        _ config: DiffusionGemmaTextConfiguration,
        vision: DiffusionGemmaVisionConfiguration? = nil
    ) {
        _languageModel.wrappedValue = EncoderLanguageModel(layerCount: config.layerTypes.count)
        if let vision {
            _visionTower.wrappedValue = VisionModel(vision)
            _embedVision.wrappedValue = MultimodalEmbedder(
                embeddingDimensions: vision.hiddenSize, textHiddenSize: config.hiddenSize,
                eps: vision.rmsNormEps)
        }
        super.init()
    }

    /// Layer `index`'s scalar.
    public func layerScalar(_ index: Int) -> MLXArray {
        languageModel.layers[index].layerScalar
    }
}

/// `model`: the decoder and the encoder.
public final class Backbone: Module {
    /// `model.decoder`, which holds every weight but the encoder's scalars.
    @ModuleInfo public var decoder: DecoderModel
    /// `model.encoder`, the encoder's layer scalars.
    @ModuleInfo public var encoder: EncoderModel

    /// Builds the decoder and the encoder from the text configuration, with the vision tower
    /// when `vision` is given.
    public init(
        _ config: DiffusionGemmaTextConfiguration,
        vision: DiffusionGemmaVisionConfiguration? = nil
    ) {
        _decoder.wrappedValue = DecoderModel(config)
        _encoder.wrappedValue = EncoderModel(config, vision: vision)
        super.init()
    }
}

/// A DiffusionGemma checkpoint's model, the root of the module tree: the text model, and the
/// vision tower when the tree reads images.
///
/// Not Sendable: it holds MLX arrays. Its caller serialises its use, as the read path will.
public final class DiffusionGemmaModel: Module, BaseLanguageModel {
    /// The tree under `model`, the prefix of every tensor that loads.
    @ModuleInfo public var model: Backbone

    /// The text configuration the tree was built from.
    public let configuration: DiffusionGemmaTextConfiguration
    /// The vision tower's configuration, nil for a text-only tree.
    public let visionConfiguration: DiffusionGemmaVisionConfiguration?
    /// `image_token_id`, the soft image token whose positions take the tower's features.
    public let imageTokenID: Int
    /// `video_token_id`, absent from the pinned checkpoint.
    public let videoTokenID: Int?

    /// `tanh(x / cap) * cap` in float32 with the configuration's `final_logit_softcapping`,
    /// compiled once, as diffusion_gemma.py compiles it at init.
    let softcap: @Sendable (MLXArray) -> MLXArray

    /// Builds a text-only tree with MLXNN's initial values. Nothing is evaluated except the five
    /// 256-entry RoPE tables, so the real-size tree costs no memory until weights replace its
    /// arrays.
    public init(_ configuration: DiffusionGemmaTextConfiguration) {
        self.configuration = configuration
        visionConfiguration = nil
        imageTokenID = 258_880
        videoTokenID = nil
        softcap = makeSoftcap(configuration.finalLogitSoftcapping)
        _model.wrappedValue = Backbone(configuration)
        super.init()
    }

    /// Builds the tree of a checkpoint's configuration with MLXNN's initial values: the text
    /// model, and the vision tower and `embed_vision` when the configuration has a
    /// `vision_config` and `vision` is true, as mlx-vlm's `load` builds them.
    public init(_ configuration: DiffusionGemmaConfiguration, vision: Bool = true) {
        self.configuration = configuration.text
        let visionConfiguration = vision ? configuration.vision : nil
        self.visionConfiguration = visionConfiguration
        imageTokenID = configuration.imageTokenID
        videoTokenID = configuration.videoTokenID
        softcap = makeSoftcap(configuration.text.finalLogitSoftcapping)
        _model.wrappedValue = Backbone(configuration.text, vision: visionConfiguration)
        super.init()
    }

    /// True when the tree has the vision tower and reads images.
    public var readsImages: Bool { encoder.visionTower != nil && encoder.embedVision != nil }

    /// `model.decoder`.
    public var decoder: DecoderModel { model.decoder }
    /// `model.encoder`.
    public var encoder: EncoderModel { model.encoder }

    // MARK: Sanitize

    /// The name a checkpoint tensor loads under, or nil when the load drops it.
    ///
    /// diffusion_gemma.py `Model.sanitize` (lines 346 to 401): `rotary_emb` buffers and the tied
    /// `lm_head.weight` are dropped; the vision tower's and `embed_vision`'s tensors are kept when
    /// the tree has a vision tower (`vision`), less the clipping bounds when the tower does not
    /// clip (`clippedLinears`), and dropped otherwise; every encoder text weight except the layer
    /// scalars is dropped; a bare `experts.gate_up_proj` or `experts.down_proj` gains `.weight`
    /// (a no-op for the pinned checkpoint, whose experts already carry it).
    public static func sanitizedName(_ key: String, vision: Bool, clippedLinears: Bool = false)
        -> String?
    {
        if key.contains("rotary_emb") || key == "lm_head.weight" {
            return nil
        }
        if key.hasPrefix("model.encoder.embed_vision.")
            || key.hasPrefix("model.encoder.vision_tower.")
        {
            guard vision else { return nil }
            // Clipping calibration tensors are only used by clipped linears.
            let bounds = ["input_max", "input_min", "output_max", "output_min"]
            if !clippedLinears, bounds.contains(where: key.contains) {
                return nil
            }
            return key
        }
        if key.hasPrefix("model.encoder.language_model.") && !key.hasSuffix(".layer_scalar") {
            return nil
        }
        if key.hasSuffix(".experts.down_proj") || key.hasSuffix(".experts.gate_up_proj") {
            return key + ".weight"
        }
        return key
    }

    /// The name a checkpoint tensor loads under in this tree, ``sanitizedName(_:vision:clippedLinears:)``
    /// with the tree's vision tower and clipping.
    public func sanitizedName(_ key: String) -> String? {
        Self.sanitizedName(
            key, vision: readsImages,
            clippedLinears: visionConfiguration?.useClippedLinears ?? false)
    }

    /// The checkpoint's tensors under the names they load as: ``sanitizedName(_:)`` of each key,
    /// without the tensors it drops. `loadWeights` calls it through MLXLMCommon's
    /// `BaseLanguageModel`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            if let name = sanitizedName(key) {
                out[name] = value
            }
        }
        return out
    }

    // MARK: Quantization

    /// Quantizes the tree as `loadWeights` does: each module whose `path.scales` is in
    /// `checkpointNames` (every quantizable module when nil) gets the per-layer map's settings,
    /// and a module the map skips stays as it is. Lazy: nothing is evaluated.
    public func quantize(
        _ perLayerQuantization: BaseConfiguration.PerLayerQuantization,
        checkpointNames: Set<String>? = nil
    ) {
        MLXNN.quantize(model: self) { path, _ in
            if let checkpointNames, !checkpointNames.contains("\(path).scales") {
                return nil
            }
            return perLayerQuantization.quantization(layer: path)?.asTuple
        }
    }

    /// The number of quantized modules in the tree.
    public var quantizedModuleCount: Int {
        leafModules().flattened().filter { $0.1 is Quantized }.count
    }

    // MARK: Prefill

    /// The encoder mask of a layer for a one-piece prefill of `length` tokens on an empty cache:
    /// `.causal`, except on a sliding layer for a prompt longer than `sliding_window`, which gets
    /// `RotatingKVCache.make_mask`'s boolean band `row >= column && row < column + window`. A
    /// single token needs none.
    public func encoderMask(
        for layerType: DiffusionGemmaTextConfiguration.LayerType, length: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard length > 1 else { return .none }
        let window = configuration.slidingWindow
        if layerType == .slidingAttention && length > window {
            let positions = MLXArray(Int32(0)..<Int32(length))
            let rows = positions[0..., .newAxis]
            let columns = positions[.newAxis, 0...]
            return .array(logicalAnd(rows .>= columns, rows .< columns + window))
        }
        return .causal
    }

    /// language.py `EncoderModel.__call__` for a text prompt in one piece, as upstream's
    /// `MlxRuntime._prefill` runs it: the embeddings, then every layer in encoder mode at RoPE
    /// offset 0 with the encoder's scalar, filling one cache per layer. The read path calls
    /// ``prefill(promptIDs:)``, which validates the ids and wraps the caches.
    ///
    /// - Parameters:
    ///   - ids: `[1, length]` token ids.
    ///   - stages: receives the embeddings and each layer's intermediate outputs, for the parity
    ///     tests.
    /// - Returns: the caches, one per layer.
    public func prefill(_ ids: MLXArray, stages: StageObserver? = nil) -> [LayerCache] {
        let embeddings = decoder.embed(ids)
        stages?("embeddings", embeddings)
        return prefill(embeddings: embeddings, stages: stages).caches
    }

    /// The prefill from given embeddings, over `layers` (every layer when nil), for the parity
    /// tests, which start from a stage dump's embeddings, and for an image prompt, whose masks
    /// are given.
    ///
    /// - Parameters:
    ///   - embeddings: `[1, length, hidden]`, the embedded prompt.
    ///   - layers: the layers to run, every layer when nil.
    ///   - given: the mask of each layer type; a type without one gets
    ///     ``encoderMask(for:length:)``.
    ///   - stages: receives each layer's intermediate outputs, for the parity tests.
    /// - Returns: the caches of the layers run and the last layer's output.
    public func prefill(
        embeddings: MLXArray, layers: Range<Int>? = nil,
        masks given: [DiffusionGemmaTextConfiguration.LayerType:
            MLXFast.ScaledDotProductAttentionMaskMode] = [:],
        stages: StageObserver? = nil
    ) -> (caches: [LayerCache], hidden: MLXArray) {
        let range = layers ?? decoder.layers.indices
        let length = embeddings.dim(1)
        typealias LayerType = DiffusionGemmaTextConfiguration.LayerType
        var masks = given
        var caches: [LayerCache] = []
        var h = embeddings
        for index in range {
            let layer = decoder.layers[index]
            let mask = masks[layer.layerType] ?? encoderMask(for: layer.layerType, length: length)
            masks[layer.layerType] = mask
            let cache = LayerCache(isFullAttention: layer.layerType == .fullAttention)
            h = layer(
                h, mask: mask, cache: cache, decoder: false, offset: 0,
                layerScalar: encoder.layerScalar(index), stages: stages)
            caches.append(cache)
        }
        return (caches, h)
    }
}
