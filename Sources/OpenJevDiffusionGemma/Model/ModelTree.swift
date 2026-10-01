// The module tree of mlx-vlm 0.6.15's DiffusionGemma for text (diffusion_gemma/language.py and
// diffusion_gemma.py, adapted from mlx-vlm, Copyright © 2025 Prince Canuma, MIT), ported from the
// spike #22 transliteration. Its paths are the checkpoint's tensor names: model.decoder.* holds
// every weight, model.encoder.language_model.layers.N.layer_scalar the encoder's 30 scalars.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `model.decoder`: the embedding, the layers, the final norm and self-conditioning. The encoder
/// runs these same layers with its own scalars.
public final class DecoderModel: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo public var layers: [DecoderLayer]
    @ModuleInfo public var norm: RMSNorm
    @ModuleInfo(key: "self_conditioning") public var selfConditioning: SelfConditioning

    /// `sqrt(hidden_size)`, a Float that MLX rounds to the embedding's dtype when it multiplies,
    /// as Python's weak scalar typing does (√2816 rounds to bfloat16).
    public let embedScale: Float

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
    @ParameterInfo(key: "layer_scalar") public var layerScalar: MLXArray

    public override init() {
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }
}

/// `model.encoder.language_model`.
public final class EncoderLanguageModel: Module {
    @ModuleInfo public var layers: [EncoderLayerScalar]

    public init(layerCount: Int) {
        _layers.wrappedValue = (0..<layerCount).map { _ in EncoderLayerScalar() }
        super.init()
    }
}

/// `model.encoder`: the encoder's layer scalars, one per layer. A text-only tree has no vision
/// tower and no `embed_vision`.
public final class EncoderModel: Module {
    @ModuleInfo(key: "language_model") public var languageModel: EncoderLanguageModel

    public init(_ config: DiffusionGemmaTextConfiguration) {
        _languageModel.wrappedValue = EncoderLanguageModel(layerCount: config.layerTypes.count)
        super.init()
    }

    /// Layer `index`'s scalar.
    public func layerScalar(_ index: Int) -> MLXArray {
        languageModel.layers[index].layerScalar
    }
}

/// `model`: the decoder and the encoder.
public final class Backbone: Module {
    @ModuleInfo public var decoder: DecoderModel
    @ModuleInfo public var encoder: EncoderModel

    public init(_ config: DiffusionGemmaTextConfiguration) {
        _decoder.wrappedValue = DecoderModel(config)
        _encoder.wrappedValue = EncoderModel(config)
        super.init()
    }
}

/// The text model of a DiffusionGemma checkpoint, the root of the module tree.
///
/// Not Sendable: it holds MLX arrays. Its caller serialises its use, as the read path will.
public final class DiffusionGemmaModel: Module, BaseLanguageModel {
    @ModuleInfo public var model: Backbone

    /// The text configuration the tree was built from.
    public let configuration: DiffusionGemmaTextConfiguration

    /// `tanh(x / cap) * cap` in float32 with the configuration's `final_logit_softcapping`,
    /// compiled once, as diffusion_gemma.py compiles it at init.
    let softcap: @Sendable (MLXArray) -> MLXArray

    /// Builds the tree with MLXNN's initial values. Nothing is evaluated except the five 256-entry
    /// RoPE tables, so the real-size tree costs no memory until weights replace its arrays.
    public init(_ configuration: DiffusionGemmaTextConfiguration) {
        self.configuration = configuration
        softcap = makeSoftcap(configuration.finalLogitSoftcapping)
        _model.wrappedValue = Backbone(configuration)
        super.init()
    }

    /// `model.decoder`.
    public var decoder: DecoderModel { model.decoder }
    /// `model.encoder`.
    public var encoder: EncoderModel { model.encoder }

    // MARK: Sanitize

    /// The name a checkpoint tensor loads under, or nil when a text-only load drops it.
    ///
    /// diffusion_gemma.py `Model.sanitize` (lines 346 to 401) for a tree without a vision tower:
    /// `rotary_emb` buffers and the tied `lm_head.weight` are dropped, as are the vision tower and
    /// its embedder and every encoder text weight except the layer scalars; a bare
    /// `experts.gate_up_proj` or `experts.down_proj` gains `.weight` (a no-op for the pinned
    /// checkpoint, whose experts already carry it).
    public static func sanitizedName(_ key: String) -> String? {
        if key.contains("rotary_emb") || key == "lm_head.weight" {
            return nil
        }
        if key.hasPrefix("model.encoder.embed_vision.")
            || key.hasPrefix("model.encoder.vision_tower.")
        {
            return nil
        }
        if key.hasPrefix("model.encoder.language_model.") && !key.hasSuffix(".layer_scalar") {
            return nil
        }
        if key.hasSuffix(".experts.down_proj") || key.hasSuffix(".experts.gate_up_proj") {
            return key + ".weight"
        }
        return key
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            if let name = Self.sanitizedName(key) {
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
    /// - Parameter ids: `[1, length]` token ids.
    /// - Returns: the caches, one per layer.
    public func prefill(_ ids: MLXArray, stages: StageObserver? = nil) -> [LayerCache] {
        let embeddings = decoder.embed(ids)
        stages?("embeddings", embeddings)
        return prefill(embeddings: embeddings, stages: stages).caches
    }

    /// The prefill from given embeddings, over `layers` (every layer when nil), for the parity
    /// tests, which start from a stage dump's embeddings.
    ///
    /// - Returns: the caches of the layers run and the last layer's output.
    public func prefill(
        embeddings: MLXArray, layers: Range<Int>? = nil, stages: StageObserver? = nil
    ) -> (caches: [LayerCache], hidden: MLXArray) {
        let range = layers ?? decoder.layers.indices
        let length = embeddings.dim(1)
        typealias LayerType = DiffusionGemmaTextConfiguration.LayerType
        var masks: [LayerType: MLXFast.ScaledDotProductAttentionMaskMode] = [:]
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
