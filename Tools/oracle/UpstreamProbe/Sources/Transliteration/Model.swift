// mlx-vlm 0.6.15's DiffusionGemma read path (mlx_vlm/models/diffusion_gemma/language.py and the
// parts of mlx_vlm/models/cache.py, base.py, rope_utils.py and switch_layers.py it uses), written
// out on ml-explore/mlx-swift 0.32.2 and mlx-swift-lm c043fb3 for spike #22. Adapted from mlx-vlm,
// Copyright © 2025 Prince Canuma, MIT. Scratch code: it keeps mlx-vlm's operations, their order,
// their shapes and their dtypes, so that it can be compared bit for bit with the Python oracle.
// Text only: no vision tower, no chunked prefill, no static cache.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct RopeParameters: Decodable {
    let ropeTheta: Float
    let ropeType: String
    let partialRotaryFactor: Float?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

struct TextConfiguration: Decodable {
    let hiddenSize: Int
    let intermediateSize: Int
    let moeIntermediateSize: Int
    let numExperts: Int
    let topKExperts: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numGlobalKeyValueHeads: Int?
    let headDim: Int
    let globalHeadDim: Int
    let layerTypes: [String]
    let slidingWindow: Int
    let rmsNormEps: Float
    let vocabSize: Int
    let finalLogitSoftcapping: Float
    let ropeParameters: [String: RopeParameters]
    let padTokenId: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case ropeParameters = "rope_parameters"
        case padTokenId = "pad_token_id"
    }
}

struct RootConfiguration: Decodable {
    let textConfig: TextConfiguration
    enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
}

// language.py:22-33. mlx.nn.gelu_approx is itself a shapeless compiled function, as
// MLXNN.geluApproximate is.
let geglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { gate, x in
    geluApproximate(gate) * x
}

func makeSoftcap(_ cap: Float) -> @Sendable (MLXArray) -> MLXArray {
    compile(shapeless: true) { x in tanh(x.asType(.float32) / cap) * cap }
}

/// TRANSLITERATION_BUG plants one deliberate porting mistake, to calibrate D-014's tolerances:
/// skip_self_conditioning (step 1 skips the module, h = embeddings), rope_offset_zero (canvas RoPE
/// from position 0), no_window (the decoder's sliding layers see every encoder position: no cut to
/// the last 1,023 and no mask), upstream_router
/// (mlx-swift-lm Gemma4TextRouter's arithmetic: the scale folded into the norm, plain softmax).
let plantedBug = ProcessInfo.processInfo.environment["TRANSLITERATION_BUG"] ?? ""

/// Set by --stages: receives the same intermediate outputs Tools/oracle/stage_dump.py records.
nonisolated(unsafe) var stageRecorder: ((String, MLXArray) -> Void)?

/// Holds an array the module tree must not treat as a parameter.
final class Constant {
    var value: MLXArray
    init(_ value: MLXArray) { self.value = value }
}

/// cache.py KVCache (full layers) and RotatingKVCache (sliding layers), for one prefill of the
/// whole prompt. RotatingKVCache._update_concat on an empty cache keeps the update as it is.
/// KVCache.update_and_fetch writes it into a zero buffer of a multiple of 256 positions and
/// returns a view of the filled part, which is what attention then reads, so that layout is kept.
final class LayerCache {
    let full: Bool
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var offset = 0

    init(full: Bool) { self.full = full }

    func update(_ newKeys: MLXArray, _ newValues: MLXArray) -> (MLXArray, MLXArray) {
        precondition(keys == nil, "one prefill per cache")
        let count = newKeys.dim(2)
        offset += count
        if !full {
            keys = newKeys
            values = newValues
            return (newKeys, newValues)
        }
        let steps = (256 + count - 1) / 256
        var keyBuffer = MLXArray.zeros(
            [newKeys.dim(0), newKeys.dim(1), steps * 256, newKeys.dim(3)], dtype: newKeys.dtype)
        var valueBuffer = MLXArray.zeros(
            [newValues.dim(0), newValues.dim(1), steps * 256, newValues.dim(3)], dtype: newValues.dtype)
        keyBuffer[.ellipsis, 0 ..< count, 0...] = newKeys
        valueBuffer[.ellipsis, 0 ..< count, 0...] = newValues
        keys = keyBuffer[.ellipsis, ..<count, 0...]
        values = valueBuffer[.ellipsis, ..<count, 0...]
        return (keys!, values!)
    }
}

final class Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let isSliding: Bool
    let headDim: Int
    let heads: Int
    let kvHeads: Int
    let eps: Float
    let slidingWindow: Int
    let ropeBase: Float
    let freqs: Constant?
    let layerIndex: Int

    init(_ config: TextConfiguration, layer: Int) {
        let kind = config.layerTypes[layer]
        layerIndex = layer
        isSliding = kind == "sliding_attention"
        headDim = isSliding ? config.headDim : config.globalHeadDim
        heads = config.numAttentionHeads
        kvHeads = isSliding ? config.numKeyValueHeads : (config.numGlobalKeyValueHeads ?? config.numKeyValueHeads)
        eps = config.rmsNormEps
        slidingWindow = config.slidingWindow
        let rope = config.ropeParameters[kind]!
        ropeBase = rope.ropeTheta
        if rope.ropeType == "proportional" {
            // rope_utils.py ProportionalRoPE: the whole head through mx.fast.rope with explicit
            // frequencies, the unrotated pairs given an infinite frequency.
            let rotatedDims = 2 * Int((rope.partialRotaryFactor ?? 1) * Float(headDim)) / 2
            let angles = rotatedDims / 2
            let exponents = MLXArray(stride(from: 0, to: 2 * angles, by: 2)).asType(.float32) / Float(headDim)
            var values = 1.0 * pow(MLXArray(rope.ropeTheta), exponents)
            let nope = headDim / 2 - angles
            if nope > 0 {
                values = concatenated([values, MLXArray.full([nope], values: MLXArray(Float.infinity))])
            }
            eval(values)
            freqs = Constant(values)
        } else {
            precondition(rope.ropeType == "default")
            freqs = nil
        }
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = isSliding ? Linear(config.hiddenSize, kvHeads * headDim, bias: false) : nil
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func rope(_ x: MLXArray, offset: Int) -> MLXArray {
        if let freqs {
            return MLXFast.RoPE(
                x, dimensions: headDim, traditional: false, base: nil, scale: 1.0, offset: offset,
                freqs: freqs.value)
        }
        return MLXFast.RoPE(
            x, dimensions: headDim, traditional: false, base: ropeBase, scale: 1.0, offset: offset)
    }

    /// language.py Attention.__call__.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: LayerCache?,
        decoder: Bool, offset: Int
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        // --stages: layers 0 and 5 record what stage_dump.py records inside mlx-vlm's attention,
        // in mlx-vlm's call order (q_proj, q_norm, rope, k_proj, [v_proj], k_norm, rope, v_norm).
        var traced = 0
        let trace: (String, MLXArray) -> Void = { name, array in
            if !decoder, self.layerIndex == 0 || self.layerIndex == 5 {
                stageRecorder?(String(format: "a%d.%02d.%@", self.layerIndex, traced, name), array)
            }
            traced += 1
        }
        let qLinear = qProj(x)
        trace("linear", qLinear)
        var queries = qLinear.reshaped(batch, length, heads, headDim)
        queries = qNorm(queries)
        trace("rmsnorm", queries)
        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)
        trace("rope", queries)
        let kLinear = kProj(x)
        trace("linear", kLinear)
        let rawKeys = kLinear.reshaped(batch, length, kvHeads, headDim)
        var rawValues = rawKeys
        if let vProj {
            let vLinear = vProj(x)
            trace("linear", vLinear)
            rawValues = vLinear.reshaped(batch, length, kvHeads, headDim)
        }
        var keys = kNorm(rawKeys)
        trace("rmsnorm", keys)
        keys = keys.transposed(0, 2, 1, 3)
        keys = rope(keys, offset: offset)
        trace("rope", keys)
        var values = MLXFast.rmsNorm(rawValues, weight: MLXArray.mlxNone, eps: eps)
        trace("rmsnorm_noscale", values)
        values = values.transposed(0, 2, 1, 3)
        var mask = mask
        if decoder {
            if let cache, var encoderKeys = cache.keys, var encoderValues = cache.values {
                if isSliding && plantedBug != "no_window" {
                    let window = max(slidingWindow - 1, 0)
                    let encoderLength = encoderKeys.dim(2)
                    if window > 0, encoderLength > window, offset >= encoderLength {
                        encoderKeys = encoderKeys[.ellipsis, (encoderLength - window)..., 0...]
                        encoderValues = encoderValues[.ellipsis, (encoderLength - window)..., 0...]
                        if case .array(let array) = mask {
                            let keep = window + length
                            mask = .array(array[.ellipsis, (array.dim(-1) - keep)...])
                        }
                    }
                }
                keys = concatenated([encoderKeys, keys], axis: 2)
                values = concatenated([encoderValues, values], axis: 2)
            }
        } else if let cache {
            (keys, values) = cache.update(keys, values)
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1.0, mask: mask)
        if let stageRecorder, !decoder {
            stageRecorder("sdpa.\(layerIndex).queries", queries)
            stageRecorder("sdpa.\(layerIndex).keys", keys)
            stageRecorder("sdpa.\(layerIndex).values", values)
            stageRecorder("sdpa.\(layerIndex).out", output)
        }
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class DenseMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: TextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geglu(gate(x), up(x))) }
}

final class Router: Module {
    @ModuleInfo var proj: Linear
    @ParameterInfo var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray
    let eps: Float
    let rootSize: Float
    let topK: Int

    init(_ config: TextConfiguration) {
        _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        _perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
        eps = config.rmsNormEps
        rootSize = Float(pow(Double(config.hiddenSize), -0.5))
        topK = config.topKExperts
    }

    /// language.py Router.__call__.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        if plantedBug == "upstream_router" {
            let normed = MLXFast.rmsNorm(x, weight: (scale * rootSize).asType(x.dtype), eps: eps)
            let scores = proj(normed)
            let indices = argPartition(scores, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
            var weights = softmax(takeAlong(scores, indices, axis: -1), axis: -1)
            weights = weights * perExpertScale[indices].asType(weights.dtype)
            return (indices, weights)
        }
        var x = MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
        x = x * scale * rootSize
        let scores = proj(x)
        let indices = argPartition(scores, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = softmax(weights, axis: -1, precise: true)
        weights = weights * perExpertScale[indices]
        return (indices, weights)
    }
}

final class Experts: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUp: SwitchLinear
    @ModuleInfo(key: "down_proj") var down: SwitchLinear
    let hiddenDims: Int

    init(_ config: TextConfiguration) {
        hiddenDims = config.moeIntermediateSize
        _gateUp.wrappedValue = SwitchLinear(
            inputDims: config.hiddenSize, outputDims: 2 * config.moeIntermediateSize,
            numExperts: config.numExperts, bias: false)
        _down.wrappedValue = SwitchLinear(
            inputDims: config.moeIntermediateSize, outputDims: config.hiddenSize,
            numExperts: config.numExperts, bias: false)
    }

    /// language.py Experts.__call__, with switch_layers.py _gather_sort and _scatter_unsort
    /// (upstream's gatherSort and scatterUnsort are the same operations).
    func callAsFunction(_ inputs: MLXArray, indices topK: MLXArray, weights: MLXArray) -> MLXArray {
        var x = expandedDimensions(inputs, axes: [-2, -3])
        let doSort = topK.size >= 64
        var indices = topK
        var inverse: MLXArray?
        if doSort {
            let sorted = gatherSort(x: x, indices: topK)
            (x, indices, inverse) = (sorted.0, sorted.1, sorted.2)
        }
        let gateUpOut = gateUp(x, indices, sortedIndices: doSort)
        let gate = gateUpOut[.ellipsis, ..<hiddenDims]
        let up = gateUpOut[.ellipsis, hiddenDims...]
        var y = down(geglu(gate, up), indices, sortedIndices: doSort)
        if let inverse {
            y = scatterUnsort(x: y, invOrder: inverse, shape: topK.shape)
        }
        y = y.squeezed(axis: -2)
        return (y * expandedDimensions(weights, axis: -1)).sum(axis: -2)
    }
}

final class DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo var mlp: DenseMLP
    @ModuleInfo var router: Router
    @ModuleInfo var experts: Experts
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray
    let layerType: String
    let index: Int

    init(_ config: TextConfiguration, layer: Int) {
        layerType = config.layerTypes[layer]
        index = layer
        _attention.wrappedValue = Attention(config, layer: layer)
        _mlp.wrappedValue = DenseMLP(config)
        _router.wrappedValue = Router(config)
        _experts.wrappedValue = Experts(config)
        func norm() -> RMSNorm { RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps) }
        _inputLayerNorm.wrappedValue = norm()
        _postAttentionLayerNorm.wrappedValue = norm()
        _preFeedforwardLayerNorm.wrappedValue = norm()
        _postFeedforwardLayerNorm.wrappedValue = norm()
        _postFeedforwardLayerNorm1.wrappedValue = norm()
        _postFeedforwardLayerNorm2.wrappedValue = norm()
        _preFeedforwardLayerNorm2.wrappedValue = norm()
        _layerScalar.wrappedValue = MLXArray.ones([1])
    }

    /// language.py DecoderLayer.__call__.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: LayerCache?,
        decoder: Bool, offset: Int, layerScalar scalar: MLXArray? = nil
    ) -> MLXArray {
        let residual = x
        var h = inputLayerNorm(x)
        h = attention(h, mask: mask, cache: cache, decoder: decoder, offset: offset)
        stageRecorder?("attn.\(index)", h)
        h = postAttentionLayerNorm(h)
        h = residual + h

        let residual2 = h
        var h1 = preFeedforwardLayerNorm(h)
        h1 = mlp(h1)
        stageRecorder?("mlp.\(index)", h1)
        h1 = postFeedforwardLayerNorm1(h1)

        let flat = residual2.reshaped(-1, residual2.dim(-1))
        let (indices, weights) = router(flat)
        stageRecorder?("router.\(index).indices", indices)
        stageRecorder?("router.\(index).weights", weights)
        var h2 = preFeedforwardLayerNorm2(flat)
        h2 = experts(h2, indices: indices, weights: weights)
        stageRecorder?("experts.\(index)", h2)
        h2 = h2.reshaped(residual2.shape)
        h2 = postFeedforwardLayerNorm2(h2)

        h = postFeedforwardLayerNorm(h1 + h2)
        h = residual2 + h
        let out = h * (scalar ?? layerScalar)
        stageRecorder?("layer.\(index)", out)
        return out
    }
}

final class SelfConditioning: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    let eps: Float

    init(_ config: TextConfiguration) {
        eps = config.rmsNormEps
        _preNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ inputsEmbeds: MLXArray, _ signal: MLXArray) -> MLXArray {
        let normed = preNorm(signal)
        let conditioning = down(geglu(gate(normed), up(normed)))
        return MLXFast.rmsNorm(inputsEmbeds + conditioning, weight: MLXArray.mlxNone, eps: eps)
    }
}

final class EncoderScalar: Module {
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray
    override init() {
        _layerScalar.wrappedValue = MLXArray.ones([1])
    }
}

final class EncoderLanguageModel: Module {
    @ModuleInfo var layers: [EncoderScalar]
    init(_ count: Int) {
        _layers.wrappedValue = (0 ..< count).map { _ in EncoderScalar() }
    }
}

final class EncoderModel: Module {
    @ModuleInfo(key: "language_model") var languageModel: EncoderLanguageModel
    init(_ config: TextConfiguration) {
        _languageModel.wrappedValue = EncoderLanguageModel(config.layerTypes.count)
    }
}

final class DecoderModel: Module {
    let config: TextConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [DecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "self_conditioning") var selfConditioning: SelfConditioning
    let embedScale: Float

    init(_ config: TextConfiguration) {
        self.config = config
        embedScale = Float(pow(Double(config.hiddenSize), 0.5))
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layerTypes.count).map { DecoderLayer(config, layer: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfConditioning.wrappedValue = SelfConditioning(config)
    }
}

final class Backbone: Module {
    @ModuleInfo var decoder: DecoderModel
    @ModuleInfo var encoder: EncoderModel
    init(_ config: TextConfiguration) {
        _decoder.wrappedValue = DecoderModel(config)
        _encoder.wrappedValue = EncoderModel(config)
    }
}

final class DiffusionGemmaReference: Module, BaseLanguageModel {
    @ModuleInfo var model: Backbone
    let config: TextConfiguration
    let softcap: @Sendable (MLXArray) -> MLXArray

    init(_ config: TextConfiguration) {
        self.config = config
        softcap = makeSoftcap(config.finalLogitSoftcapping)
        _model.wrappedValue = Backbone(config)
    }

    /// diffusion_gemma.py Model.sanitize for a text-only load: the vision tower and its
    /// embedder are dropped, as is every encoder text weight except the layer scalars.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("rotary_emb") || key == "lm_head.weight" { continue }
            if key.hasPrefix("model.encoder.embed_vision.") || key.hasPrefix("model.encoder.vision_tower.") {
                continue
            }
            if key.hasPrefix("model.encoder.language_model.") && !key.hasSuffix(".layer_scalar") { continue }
            var name = key
            if name.hasSuffix(".experts.down_proj") || name.hasSuffix(".experts.gate_up_proj") {
                name += ".weight"
            }
            out[name] = value
        }
        return out
    }

    var decoder: DecoderModel { model.decoder }

    /// language.py EncoderModel.__call__ for a text prompt in one piece (MlxRuntime._prefill).
    func prefill(_ ids: MLXArray) -> [LayerCache] {
        let h0 = decoder.embedTokens(ids) * decoder.embedScale
        stageRecorder?("embeddings", h0)
        let caches = config.layerTypes.map { LayerCache(full: $0 == "full_attention") }
        let length = h0.dim(1)
        var h = h0
        for (index, layer) in decoder.layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if layer.layerType == "sliding_attention" {
                // RotatingKVCache.make_mask with window_size = sliding_window on an empty cache
                if length > 1 && length > config.slidingWindow {
                    let positions = MLXArray(Int32(0) ..< Int32(length))
                    let rows = positions[0..., .newAxis]
                    let columns = positions[.newAxis, 0...]
                    mask = .array(logicalAnd(rows .>= columns, rows .< columns + config.slidingWindow))
                } else {
                    mask = length > 1 ? .causal : .none
                }
            } else {
                mask = length > 1 ? .causal : .none
            }
            h = layer(
                h, mask: mask, cache: caches[index], decoder: false, offset: 0,
                layerScalar: model.encoder.languageModel.layers[index].layerScalar)
        }
        return caches
    }

    /// language.py DecoderModel._make_decoder_masks with no decoder_attention_mask.
    func decoderMasks(canvasLength: Int, caches: [LayerCache]) -> [String: MLXFast.ScaledDotProductAttentionMaskMode] {
        var masks = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
        for kind in Set(config.layerTypes) {
            let index = config.layerTypes.firstIndex(of: kind)!
            let cache = caches[index]
            let encoderLength = cache.keys?.dim(2) ?? 0
            let valid = min(cache.offset, encoderLength)
            if kind == "full_attention" {
                precondition(encoderLength == valid)
                masks[kind] = MLXFast.ScaledDotProductAttentionMaskMode.none
                continue
            }
            let prefix = max(config.slidingWindow - 1, 0)
            if plantedBug == "no_window" {
                // the planted mistake: no window for the canvas, every encoder position visible
                masks[kind] = MLXFast.ScaledDotProductAttentionMaskMode.none
                continue
            }
            if encoderLength == valid && encoderLength <= prefix {
                masks[kind] = MLXFast.ScaledDotProductAttentionMaskMode.none
                continue
            }
            let start = max(0, valid - prefix)
            let positions = MLXArray(Int32(0) ..< Int32(encoderLength))
            let encoderMask = logicalAnd(positions .>= start, positions .< valid)
            let row = concatenated([encoderMask, MLXArray.ones([canvasLength], dtype: .bool)], axis: 0)
            masks[kind] = .array(
                broadcast(row[.newAxis, .newAxis, .newAxis, 0...], to: [1, 1, canvasLength, encoderLength + canvasLength]))
        }
        return masks
    }

    /// Model.diffusion_decoder_logits: the canvas embedded with self-conditioning from the
    /// previous logits (prefers_logits_self_conditioning: the embedding is quantized), the
    /// decoder layers over the cache, the final norm, the tied head and the float32 softcap.
    func decoderLogits(
        _ canvas: MLXArray, caches: [LayerCache], selfConditioningLogits: MLXArray?,
        masks: [String: MLXFast.ScaledDotProductAttentionMaskMode]
    ) -> MLXArray {
        let embeddings = decoder.embedTokens(canvas) * decoder.embedScale
        let soft: MLXArray
        if let logits = selfConditioningLogits {
            let probabilities = softmax(logits, axis: -1, precise: true)
            guard let packed = decoder.embedTokens as? QuantizedEmbedding else {
                fatalError("the checkpoint's embedding is quantized")
            }
            soft = quantizedMM(
                probabilities.asType(embeddings.dtype), packed.weight, scales: packed.scales,
                biases: packed.biases, transpose: false, groupSize: packed.groupSize, bits: packed.bits,
                mode: packed.mode
            ).asType(embeddings.dtype) * decoder.embedScale
        } else {
            soft = zeros(like: embeddings)
        }
        var h =
            plantedBug == "skip_self_conditioning" && selfConditioningLogits == nil
            ? embeddings : decoder.selfConditioning(embeddings, soft)
        let offset = plantedBug == "rope_offset_zero" ? 0 : (caches.first?.offset ?? 0)
        for (index, layer) in decoder.layers.enumerated() {
            h = layer(h, mask: masks[layer.layerType]!, cache: caches[index], decoder: true, offset: offset)
        }
        h = decoder.norm(h)
        return softcap(decoder.embedTokens.asLinear(h))
    }
}
