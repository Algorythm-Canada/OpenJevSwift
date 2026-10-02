// mlx-vlm 0.6.15's diffusion_gemma/language.py Attention (adapted from mlx-vlm, Copyright © 2025
// Prince Canuma, MIT), with the RoPE of rope_utils.py, ported from the spike #22 transliteration
// (Tools/oracle/UpstreamProbe/Sources/Transliteration/Model.swift), which matched mlx-vlm bit for
// bit. The operations, their order, shapes and dtypes are mlx-vlm's.

import MLX
import MLXNN

/// Holds an array the module tree must not report as a parameter.
final class Constant {
    var value: MLXArray
    init(_ value: MLXArray) { self.value = value }
}

/// One layer's self-attention.
///
/// Sliding layers have `head_dim` heads, `num_key_value_heads` KV heads, a `v_proj` and the
/// default RoPE. Full layers have `global_head_dim` heads, `num_global_key_value_heads` KV heads
/// and no `v_proj`: their values are the raw keys (before `k_norm`). Every layer's values go
/// through the parameter-free RMSNorm. The attention scale is 1.
///
/// Two modes. Encoder mode (`decoder: false`) attends over the input with the given mask and
/// stores the keys and values in the layer's cache. Decoder mode (`decoder: true`) puts the
/// cached encoder keys and values ahead of the canvas's; a sliding layer keeps only the last
/// `sliding_window − 1` encoder positions when the cache is longer than that and the offset is
/// at least its length, and an array mask is cut to its last `sliding_window − 1 + canvas`
/// columns to match (language.py lines 225 to 246).
public final class Attention: Module {
    /// The query projection, the checkpoint's `q_proj`.
    @ModuleInfo(key: "q_proj") public var qProj: Linear
    /// The key projection, `k_proj`.
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    /// The value projection, `v_proj`: sliding layers only, since a full layer's values are its
    /// keys.
    @ModuleInfo(key: "v_proj") public var vProj: Linear?
    /// The output projection, `o_proj`.
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    /// The query norm, `q_norm`, an RMSNorm with a weight.
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    /// The key norm, `k_norm`, an RMSNorm with a weight.
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    /// The layer's type.
    public let layerType: DiffusionGemmaTextConfiguration.LayerType
    /// The layer's index in the decoder.
    public let layerIndex: Int
    /// The size of one head.
    public let headDim: Int
    /// The query heads.
    public let heads: Int
    /// The key-value heads.
    public let keyValueHeads: Int
    let eps: Float
    let slidingWindow: Int
    let ropeBase: Float
    /// The proportional RoPE frequencies, nil for the default RoPE.
    private let frequencies: Constant?

    /// Builds layer `layerIndex`'s attention from the text configuration: the head size, the
    /// key-value heads, the `v_proj` and the RoPE follow the layer's type.
    public init(_ config: DiffusionGemmaTextConfiguration, layerIndex: Int) {
        let layerType = config.layerTypes[layerIndex]
        self.layerType = layerType
        self.layerIndex = layerIndex
        headDim = config.headDim(for: layerType)
        heads = config.numAttentionHeads
        keyValueHeads = config.keyValueHeads(for: layerType)
        eps = config.rmsNormEps
        slidingWindow = config.slidingWindow
        let rope = config.ropeParameters(for: layerType)
        ropeBase = rope.ropeTheta
        if rope.ropeType == "proportional" {
            frequencies = Constant(
                Self.proportionalFrequencies(
                    headDim: headDim, theta: rope.ropeTheta,
                    partialRotaryFactor: rope.partialRotaryFactor ?? 1))
        } else {
            frequencies = nil
        }
        let hidden = config.hiddenSize
        _qProj.wrappedValue = Linear(hidden, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(hidden, keyValueHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue =
            layerType == .slidingAttention
            ? Linear(hidden, keyValueHeads * headDim, bias: config.attentionBias) : nil
        _oProj.wrappedValue = Linear(heads * headDim, hidden, bias: config.attentionBias)
        _qNorm.wrappedValue = rmsNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = rmsNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// rope_utils.py `ProportionalRoPE`'s table: the whole head goes through `mx.fast.rope` with
    /// explicit frequencies `1.0 * pow(theta, arange(0, rotated, 2) / headDim)`, and the pairs it
    /// does not rotate get an infinite frequency. 256 entries for the pinned checkpoint, 64 of
    /// them finite. Computed with MLX `pow` on the default device and evaluated now, as mlx-vlm
    /// computes it at load time.
    static func proportionalFrequencies(
        headDim: Int, theta: Float, partialRotaryFactor: Float
    ) -> MLXArray {
        let rotatedDims = 2 * Int(partialRotaryFactor * Float(headDim)) / 2
        let angles = rotatedDims / 2
        let exponents =
            MLXArray(stride(from: 0, to: 2 * angles, by: 2)).asType(.float32) / Float(headDim)
        var values = 1.0 * pow(MLXArray(theta), exponents)
        let unrotated = headDim / 2 - angles
        if unrotated > 0 {
            values = concatenated([
                values, MLXArray.full([unrotated], values: MLXArray(Float.infinity)),
            ])
        }
        eval(values)
        return values
    }

    /// The proportional RoPE table of a full-attention layer, nil on a sliding layer. It is the
    /// table computed with MLX `pow` at init; D-014's exact tier sets the oracle's table here
    /// (`rope` in Fixtures/oracle/reads.json), because 23 of the 64 finite entries differ in the
    /// last bit under mlx-swift's kernels. Setting it on a sliding layer has no effect.
    public var fullAttentionFrequencies: MLXArray? {
        get { frequencies?.value }
        set {
            if let frequencies, let newValue {
                frequencies.value = newValue
            }
        }
    }

    func rope(_ x: MLXArray, offset: Int) -> MLXArray {
        if let frequencies {
            return MLXFast.RoPE(
                x, dimensions: headDim, traditional: false, base: nil, scale: 1.0,
                offset: offset, freqs: frequencies.value)
        }
        return MLXFast.RoPE(
            x, dimensions: headDim, traditional: false, base: ropeBase, scale: 1.0, offset: offset)
    }

    /// language.py `Attention.__call__`.
    ///
    /// - Parameters:
    ///   - x: `[batch, length, hidden]`.
    ///   - mask: the encoder mask in encoder mode; the decoder mask in decoder mode.
    ///   - cache: the layer's encoder cache: written in encoder mode, read in decoder mode.
    ///   - decoder: the mode.
    ///   - offset: the RoPE position of `x`'s first token: 0 for a one-piece prefill, the cache
    ///     offset (the prompt length) for the canvas.
    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: LayerCache?,
        decoder: Bool, offset: Int
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qProj(x).reshaped(batch, length, heads, headDim)
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        let rawKeys = kProj(x).reshaped(batch, length, keyValueHeads, headDim)
        let rawValues =
            vProj.map { $0(x).reshaped(batch, length, keyValueHeads, headDim) } ?? rawKeys
        var keys = kNorm(rawKeys).transposed(0, 2, 1, 3)
        keys = rope(keys, offset: offset)
        var values = rmsNormNoScale(rawValues, eps: eps).transposed(0, 2, 1, 3)

        var mask = mask
        if decoder {
            if let cache, var encoderKeys = cache.keys, var encoderValues = cache.values {
                if layerType == .slidingAttention {
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
            (keys, values) = cache.update(keys: keys, values: values)
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1.0, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}
