// mlx-vlm 0.6.15's diffusion_gemma/language.py DecoderLayer (adapted from mlx-vlm, Copyright ©
// 2025 Prince Canuma, MIT), ported from the spike #22 transliteration. The residual structure is
// language.py lines 290 to 320, operation for operation.

import MLX
import MLXNN

/// Receives a layer's intermediate outputs under the names Tools/oracle/stage_dump.py records
/// them: `attn.N`, `mlp.N`, `router.N.indices`, `router.N.weights`, `experts.N` and `layer.N`.
public typealias StageObserver = (_ name: String, _ value: MLXArray) -> Void

/// One transformer layer, shared by the encoder and the decoder.
///
/// The attention block with its input and post-attention norms; a dense branch
/// (`pre_feedforward_layernorm`, `mlp`, `post_feedforward_layernorm_1`) and a mixture-of-experts
/// branch on the flattened residual (`router` on the residual itself, `pre_feedforward_layernorm_2`,
/// `experts`, `post_feedforward_layernorm_2`); their sum through `post_feedforward_layernorm`; the
/// residual add; then the product with a layer scalar, the decoder's own `layer_scalar` or the
/// encoder's passed in.
public final class DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttention: Attention
    @ModuleInfo public var mlp: DenseMLP
    @ModuleInfo public var router: Router
    @ModuleInfo public var experts: Experts
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm
    @ParameterInfo(key: "layer_scalar") public var layerScalar: MLXArray

    /// The layer's type.
    public let layerType: DiffusionGemmaTextConfiguration.LayerType
    /// The layer's index in the decoder.
    public let index: Int

    public init(_ config: DiffusionGemmaTextConfiguration, layerIndex: Int) {
        layerType = config.layerTypes[layerIndex]
        index = layerIndex
        _selfAttention.wrappedValue = Attention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = DenseMLP(config)
        _router.wrappedValue = Router(config)
        _experts.wrappedValue = Experts(config)
        func norm() -> RMSNorm { rmsNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps) }
        _inputLayerNorm.wrappedValue = norm()
        _postAttentionLayerNorm.wrappedValue = norm()
        _preFeedforwardLayerNorm.wrappedValue = norm()
        _postFeedforwardLayerNorm.wrappedValue = norm()
        _postFeedforwardLayerNorm1.wrappedValue = norm()
        _postFeedforwardLayerNorm2.wrappedValue = norm()
        _preFeedforwardLayerNorm2.wrappedValue = norm()
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    /// language.py `DecoderLayer.__call__`.
    ///
    /// - Parameters:
    ///   - x: `[batch, length, hidden]`.
    ///   - mask, cache, decoder, offset: as ``Attention/callAsFunction(_:mask:cache:decoder:offset:)``.
    ///   - layerScalar: the encoder's scalar for this layer in encoder mode; nil uses the
    ///     decoder's own.
    ///   - stages: receives the intermediate outputs, for the parity tests.
    /// - Returns: `[batch, length, hidden]`.
    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: LayerCache?,
        decoder: Bool, offset: Int, layerScalar scalar: MLXArray? = nil,
        stages: StageObserver? = nil
    ) -> MLXArray {
        let residual = x
        var h = inputLayerNorm(x)
        h = selfAttention(h, mask: mask, cache: cache, decoder: decoder, offset: offset)
        stages?("attn.\(index)", h)
        h = postAttentionLayerNorm(h)
        h = residual + h

        let residual2 = h
        var h1 = preFeedforwardLayerNorm(h)
        h1 = mlp(h1)
        stages?("mlp.\(index)", h1)
        h1 = postFeedforwardLayerNorm1(h1)

        let flat = residual2.reshaped(-1, residual2.dim(-1))
        let (indices, weights) = router(flat)
        stages?("router.\(index).indices", indices)
        stages?("router.\(index).weights", weights)
        var h2 = preFeedforwardLayerNorm2(flat)
        h2 = experts(h2, indices: indices, weights: weights)
        stages?("experts.\(index)", h2)
        h2 = h2.reshaped(residual2.shape)
        h2 = postFeedforwardLayerNorm2(h2)

        h = postFeedforwardLayerNorm(h1 + h2)
        h = residual2 + h
        let out = h * (scalar ?? layerScalar)
        stages?("layer.\(index)", out)
        return out
    }
}
