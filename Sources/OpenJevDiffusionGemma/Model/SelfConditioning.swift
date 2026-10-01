// mlx-vlm 0.6.15's diffusion_gemma/language.py SelfConditioning (lines 351 to 370, adapted from
// mlx-vlm, Copyright © 2025 Prince Canuma, MIT), ported from the spike #22 transliteration.

import MLX
import MLXNN

/// `model.decoder.self_conditioning`: mixes a soft embedding of the previous step's prediction
/// into the canvas embeddings.
///
/// `post_norm(embeddings + down_proj(geglu(gate_proj(pre_norm(signal)), up_proj(pre_norm(signal)))))`.
/// `pre_norm` is a weighted RMSNorm, the checkpoint's `pre_norm.weight`. `post_norm` is
/// language.py's `RMSNormNoScale`, a norm without a weight, which is why the checkpoint has no
/// `post_norm` tensor. On a read's first step the signal is zeros and the module still runs, as
/// mlx-vlm runs it.
public final class SelfConditioning: Module {
    @ModuleInfo(key: "pre_norm") public var preNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear
    let eps: Float

    public init(_ config: DiffusionGemmaTextConfiguration) {
        eps = config.rmsNormEps
        _preNorm.wrappedValue = rmsNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    /// language.py `SelfConditioning.__call__`.
    ///
    /// - Parameters:
    ///   - embeddings: the canvas embeddings, `[batch, canvas, hidden]`.
    ///   - signal: the soft embeddings of the previous step, or zeros, shaped as `embeddings`.
    /// - Returns: `[batch, canvas, hidden]`.
    public func callAsFunction(_ embeddings: MLXArray, signal: MLXArray) -> MLXArray {
        let normed = preNorm(signal)
        let conditioning = downProj(geglu(gateProj(normed), upProj(normed)))
        return rmsNormNoScale(embeddings + conditioning, eps: eps)
    }
}
