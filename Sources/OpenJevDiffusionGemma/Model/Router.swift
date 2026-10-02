// mlx-vlm 0.6.15's diffusion_gemma/language.py Router (adapted from mlx-vlm, Copyright © 2025
// Prince Canuma, MIT), ported from the spike #22 transliteration. Not mlx-swift-lm's
// Gemma4TextRouter, which folds the scale into the norm weight and uses a plain softmax: exact in
// real numbers, but not mlx-vlm's rounding (docs/spikes/backend-validation.md, R20).

import Foundation
import MLX
import MLXNN

/// Picks each token's experts and their weights.
///
/// `rms_norm(x, None, eps)`, then `x * scale * hidden^-0.5` as two multiplies in the input's
/// dtype, `proj`, the top `top_k_experts` scores by `argpartition`, a precise softmax over them,
/// then times `per_expert_scale[indices]`.
public final class Router: Module {
    /// The projection to one score per expert, `proj`.
    @ModuleInfo public var proj: Linear
    /// The per-dimension input scale, `scale`.
    @ParameterInfo public var scale: MLXArray
    /// The weight each chosen expert's softmax share is multiplied by, `per_expert_scale`.
    @ParameterInfo(key: "per_expert_scale") public var perExpertScale: MLXArray

    let eps: Float
    let rootSize: Float
    /// The experts each token goes to.
    public let topK: Int

    /// Builds the router with the configuration's hidden size, expert count and `top_k_experts`.
    public init(_ config: DiffusionGemmaTextConfiguration) {
        _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        _perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
        eps = config.rmsNormEps
        // A Float, which MLX rounds to the input's dtype when it multiplies, as Python does.
        rootSize = Float(pow(Double(config.hiddenSize), -0.5))
        topK = config.topKExperts
        super.init()
    }

    /// language.py `Router.__call__`.
    ///
    /// - Parameter x: `[tokens, hidden]`.
    /// - Returns: the expert indices and their weights, both `[tokens, topK]`.
    public func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        var x = rmsNormNoScale(x, eps: eps)
        x = x * scale * rootSize
        let scores = proj(x)
        let indices = argPartition(scores, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = softmax(weights, axis: -1, precise: true)
        weights = weights * perExpertScale[indices]
        return (indices, weights)
    }
}
