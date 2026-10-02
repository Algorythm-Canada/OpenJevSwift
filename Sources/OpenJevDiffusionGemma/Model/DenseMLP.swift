// mlx-vlm 0.6.15's diffusion_gemma/language.py MLP and its compiled GeGLU (adapted from mlx-vlm,
// Copyright © 2025 Prince Canuma, MIT), ported from the spike #22 transliteration.

import MLX
import MLXNN

/// `gelu_approx(gate) * x`, compiled shapeless as language.py compiles it. `MLXNN.geluApproximate`
/// is the same expression as `mlx.nn.gelu_approx`. The dense MLP, the experts and the
/// self-conditioning module (#28) share it.
let geglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { gate, x in
    geluApproximate(gate) * x
}

/// A layer's dense branch: `down_proj(geglu(gate_proj(x), up_proj(x)))`.
public final class DenseMLP: Module {
    /// The gate projection, `gate_proj`.
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    /// The up projection, `up_proj`.
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    /// The down projection back to the hidden size, `down_proj`.
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    /// Builds the branch with the configuration's hidden and intermediate sizes, without biases.
    public init(_ config: DiffusionGemmaTextConfiguration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    /// `down_proj(geglu(gate_proj(x), up_proj(x)))`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geglu(gateProj(x), upProj(x)))
    }
}
