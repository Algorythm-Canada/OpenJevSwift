// mlx-vlm 0.6.15's diffusion_gemma/language.py Experts with switch_layers.py's _gather_sort and
// _scatter_unsort (adapted from mlx-vlm, Copyright © 2025 Prince Canuma, MIT), ported from the
// spike #22 transliteration on mlx-swift-lm's SwitchLinear, gatherSort and scatterUnsort, which
// are the same operations.

import MLX
import MLXLMCommon
import MLXNN

/// A layer's mixture of experts.
///
/// `gate_up_proj` is one `SwitchLinear` with `2 × moe_intermediate_size` outputs (1,408 for the
/// pinned checkpoint), split at `moe_intermediate_size` into gate and up; `down_proj` maps back
/// to the hidden size. Loading quantizes both to mlx-swift-lm's `QuantizedSwitchLinear`, whose
/// gathered quantized matmuls are mlx-vlm's. Not mlx-swift-lm's `Gemma4TextExperts`, whose
/// `SwitchGLU` keeps gate and up apart.
public final class Experts: Module {
    /// Every expert's gate and up projections in one `SwitchLinear`, `gate_up_proj`, split at
    /// ``hiddenDims``.
    @ModuleInfo(key: "gate_up_proj") public var gateUpProj: SwitchLinear
    /// Every expert's down projection, `down_proj`.
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

    /// `moe_intermediate_size`, where `gate_up_proj`'s output splits.
    public let hiddenDims: Int

    /// The assignment count from which the tokens are sorted by expert before the gathered
    /// matmuls, as switch_layers.py does.
    public static let sortThreshold = 64

    /// Builds the experts with the configuration's expert count and sizes, without biases.
    public init(_ config: DiffusionGemmaTextConfiguration) {
        hiddenDims = config.moeIntermediateSize
        _gateUpProj.wrappedValue = SwitchLinear(
            inputDims: config.hiddenSize, outputDims: 2 * config.moeIntermediateSize,
            numExperts: config.numExperts, bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: config.moeIntermediateSize, outputDims: config.hiddenSize,
            numExperts: config.numExperts, bias: false)
        super.init()
    }

    /// language.py `Experts.__call__`.
    ///
    /// - Parameters:
    ///   - inputs: `[tokens, hidden]`.
    ///   - indices: `[tokens, topK]`, from the router.
    ///   - weights: `[tokens, topK]`, from the router.
    ///   - sort: whether to sort the assignments by expert; nil sorts when there are at least
    ///     ``sortThreshold`` of them, as mlx-vlm does. Tests force each path.
    /// - Returns: `[tokens, hidden]`.
    public func callAsFunction(
        _ inputs: MLXArray, indices: MLXArray, weights: MLXArray, sort: Bool? = nil
    ) -> MLXArray {
        var x = expandedDimensions(inputs, axes: [-2, -3])
        let doSort = sort ?? (indices.size >= Self.sortThreshold)
        var routed = indices
        var inverse: MLXArray?
        if doSort {
            let sorted = gatherSort(x: x, indices: indices)
            (x, routed, inverse) = (sorted.0, sorted.1, sorted.2)
        }
        let gateUp = gateUpProj(x, routed, sortedIndices: doSort)
        let gate = gateUp[.ellipsis, ..<hiddenDims]
        let up = gateUp[.ellipsis, hiddenDims...]
        var y = downProj(geglu(gate, up), routed, sortedIndices: doSort)
        if let inverse {
            y = scatterUnsort(x: y, invOrder: inverse, shape: indices.shape)
        }
        y = y.squeezed(axis: -2)
        return (y * expandedDimensions(weights, axis: -1)).sum(axis: -2)
    }
}
