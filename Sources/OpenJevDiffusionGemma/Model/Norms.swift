// The two RMSNorms of mlx-vlm 0.6.15's diffusion_gemma/language.py (adapted from mlx-vlm,
// Copyright © 2025 Prince Canuma, MIT): `nn.RMSNorm`, which multiplies by a learned weight, and
// `RMSNormNoScale`, which has none.

import MLX
import MLXNN

/// MLXNN's `RMSNorm(dimensions:eps:)`, the weighted norm every `*_layernorm`, `q_norm`, `k_norm`
/// and the final `norm` are. Its one parameter is `weight`, as in the checkpoint.
func rmsNorm(dimensions: Int, eps: Float) -> RMSNorm {
    RMSNorm(dimensions: dimensions, eps: eps)
}

/// language.py's `RMSNormNoScale`: `mx.fast.rms_norm(x, None, eps)`, a norm with no weight. The
/// full-attention layers' values and the router's input go through it.
func rmsNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
}
