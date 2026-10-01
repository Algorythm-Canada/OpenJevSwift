// The logit softcap of mlx-vlm 0.6.15's diffusion_gemma/language.py (adapted from mlx-vlm,
// Copyright © 2025 Prince Canuma, MIT), compiled as mlx-vlm compiles it.

import MLX

/// `tanh(x / cap) * cap` in float32, compiled shapeless. The pinned checkpoint's
/// `final_logit_softcapping` is 30. The decoder read pass (#26) applies it to the tied head's
/// logits.
public func makeSoftcap(_ cap: Float) -> @Sendable (MLXArray) -> MLXArray {
    compile(shapeless: true) { x in tanh(x.asType(.float32) / cap) * cap }
}
