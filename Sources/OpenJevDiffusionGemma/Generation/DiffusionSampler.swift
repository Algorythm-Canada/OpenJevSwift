// The sampling pieces of a diffusion block (issue #50): mlx-vlm 0.6.15's generate/diffusion.py
// `_diffusion_initialize_canvas`, `_diffusion_linear_temperature`, `_diffusion_sample_canvas`,
// `_diffusion_token_probability`, `_diffusion_token_entropy`,
// `_diffusion_confidence_transfer_mask`, `_diffusion_entropy_transfer_mask` and
// `_diffusion_stable_and_confident` (lines 289 to 474), adapted from mlx-vlm, Copyright © 2025
// Prince Canuma, MIT. The operations, their order and their dtypes are mlx-vlm's, so each function
// gives mlx-vlm's output bit for bit (Fixtures/generation/generation.json `sampler`).

import MLX

/// The sampling functions of mlx-vlm's diffusion loop, as functions over `MLXArray`s.
///
/// Logits are `[batch, canvas, vocab]` and canvases `[batch, canvas]` int32. The random draws take
/// an `MLXRandom.RandomState`: mlx-vlm draws from MLX's global generator, and a state seeded
/// with `s` gives the draws `mx.random.seed(s)` gives, because it splits its key as MLX's global
/// `KeySequence` does.
public enum DiffusionSampler {
    /// `_diffusion_initialize_canvas`: `randint(0, vocabularySize)` ids, `[batch, length]` int32,
    /// from `random`'s next key.
    public static func initialCanvas(
        batch: Int = 1, length: Int, vocabularySize: Int, random: MLXRandom.RandomState
    ) -> MLXArray {
        MLXRandom.randInt(Int32(0)..<Int32(vocabularySize), [batch, length], key: random)
    }

    /// `_diffusion_linear_temperature`: `tMin + (tMax − tMin) × step / maxSteps`, computed in
    /// Double as Python computes it, then rounded to the float32 the logits are divided by; nil
    /// without a schedule.
    ///
    /// - Parameters:
    ///   - step: the step countdown, `maxSteps` on the first step down to 1 on the last.
    ///   - maxSteps: `max_denoising_steps`.
    ///   - schedule: `(tMin, tMax)`, the checkpoint's 0.4 and 0.8.
    public static func linearTemperature(
        step: Int, maxSteps: Int, schedule: (tMin: Double, tMax: Double)?
    ) -> Float? {
        guard let schedule else { return nil }
        return Float(
            schedule.tMin + (schedule.tMax - schedule.tMin) * (Double(step) / Double(maxSteps)))
    }

    /// `_diffusion_sample_canvas`: the argmax at temperature 0 or below; otherwise a categorical
    /// draw from the float32 logits divided by `temperature` (not divided at 1), with `random`'s
    /// next key. int32.
    public static func sampleCanvas(
        _ logits: MLXArray, temperature: Float, random: MLXRandom.RandomState
    ) -> MLXArray {
        var logits = logits.asType(.float32)
        if temperature <= 0 {
            return argMax(logits, axis: -1).asType(.int32)
        }
        if temperature != 1 {
            logits = logits / temperature
        }
        return MLXRandom.categorical(logits, key: random).asType(.int32)
    }

    /// `_diffusion_token_probability`: the probability of `tokenIDs` under the float32 softmax of
    /// `logits`, `exp(logit − logsumexp)`, `[batch, canvas]`.
    public static func tokenProbability(_ logits: MLXArray, tokenIDs: MLXArray) -> MLXArray {
        let logits = logits.asType(.float32)
        let tokenLogits = takeAlong(logits, tokenIDs[.ellipsis, .newAxis], axis: -1)
            .squeezed(axis: -1)
        return exp(tokenLogits - logSumExp(logits, axis: -1))
    }

    /// `_diffusion_token_entropy`: the entropy of each position's float32 softmax, in nats,
    /// `[batch, canvas]`.
    public static func tokenEntropy(_ logits: MLXArray) -> MLXArray {
        let logits = logits.asType(.float32)
        let logProbabilities = logits - logSumExp(logits, axis: -1, keepDims: true)
        let probabilities = exp(logProbabilities)
        return -sum(probabilities * logProbabilities, axis: -1)
    }

    /// `_diffusion_confidence_transfer_mask`, the `confidence-threshold` sampler's acceptance:
    /// the unrevealed positions whose confidence is at least `threshold`; in a row where some
    /// position is unrevealed and none passes, the unrevealed position of highest confidence (the
    /// first on a tie); every unrevealed position when `forceAll`.
    public static func confidenceTransferMask(
        confidence: MLXArray, unrevealed: MLXArray, threshold: Float, forceAll: Bool = false
    ) -> MLXArray {
        if forceAll {
            return unrevealed
        }
        let transfer = logicalAnd(unrevealed, confidence .>= threshold)
        let hasUnrevealed = any(unrevealed, axis: -1)
        let hasTransfer = any(transfer, axis: -1)
        let needsForce = logicalAnd(hasUnrevealed, logicalNot(hasTransfer))
        let masked = which(unrevealed, confidence, MLXArray(-Float.infinity))
        let best = argMax(masked, axis: -1)
        let positions = MLXArray(Int32(0)..<Int32(confidence.dim(-1)))[.newAxis, 0...]
        let forced = logicalAnd(
            positions .== best[0..., .newAxis], needsForce[0..., .newAxis])
        return logicalOr(transfer, forced)
    }

    /// `_diffusion_entropy_transfer_mask`, the checkpoint's `EntropyBoundSamplerConfig`: in
    /// ascending entropy order, the positions whose cumulative entropy less the largest entropy
    /// so far is at most `bound`.
    public static func entropyTransferMask(entropy: MLXArray, bound: Float) -> MLXArray {
        let order = argSort(entropy, axis: -1)
        let sorted = takeAlong(entropy, order, axis: -1)
        let cumulative = cumsum(sorted, axis: -1)
        let maximum = cummax(sorted, axis: -1)
        let selected = (cumulative - maximum) .<= bound
        return putAlong(zeros(like: selected), order, values: selected, axis: -1)
    }

    /// The stable-and-confident rule's settings: the checkpoint's `stability_threshold` 1 and
    /// `confidence_threshold` 0.005.
    public struct StoppingRule: Sendable, Hashable {
        /// `stability_threshold`: how many earlier steps the argmax canvas must equal.
        public var stabilityThreshold: Int
        /// `confidence_threshold`: the mean entropy, in nats, the canvas must be below.
        public var confidenceThreshold: Double

        /// Creates a rule; the defaults are `_diffusion_stable_and_confident`'s.
        public init(stabilityThreshold: Int = 1, confidenceThreshold: Double = 0.005) {
            self.stabilityThreshold = stabilityThreshold
            self.confidenceThreshold = confidenceThreshold
        }
    }

    /// `_diffusion_stable_and_confident`: true when `canvas` equals each of the last
    /// `stabilityThreshold` canvases in `history` and the mean entropy of `logits` is below
    /// `confidenceThreshold`; false without a rule. Appends `canvas` to `history`, which keeps the
    /// last `stabilityThreshold` canvases, as one block's steps share it.
    public static func stableAndConfident(
        canvas: MLXArray, logits: MLXArray, history: inout [MLXArray], rule: StoppingRule?
    ) -> Bool {
        guard let rule else { return false }
        let stable =
            history.count == rule.stabilityThreshold
            && history.allSatisfy { all(canvas .== $0).item(Bool.self) }
        history.append(canvas)
        if history.count > rule.stabilityThreshold {
            history.removeFirst()
        }
        guard stable else { return false }
        let entropy = tokenEntropy(logits)
        return (mean(entropy) .< Float(rule.confidenceThreshold)).item(Bool.self)
    }
}
