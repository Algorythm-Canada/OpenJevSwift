// The decoder pass of a read (issues #26 and #28): mlx-vlm 0.6.15's diffusion_gemma/language.py
// DecoderModel (_embed_canvas, diffusion_self_conditioning and _make_decoder_masks, lines 383 to
// 553) and diffusion_gemma.py's diffusion_decoder_logits (lines 183 to 202), adapted from mlx-vlm,
// Copyright © 2025 Prince Canuma, MIT, ported from the spike #22 transliteration.

import MLX
import MLXNN

extension DiffusionGemmaModel {
    /// The decoder's attention mask for each layer type.
    public typealias DecoderMasks = [DiffusionGemmaTextConfiguration.LayerType:
        MLXFast.ScaledDotProductAttentionMaskMode]

    /// language.py `DecoderModel._make_decoder_masks` without a `decoder_attention_mask`.
    ///
    /// For each layer type, from the cache of its first layer: a full-attention layer gets
    /// `.none` while every cached position is valid (always, for a prefill this port makes), else
    /// a row hiding the invalid trailing positions. A sliding layer gets `.none` while the prompt
    /// is at most `sliding_window − 1` tokens; past that, a boolean row allowing the encoder
    /// positions in `[valid − (window − 1), valid)` and every canvas position. Rows are broadcast
    /// to `[1, 1, canvas, encoder + canvas]`; a sliding layer's attention cuts the row to its last
    /// `window − 1 + canvas` columns when it cuts the cache. A read computes these once and
    /// passes them to every step.
    public func decoderMasks(canvasLength: Int, cache: PromptCache) -> DecoderMasks {
        var masks: DecoderMasks = [:]
        for (index, layerType) in configuration.layerTypes.enumerated()
        where masks[layerType] == nil {
            let layerCache = cache.layers[index]
            let encoderLength = layerCache.keys?.dim(2) ?? 0
            let valid = min(layerCache.offset, encoderLength)
            let keyLength = encoderLength + canvasLength
            let positions = MLXArray(Int32(0)..<Int32(encoderLength))
            let encoderRow: MLXArray
            switch layerType {
            case .fullAttention:
                if encoderLength == valid {
                    masks[layerType] = MLXFast.ScaledDotProductAttentionMaskMode.none
                    continue
                }
                encoderRow = less(positions, MLXArray(Int32(valid)))
            case .slidingAttention:
                let prefix = max(configuration.slidingWindow - 1, 0)
                if encoderLength == valid && encoderLength <= prefix {
                    masks[layerType] = MLXFast.ScaledDotProductAttentionMaskMode.none
                    continue
                }
                let start = max(0, valid - prefix)
                encoderRow = logicalAnd(positions .>= start, positions .< valid)
            }
            let row = concatenated(
                [encoderRow, MLXArray.ones([canvasLength], dtype: .bool)], axis: 0)
            masks[layerType] = .array(
                broadcast(
                    row[.newAxis, .newAxis, .newAxis, 0...],
                    to: [1, 1, canvasLength, keyLength]))
        }
        return masks
    }

    /// The soft embeddings of the previous step's prediction, the self-conditioning signal.
    ///
    /// With a quantized embedding (the pinned checkpoint, mlx-vlm's
    /// `prefers_logits_self_conditioning`), language.py `_embed_canvas`: `softmax(logits,
    /// precise: true)` in the logits' float32, cast to the embedding dtype, `quantizedMM` against
    /// the packed embedding with `transpose: false` and its group size, bits and mode, cast to
    /// the embedding dtype and times the embedding scale. With a dense embedding,
    /// `diffusion_self_conditioning`: the logits cast to the weight's dtype, the precise softmax,
    /// `probs @ weight` in that dtype, times the scale, then cast to the embedding dtype.
    ///
    /// - Parameters:
    ///   - logits: the previous step's full logits, `[1, canvas, vocab]` float32.
    ///   - dtype: the canvas embeddings' dtype.
    func selfConditioningSignal(logits: MLXArray, dtype: DType) -> MLXArray {
        let embedding = decoder.embedTokens
        if let packed = embedding as? QuantizedEmbedding {
            let probabilities = softmax(logits, axis: -1, precise: true)
            return quantizedMM(
                probabilities.asType(dtype), packed.weight, scales: packed.scales,
                biases: packed.biases, transpose: false, groupSize: packed.groupSize,
                bits: packed.bits, mode: packed.mode
            ).asType(dtype) * decoder.embedScale
        }
        let weight = embedding.weight
        let probabilities = softmax(logits.asType(weight.dtype), axis: -1, precise: true)
        let soft = matmul(probabilities.asType(weight.dtype), weight).asType(weight.dtype)
        return (soft * decoder.embedScale).asType(dtype)
    }

    /// language.py `DecoderModel.__call__`: the canvas embedded and self-conditioned, every layer
    /// in decoder mode over its cache at RoPE offset = the prompt length, the final norm.
    ///
    /// - Returns: `norm(h)`, `[1, canvas, hidden]`.
    func decoderHidden(
        canvas: MLXArray, cache: PromptCache, conditioning: MLXArray?, masks: DecoderMasks
    ) -> MLXArray {
        let embeddings = decoder.embed(canvas)
        let signal =
            conditioning.map { selfConditioningSignal(logits: $0, dtype: embeddings.dtype) }
            ?? zeros(like: embeddings)
        var h = decoder.selfConditioning(embeddings, signal: signal)
        let offset = cache.layers.first?.offset ?? 0
        for (index, layer) in decoder.layers.enumerated() {
            h = layer(
                h, mask: masks[layer.layerType] ?? .none, cache: cache.layers[index],
                decoder: true, offset: offset)
        }
        return decoder.norm(h)
    }

    /// diffusion_gemma.py `diffusion_decoder_logits`: the full logits of one decoder pass,
    /// `softcap(embed_tokens.as_linear(norm(h)))`.
    ///
    /// - Parameters:
    ///   - canvas: the canvas ids, `[1, canvas]` int32.
    ///   - cache: the prompt's prefill.
    ///   - conditioning: nil on a read's first step, where the module runs on a zero signal; the
    ///     previous step's full logits afterwards.
    ///   - masks: ``decoderMasks(canvasLength:cache:)`` for this canvas length.
    /// - Returns: `[1, canvas, vocab]` float32.
    public func decoderLogits(
        canvas: MLXArray, cache: PromptCache, conditioning: MLXArray?, masks: DecoderMasks
    ) -> MLXArray {
        let h = decoderHidden(
            canvas: canvas, cache: cache, conditioning: conditioning, masks: masks)
        return softcap(decoder.embedTokens.asLinear(h))
    }

    /// The logits of the slot rows only (D-015): the slot rows of `norm(h)` through the tied head
    /// and the softcap. Everything before the head is the full pass. Close to the full rows but
    /// not bit-identical (``SlotProjection/slotsOnly``), so reads do not use it.
    ///
    /// - Parameters:
    ///   - canvas: the canvas ids, `[1, canvas]` int32.
    ///   - cache: the prompt's prefill.
    ///   - conditioning: nil on a read's first step, where the module runs on a zero signal; the
    ///     previous step's full logits afterwards.
    ///   - masks: ``decoderMasks(canvasLength:cache:)`` for this canvas length.
    ///   - positions: the slot positions in the canvas.
    /// - Returns: `[1, positions.count, vocab]` float32, in `positions` order.
    public func decoderSlotLogits(
        canvas: MLXArray, cache: PromptCache, conditioning: MLXArray?, masks: DecoderMasks,
        positions: [Int]
    ) -> MLXArray {
        let h = decoderHidden(
            canvas: canvas, cache: cache, conditioning: conditioning, masks: masks)
        let rows = take(h, MLXArray(positions.map(Int32.init)), axis: 1)
        return softcap(decoder.embedTokens.asLinear(rows))
    }
}
