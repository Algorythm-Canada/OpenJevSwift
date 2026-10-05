// The prefill of a prompt with images (issue #47): mlx-vlm 0.6.15's diffusion_gemma/language.py
// EncoderModel `_embed_inputs`, `_vision_block_overlay`, `_make_encoder_masks` and
// `chunked_prefill_policy` (lines 555 to 771), adapted from mlx-vlm, Copyright © 2025 Prince
// Canuma, MIT, in one piece as upstream's MlxRuntime._prefill runs it for an ImagePrompt.

import Foundation
import MLX

extension DiffusionGemmaModel {
    /// language.py's `chunked_prefill_policy` for the prompts the runtime builds: false when the
    /// prompt has pixel values or any image or video soft token (`mm_token_type_ids` 1 or 2),
    /// because an image block attends to itself in both directions and a chunk boundary would
    /// cut it. The policy's other conditions (padding, a partial attention mask, a static cache)
    /// never hold for the runtime's prompts.
    ///
    /// The read path never chunks a prefill, text or image (see ``prefill(promptIDs:)``); this is
    /// the rule that generation (milestone 5), which may chunk long text prompts, must apply.
    public static func allowsChunkedPrefill(mmTokenTypeIDs: [Int]?, hasPixelValues: Bool)
        -> Bool
    {
        if hasPixelValues { return false }
        if let mmTokenTypeIDs, mmTokenTypeIDs.contains(where: { $0 == 1 || $0 == 2 }) {
            return false
        }
        return true
    }

    /// language.py's `_embed_inputs`: the image and video soft token positions (by id, or
    /// marked 1 or 2 in `mm_token_type_ids`) replaced by `pad` before the embedding, then the
    /// vision tower's projected features scattered into them in order (`masked_scatter`).
    ///
    /// - Parameters:
    ///   - ids: `[1, n]` int32 prompt ids.
    ///   - mmTokenTypeIDs: `[1, n]` int32 types.
    ///   - pixelValues: the images, as ``VisionModel/callAsFunction(_:)`` takes them.
    ///   - stages: receives `embeddings`, the text embeddings before the scatter, and
    ///     `image_features`, the projected soft tokens.
    func embedInputs(
        ids: MLXArray, mmTokenTypeIDs: MLXArray, pixelValues: [MLXArray],
        stages: StageObserver? = nil
    ) throws -> MLXArray {
        guard let visionTower = encoder.visionTower, let embedVision = encoder.embedVision else {
            throw ReadInputError.noVisionTower
        }
        var visionMask = ids .== Int32(imageTokenID)
        if let videoTokenID {
            visionMask = logicalOr(visionMask, ids .== Int32(videoTokenID))
        }
        if mmTokenTypeIDs.shape == ids.shape {
            let marked = logicalOr(mmTokenTypeIDs .== Int32(1), mmTokenTypeIDs .== Int32(2))
            visionMask = logicalOr(visionMask, marked)
        }
        let textIDs = which(visionMask, MLXArray(Int32(configuration.padTokenID)), ids)
        let embeddings = decoder.embed(textIDs)
        stages?("embeddings", embeddings)
        let features = embedVision(visionTower(pixelValues, stages: stages))
            .asType(embeddings.dtype)
        stages?("image_features", features)
        let expanded = broadcast(expandedDimensions(visionMask, axis: -1), to: embeddings.shape)
        return maskedScatter(embeddings, mask: expanded, source: features)
    }

    /// language.py's `_vision_block_overlay`: `[B, n, n]`, true where the query and the key are
    /// in the same contiguous run of image or video soft tokens, or nil when the configuration's
    /// `use_bidirectional_attention` is not `vision`, the prompt is one token, or it has no soft
    /// token.
    func visionBlockOverlay(_ mmTokenTypeIDs: MLXArray, length: Int) -> MLXArray? {
        guard configuration.useBidirectionalAttention == "vision", length > 1,
            mmTokenTypeIDs.dim(-1) == length
        else { return nil }
        let isVision = logicalOr(mmTokenTypeIDs .== Int32(1), mmTokenTypeIDs .== Int32(2))
        guard isVision.any().item(Bool.self) else { return nil }
        let previous = concatenated(
            [zeros(like: isVision[0..., ..<1]), isVision[0..., ..<(length - 1)]], axis: 1)
        let starts = logicalAnd(isVision, logicalNot(previous))
        let groups = cumsum(starts.asType(.int32), axis: 1) - 1
        let blocks = which(isVision, groups, zeros(like: groups) - 1)
        let queries = expandedDimensions(blocks, axis: -1)
        let keys = expandedDimensions(blocks, axis: -2)
        return logicalAnd(queries .!= Int32(-1), queries .== keys)
    }

    /// language.py's `_make_encoder_masks` for a prompt prefill on an empty cache with the
    /// processor's attention mask (all ones): for each layer type, the boolean causal mask, cut
    /// to the window on a sliding layer, or-ed with `overlay`, `[B, 1, n, n]`.
    func imageEncoderMasks(length: Int, batch: Int, overlay: MLXArray?)
        -> [DiffusionGemmaTextConfiguration.LayerType: MLXFast.ScaledDotProductAttentionMaskMode]
    {
        let keyMask = MLXArray.ones([batch, length], dtype: .bool)
        let positions = MLXArray(Int32(0)..<Int32(length))
        let queryPositions = positions[0..., .newAxis]
        let base = queryPositions .>= positions[.newAxis, 0...]
        var masks:
            [DiffusionGemmaTextConfiguration.LayerType: MLXFast.ScaledDotProductAttentionMaskMode] =
                [:]
        for layerType in Set(configuration.layerTypes) {
            var mask = base
            if layerType == .slidingAttention {
                let window =
                    queryPositions .< positions[.newAxis, 0...] + configuration.slidingWindow
                mask = logicalAnd(mask, window)
            }
            mask = mask[.newAxis, .newAxis, 0..., 0...]
            if let overlay {
                mask = logicalOr(mask, overlay[0..., .newAxis, 0..., 0...])
            }
            mask = logicalAnd(mask, keyMask[0..., .newAxis, .newAxis, 0...])
            masks[layerType] = .array(broadcast(mask, to: [batch, 1, length, length]))
        }
        return masks
    }

    /// Prefills a prompt with images in one piece, as upstream's `MlxRuntime._prefill` calls
    /// `diffusion_prefill_cache` with the processor's `input_ids`, `pixel_values`,
    /// `mm_token_type_ids` and `attention_mask`, and evaluates the caches: the embeddings with
    /// the image features scattered in (``embedInputs(ids:mmTokenTypeIDs:pixelValues:stages:)``),
    /// then every layer in encoder mode with the explicit masks and the overlay, so each image's
    /// soft tokens attend to each other in both directions. Never chunked
    /// (``allowsChunkedPrefill(mmTokenTypeIDs:hasPixelValues:)``).
    ///
    /// - Parameters:
    ///   - inputs: the prompt and pixels from ``ImageReadInputs``.
    ///   - stages: receives the embeddings, the image features and each layer's intermediate
    ///     outputs, for the parity tests.
    /// - Returns: the caches, their offset and the prompt tokens, the expanded prompt's length
    ///   with the image tokens.
    /// - Throws: ``ReadInputError/noVisionTower`` for a text-only tree,
    ///   ``ReadInputError/emptyPrompt``,
    ///   ``ReadInputError/promptTokenOutOfRange(index:id:vocabularySize:)``, or
    ///   ``ReadInputError/tokenTypesMismatch(ids:types:)``.
    public func prefill(image inputs: ImageReadInputs, stages: StageObserver? = nil) throws
        -> PromptCache
    {
        guard readsImages else { throw ReadInputError.noVisionTower }
        let promptIDs = inputs.prompt.ids
        guard !promptIDs.isEmpty else { throw ReadInputError.emptyPrompt }
        let vocabularySize = configuration.vocabSize
        if let index = promptIDs.firstIndex(where: { $0 < 0 || $0 >= vocabularySize }) {
            throw ReadInputError.promptTokenOutOfRange(
                index: index, id: promptIDs[index], vocabularySize: vocabularySize)
        }
        let types = inputs.prompt.mmTokenTypeIDs
        guard types.count == promptIDs.count else {
            throw ReadInputError.tokenTypesMismatch(ids: promptIDs.count, types: types.count)
        }
        let length = promptIDs.count
        let ids = MLXArray(promptIDs.map(Int32.init)).reshaped(1, length)
        let typeArray = MLXArray(types.map(Int32.init)).reshaped(1, length)
        let embeddings = try embedInputs(
            ids: ids, mmTokenTypeIDs: typeArray, pixelValues: inputs.pixelValues, stages: stages)
        stages?("inputs_embeds", embeddings)
        let overlay = visionBlockOverlay(typeArray, length: length)
        let masks = imageEncoderMasks(length: length, batch: 1, overlay: overlay)
        for (layerType, name) in [
            (DiffusionGemmaTextConfiguration.LayerType.slidingAttention, "mask.0"),
            (.fullAttention, "mask.5"),
        ] {
            if case .array(let mask) = masks[layerType] {
                stages?(name, mask)
            }
        }
        let layers = prefill(embeddings: embeddings, masks: masks, stages: stages).caches
        eval(layers.flatMap { [$0.keys, $0.values].compactMap { $0 } })
        return PromptCache(layers: layers, offset: length, promptTokens: length)
    }
}
