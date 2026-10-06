// The prompt prefill of a read (issue #25): mlx-vlm 0.6.15's diffusion_gemma/language.py
// EncoderModel for a text prompt (lines 555 to 771) and the Backbone's diffusion_prefill_cache
// (lines 780 to 810), adapted from mlx-vlm, Copyright © 2025 Prince Canuma, MIT, in one piece as
// upstream's MlxRuntime._prefill runs it.

import CryptoKit
import Foundation
import MLX

/// An input the read path refuses, before anything reaches MLX.
public enum ReadInputError: Error, Equatable, Sendable, CustomStringConvertible {
    /// The prompt has no tokens.
    case emptyPrompt
    /// A prompt token is outside the vocabulary.
    case promptTokenOutOfRange(index: Int, id: Int, vocabularySize: Int)
    /// The canvas has no tokens.
    case emptyCanvas
    /// A canvas token is outside the vocabulary.
    case canvasTokenOutOfRange(index: Int, id: Int, vocabularySize: Int)
    /// The read names no slot.
    case noSlots
    /// A slot position is outside the canvas.
    case slotOutOfRange(slot: Int, position: Int, canvasLength: Int)
    /// A slot has no labels.
    case noLabels(slot: Int)
    /// A label id is outside the vocabulary.
    case labelOutOfRange(slot: Int, id: Int, vocabularySize: Int)
    /// `steps` is below 1.
    case stepsOutOfRange(Int)
    /// `topK` is not in `1 ..< vocabularySize`.
    case topKOutOfRange(topK: Int, vocabularySize: Int)
    /// The cache was made by a model with another layer count.
    case cacheLayerMismatch(cacheLayers: Int, modelLayers: Int)
    /// An image prompt reached a tree without a vision tower.
    case noVisionTower
    /// An image prompt's `mm_token_type_ids` do not have one entry per id.
    case tokenTypesMismatch(ids: Int, types: Int)

    /// What was refused, with the index, the id and the bounds involved.
    public var description: String {
        switch self {
        case .emptyPrompt:
            return "the prompt has no tokens"
        case .promptTokenOutOfRange(let index, let id, let vocabularySize):
            return "prompt token \(index) is \(id), outside the vocabulary of \(vocabularySize)"
        case .emptyCanvas:
            return "the canvas has no tokens"
        case .canvasTokenOutOfRange(let index, let id, let vocabularySize):
            return "canvas token \(index) is \(id), outside the vocabulary of \(vocabularySize)"
        case .noSlots:
            return "the read names no slot"
        case .slotOutOfRange(let slot, let position, let canvasLength):
            return "slot \(slot) is at position \(position), outside the canvas of \(canvasLength)"
        case .noLabels(let slot):
            return "slot \(slot) has no labels"
        case .labelOutOfRange(let slot, let id, let vocabularySize):
            return "slot \(slot) has label \(id), outside the vocabulary of \(vocabularySize)"
        case .stepsOutOfRange(let steps):
            return "steps is \(steps); a read takes at least 1"
        case .topKOutOfRange(let topK, let vocabularySize):
            return "topK is \(topK); it must be at least 1 and below \(vocabularySize)"
        case .cacheLayerMismatch(let cacheLayers, let modelLayers):
            return "the cache has \(cacheLayers) layers and the model \(modelLayers)"
        case .noVisionTower:
            return "the model was loaded without a vision tower, so it cannot read images"
        case .tokenTypesMismatch(let ids, let types):
            return "the image prompt has \(ids) ids and \(types) token types"
        }
    }
}

/// A prefilled prompt: one encoder cache per layer, which every read of the prompt shares.
///
/// Reads never write to it, so a read with any canvas, slots or steps can reuse it, as upstream's
/// prefill cache does. Not Sendable: it holds MLX arrays, and its caller serialises its use with
/// the model's.
public final class PromptCache {
    /// The caches, one per decoder layer, in layer order.
    public let layers: [LayerCache]
    /// The number of positions the prefill wrote, the RoPE offset of the canvas.
    public let offset: Int
    /// The prompt tokens the prefill processed, which a read reports: the prompt's length, for an
    /// image prompt the expanded prompt with its image tokens, as upstream bills it.
    public let promptTokens: Int

    init(layers: [LayerCache], offset: Int, promptTokens: Int) {
        self.layers = layers
        self.offset = offset
        self.promptTokens = promptTokens
    }
}

extension DiffusionGemmaModel {
    /// Prefills a text prompt in one piece, as upstream's `MlxRuntime._prefill` calls
    /// `diffusion_prefill_cache` without `chunk_prefill`, and evaluates the caches.
    ///
    /// The prompt is never chunked: a chunked prefill is exact in real arithmetic, but in
    /// bfloat16 it moves read probabilities by up to 0.62 (spike #22), so reads take none.
    ///
    /// Image prompts take ``prefill(image:stages:)``, which scatters the vision tower's features
    /// into the embeddings and adds the bidirectional overlay to the masks. Hooks of mlx-vlm's
    /// encoder that later milestones add here, and that this path leaves out on purpose:
    /// - Chunked prefill (generation, milestone 5): `diffusion_prefill_cache` with
    ///   `prefill_step_size`, evaluating and clearing the cache between chunks, for prompts that
    ///   do not fit one pass. Not for reads, for the reason above.
    ///
    /// Generation appends each committed block with ``updateCache(_:tokens:)``, mlx-vlm's
    /// `diffusion_update_cache`, on a copy of these caches, so a prefill that reads share is never
    /// changed by a reply.
    ///
    /// - Parameter promptIDs: the prompt's token ids, `Engine.chat_prompt_ids` for a read.
    /// - Throws: ``ReadInputError/emptyPrompt`` or ``ReadInputError/promptTokenOutOfRange(index:id:vocabularySize:)``.
    public func prefill(promptIDs: [Int]) throws -> PromptCache {
        guard !promptIDs.isEmpty else { throw ReadInputError.emptyPrompt }
        let vocabularySize = configuration.vocabSize
        if let index = promptIDs.firstIndex(where: { $0 < 0 || $0 >= vocabularySize }) {
            throw ReadInputError.promptTokenOutOfRange(
                index: index, id: promptIDs[index], vocabularySize: vocabularySize)
        }
        let ids = MLXArray(promptIDs.map(Int32.init)).reshaped(1, promptIDs.count)
        let layers = prefill(ids)
        eval(layers.flatMap { [$0.keys, $0.values].compactMap { $0 } })
        return PromptCache(layers: layers, offset: promptIDs.count, promptTokens: promptIDs.count)
    }
}

extension DiffusionGemmaModel {
    /// mlx-vlm's `diffusion_update_cache`: the encoder over `tokens` after the cached prompt, in
    /// one piece, which appends their keys and values to a copy of `cache`; `cache` itself is not
    /// changed.
    ///
    /// language.py `EncoderModel.__call__` with a filled cache: image and video soft token ids
    /// replaced by `pad` before the embedding (`_embed_inputs`), RoPE from each layer's offset,
    /// and each layer's `make_mask` for `n` new positions: `.causal` on a full layer; on a sliding
    /// layer `.causal` while the last `window − 1` cached positions and the new ones fit the
    /// window, else the band `create_causal_mask(n, min(window − 1, offset), window)`. Generation
    /// calls it with the canvas each block committed, before the next block.
    ///
    /// - Parameters:
    ///   - cache: the prompt's prefill, or the cache an earlier update returned.
    ///   - tokens: the committed canvas.
    /// - Returns: a cache with `tokens` after the cached positions. Its ``PromptCache/promptTokens``
    ///   is `cache`'s: the committed tokens are generated, not prompt.
    /// - Throws: ``ReadInputError/emptyCanvas``,
    ///   ``ReadInputError/canvasTokenOutOfRange(index:id:vocabularySize:)`` or
    ///   ``ReadInputError/cacheLayerMismatch(cacheLayers:modelLayers:)``.
    public func updateCache(_ cache: PromptCache, tokens: [Int]) throws -> PromptCache {
        guard !tokens.isEmpty else { throw ReadInputError.emptyCanvas }
        let vocabularySize = configuration.vocabSize
        if let index = tokens.firstIndex(where: { $0 < 0 || $0 >= vocabularySize }) {
            throw ReadInputError.canvasTokenOutOfRange(
                index: index, id: tokens[index], vocabularySize: vocabularySize)
        }
        guard cache.layers.count == decoder.layers.count else {
            throw ReadInputError.cacheLayerMismatch(
                cacheLayers: cache.layers.count, modelLayers: decoder.layers.count)
        }
        let count = tokens.count
        let ids = MLXArray(tokens.map(Int32.init)).reshaped(1, count)
        var visionMask = ids .== Int32(imageTokenID)
        if let videoTokenID {
            visionMask = logicalOr(visionMask, ids .== Int32(videoTokenID))
        }
        let textIDs = which(visionMask, MLXArray(Int32(configuration.padTokenID)), ids)
        var h = decoder.embed(textIDs)
        let layers = cache.layers.map { $0.extended() }
        for (index, layer) in decoder.layers.enumerated() {
            let layerCache = layers[index]
            let mask = continuationMask(
                for: layer.layerType, length: count, cachedOffset: layerCache.offset)
            h = layer(
                h, mask: mask, cache: layerCache, decoder: false, offset: layerCache.offset,
                layerScalar: encoder.layerScalar(index))
        }
        eval(layers.flatMap { [$0.keys, $0.values].compactMap { $0 } })
        return PromptCache(
            layers: layers, offset: cache.offset + count, promptTokens: cache.promptTokens)
    }

    /// The encoder mask of `length` new positions after `cachedOffset` cached ones: cache.py
    /// `KVCache.make_mask` (`.causal`) on a full layer, `RotatingKVCache.make_mask` on a sliding
    /// one.
    func continuationMask(
        for layerType: DiffusionGemmaTextConfiguration.LayerType, length: Int, cachedOffset: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard length > 1 else { return .none }
        guard layerType == .slidingAttention else { return .causal }
        let window = configuration.slidingWindow
        let offset = min(window - 1, cachedOffset)
        guard offset + length > window else { return .causal }
        let rows = MLXArray(Int32(offset)..<Int32(offset + length))[0..., .newAxis]
        let columns = MLXArray(Int32(0)..<Int32(offset + length))[.newAxis, 0...]
        return .array(logicalAnd(rows .>= columns, rows .< columns + window))
    }
}

// MARK: Digests

/// The digest of one cached tensor, as Fixtures/oracle/reads.json records mlx-vlm's.
public struct TensorDigest: Equatable, Sendable, CustomStringConvertible {
    /// The shape.
    public var shape: [Int]
    /// The dtype's name as MLX Python prints it, such as `bfloat16`.
    public var dtype: String
    /// SHA-256 of the raw bytes in C order, as lowercase hex.
    public var sha256: String
    /// The sum of the values, accumulated in Double in C order.
    public var sum: Double
    /// The sum of the squared values, accumulated in Double in C order.
    public var sumOfSquares: Double

    /// The digest of `array`, which is evaluated.
    public init(_ array: MLXArray) {
        shape = array.shape
        dtype = Self.name(array.dtype)
        let bytes = array.asData(access: .copy).data
        sha256 = CryptoKit.SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        var sum = 0.0
        var squares = 0.0
        for value in array.asArray(Float.self) {
            let value = Double(value)
            sum += value
            squares += value * value
        }
        self.sum = sum
        sumOfSquares = squares
    }

    /// The dtype and shape, the first 16 hex digits of the SHA-256, the sum and the sum of squares.
    public var description: String {
        "\(dtype)\(shape) sha256 \(sha256.prefix(16)), sum \(sum), sum of squares \(sumOfSquares)"
    }

    static func name(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: return "bfloat16"
        case .float16: return "float16"
        case .float32: return "float32"
        default: return "\(dtype)"
        }
    }
}

extension LayerCache {
    /// The digests of the cached keys and values, nil before the prefill.
    public var digest: (keys: TensorDigest, values: TensorDigest)? {
        guard let keys, let values else { return nil }
        return (TensorDigest(keys), TensorDigest(values))
    }

    /// The digests of what a decoder pass reads from this cache: for a sliding layer the last
    /// `window − 1` positions (all of them when there are fewer), for a full layer every
    /// position. The oracle's `decoder_view` digests this for layer 0 with `window` 1,024.
    public func decoderViewDigest(slidingWindow window: Int) -> (
        keys: TensorDigest, values: TensorDigest
    )? {
        guard let keys, let values else { return nil }
        guard !isFullAttention else { return digest }
        let start = max(keys.dim(2) - max(window - 1, 0), 0)
        return (
            TensorDigest(keys[.ellipsis, start..., 0...]),
            TensorDigest(values[.ellipsis, start..., 0...])
        )
    }
}
