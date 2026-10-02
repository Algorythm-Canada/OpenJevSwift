// A read (issues #26 and #28): upstream OpenJev's MlxRuntime.read (razorback16/openjev at dcd2094,
// openjev/mlx_backend.py lines 175 to 208, Apache-2.0, see THIRD_PARTY.md) on the decoder pass,
// ported from the spike #22 transliteration's read loop.

import MLX
import OpenJevCore

/// One slot of a read: where it is in the canvas and the token ids of its labels.
public struct SlotRequest: Equatable, Sendable {
    /// The slot's position in the canvas.
    public var position: Int
    /// The label token ids, in option order.
    public var labelIDs: [Int]

    /// Creates a slot.
    public init(position: Int, labelIDs: [Int]) {
        self.position = position
        self.labelIDs = labelIDs
    }
}

/// What a read returns: upstream's `(logprobs, prompt tokens)` and the argmaxes it wrote.
public struct ReadOutput: Sendable {
    /// Per slot, in slot order, `(token id, logprob)` for the top-k tokens and every label,
    /// sorted by token id: what `MlxRuntime.read` returns as a dict. Each logprob is the float32
    /// log-softmax value widened to Double, as Python's `tolist()` gives it.
    public var slots: [[(tokenID: Int, logprob: Double)]]
    /// For each step but the last, the argmaxes written into the slot positions, in slot order.
    /// Empty for a single-step read.
    public var written: [[Int]]
    /// The prompt tokens the prefill processed. Steps do not change it.
    public var promptTokens: Int

    /// Creates a read's output.
    public init(slots: [[(tokenID: Int, logprob: Double)]], written: [[Int]], promptTokens: Int) {
        self.slots = slots
        self.written = written
        self.promptTokens = promptTokens
    }

    /// The engine's ``/OpenJevCore/ReadResult``: each slot's map through `slot_distribution`, as
    /// upstream's `one_read` does with what `MlxRuntime.read` returns.
    ///
    /// - Parameter slots: the read's slots, for their label ids.
    public func readResult(for slots: [SlotRequest]) -> ReadResult {
        ReadResult(tops: self.slots, labelIDs: slots.map(\.labelIDs), promptTokens: promptTokens)
    }
}

extension DiffusionGemmaModel {
    /// Which rows the last step projects through the tied head. Every step before the last
    /// needs the full logits for self-conditioning.
    public enum SlotProjection: Sendable {
        /// Every canvas row, as upstream and mlx-vlm do. Reads use it.
        case full
        /// Only the slot rows (D-015). Not bit-identical to ``full``: the smaller quantized
        /// matmul rounds differently, and in D-014's exact tier only 29 of the oracle's 156 slots
        /// matched (40 under native kernels, logprobs up to 0.16 apart). Kept for measurement.
        case slotsOnly
    }

    /// `MlxRuntime.read` over a prefilled prompt.
    ///
    /// The masks are made once. Each step runs the decoder pass; after every step but the last,
    /// the argmax of each slot row is written into that slot's position only (the template is
    /// never overwritten, upstream's `diffusion_pinned`), the step's full logits become the next
    /// step's self-conditioning, and the canvas and logits are evaluated. The last step's slot
    /// rows give the maps (``slotLogprobs(row:labelIDs:topK:)``).
    ///
    /// - Parameters:
    ///   - canvas: the canvas token ids.
    ///   - slots: the slots, each a canvas position and its labels.
    ///   - cache: ``prefill(promptIDs:)``'s result for the prompt.
    ///   - steps: the denoising passes, at least 1.
    ///   - topK: the top tokens kept per slot beside the labels, upstream's `TOPK`.
    ///   - projection: the last step's projection, ``SlotProjection/full`` unless measuring.
    /// - Throws: ``ReadInputError`` for an input upstream would not send.
    public func read(
        canvas: [Int], slots: [SlotRequest], cache: PromptCache, steps: Int, topK: Int = 20,
        projection: SlotProjection = .full
    ) throws -> ReadOutput {
        try validate(canvas: canvas, slots: slots, cache: cache, steps: steps, topK: topK)
        let ids = MLXArray(canvas.map(Int32.init)).reshaped(1, canvas.count)
        let masks = decoderMasks(canvasLength: canvas.count, cache: cache)
        let slotPositions = slots.map(\.position)
        let positions = MLXArray(slotPositions.map(Int32.init))
        var conditioning: MLXArray?
        var written: [[Int]] = []
        for _ in 1..<steps {
            let logits = decoderLogits(
                canvas: ids, cache: cache, conditioning: conditioning, masks: masks)
            ids[0, positions] = argMax(logits[0, positions], axis: -1).asType(ids.dtype)
            // diffusion_self_conditioning with a quantized embedding: the logits themselves.
            conditioning = logits
            eval(ids, logits)
            written.append(ids[0, positions].asArray(Int32.self).map(Int.init))
        }
        let maps: [[(tokenID: Int, logprob: Double)]]
        switch projection {
        case .full:
            let logits = decoderLogits(
                canvas: ids, cache: cache, conditioning: conditioning, masks: masks)
            maps = slots.map {
                Self.slotLogprobs(row: logits[0, $0.position], labelIDs: $0.labelIDs, topK: topK)
            }
        case .slotsOnly:
            let rows = decoderSlotLogits(
                canvas: ids, cache: cache, conditioning: conditioning, masks: masks,
                positions: slotPositions)
            maps = slots.indices.map {
                Self.slotLogprobs(row: rows[0, $0], labelIDs: slots[$0].labelIDs, topK: topK)
            }
        }
        return ReadOutput(slots: maps, written: written, promptTokens: cache.promptTokens)
    }

    /// One slot's map, mlx_backend.py lines 203 to 207: the row in float32, `lp = row −
    /// logsumexp(row)`, the ids of `argpartition(−lp, topK)[:topK]` united with the labels and
    /// sorted, and each kept id's `lp` widened to Double.
    ///
    /// - Parameters:
    ///   - row: one slot's logits, `[vocab]`.
    ///   - labelIDs: the slot's labels.
    ///   - topK: the top tokens to keep, below the vocabulary size.
    public static func slotLogprobs(
        row: MLXArray, labelIDs: [Int], topK: Int
    ) -> [(tokenID: Int, logprob: Double)] {
        let row = row.asType(.float32)
        let logprobs = row - logSumExp(row)
        let top = argPartition(-logprobs, kth: topK)[0..<topK].asType(.int32).asArray(Int32.self)
        let keep = Set(top.map(Int.init)).union(labelIDs).sorted()
        let values = logprobs[MLXArray(keep.map(Int32.init))].asArray(Float.self)
        return zip(keep, values).map { (tokenID: $0, logprob: Double($1)) }
    }

    private func validate(
        canvas: [Int], slots: [SlotRequest], cache: PromptCache, steps: Int, topK: Int
    ) throws {
        let vocabularySize = configuration.vocabSize
        guard cache.layers.count == decoder.layers.count else {
            throw ReadInputError.cacheLayerMismatch(
                cacheLayers: cache.layers.count, modelLayers: decoder.layers.count)
        }
        guard !canvas.isEmpty else { throw ReadInputError.emptyCanvas }
        if let index = canvas.firstIndex(where: { $0 < 0 || $0 >= vocabularySize }) {
            throw ReadInputError.canvasTokenOutOfRange(
                index: index, id: canvas[index], vocabularySize: vocabularySize)
        }
        guard !slots.isEmpty else { throw ReadInputError.noSlots }
        for (index, slot) in slots.enumerated() {
            guard canvas.indices.contains(slot.position) else {
                throw ReadInputError.slotOutOfRange(
                    slot: index, position: slot.position, canvasLength: canvas.count)
            }
            guard !slot.labelIDs.isEmpty else { throw ReadInputError.noLabels(slot: index) }
            if let id = slot.labelIDs.first(where: { $0 < 0 || $0 >= vocabularySize }) {
                throw ReadInputError.labelOutOfRange(
                    slot: index, id: id, vocabularySize: vocabularySize)
            }
        }
        guard steps >= 1 else { throw ReadInputError.stepsOutOfRange(steps) }
        guard topK >= 1 && topK < vocabularySize else {
            throw ReadInputError.topKOutOfRange(topK: topK, vocabularySize: vocabularySize)
        }
    }
}
