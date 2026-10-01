import Foundation
import MLX
import MLXNN
import OpenJevDiffusionGemma
import Testing

/// The tiny tree with float32 weights drawn from `seed`, quantized as the tiny checkpoint when
/// `quantized` (an 8-bit embedding, so self-conditioning takes the quantized path).
private func tinyModel(seed: UInt64 = 25, quantized: Bool = false) throws -> DiffusionGemmaModel {
    MetalLibrary.configure()
    MLXRandom.seed(seed)
    let model = DiffusionGemmaModel(try ModelFixtures.tinyText())
    if quantized {
        model.quantize(try ModelFixtures.tinyPerLayerQuantization())
    }
    eval(model)
    return model
}

/// `count` token ids of the tiny vocabulary, from a fixed formula.
private func tinyIDs(_ count: Int, salt: Int = 0) -> [Int] {
    (0..<count).map { ($0 * 37 + salt * 11 + 5) % 128 }
}

/// The boolean values of an array.
private func bools(_ array: MLXArray) -> [Bool] {
    array.asType(.int32).asArray(Int32.self).map { $0 != 0 }
}

/// True when two maps have the same ids and bit-identical logprobs.
private func identical(
    _ a: [(tokenID: Int, logprob: Double)], _ b: [(tokenID: Int, logprob: Double)]
) -> Bool {
    a.count == b.count
        && zip(a, b).allSatisfy {
            $0.tokenID == $1.tokenID && $0.logprob.bitPattern == $1.logprob.bitPattern
        }
}

extension MLXTests {
    @Suite("DiffusionGemma prefill and read pass on a tiny configuration")
    struct ReadPassTests {
        @Test("prefill(promptIDs:) fills one cache per layer at the prompt's offset")
        func prefillCache() throws {
            let model = try tinyModel()
            let cache = try model.prefill(promptIDs: tinyIDs(12))
            #expect(cache.layers.count == 2)
            #expect(cache.offset == 12)
            #expect(cache.promptTokens == 12)
            #expect(cache.layers.map(\.offset) == [12, 12])
            #expect(cache.layers[0].keys?.shape == [1, 1, 12, 16])
            #expect(cache.layers[1].values?.shape == [1, 1, 12, 32])

            let digest = try #require(cache.layers[0].digest)
            #expect(digest.keys.shape == [1, 1, 12, 16])
            #expect(digest.keys.dtype == "float32")
            #expect(digest.keys.sha256.count == 64)
            let keys = try #require(cache.layers[0].keys).asArray(Float.self).map(Double.init)
            #expect(abs(digest.keys.sum - keys.reduce(0, +)) <= 1e-9)
            // Window 8: the decoder view of the sliding layer is its last 7 positions.
            let view = try #require(cache.layers[0].decoderViewDigest(slidingWindow: 8))
            #expect(view.keys.shape == [1, 1, 7, 16])
            #expect(view.keys.sha256 != digest.keys.sha256)
            let fullView = try #require(cache.layers[1].decoderViewDigest(slidingWindow: 8))
            #expect(fullView.keys == (try #require(cache.layers[1].digest)).keys)
            // The same prompt digests the same.
            let again = try model.prefill(promptIDs: tinyIDs(12))
            #expect(again.layers[0].digest?.keys == digest.keys)
        }

        @Test("prefill(promptIDs:) refuses an empty prompt and ids outside the vocabulary")
        func prefillRefuses() throws {
            let model = try tinyModel()
            #expect(throws: ReadInputError.emptyPrompt) { try model.prefill(promptIDs: []) }
            #expect(
                throws: ReadInputError.promptTokenOutOfRange(
                    index: 1, id: 128, vocabularySize: 128)
            ) { try model.prefill(promptIDs: [3, 128]) }
            #expect(
                throws: ReadInputError.promptTokenOutOfRange(index: 0, id: -1, vocabularySize: 128)
            ) { try model.prefill(promptIDs: [-1]) }
            #expect(
                ReadInputError.promptTokenOutOfRange(index: 1, id: 128, vocabularySize: 128)
                    .description == "prompt token 1 is 128, outside the vocabulary of 128")
        }

        @Test("read refuses inputs upstream would not send")
        func readRefuses() throws {
            let model = try tinyModel()
            let cache = try model.prefill(promptIDs: tinyIDs(6))
            let canvas = tinyIDs(4, salt: 1)
            let slot = SlotRequest(position: 1, labelIDs: [3, 4])
            #expect(throws: ReadInputError.emptyCanvas) {
                try model.read(canvas: [], slots: [slot], cache: cache, steps: 1)
            }
            #expect(
                throws: ReadInputError.canvasTokenOutOfRange(index: 0, id: 200, vocabularySize: 128)
            ) { try model.read(canvas: [200], slots: [slot], cache: cache, steps: 1) }
            #expect(throws: ReadInputError.noSlots) {
                try model.read(canvas: canvas, slots: [], cache: cache, steps: 1)
            }
            #expect(throws: ReadInputError.slotOutOfRange(slot: 0, position: 4, canvasLength: 4)) {
                try model.read(
                    canvas: canvas, slots: [SlotRequest(position: 4, labelIDs: [3])],
                    cache: cache, steps: 1)
            }
            #expect(throws: ReadInputError.noLabels(slot: 0)) {
                try model.read(
                    canvas: canvas, slots: [SlotRequest(position: 0, labelIDs: [])],
                    cache: cache, steps: 1)
            }
            let outsideLabel = ReadInputError.labelOutOfRange(
                slot: 0, id: 128, vocabularySize: 128)
            #expect(throws: outsideLabel) {
                try model.read(
                    canvas: canvas, slots: [SlotRequest(position: 0, labelIDs: [128])],
                    cache: cache, steps: 1)
            }
            #expect(throws: ReadInputError.stepsOutOfRange(0)) {
                try model.read(canvas: canvas, slots: [slot], cache: cache, steps: 0)
            }
            #expect(throws: ReadInputError.topKOutOfRange(topK: 128, vocabularySize: 128)) {
                try model.read(canvas: canvas, slots: [slot], cache: cache, steps: 1, topK: 128)
            }
        }

        @Test("Decoder masks for a prompt within the window: none for both layer types")
        func masksWithinWindow() throws {
            let model = try tinyModel()
            let cache = try model.prefill(promptIDs: tinyIDs(6))
            let masks = model.decoderMasks(canvasLength: 4, cache: cache)
            #expect(masks.count == 2)
            guard case .some(.none) = masks[.slidingAttention],
                case .some(.none) = masks[.fullAttention]
            else {
                Issue.record("a mask within the window is not .none: \(masks)")
                return
            }
        }

        @Test("Decoder masks past the window: the sliding row allows [valid − 7, valid) and canvas")
        func masksPastWindow() throws {
            let model = try tinyModel()
            let cache = try model.prefill(promptIDs: tinyIDs(12))
            let masks = model.decoderMasks(canvasLength: 4, cache: cache)
            guard case .some(.none) = masks[.fullAttention] else {
                Issue.record("the full layer's mask is not .none")
                return
            }
            guard case .some(.array(let mask)) = masks[.slidingAttention] else {
                Issue.record("the sliding layer's mask is not an array")
                return
            }
            #expect(mask.shape == [1, 1, 4, 16])
            #expect(mask.dtype == .bool)
            let values = bools(mask)
            for row in 0..<4 {
                for column in 0..<16 {
                    let expected = column >= 12 || (column >= 5 && column < 12)
                    #expect(values[row * 16 + column] == expected, "row \(row), column \(column)")
                }
            }
        }

        @Test("The decoder pass returns [1, canvas, vocab] float32 logits within the softcap")
        func decoderPassShape() throws {
            for quantized in [false, true] {
                let model = try tinyModel(quantized: quantized)
                let cache = try model.prefill(promptIDs: tinyIDs(12))
                let masks = model.decoderMasks(canvasLength: 8, cache: cache)
                let canvas = MLXArray(tinyIDs(8, salt: 2).map(Int32.init)).reshaped(1, 8)
                let logits = model.decoderLogits(
                    canvas: canvas, cache: cache, conditioning: nil, masks: masks)
                #expect(logits.shape == [1, 8, 128])
                #expect(logits.dtype == .float32)
                let largest = abs(logits).max().item(Float.self)
                #expect(largest.isFinite && largest <= 30)
                #expect(cache.layers.map(\.offset) == [12, 12], "a read leaves the cache as it is")
            }
        }

        @Test("Slot extraction: the sorted union of the top 20 and labels, lp = row − logsumexp")
        func slotExtraction() throws {
            MetalLibrary.configure()
            // 128 distinct values in a scrambled order: value(i) = ((i * 53) % 128) / 16 - 3.
            let values = (0..<128).map { Float(($0 * 53) % 128) / 16 - 3 }
            let labels = [0, 7, 127, 64]
            let map = DiffusionGemmaModel.slotLogprobs(
                row: MLXArray(values), labelIDs: labels, topK: 20)

            let top = values.indices.sorted { values[$0] > values[$1] }.prefix(20)
            let expectedIDs = Set(top).union(labels).sorted()
            #expect(map.map(\.tokenID) == expectedIDs)
            let largest = Double(values.max() ?? 0)
            let logSumExp =
                largest + log(values.map { exp(Double($0) - largest) }.reduce(0, +))
            for entry in map {
                let expected = Double(values[entry.tokenID]) - logSumExp
                #expect(abs(entry.logprob - expected) <= 1e-5, "token \(entry.tokenID)")
                // A float32 value widened to Double.
                #expect(Double(Float(entry.logprob)) == entry.logprob)
            }
            // Labels already in the top 20 are not repeated.
            let inTop = DiffusionGemmaModel.slotLogprobs(
                row: MLXArray(values), labelIDs: [top[0], top[1]], topK: 20)
            #expect(inTop.count == 20)
        }

        @Test("Self-conditioning runs on a zero signal and on a logits signal")
        func selfConditioning() throws {
            for quantized in [false, true] {
                let model = try tinyModel(seed: 28, quantized: quantized)
                let module = model.decoder.selfConditioning
                let embeddings = MLXRandom.normal([1, 4, 64])

                // The zero signal still goes through the module and its norms.
                let zero = module(embeddings, signal: zeros(like: embeddings))
                #expect(zero.shape == [1, 4, 64])
                let normed = module.preNorm(zeros(like: embeddings))
                let conditioning = module.downProj(
                    geluApproximate(module.gateProj(normed)) * module.upProj(normed))
                let expected = MLXFast.rmsNorm(
                    embeddings + conditioning, weight: MLXArray.mlxNone, eps: 1e-6)
                #expect(abs(zero - expected).max().item(Float.self) <= 1e-5)
                #expect(abs(zero - embeddings).max().item(Float.self) > 1e-3)

                // A logits signal changes the pass: the conditioning is the previous logits.
                let cache = try model.prefill(promptIDs: tinyIDs(6))
                let masks = model.decoderMasks(canvasLength: 4, cache: cache)
                let canvas = MLXArray(tinyIDs(4, salt: 3).map(Int32.init)).reshaped(1, 4)
                let first = model.decoderLogits(
                    canvas: canvas, cache: cache, conditioning: nil, masks: masks)
                let second = model.decoderLogits(
                    canvas: canvas, cache: cache, conditioning: first, masks: masks)
                #expect(second.shape == [1, 4, 128])
                #expect(second.dtype == .float32)
                #expect(abs(second - first).max().item(Float.self) > 1e-4, "quantized \(quantized)")
            }
        }

        @Test("A three-step read writes argmaxes into the slot positions only")
        func multiStepRead() throws {
            for quantized in [false, true] {
                let model = try tinyModel(seed: 26, quantized: quantized)
                let cache = try model.prefill(promptIDs: tinyIDs(12))
                let canvas = tinyIDs(10, salt: 4)
                let slots = [
                    SlotRequest(position: 2, labelIDs: [5, 6]),
                    SlotRequest(position: 7, labelIDs: [9, 10, 11]),
                ]
                let output = try model.read(canvas: canvas, slots: slots, cache: cache, steps: 3)
                #expect(output.written.count == 2)
                #expect(output.promptTokens == 12)
                #expect(output.slots.count == 2)

                // Replay: each step's canvas is the template with only the slot positions
                // replaced by the previous step's argmaxes, conditioned on its full logits.
                let masks = model.decoderMasks(canvasLength: 10, cache: cache)
                var current = canvas
                var conditioning: MLXArray?
                for step in 0..<3 {
                    let ids = MLXArray(current.map(Int32.init)).reshaped(1, 10)
                    let logits = model.decoderLogits(
                        canvas: ids, cache: cache, conditioning: conditioning, masks: masks)
                    if step == 2 {
                        for (index, slot) in slots.enumerated() {
                            let map = DiffusionGemmaModel.slotLogprobs(
                                row: logits[0, slot.position], labelIDs: slot.labelIDs,
                                topK: 20)
                            #expect(identical(map, output.slots[index]), "slot \(index)")
                        }
                        break
                    }
                    let argmaxes = slots.map {
                        argMax(logits[0, $0.position], axis: -1).item(Int.self)
                    }
                    #expect(output.written[step] == argmaxes, "step \(step)")
                    var next = canvas
                    for (slot, id) in zip(slots, argmaxes) {
                        next[slot.position] = id
                    }
                    current = next
                    conditioning = logits
                }

                // One step writes nothing and reports the same prompt tokens.
                let single = try model.read(canvas: canvas, slots: slots, cache: cache, steps: 1)
                #expect(single.written.isEmpty)
                #expect(single.promptTokens == output.promptTokens)
            }
        }

        @Test("Slot-only and full projection agree within 1e-3 on the random tiny model")
        func slotOnlyProjection() throws {
            for quantized in [false, true] {
                let model = try tinyModel(seed: 15, quantized: quantized)
                let cache = try model.prefill(promptIDs: tinyIDs(12))
                let masks = model.decoderMasks(canvasLength: 16, cache: cache)
                let canvas = MLXArray(tinyIDs(16, salt: 5).map(Int32.init)).reshaped(1, 16)
                let positions = [1, 6, 15]
                let full = model.decoderLogits(
                    canvas: canvas, cache: cache, conditioning: nil, masks: masks)
                let rows = model.decoderSlotLogits(
                    canvas: canvas, cache: cache, conditioning: nil, masks: masks,
                    positions: positions)
                #expect(rows.shape == [1, 3, 128])
                let fullRows = take(full, MLXArray(positions.map(Int32.init)), axis: 1)
                #expect(abs(rows - fullRows).max().item(Float.self) <= 1e-3)

                let slots = positions.map { SlotRequest(position: $0, labelIDs: [1, 2]) }
                let viaFull = try model.read(
                    canvas: tinyIDs(16, salt: 5), slots: slots, cache: cache, steps: 2,
                    projection: .full)
                let viaSlots = try model.read(
                    canvas: tinyIDs(16, salt: 5), slots: slots, cache: cache, steps: 2,
                    projection: .slotsOnly)
                #expect(viaFull.written == viaSlots.written)
                for (a, b) in zip(viaFull.slots, viaSlots.slots) {
                    for (x, y) in zip(a, b) where x.tokenID == y.tokenID {
                        #expect(abs(x.logprob - y.logprob) <= 1e-3)
                    }
                }
            }
        }

        @Test("ReadOutput maps onto the engine's ReadResult through slot_distribution")
        func readResult() throws {
            let model = try tinyModel()
            let cache = try model.prefill(promptIDs: tinyIDs(6))
            let slots = [SlotRequest(position: 0, labelIDs: [3, 4, 5])]
            let output = try model.read(
                canvas: tinyIDs(4, salt: 6), slots: slots, cache: cache, steps: 1)
            let result = output.readResult(for: slots)
            #expect(result.promptTokens == 6)
            #expect(result.slots.count == 1)
            #expect(result.slots[0].probabilities.count == 3)
            #expect(abs(result.slots[0].probabilities.reduce(0, +) - 1) <= 1e-12)
        }
    }
}
