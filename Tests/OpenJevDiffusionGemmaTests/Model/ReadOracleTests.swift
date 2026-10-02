import Foundation
import MLX
import OpenJevCore
import OpenJevDiffusionGemma
import Testing

/// Fixtures/oracle/reads.json's prompts, with their cache digests, and its 27 reads.
private struct OracleReads: Decodable {
    struct Digest: Decodable {
        let shape: [Int]
        let dtype: String
        let sha256: String
        let sum: Double
        let sumOfSquares: Double
        enum CodingKeys: String, CodingKey {
            case shape, dtype, sha256, sum
            case sumOfSquares = "sum_of_squares"
        }
    }
    struct View: Decodable {
        let keys: Digest
        let values: Digest
    }
    struct LayerDigest: Decodable {
        let layer: Int
        let offset: Int
        let keys: Digest
        let values: Digest
        let decoderView: View?
        enum CodingKeys: String, CodingKey {
            case layer, offset, keys, values
            case decoderView = "decoder_view"
        }
    }
    struct Prompt: Decodable {
        let ids: [Int]
        let cache: [LayerDigest]
    }
    struct Slot: Decodable {
        let pos: Int
        let labelIDs: [Int]
        enum CodingKeys: String, CodingKey {
            case pos
            case labelIDs = "label_ids"
        }
    }
    struct Distribution: Decodable {
        let probs: [Double]
        let entropy: Double
    }
    struct Read: Decodable {
        let id: String
        let prompt: String
        let width: Int
        let canvas: [Int]
        let slots: [Slot]
        let steps: Int
        let promptTokens: Int
        let logprobs: [[[Double]]]
        let distributions: [Distribution]
        let written: [[Int]]
        enum CodingKeys: String, CodingKey {
            case id, prompt, width, canvas, slots, steps, logprobs, distributions, written
            case promptTokens = "prompt_tokens"
        }

        var requests: [SlotRequest] {
            slots.map { SlotRequest(position: $0.pos, labelIDs: $0.labelIDs) }
        }

        /// The recorded maps as `(token id, logprob)` pairs.
        var maps: [[(tokenID: Int, logprob: Double)]] {
            logprobs.map { slot in slot.map { (tokenID: Int($0[0]), logprob: $0[1]) } }
        }
    }
    let prompts: [String: Prompt]
    let reads: [Read]

    static func load() throws -> OracleReads {
        let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent("oracle/reads.json")
        return try JSONDecoder().decode(OracleReads.self, from: Data(contentsOf: url))
    }
}

/// True when two maps have the same ids and the same float32 logprobs, bit for bit.
private func identical(
    _ a: [(tokenID: Int, logprob: Double)], _ b: [(tokenID: Int, logprob: Double)]
) -> Bool {
    a.count == b.count
        && zip(a, b).allSatisfy {
            $0.tokenID == $1.tokenID && Float($0.logprob).bitPattern == Float($1.logprob).bitPattern
                && $0.logprob == $1.logprob
        }
}

/// Installs the oracle's RoPE table in the full-attention layers when the exact tier is on
/// (`OPENJEV_MLX_METALLIB` taken), runs `body`, and restores the model's own tables.
private func withTier<T>(
    _ model: DiffusionGemmaModel, _ body: (_ exact: Bool) throws -> T
) throws -> T {
    let exact = MetalLibrary.override != nil
    let fullLayers = model.decoder.layers.filter { $0.layerType == .fullAttention }
    var restore: [MLXArray] = []
    if exact {
        let bits = try ModelFixtures.oracle().rope.float32Bits
        let table = MLXArray(bits.map { Float(bitPattern: $0) })
        for layer in fullLayers {
            restore.append(try #require(layer.selfAttention.fullAttentionFrequencies))
            layer.selfAttention.fullAttentionFrequencies = table
        }
    }
    defer {
        for (layer, table) in zip(fullLayers, restore) {
            layer.selfAttention.fullAttentionFrequencies = table
        }
    }
    return try body(exact)
}

/// The D-014 aggregates of one comparison.
private struct Aggregates {
    var labelDifferences: [Double] = []
    var longLabelDifferences: [Double] = []
    var entropyDifferences: [Double] = []
    var longEntropyDifferences: [Double] = []
    var slots = 0
    var topAgree = 0
    var confidentSlots = 0
    var confidentAgree = 0

    static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
    }
}

extension MLXTests {
    @Suite(
        "DiffusionGemma reads against the oracle (opt-in)",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage))
    struct ReadOracleTests {
        /// D-014 on all 27 oracle reads. Each of the 12 prompts is prefilled once and its reads
        /// share the cache. Exact tier (`OPENJEV_MLX_METALLIB` set to the wheel's metallib, the
        /// oracle's RoPE table installed): every map, written list, prompt token count and cache
        /// digest identical. Native tier:
        /// D-014's six bounds on the label probabilities and entropies through
        /// `SlotDistribution.compute`; per-read maxima and the written argmaxes are printed. In both,
        /// how often the slot-only projection (D-015) matches the full one is printed.
        @Test("The 27 oracle reads meet D-014")
        func oracleReads() async throws {
            let live = try await LiveCheckpoint.shared()
            let model = live.loaded.model
            let oracle = try OracleReads.load()
            #expect(oracle.reads.count == 27)
            #expect(oracle.prompts.count == 12)

            try withTier(model) { exact in
                var order: [String] = []
                for read in oracle.reads where !order.contains(read.prompt) {
                    order.append(read.prompt)
                }
                var aggregates = Aggregates()
                // The same figures per steps value (#43): reads, label differences, top labels,
                // and the written argmaxes of the multi-step reads.
                var bySteps: [Int: (reads: Int, figures: Aggregates, written: Int)] = [:]
                var lines: [String] = []
                var moved: [String] = []
                var identicalReads = 0
                var slotOnlyIdentical = 0
                var slotOnlyMax = 0.0
                var writtenEqual = 0
                var writtenTotal = 0
                var digestsEqual = 0
                var digestsTotal = 0
                // Reported, not bounded: the native tier bounds only the reads.
                var digestMaxRelative = 0.0
                let clock = ContinuousClock()

                for key in order {
                    let prompt = try #require(oracle.prompts[key])
                    let started = clock.now
                    let cache = try model.prefill(promptIDs: prompt.ids)
                    let prefillTime = clock.now - started
                    let long = prompt.ids.count > 1024

                    for want in prompt.cache {
                        let layer = cache.layers[want.layer]
                        let got = try #require(layer.digest)
                        #expect(layer.offset == want.offset)
                        #expect(got.keys.shape == want.keys.shape)
                        #expect(got.keys.dtype == want.keys.dtype)
                        var pairs = [(got.keys, want.keys), (got.values, want.values)]
                        if let view = want.decoderView {
                            let gotView = try #require(
                                layer.decoderViewDigest(
                                    slidingWindow: model.configuration.slidingWindow))
                            #expect(gotView.keys.shape == view.keys.shape)
                            pairs += [(gotView.keys, view.keys), (gotView.values, view.values)]
                        }
                        for (index, (mine, theirs)) in pairs.enumerated() {
                            digestsTotal += 1
                            let equal = mine.sha256 == theirs.sha256
                            digestsEqual += equal ? 1 : 0
                            let name = ["keys", "values", "view keys", "view values"][index]
                            let relative =
                                abs(mine.sumOfSquares - theirs.sumOfSquares) / theirs.sumOfSquares
                            digestMaxRelative = max(digestMaxRelative, relative)
                            if exact {
                                #expect(equal, "\(key) layer \(want.layer) \(name)")
                            }
                        }
                    }

                    for read in oracle.reads where read.prompt == key {
                        let started = clock.now
                        let output = try model.read(
                            canvas: read.canvas, slots: read.requests, cache: cache,
                            steps: read.steps, topK: 20, projection: .full)
                        let readTime = clock.now - started
                        let slotOnly = try model.read(
                            canvas: read.canvas, slots: read.requests, cache: cache,
                            steps: read.steps, topK: 20, projection: .slotsOnly)

                        #expect(output.promptTokens == read.promptTokens, "\(read.id)")
                        #expect(output.slots.count == read.slots.count, "\(read.id)")
                        bySteps[read.steps, default: (0, Aggregates(), 0)].reads += 1
                        if read.steps > 1 {
                            writtenTotal += 1
                            writtenEqual += output.written == read.written ? 1 : 0
                            bySteps[read.steps]?.written += output.written == read.written ? 1 : 0
                        }
                        let recorded = read.maps
                        let same = zip(output.slots, recorded).allSatisfy { identical($0, $1) }
                        identicalReads += same ? 1 : 0
                        for (full, slots) in zip(output.slots, slotOnly.slots) {
                            slotOnlyIdentical += identical(full, slots) ? 1 : 0
                            let lookup = Dictionary(
                                uniqueKeysWithValues: slots.map { ($0.tokenID, $0.logprob) })
                            for entry in full {
                                if let other = lookup[entry.tokenID] {
                                    slotOnlyMax = max(slotOnlyMax, abs(other - entry.logprob))
                                }
                            }
                        }
                        if exact {
                            #expect(same, "\(read.id): the maps differ from the oracle's")
                            #expect(output.written == read.written, "\(read.id)")
                        }

                        var readMaxDifference = 0.0
                        var readMaxEntropy = 0.0
                        var compared: [ReadDivergence.Slot] = []
                        for (index, slot) in read.slots.enumerated() {
                            let want = SlotDistribution.compute(
                                top: recorded[index], labelIDs: slot.labelIDs)
                            let got = SlotDistribution.compute(
                                top: output.slots[index], labelIDs: slot.labelIDs)
                            // The core's slot_distribution reproduces the recorded one.
                            let stored = read.distributions[index]
                            #expect(
                                zip(want.probabilities, stored.probs).allSatisfy {
                                    abs($0 - $1) <= 1e-12
                                }, "\(read.id) slot \(index)")
                            #expect(abs(want.entropy - stored.entropy) <= 1e-12)

                            let differences = zip(got.probabilities, want.probabilities).map {
                                abs($0 - $1)
                            }
                            let entropy = abs(got.entropy - want.entropy)
                            aggregates.labelDifferences += differences
                            aggregates.entropyDifferences.append(entropy)
                            bySteps[read.steps]?.figures.labelDifferences += differences
                            if long {
                                aggregates.longLabelDifferences += differences
                                aggregates.longEntropyDifferences.append(entropy)
                            }
                            readMaxDifference = max(readMaxDifference, differences.max() ?? 0)
                            readMaxEntropy = max(readMaxEntropy, entropy)
                            compared.append(
                                .init(
                                    labelIDs: slot.labelIDs, ours: got.probabilities,
                                    oracle: want.probabilities))

                            let agree =
                                ReadDivergence.firstLargest(got.probabilities)
                                == ReadDivergence.firstLargest(want.probabilities)
                            aggregates.slots += 1
                            aggregates.topAgree += agree ? 1 : 0
                            bySteps[read.steps]?.figures.slots += 1
                            bySteps[read.steps]?.figures.topAgree += agree ? 1 : 0
                            let ranked = want.probabilities.sorted(by: >)
                            let margin = ranked[0] - (ranked.count > 1 ? ranked[1] : 0)
                            if margin >= 0.5 {
                                aggregates.confidentSlots += 1
                                aggregates.confidentAgree += agree ? 1 : 0
                            }
                        }
                        if let report = ReadDivergence.report(
                            id: read.id, prompt: read.prompt, width: read.width,
                            steps: read.steps, slots: compared)
                        {
                            moved.append(report)
                        }
                        let writtenNote =
                            read.steps > 1
                            ? ", written \(output.written == read.written ? "equal" : "differs") \(output.written)"
                            : ""
                        lines.append(
                            String(
                                format:
                                    "%@ (%d tokens, width %d, steps %d): %@, max |dp| %.4f, max |dH| %.4f, prefill %@, read %@%@",
                                read.id, read.promptTokens, read.width, read.steps,
                                same ? "identical" : "differs", readMaxDifference,
                                readMaxEntropy, "\(prefillTime)", "\(readTime)", writtenNote))
                    }
                }

                let meanP = Aggregates.mean(aggregates.labelDifferences)
                let longMeanP = Aggregates.mean(aggregates.longLabelDifferences)
                let meanH = Aggregates.mean(aggregates.entropyDifferences)
                let longMeanH = Aggregates.mean(aggregates.longEntropyDifferences)
                let topShare = Double(aggregates.topAgree) / Double(aggregates.slots)
                let confidentShare =
                    Double(aggregates.confidentAgree) / Double(aggregates.confidentSlots)
                let slotCount = aggregates.slots
                let stepLines = bySteps.keys.sorted().map { steps in
                    let (reads, figures, written) = bySteps[steps] ?? (0, Aggregates(), 0)
                    let writtenNote =
                        steps > 1 ? ", written argmaxes equal on \(written)/\(reads) reads" : ""
                    return String(
                        format:
                            "steps %d: %d reads, mean |dp| %.4f over %d labels, top label %d/%d%@",
                        steps, reads, Aggregates.mean(figures.labelDifferences),
                        figures.labelDifferences.count, figures.topAgree, figures.slots,
                        writtenNote)
                }
                let report =
                    """
                    \(exact ? "exact tier (wheel metallib, oracle RoPE)" : "native kernels"): \
                    \(identicalReads)/27 reads bit-identical, \(digestsEqual)/\(digestsTotal) cache digests equal \
                    (largest relative sum-of-squares difference \(digestMaxRelative)), \
                    written argmaxes equal on \(writtenEqual)/\(writtenTotal) multi-step reads, \
                    slot-only projection identical on \(slotOnlyIdentical)/\(slotCount) slots (max |d lp| \(slotOnlyMax))
                    labels \(aggregates.labelDifferences.count), long-prompt labels \(aggregates.longLabelDifferences.count), \
                    slots \(slotCount), long-prompt slots \(aggregates.longEntropyDifferences.count)
                    mean |dp| all labels \(String(format: "%.4f", meanP)) (bound 0.02)
                    mean |dp| long prompts \(String(format: "%.4f", longMeanP)) (bound 0.01)
                    mean |dH| all slots \(String(format: "%.4f", meanH)) (bound 0.2)
                    mean |dH| long prompts \(String(format: "%.4f", longMeanH)) (bound 0.2)
                    top label \(aggregates.topAgree)/\(slotCount) \(String(format: "%.1f%%", topShare * 100)) (bound 90%)
                    top label, oracle margin >= 0.5: \(aggregates.confidentAgree)/\(aggregates.confidentSlots) \
                    \(String(format: "%.1f%%", confidentShare * 100)) (bound 97%)
                    """
                    + "\n" + stepLines.joined(separator: "\n")
                    + "\n" + lines.joined(separator: "\n")
                    + "\nreads with a slot whose top label moved or whose max |dp| exceeds "
                    + "\(ReadDivergence.threshold): \(moved.count)\n"
                    + moved.joined(separator: "\n")
                print(report)
                SpikeReport.record("read-parity", report)

                #expect(aggregates.labelDifferences.count == 1_763)
                #expect(aggregates.longLabelDifferences.isEmpty == false)
                #expect(slotCount == 156)
                #expect(aggregates.longEntropyDifferences.count == 50)
                // The slot-only projection is measured, not required: on 2026-10-01 it was
                // identical on 29 of 156 slots in the exact tier and 40 natively, so D-015's
                // condition fails and reads project every row.
                if exact {
                    #expect(identicalReads == 27)
                    #expect(digestsEqual == digestsTotal)
                }
                #expect(meanP <= 0.02)
                #expect(longMeanP <= 0.01)
                #expect(meanH <= 0.2)
                #expect(longMeanH <= 0.2)
                #expect(topShare >= 0.90)
                #expect(confidentShare >= 0.97)
            }
        }

        /// The cache a read reuses gives the same bits as a fresh prefill of the same prompt, in
        /// one process with the same kernels, on a two-step read so self-conditioning runs too.
        @Test("A cached and a cold prefill give identical logprobs")
        func cachedAndCold() async throws {
            let live = try await LiveCheckpoint.shared()
            let model = live.loaded.model
            let oracle = try OracleReads.load()
            let read = try #require(
                oracle.reads.first { $0.prompt == "quickstart/g0" && $0.steps == 2 })
            let ids = try #require(oracle.prompts[read.prompt]).ids
            try withTier(model) { _ in
                let cache = try model.prefill(promptIDs: ids)
                let first = try model.read(
                    canvas: read.canvas, slots: read.requests, cache: cache, steps: read.steps)
                let cached = try model.read(
                    canvas: read.canvas, slots: read.requests, cache: cache, steps: read.steps)
                let cold = try model.read(
                    canvas: read.canvas, slots: read.requests,
                    cache: try model.prefill(promptIDs: ids), steps: read.steps)
                #expect(first.written == cached.written)
                #expect(first.written == cold.written)
                for index in read.slots.indices {
                    #expect(identical(first.slots[index], cached.slots[index]), "slot \(index)")
                    #expect(identical(first.slots[index], cold.slots[index]), "slot \(index)")
                }
            }
        }
    }
}
