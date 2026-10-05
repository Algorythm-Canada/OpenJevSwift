import Foundation
import MLX
import OpenJevCore
import OpenJevDiffusionGemma
import Testing

/// Fixtures/vision/reads.json: upstream's `MlxRuntime.read` of the hot dog photo for two requests,
/// each at canvas index 0 and 1 (#46's oracle for #47).
private struct ImageOracleReads: Decodable {
    struct Prompt: Decodable {
        let system: String
        let state: String
        let images: [String]
        let ids: [Int]
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
        enum CodingKeys: String, CodingKey {
            case id, prompt, width, canvas, slots, steps, logprobs, distributions
            case promptTokens = "prompt_tokens"
        }

        var requests: [SlotRequest] {
            slots.map { SlotRequest(position: $0.pos, labelIDs: $0.labelIDs) }
        }

        var maps: [[(tokenID: Int, logprob: Double)]] {
            logprobs.map { slot in slot.map { (tokenID: Int($0[0]), logprob: $0[1]) } }
        }
    }
    let prompts: [String: Prompt]
    let reads: [Read]

    static func load() throws -> ImageOracleReads {
        let url = VisionFixtures.directory.appendingPathComponent("reads.json")
        return try JSONDecoder().decode(ImageOracleReads.self, from: Data(contentsOf: url))
    }
}

/// True when two maps have the same ids and the same float32 logprobs, bit for bit.
private func identical(
    _ a: [(tokenID: Int, logprob: Double)], _ b: [(tokenID: Int, logprob: Double)]
) -> Bool {
    a.count == b.count
        && zip(a, b).allSatisfy {
            $0.tokenID == $1.tokenID && Float($0.logprob).bitPattern == Float($1.logprob).bitPattern
        }
}

extension MLXTests {
    @Suite(
        "DiffusionGemma image reads against the oracle (opt-in)",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage),
        .enabled(if: VisionFixtures.hotdogAvailable, VisionFixtures.hotdogMessage))
    struct ImageReadOracleTests {
        /// The hot dog prompts are expanded to the oracle's ids, prefilled through the ported
        /// vision tower and read at both canvases. Exact tier (`OPENJEV_MLX_METALLIB` set to the
        /// wheel's metallib, the oracle's RoPE table installed): every map identical, bit for
        /// bit. Native tier: D-014's bounds over the four reads' slots, with per-read figures
        /// printed.
        @Test("The four hot dog reads of Fixtures/vision/reads.json meet D-014")
        func hotdogReads() async throws {
            let live = try await LiveCheckpoint.shared()
            let model = live.loaded.model
            let tokenizer = try #require(live.runtime.tokenizer as? SwiftTransformersTokenizer)
            let oracle = try ImageOracleReads.load()
            #expect(oracle.reads.count == 4)
            let hotdog = try Data(contentsOf: VisionFixtures.hotdog)
            let part = ImagePart(contentType: "image/jpeg", base64: hotdog.base64EncodedString())

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

            var labelDifferences: [Double] = []
            var entropyDifferences: [Double] = []
            var slots = 0
            var topAgree = 0
            var confident = 0
            var confidentAgree = 0
            var identicalReads = 0
            var lines: [String] = []
            for key in ["hotdog", "readme_hotdog"] {
                let prompt = try #require(oracle.prompts[key])
                #expect(prompt.images == ["hotdog"])
                let inputs = try ImageReadInputs(
                    system: prompt.system, state: prompt.state, parts: [part],
                    tokenizer: tokenizer)
                #expect(inputs.prompt.ids == prompt.ids, "\(key): the expanded prompt differs")
                let cache = try model.prefill(image: inputs)
                #expect(cache.promptTokens == prompt.ids.count)

                for read in oracle.reads where read.prompt == key {
                    let output = try model.read(
                        canvas: read.canvas, slots: read.requests, cache: cache,
                        steps: read.steps, topK: 20, projection: .full)
                    #expect(output.promptTokens == read.promptTokens, "\(read.id)")
                    let recorded = read.maps
                    let same =
                        output.slots.count == recorded.count
                        && zip(output.slots, recorded).allSatisfy { identical($0, $1) }
                    identicalReads += same ? 1 : 0
                    if exact {
                        #expect(same, "\(read.id): the maps differ from the oracle's")
                    }
                    var readMax = 0.0
                    var readEntropy = 0.0
                    var tops: [String] = []
                    for (index, slot) in read.slots.enumerated() {
                        let want = SlotDistribution.compute(
                            top: recorded[index], labelIDs: slot.labelIDs)
                        let got = SlotDistribution.compute(
                            top: output.slots[index], labelIDs: slot.labelIDs)
                        let stored = read.distributions[index]
                        #expect(
                            zip(want.probabilities, stored.probs).allSatisfy {
                                abs($0 - $1) <= 1e-12
                            }, "\(read.id) slot \(index)")
                        let differences = zip(got.probabilities, want.probabilities).map {
                            abs($0 - $1)
                        }
                        labelDifferences += differences
                        entropyDifferences.append(abs(got.entropy - want.entropy))
                        readMax = max(readMax, differences.max() ?? 0)
                        readEntropy = max(readEntropy, abs(got.entropy - want.entropy))
                        let agree =
                            ReadDivergence.firstLargest(got.probabilities)
                            == ReadDivergence.firstLargest(want.probabilities)
                        slots += 1
                        topAgree += agree ? 1 : 0
                        let ranked = want.probabilities.sorted(by: >)
                        if ranked[0] - (ranked.count > 1 ? ranked[1] : 0) >= 0.5 {
                            confident += 1
                            confidentAgree += agree ? 1 : 0
                        }
                        tops.append(
                            String(
                                format: "%.4f/%.4f", got.probabilities[0], want.probabilities[0]))
                    }
                    lines.append(
                        String(
                            format:
                                "%@ (%d tokens, width %d): %@, max |dp| %.6f, max |dH| %.6f, first label ours/oracle %@",
                            read.id, read.promptTokens, read.width,
                            same ? "identical" : "differs", readMax, readEntropy,
                            tops.joined(separator: " ")))
                }
            }

            let mean = { (values: [Double]) in values.reduce(0, +) / Double(max(values.count, 1)) }
            let report =
                """
                image reads, \(exact ? "exact tier (wheel metallib, oracle RoPE)" : "native kernels"): \
                \(identicalReads)/\(oracle.reads.count) reads bit-identical
                mean |dp| \(String(format: "%.6f", mean(labelDifferences))) over \(labelDifferences.count) labels (bound 0.02)
                mean |dH| \(String(format: "%.6f", mean(entropyDifferences))) over \(slots) slots (bound 0.2)
                top label \(topAgree)/\(slots) (bound 90%), oracle margin >= 0.5: \(confidentAgree)/\(confident) (bound 97%)
                """
                + "\n" + lines.joined(separator: "\n")
            print(report)
            SpikeReport.record("image-read-parity", report)

            if exact {
                #expect(identicalReads == oracle.reads.count)
            }
            #expect(mean(labelDifferences) <= 0.02)
            #expect(mean(entropyDifferences) <= 0.2)
            #expect(Double(topAgree) >= 0.9 * Double(slots))
            #expect(Double(confidentAgree) >= 0.97 * Double(confident))
        }

        /// Two images: of two sizes (the hot dog and small.png, `preprocessing.json`'s
        /// `hotdog+small`), which the tower reads one by one, and of one size (the hot dog twice),
        /// which it reads as a batch. Each prompt expands to the recorded ids or to both images'
        /// blocks, prefills with every soft token filled, and reads. No oracle read exists for two
        /// images, so this shows the paths run and count, not their values.
        @Test("Two images, of two sizes and of one, prefill and read")
        func twoImages() async throws {
            let live = try await LiveCheckpoint.shared()
            let model = live.loaded.model
            let tokenizer = try #require(live.runtime.tokenizer as? SwiftTransformersTokenizer)
            let preprocessing = try VisionFixtures.preprocessing()
            let recorded = try #require(preprocessing.prompts.first { $0.key == "hotdog+small" })
            let parts = try recorded.images.map { name in
                let image = try #require(preprocessing.images.first { $0.name == name })
                return ImagePart(
                    contentType: image.contentType, base64: try image.data().base64EncodedString())
            }
            let oracle = try ImageOracleReads.load()
            let read = try #require(oracle.reads.first { $0.prompt == "hotdog" })
            for (label, images) in [("two sizes", parts), ("one size", [parts[0], parts[0]])] {
                let inputs = try ImageReadInputs(
                    system: recorded.system, state: recorded.state, parts: images,
                    tokenizer: tokenizer)
                if label == "two sizes" {
                    #expect(inputs.prompt.ids == recorded.ids)
                    #expect(inputs.pixelValues.count == 2)
                } else {
                    #expect(inputs.pixelValues.count == 1 && inputs.pixelValues[0].dim(0) == 2)
                }
                var features: MLXArray?
                let cache = try model.prefill(image: inputs) { name, value in
                    if name == "image_features" { features = value }
                }
                let soft = inputs.prompt.mmTokenTypeIDs.filter { $0 == 1 }.count
                #expect(soft == inputs.prompt.softTokens.reduce(0, +))
                #expect(features?.dim(1) == soft, "\(label)")
                let output = try model.read(
                    canvas: read.canvas, slots: read.requests, cache: cache, steps: 1, topK: 20)
                #expect(output.promptTokens == inputs.prompt.ids.count)
                print("\(label): \(inputs.prompt.ids.count) prompt tokens, \(soft) soft tokens")
            }
        }
    }
}
