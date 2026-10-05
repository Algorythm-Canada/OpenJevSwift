import Foundation
import MLX
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// mlx-vlm's own stages of the hot dog prefill, from
/// `Tools/oracle/stage_dump.py --image hotdog Tools/oracle/results/vision/hotdog.stages.safetensors`.
private enum ImageStages {
    static let url = VisionFixtures.tensors.appendingPathComponent("hotdog.stages.safetensors")

    static var available: Bool { FileManager.default.fileExists(atPath: url.path) }

    static let message = Comment(
        rawValue: "\(TokenizerFixtures.modelVariable) tests: \(url.lastPathComponent) is missing; "
            + "run Tools/oracle/stage_dump.py --image hotdog")
}

extension MLXTests {
    @Suite(
        "DiffusionGemma image prefill, stage by stage against mlx-vlm (opt-in)",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage),
        .enabled(if: VisionFixtures.hotdogAvailable, VisionFixtures.hotdogMessage),
        .enabled(if: ImageStages.available, ImageStages.message))
    struct ImageStageTests {
        /// Every recorded stage of the hot dog prefill, in the order the prefill reaches it: the
        /// pixels, the tower's patches, blocks, pool and output, the projected features, the
        /// embeddings after the scatter, the masks, each encoder layer and the first and last
        /// caches. Prints each stage's largest difference and names the first that is not equal.
        /// Exact tier: every stage equal, bit for bit. Native tier: reported only.
        @Test("The hot dog prefill's stages against mlx-vlm's")
        func stages() async throws {
            let live = try await LiveCheckpoint.shared()
            let model = live.loaded.model
            let tokenizer = try #require(live.runtime.tokenizer as? SwiftTransformersTokenizer)
            let reads = try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: VisionFixtures.directory.appendingPathComponent("reads.json")))
            let prompt = try #require(
                ((reads as? [String: Any])?["prompts"] as? [String: Any])?["hotdog"]
                    as? [String: Any])
            let system = try #require(prompt["system"] as? String)
            let state = try #require(prompt["state"] as? String)
            let hotdog = try Data(contentsOf: VisionFixtures.hotdog)
            let inputs = try ImageReadInputs(
                system: system, state: state,
                parts: [ImagePart(contentType: "image/jpeg", base64: hotdog.base64EncodedString())],
                tokenizer: tokenizer)
            let theirs = try MLX.loadArrays(url: ImageStages.url)

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

            var ours: [String: MLXArray] = ["pixel_values": inputs.pixelValues[0]]
            let cache = try model.prefill(image: inputs) { name, value in
                eval(value)
                ours[name] = value
            }
            let last = cache.layers.count - 1
            for index in [0, last] {
                ours["cache.\(index).keys"] = cache.layers[index].keys
                ours["cache.\(index).values"] = cache.layers[index].values
            }

            let vision = model.encoder.visionTower?.encoder.layers.count ?? 0
            let order =
                ["pixel_values", "vision.patches"] + (0..<vision).map { "vision.layer.\($0)" }
                + [
                    "vision.pooled", "vision.out", "image_features", "inputs_embeds", "mask.0",
                    "mask.5",
                ]
                + (0...last).map { "layer.\($0)" }
                + ["cache.0.keys", "cache.0.values", "cache.\(last).keys", "cache.\(last).values"]
            var lines: [String] = []
            var firstDifference: String?
            for name in order {
                guard let mine = ours[name], let recorded = theirs[name] else {
                    lines.append(
                        "\(name): missing (ours \(ours[name] != nil), theirs \(theirs[name] != nil))"
                    )
                    firstDifference = firstDifference ?? name
                    continue
                }
                let sameShape = mine.shape == recorded.shape && mine.dtype == recorded.dtype
                let equal = sameShape && arrayEqual(mine, recorded).item(Bool.self)
                var largest = Float.nan
                if mine.shape == recorded.shape, mine.dtype != .bool {
                    largest = abs(mine.asType(.float32) - recorded.asType(.float32)).max()
                        .item(Float.self)
                }
                lines.append(
                    "\(name) \(mine.dtype)\(mine.shape): \(equal ? "equal" : "differs"), max |d| \(largest)"
                        + (sameShape ? "" : " (theirs \(recorded.dtype)\(recorded.shape))"))
                if !equal, firstDifference == nil {
                    firstDifference = name
                }
            }
            let report =
                "image prefill stages, \(exact ? "exact tier" : "native kernels"): first difference "
                + (firstDifference ?? "none") + "\n" + lines.joined(separator: "\n")
            print(report)
            SpikeReport.record("image-stages", report)
            if exact {
                #expect(firstDifference == nil, "first difference at \(firstDifference ?? "")")
            }
        }

        /// The first vision block one operation at a time, each fed mlx-vlm's own input for it
        /// (so each operation is judged on its own), through the port's modules and functions:
        /// names the operations whose output differs. The RoPE is also run with mlx-vlm's
        /// timescale (`b0.q_rope with mlx-vlm's timescale`), which separates the `pow` that
        /// computes it from the rest. This is how the `pow` difference ``precisePow(_:_:)`` fixes
        /// was found. Exact tier: every operation equal.
        @Test("The first vision block, operation by operation, against mlx-vlm's")
        func firstBlock() async throws {
            let live = try await LiveCheckpoint.shared()
            let tower = try #require(live.loaded.model.encoder.visionTower)
            let theirs = try MLX.loadArrays(url: ImageStages.url)
            func stage(_ name: String) throws -> MLXArray { try #require(theirs[name]) }
            let x = try stage("vision.patches")
            let positions = try stage("b0.positions")
            let mask = try stage("b0.mask")
            let block = tower.encoder.layers[0]
            let a = block.selfAttention
            let (batch, length) = (x.dim(0), x.dim(1))
            let normed = try stage("b0.normed")
            let timescale = try stage("b0.rope_timescale")
            /// visionMultidimensionalRoPE with a given timescale.
            func rope(_ inputs: MLXArray, timescale: MLXArray) -> MLXArray {
                let channels = inputs.dim(-1) / 2
                var parts: [MLXArray] = []
                for d in 0..<2 {
                    let part = inputs[.ellipsis, (d * channels)..<((d + 1) * channels)]
                    let sinusoid = positions[.ellipsis, d..<(d + 1)].asType(.float32) / timescale
                    let cosine = expandedDimensions(
                        concatenated([cos(sinusoid), cos(sinusoid)], axis: -1).asType(inputs.dtype),
                        axis: 2)
                    let sine = expandedDimensions(
                        concatenated([sin(sinusoid), sin(sinusoid)], axis: -1).asType(inputs.dtype),
                        axis: 2)
                    parts.append(part * cosine + visionRotateHalf(part) * sine)
                }
                return concatenated(parts, axis: -1)
            }
            let step = Float(2.0 / Double(a.headDim / 2))
            let ownTimescale = precisePow(
                MLXArray(a.ropeBase),
                step * MLXArray(Int32(0)..<Int32(a.headDim / 4)).asType(.float32))
            let qNorm = try stage("b0.q_norm")
            let kNorm = try stage("b0.k_norm")
            let vNorm = try stage("b0.v_norm")
            let qRope = try stage("b0.q_rope")
            let kRope = try stage("b0.k_rope")
            let attention = try stage("b0.sdpa")
            let h = try stage("b0.h")
            let n = try stage("b0.pre_feedforward")
            let ours: [(String, String, MLXArray)] = [
                ("b0.normed", "b0.normed", block.inputLayerNorm(x)),
                ("b0.q", "b0.q", a.qProj(normed).reshaped(batch, length, a.heads, a.headDim)),
                (
                    "b0.k", "b0.k",
                    a.kProj(normed).reshaped(batch, length, a.keyValueHeads, a.headDim)
                ),
                (
                    "b0.v", "b0.v",
                    a.vProj(normed).reshaped(batch, length, a.keyValueHeads, a.headDim)
                ),
                ("b0.q_norm", "b0.q_norm", a.qNorm(try stage("b0.q"))),
                ("b0.k_norm", "b0.k_norm", a.kNorm(try stage("b0.k"))),
                ("b0.v_norm", "b0.v_norm", visionRMSNormNoScale(try stage("b0.v"), eps: 1e-6)),
                ("b0.rope_timescale", "b0.rope_timescale", ownTimescale),
                (
                    "b0.q_rope", "b0.q_rope",
                    visionMultidimensionalRoPE(
                        qNorm, positions: positions, baseFrequency: a.ropeBase)
                ),
                (
                    "b0.q_rope with mlx-vlm's timescale", "b0.q_rope",
                    rope(qNorm, timescale: timescale)
                ),
                (
                    "b0.k_rope with mlx-vlm's timescale", "b0.k_rope",
                    rope(kNorm, timescale: timescale)
                ),
                (
                    "b0.sdpa", "b0.sdpa",
                    visionFusedAttention(
                        qRope.transposed(0, 2, 1, 3), kRope.transposed(0, 2, 1, 3),
                        vNorm.transposed(0, 2, 1, 3), mask: mask)
                ),
                (
                    "b0.o_proj", "b0.o_proj",
                    a.oProj(attention.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
                ),
                (
                    "b0.post_attention", "b0.post_attention",
                    block.postAttentionLayerNorm(try stage("b0.o_proj"))
                ),
                ("b0.h", "b0.h", x + (try stage("b0.post_attention"))),
                ("b0.pre_feedforward", "b0.pre_feedforward", block.preFeedforwardLayerNorm(h)),
                ("b0.gate", "b0.gate", block.mlp.gateProj(n)),
                ("b0.up", "b0.up", block.mlp.upProj(n)),
                ("b0.mlp", "b0.mlp", block.mlp(n)),
                (
                    "b0.post_feedforward", "b0.post_feedforward",
                    block.postFeedforwardLayerNorm(try stage("b0.mlp"))
                ),
                ("b0.out", "b0.out", h + (try stage("b0.post_feedforward"))),
            ]
            var lines: [String] = []
            var differing: [String] = []
            for (name, key, mine) in ours {
                guard let recorded = theirs[key] else {
                    lines.append("\(name): not in the dump")
                    continue
                }
                let equal =
                    mine.shape == recorded.shape && mine.dtype == recorded.dtype
                    && arrayEqual(mine, recorded).item(Bool.self)
                let largest =
                    mine.shape == recorded.shape
                    ? abs(mine.asType(.float32) - recorded.asType(.float32)).max().item(Float.self)
                    : .nan
                lines.append(
                    "\(name) \(mine.dtype)\(mine.shape) vs \(recorded.dtype)\(recorded.shape): "
                        + "\(equal ? "equal" : "differs"), max |d| \(largest)")
                if !equal { differing.append(name) }
            }
            let report =
                "first vision block, each operation on mlx-vlm's input, "
                + "\(MetalLibrary.override != nil ? "exact tier" : "native kernels"): differing "
                + (differing.isEmpty ? "none" : differing.joined(separator: ", ")) + "\n"
                + lines.joined(separator: "\n")
            print(report)
            SpikeReport.record("image-first-block", report)
            if MetalLibrary.override != nil {
                #expect(differing.isEmpty, "differing: \(differing)")
            }
        }
    }
}
