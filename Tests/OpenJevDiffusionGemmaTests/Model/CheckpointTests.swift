import Foundation
import MLX
import MLXLMCommon
import MLXNN
import OpenJevDiffusionGemma
import Testing

/// One recorded stage against the dump.
private struct StageComparison: CustomStringConvertible {
    let name: String
    let identical: Bool
    /// The largest absolute difference (for router indices, the entries that differ).
    let maxAbsolute: Float
    /// ``maxAbsolute`` over the dump's largest magnitude.
    let maxRelative: Float

    var description: String {
        let state = identical ? "identical" : "differs"
        if name.hasSuffix(".indices") {
            return "\(name): \(state), \(Int(maxAbsolute)) entries differ"
        }
        return String(
            format: "%@: %@, max |d| %.6g, relative %.3g", name, state, maxAbsolute, maxRelative)
    }
}

/// Compares `got` with `want`. Router rows are put in expert order first, since argpartition's
/// order within the top 8 is not part of the result.
private func compare(
    _ name: String, _ got: MLXArray, _ want: MLXArray, order: (got: MLXArray, want: MLXArray)?
) -> StageComparison {
    var got = got
    var want = want
    if let order {
        got = takeAlong(got, argSort(order.got, axis: -1), axis: -1)
        want = takeAlong(want, argSort(order.want, axis: -1), axis: -1)
    }
    let identical =
        got.shape == want.shape && got.dtype == want.dtype && arrayEqual(got, want).item(Bool.self)
    if name.hasSuffix(".indices") {
        let differing = (got.asType(.int64) .!= want.asType(.int64)).sum().item(Int.self)
        return StageComparison(
            name: name, identical: identical, maxAbsolute: Float(differing),
            maxRelative: Float(differing) / Float(want.size))
    }
    let difference = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
    let scale = abs(want.asType(.float32)).max().item(Float.self)
    return StageComparison(
        name: name, identical: identical, maxAbsolute: difference,
        maxRelative: scale > 0 ? difference / scale : difference)
}

/// D-014's native tier: the bounds the parity test asserts, twice the largest difference measured
/// on 2026-10-01 over layers 0 to 5 of both dumps (quickstart/g0 and indexed_12_mixed/g0) under
/// mlx-swift's own kernels, with the measurement beside each bound. Layer 0's differences are
/// bfloat16 last bits (attn.0 0.031, relative 3.8e-4); they grow with depth because a last-bit
/// change in a router score flips an expert, which is a discrete change. The same comparison is
/// bit-identical on all 37 stages of both dumps in the exact tier. A wrong module shows as a
/// relative difference near 1.
private enum NativeBounds {
    static let absolute: [String: Float] = [
        "embeddings": 0,  // measured 0: the quantized embedding gathers exactly
        "attn": 6,  // measured 3 (attn.5, indexed_12_mixed)
        "mlp": 12,  // measured 6 (mlp.5, indexed_12_mixed)
        "router.weights": 1.17,  // measured 0.583 (router.3, indexed_12_mixed)
        "experts": 2.15,  // measured 1.077 (experts.4, indexed_12_mixed)
        "layer": 8.5,  // measured 4.25 (layer.5, indexed_12_mixed)
    ]
    static let relative: [String: Float] = [
        "embeddings": 0,  // measured 0
        "attn": 0.0906,  // measured 0.0453 (attn.4, indexed_12_mixed)
        "mlp": 0.0538,  // measured 0.0269 (mlp.5, indexed_12_mixed)
        "router.weights": 1.29,  // measured 0.644 (router.5, indexed_12_mixed): flipped experts
        "experts": 0.021,  // measured 0.0105 (experts.4, indexed_12_mixed)
        "layer": 0.114,  // measured 0.057 (layer.5, indexed_12_mixed)
    ]
    /// The fraction of router index entries that differ, rows in expert order.
    static let differingIndices: [String: Float] = [
        "router.indices": 0.19  // measured 0.095 (router.5, indexed_12_mixed: 1,195 of 12,576)
    ]
}

extension MLXTests {
    @Suite(
        "DiffusionGemma checkpoint (opt-in)",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage))
    struct CheckpointTests {
        @Test("The checkpoint loads strictly: no missing, no unexpected, 300 quantized modules")
        func loads() async throws {
            let live = try await LiveCheckpoint.shared()
            let metrics = live.loaded.metrics
            let weightMap = try ModelFixtures.checkpointWeightMap()
            let kept = weightMap.keys.compactMap {
                DiffusionGemmaModel.sanitizedName($0, vision: true)
            }
            #expect(metrics.shardCount == 4)
            // The index's total_size counts tensor bytes; the shards add their headers.
            #expect(metrics.mappedBytes >= 16_542_844_632)
            #expect(metrics.mappedBytes < 16_542_844_632 + 64 * 1024 * 1024)
            #expect(metrics.tensorCount == kept.count)
            // Every tensor loads, the vision tower's included (#47).
            #expect(metrics.tensorCount == 1_647)
            #expect(metrics.droppedTensorCount == 0)
            #expect(metrics.quantizedModuleCount == 300)
            #expect(live.loaded.model.readsImages)
            let tower = try #require(live.loaded.model.encoder.visionTower)
            #expect(tower.patchEmbedder.positionEmbeddingTable.dtype == .bfloat16)
            #expect(tower.encoder.layers.count == 27)
            let projection = try #require(
                live.loaded.model.encoder.embedVision?.embeddingProjection as? QuantizedLinear)
            #expect(projection.bits == 4)
            #expect(live.loaded.model.parameters().flattened().count == kept.count)
            let embedding = try #require(
                live.loaded.model.decoder.embedTokens as? QuantizedEmbedding)
            #expect(embedding.bits == 8)
            let experts = try #require(
                live.loaded.model.decoder.layers[0].experts.gateUpProj as? QuantizedSwitchLinear)
            #expect(experts.bits == 4)

            let gib = { (bytes: Int) in String(format: "%.2f GiB", Double(bytes) / 1_073_741_824) }
            let report = """
                load: \(metrics.wallTime.formatted(.units(allowed: [.seconds], fractionalPart: .show(length: 2))))
                shards: \(metrics.shardCount), \(metrics.mappedBytes) bytes (\(gib(metrics.mappedBytes)))
                tensors: \(metrics.tensorCount) loaded, \(metrics.droppedTensorCount) dropped by sanitize
                quantized modules: \(metrics.quantizedModuleCount)
                vision tower and embed_vision: \(metrics.visionParameterCount) parameters \
                (as stored), \(metrics.visionBytes) bytes (\(gib(metrics.visionBytes)))
                resident before: \(gib(metrics.residentBytesBefore))
                resident after: \(gib(metrics.residentBytesAfter))
                resident added: \(gib(metrics.residentBytesAdded))
                peak resident: \(gib(metrics.peakResidentBytes))
                MLX active: \(gib(metrics.mlxActiveBytes))
                """
            print(report)
            SpikeReport.record("model-load", report)
        }

        /// D-014's two tiers on layers 0 to 5 of a one-piece prefill, from the dump's own
        /// embeddings. With `OPENJEV_MLX_METALLIB` taken (the wheel's kernels) and the oracle's
        /// RoPE table installed, every stage must be bit-identical; otherwise each stage must stay
        /// within the native bounds.
        @Test(
            "Layers 0 to 5 match mlx-vlm's stage dump",
            .enabled(
                if: ModelFixtures.stageDumps.contains {
                    FileManager.default.fileExists(atPath: $0.url.path)
                },
                Comment(
                    rawValue: "No stage dump under Tools/oracle/results or in "
                        + "\(ModelFixtures.stagesVariable); with the checkpoint at "
                        + "\(TokenizerFixtures.modelVariable), run Tools/oracle/stage_dump.py "
                        + "as docs/spikes/backend-validation.md describes")),
            arguments: ModelFixtures.stageDumps.filter {
                FileManager.default.fileExists(atPath: $0.url.path)
            })
        func layerParity(dump: ModelFixtures.StageDump) async throws {
            let live = try await LiveCheckpoint.shared()
            let model = live.loaded.model
            let reference = try loadArrays(url: dump.url)
            let embeddings = try #require(reference["embeddings"])
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

            var recorded: [(String, MLXArray)] = []
            if let key = dump.promptKey, let ids = try ModelFixtures.oracle().prompts[key]?.ids {
                #expect(ids.count == embeddings.dim(1))
                let own = model.decoder.embed(MLXArray(ids.map(Int32.init)).reshaped(1, ids.count))
                recorded.append(("embeddings", own))
            }
            _ = model.prefill(embeddings: embeddings, layers: 0..<6) { recorded.append(($0, $1)) }
            eval(recorded.map(\.1))

            let values = Dictionary(recorded, uniquingKeysWith: { first, _ in first })
            var results: [StageComparison] = []
            for (name, got) in recorded {
                let want = try #require(
                    reference[name], "\(name) is not in \(dump.url.lastPathComponent)")
                var order: (got: MLXArray, want: MLXArray)?
                if name.hasPrefix("router.") {
                    let layer = name.split(separator: ".")[1]
                    order = (
                        try #require(values["router.\(layer).indices"]),
                        try #require(reference["router.\(layer).indices"])
                    )
                }
                results.append(compare(name, got, want, order: order))
            }
            let identical = results.filter(\.identical).count
            let report =
                "\(dump.url.lastPathComponent), \(exact ? "exact tier (wheel metallib, oracle RoPE)" : "native kernels"): "
                + "\(identical)/\(results.count) stages bit-identical\n"
                + results.map { "  \($0)" }.joined(separator: "\n")
            print(report)
            SpikeReport.record("layer-parity", report)

            if exact {
                for result in results {
                    #expect(result.identical, "\(result)")
                }
                return
            }
            for result in results {
                let stage = Self.stageKind(result.name)
                if result.name.hasSuffix(".indices") {
                    let bound = try #require(NativeBounds.differingIndices[stage], "\(stage)")
                    #expect(result.maxRelative <= bound, "\(result)")
                } else {
                    let absolute = try #require(NativeBounds.absolute[stage], "\(stage)")
                    let relative = try #require(NativeBounds.relative[stage], "\(stage)")
                    #expect(result.maxAbsolute <= absolute, "\(result)")
                    #expect(result.maxRelative <= relative, "\(result)")
                }
            }
        }

        /// `attn.3` to `attn`, `router.3.weights` to `router.weights`.
        static func stageKind(_ name: String) -> String {
            name.split(separator: ".").filter { Int($0) == nil }.joined(separator: ".")
        }
    }
}
