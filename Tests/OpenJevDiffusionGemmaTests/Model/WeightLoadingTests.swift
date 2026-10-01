import Foundation
import MLX
import MLXLMCommon
import MLXNN
import OpenJevDiffusionGemma
import Testing

/// A tiny checkpoint written from a quantized tiny tree: two shards, an index and config.json.
private struct TinyCheckpoint {
    static let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

    let directory: URL
    /// The tree the tensors came from.
    let source: DiffusionGemmaModel
    /// Tensor name to the shard the index places it in.
    let index: [String: String]

    /// Writes the tree's tensors, after `edit`, to a new temporary directory. The index lists
    /// the tensors before the edit, as a partial checkpoint's would.
    init(edit: (inout [String: MLXArray]) -> Void = { _ in }) throws {
        MetalLibrary.configure()
        MLXRandom.seed(27)
        source = DiffusionGemmaModel(try ModelFixtures.tinyText())
        source.quantize(try ModelFixtures.tinyPerLayerQuantization())
        eval(source)
        let original = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let names = original.keys.sorted()
        var index: [String: String] = [:]
        for (position, name) in names.enumerated() {
            index[name] = Self.shards[position < names.count / 2 ? 0 : 1]
        }
        self.index = index

        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-tiny-checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var tensors = original
        edit(&tensors)
        for shard in Self.shards {
            let part = tensors.filter { (index[$0.key] ?? Self.shards[1]) == shard }
            try save(
                arrays: part, metadata: ["format": "mlx"],
                url: directory.appendingPathComponent(shard))
        }
        let indexObject: [String: Any] = ["metadata": [:] as [String: Any], "weight_map": index]
        try JSONSerialization.data(withJSONObject: indexObject).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        try Data(ModelFixtures.tinyConfigJSON.utf8).write(
            to: directory.appendingPathComponent("config.json"))
    }

    func remove() {
        try? FileManager.default.removeItem(at: directory)
    }

    /// The coverage error loading this checkpoint throws.
    func coverageError() async throws -> (
        missing: [WeightLoadingError.Tensor], unexpected: [WeightLoadingError.Tensor],
        mismatched: [WeightLoadingError.ShapeMismatch], description: String
    ) {
        do {
            _ = try await DiffusionGemmaModel.load(from: directory)
        } catch let error as WeightLoadingError {
            guard case .coverage(let missing, let unexpected, let mismatched) = error else {
                Issue.record("not a coverage error: \(error)")
                throw error
            }
            return (missing, unexpected, mismatched, error.description)
        }
        Issue.record("the checkpoint loaded")
        throw CancellationError()
    }
}

extension MLXTests {
    @Suite("DiffusionGemma weight loading")
    struct WeightLoadingTests {
        @Test("Sanitize: what a text-only load drops and renames")
        func sanitizeNames() {
            let cases: [(String, String?)] = [
                (
                    "model.decoder.layers.0.self_attn.q_proj.weight",
                    "model.decoder.layers.0.self_attn.q_proj.weight"
                ),
                ("model.decoder.layers.0.self_attn.rotary_emb.inv_freq", nil),
                ("lm_head.weight", nil),
                ("model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight", nil),
                ("model.encoder.embed_vision.embedding_projection.weight", nil),
                (
                    "model.encoder.language_model.layers.3.layer_scalar",
                    "model.encoder.language_model.layers.3.layer_scalar"
                ),
                ("model.encoder.language_model.layers.3.mlp.up_proj.weight", nil),
                ("model.encoder.language_model.norm.weight", nil),
                (
                    "model.decoder.layers.2.experts.gate_up_proj",
                    "model.decoder.layers.2.experts.gate_up_proj.weight"
                ),
                (
                    "model.decoder.layers.2.experts.down_proj",
                    "model.decoder.layers.2.experts.down_proj.weight"
                ),
                (
                    "model.decoder.layers.2.experts.down_proj.scales",
                    "model.decoder.layers.2.experts.down_proj.scales"
                ),
            ]
            for (name, expected) in cases {
                #expect(DiffusionGemmaModel.sanitizedName(name) == expected, "\(name)")
            }
        }

        /// The real-size tree, built and quantized without evaluating anything, against the
        /// 1,647 names of Fixtures/model/weight_map.json.
        @Test("The checkpoint's names after sanitize are exactly the real tree's parameters")
        func checkpointCoverage() throws {
            MetalLibrary.configure()
            let configuration = try ModelFixtures.checkpointConfiguration()
            let weightMap = try ModelFixtures.checkpointWeightMap()
            #expect(weightMap.count == 1_647)
            let sanitized = Set(weightMap.keys.compactMap(DiffusionGemmaModel.sanitizedName))
            let dropped = weightMap.keys.filter { DiffusionGemmaModel.sanitizedName($0) == nil }
            // 355 vision tower tensors and embed_vision's projection (weight, scales, biases).
            // The pinned checkpoint has no rotary_emb, no lm_head.weight and no encoder text
            // weight besides the 30 scalars, so those rules drop nothing here.
            #expect(dropped.count == 358)
            #expect(dropped.filter { $0.hasPrefix("model.encoder.vision_tower.") }.count == 355)
            #expect(dropped.filter { $0.hasPrefix("model.encoder.embed_vision.") }.count == 3)
            #expect(sanitized.count == 1_289)

            let before = ProcessMemory.current()
            let model = DiffusionGemmaModel(configuration.text)
            let perLayer = try #require(configuration.quantization).perLayerQuantization
            model.quantize(perLayer, checkpointNames: sanitized)
            let parameters = Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
            let after = ProcessMemory.current()

            let missing = Set(parameters.keys).subtracting(sanitized).sorted()
            let unexpected = sanitized.subtracting(parameters.keys).sorted()
            #expect(missing.isEmpty, "missing: \(missing.prefix(5))")
            #expect(unexpected.isEmpty, "unexpected: \(unexpected.prefix(5))")
            // Every .scales of the checkpoint but embed_vision's: 299 of 300.
            #expect(weightMap.keys.filter { $0.hasSuffix(".scales") }.count == 300)
            #expect(model.quantizedModuleCount == 299)

            // Shapes from the configuration: 8 bits pack 4 values per uint32, 4 bits 8, in
            // groups of 64.
            let expectedShapes: [String: [Int]] = [
                "model.decoder.embed_tokens.weight": [262_144, 704],
                "model.decoder.embed_tokens.scales": [262_144, 44],
                "model.decoder.layers.0.self_attn.q_proj.weight": [4096, 704],
                "model.decoder.layers.0.self_attn.v_proj.weight": [2048, 704],
                "model.decoder.layers.5.self_attn.k_proj.weight": [1024, 704],
                "model.decoder.layers.5.self_attn.o_proj.weight": [2816, 2048],
                "model.decoder.layers.5.self_attn.q_norm.weight": [512],
                "model.decoder.layers.0.mlp.down_proj.weight": [2816, 528],
                "model.decoder.layers.0.router.proj.weight": [128, 704],
                "model.decoder.layers.0.router.scale": [2816],
                "model.decoder.layers.0.router.per_expert_scale": [128],
                "model.decoder.layers.0.experts.gate_up_proj.weight": [128, 1408, 352],
                "model.decoder.layers.0.experts.gate_up_proj.scales": [128, 1408, 44],
                "model.decoder.layers.0.experts.down_proj.weight": [128, 2816, 88],
                "model.decoder.layers.0.layer_scalar": [1],
                "model.decoder.self_conditioning.gate_proj.weight": [2112, 352],
                "model.encoder.language_model.layers.29.layer_scalar": [1],
            ]
            for (name, shape) in expectedShapes {
                #expect(parameters[name] == shape, "\(name)")
            }
            #expect(parameters["model.decoder.layers.5.self_attn.v_proj.weight"] == nil)

            // Lazy: building and quantizing the 26B tree allocates no weights.
            let added = after.residentBytes - before.residentBytes
            print("real tree built lazily: resident added \(ProcessMemory.megabytes(added))")
            #expect(added < 512 * 1024 * 1024)
        }

        @Test("A complete tiny checkpoint loads strictly and gives the source's outputs")
        func tinyCheckpointLoads() async throws {
            let checkpoint = try TinyCheckpoint { tensors in
                // Tensors a text-only load drops.
                tensors["lm_head.weight"] = MLXArray.zeros([4])
                tensors["model.encoder.vision_tower.patch_embedder.weight"] = MLXArray.zeros([4])
                tensors["model.encoder.language_model.layers.0.mlp.up_proj.weight"] =
                    MLXArray.zeros([4])
            }
            defer { checkpoint.remove() }
            let loaded = try await DiffusionGemmaModel.load(from: checkpoint.directory)
            let metrics = loaded.metrics
            #expect(metrics.shardCount == 2)
            #expect(metrics.tensorCount == 93)
            #expect(metrics.droppedTensorCount == 3)
            #expect(metrics.quantizedModuleCount == 21)
            let sizes = try TinyCheckpoint.shards.map {
                try #require(
                    FileManager.default.attributesOfItem(
                        atPath: checkpoint.directory.appendingPathComponent($0).path)[.size]
                        as? NSNumber
                ).intValue
            }
            #expect(metrics.mappedBytes == sizes.reduce(0, +))

            let ids = MLXArray(Int32(0)..<Int32(12)).reshaped(1, 12)
            let embeddings = checkpoint.source.decoder.embed(ids)
            let expected = checkpoint.source.prefill(embeddings: embeddings).hidden
            let got = loaded.model.prefill(embeddings: loaded.model.decoder.embed(ids)).hidden
            #expect(arrayEqual(got, expected).item(Bool.self))
        }

        @Test("A missing tensor is named with the shard the index places it in")
        func missingTensor() async throws {
            let name = "model.decoder.layers.1.router.per_expert_scale"
            let checkpoint = try TinyCheckpoint { $0[name] = nil }
            defer { checkpoint.remove() }
            let error = try await checkpoint.coverageError()
            let shard = try #require(checkpoint.index[name])
            #expect(error.missing == [WeightLoadingError.Tensor(name: name, shard: shard)])
            #expect(error.unexpected.isEmpty)
            #expect(error.description.contains(name))
            #expect(error.description.contains(shard))
        }

        @Test("A shard absent from disk makes its tensors missing, each naming it")
        func missingShard() async throws {
            let checkpoint = try TinyCheckpoint()
            defer { checkpoint.remove() }
            let shard = TinyCheckpoint.shards[1]
            try FileManager.default.removeItem(
                at: checkpoint.directory.appendingPathComponent(shard))
            let error = try await checkpoint.coverageError()
            let expected = checkpoint.index.filter { $0.value == shard }.keys.sorted()
            #expect(error.missing.map(\.name) == expected)
            #expect(error.missing.allSatisfy { $0.shard == shard })
            #expect(error.description.contains(try #require(expected.first)))
        }

        @Test("An unexpected tensor is named with its shard")
        func unexpectedTensor() async throws {
            let name = "model.decoder.layers.0.extra.weight"
            let checkpoint = try TinyCheckpoint { $0[name] = MLXArray.zeros([2]) }
            defer { checkpoint.remove() }
            let error = try await checkpoint.coverageError()
            #expect(error.missing.isEmpty)
            let tensor = WeightLoadingError.Tensor(name: name, shard: TinyCheckpoint.shards[1])
            #expect(error.unexpected == [tensor])
            #expect(error.description.contains(name))
        }

        @Test("A tensor of the wrong shape is reported with both shapes")
        func wrongShape() async throws {
            let name = "model.decoder.layers.0.layer_scalar"
            let checkpoint = try TinyCheckpoint { $0[name] = MLXArray.ones([2]) }
            defer { checkpoint.remove() }
            let error = try await checkpoint.coverageError()
            #expect(error.mismatched.map(\.name) == [name])
            #expect(error.mismatched.first?.expected == [1])
            #expect(error.mismatched.first?.found == [2])
        }
    }
}
