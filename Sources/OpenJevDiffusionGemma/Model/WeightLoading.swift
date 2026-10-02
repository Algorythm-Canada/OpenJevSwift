// Loads a DiffusionGemma checkpoint directory into the module tree with mlx-swift-lm's
// loadWeights, after a strict, header-only coverage check that names the first missing or
// unexpected tensor and its shard.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Why a checkpoint did not load.
public enum WeightLoadingError: Error, Equatable, Sendable, CustomStringConvertible {
    /// One tensor and the shard it is in, or for a missing tensor the shard the index places it
    /// in (nil when the index does not name it).
    public struct Tensor: Equatable, Sendable {
        /// The tensor's name after sanitize.
        public var name: String
        /// The shard's file name, or nil when neither a shard nor the index names it.
        public var shard: String?

        /// Creates a tensor reference.
        public init(name: String, shard: String?) {
            self.name = name
            self.shard = shard
        }
    }

    /// A tensor whose shape in the checkpoint differs from the module tree's.
    public struct ShapeMismatch: Equatable, Sendable {
        /// The tensor's name after sanitize.
        public var name: String
        /// The shard that holds it.
        public var shard: String
        /// The shape the module tree has.
        public var expected: [Int]
        /// The shape the checkpoint has.
        public var found: [Int]
    }

    /// The directory holds no safetensors file.
    case noWeightFiles(directory: String)
    /// A shard's header could not be read.
    case unreadableShard(String, reason: String)
    /// The checkpoint's tensors, after sanitize, are not exactly the tree's parameters. Each list
    /// is sorted by name; the first entry is the one the description names.
    case coverage(missing: [Tensor], unexpected: [Tensor], mismatched: [ShapeMismatch])

    /// What failed: for a coverage error the counts, and the first missing, unexpected and
    /// mismatched tensor with its shard.
    public var description: String {
        switch self {
        case .noWeightFiles(let directory):
            return "\(directory) holds no .safetensors file"
        case .unreadableShard(let shard, let reason):
            return "cannot read the safetensors header of \(shard): \(reason)"
        case .coverage(let missing, let unexpected, let mismatched):
            var parts: [String] = []
            if let first = missing.first {
                let place =
                    first.shard.map { "the index places it in \($0)" }
                    ?? "no shard or index entry has it"
                parts.append(
                    "\(missing.count) tensor(s) missing, first \(first.name) (\(place))")
            }
            if let first = unexpected.first {
                parts.append(
                    "\(unexpected.count) unexpected tensor(s), first \(first.name) in "
                        + (first.shard ?? "an unknown shard"))
            }
            if let first = mismatched.first {
                parts.append(
                    "\(mismatched.count) shape mismatch(es), first \(first.name) in \(first.shard): "
                        + "expected \(first.expected), found \(first.found)")
            }
            return "the checkpoint does not match the model: " + parts.joined(separator: "; ")
        }
    }
}

extension DiffusionGemmaModel {
    /// What loading a checkpoint cost, measured by ``load(from:configuration:progress:)``.
    ///
    /// The memory figures are of the whole process (`task_info` resident size and `getrusage`'s
    /// peak), so they mean most in a process that has done little else.
    public struct LoadMetrics: Sendable, Hashable {
        /// The wall time from reading `config.json` to the evaluated parameters.
        public var wallTime: Duration
        /// The total size of the shards loaded.
        public var mappedBytes: Int
        /// The number of shards loaded.
        public var shardCount: Int
        /// The tensors loaded, after sanitize.
        public var tensorCount: Int
        /// The tensors sanitize dropped.
        public var droppedTensorCount: Int
        /// The quantized modules in the loaded tree.
        public var quantizedModuleCount: Int
        /// The process's resident memory in bytes just before loading.
        public var residentBytesBefore: Int
        /// The process's resident memory in bytes after loading.
        public var residentBytesAfter: Int
        /// The process's peak resident memory in bytes after loading.
        public var peakResidentBytes: Int
        /// MLX's active memory in bytes after loading (`Memory.activeMemory`): the arrays the
        /// tree holds, which the resident size undercounts because the weights are GPU buffers.
        public var mlxActiveBytes: Int

        /// The resident memory loading added, in bytes.
        public var residentBytesAdded: Int { residentBytesAfter - residentBytesBefore }
    }

    /// The stages ``load(from:configuration:progress:)`` reports.
    public enum LoadStage: Sendable, Hashable {
        /// Reading `config.json` and building the module tree.
        case configuring
        /// Reading the shard headers and checking them against the tree.
        case checkingCoverage
        /// Reading the tensors through `loadWeights`.
        case loadingWeights
        /// Loaded and evaluated.
        case finished
    }

    /// A loaded model with its configuration and what loading cost. Not Sendable, as the model
    /// is not.
    public struct LoadedModel {
        /// The module tree with the checkpoint's weights, evaluated.
        public let model: DiffusionGemmaModel
        /// The checkpoint's configuration.
        public let configuration: DiffusionGemmaConfiguration
        /// What loading took.
        public let metrics: LoadMetrics
    }

    /// Loads a checkpoint directory.
    ///
    /// Reads `config.json` when no configuration is passed and builds the text tree. Then, from
    /// the shard headers alone, it applies ``sanitizedName(_:)``, quantizes the tree as the
    /// configuration's per-layer map and the checkpoint's `.scales` tensors ask, and requires the
    /// tree's parameters and the checkpoint's tensors to be the same names with the same shapes.
    /// Only then does mlx-swift-lm's `loadWeights(modelDirectory:model:perLayerQuantization:)`
    /// read the tensors, sanitize them (``sanitize(weights:)``), update the tree with
    /// `verify: .all` and evaluate it.
    ///
    /// The shards are those `model.safetensors.index.json` names, else every top-level
    /// `model*.safetensors` file. A shard the index names but the directory lacks makes its
    /// tensors missing, each naming that shard.
    ///
    /// - Throws: ``WeightLoadingError``, a ``DiffusionGemmaConfigurationError``, or what
    ///   `loadWeights` throws.
    public static func load(
        from directory: URL, configuration: DiffusionGemmaConfiguration? = nil,
        progress: (@Sendable (LoadStage) -> Void)? = nil
    ) async throws -> LoadedModel {
        let before = ResourceUsage.current()
        let clock = ContinuousClock()
        let start = clock.now

        progress?(.configuring)
        let configuration = try configuration ?? DiffusionGemmaConfiguration.load(from: directory)
        let model = DiffusionGemmaModel(configuration.text)
        let perLayer = configuration.quantization?.perLayerQuantization

        progress?(.checkingCoverage)
        let checkpoint = try CheckpointTensors(directory: directory)
        if let perLayer {
            // The index's names count too, so that a module whose shard is absent is still
            // quantized and its tensors are reported missing under their quantized names.
            model.quantize(
                perLayer,
                checkpointNames: Set(checkpoint.tensors.keys).union(checkpoint.indexed.keys))
        }
        try checkpoint.verify(against: model)

        progress?(.loadingWeights)
        try await loadWeights(
            modelDirectory: directory, model: model, perLayerQuantization: perLayer)
        model.train(false)

        let after = ResourceUsage.current()
        progress?(.finished)
        let metrics = LoadMetrics(
            wallTime: clock.now - start, mappedBytes: checkpoint.shardBytes,
            shardCount: checkpoint.shards.count, tensorCount: checkpoint.tensors.count,
            droppedTensorCount: checkpoint.droppedCount,
            quantizedModuleCount: model.quantizedModuleCount,
            residentBytesBefore: before.residentBytes, residentBytesAfter: after.residentBytes,
            peakResidentBytes: after.peakResidentBytes, mlxActiveBytes: Memory.activeMemory)
        return LoadedModel(model: model, configuration: configuration, metrics: metrics)
    }
}

/// The tensor names and shapes of a checkpoint's shards, from their headers, after sanitize.
struct CheckpointTensors: Sendable {
    struct Entry: Sendable {
        var shard: String
        var shape: [Int]
    }

    /// The shard file names, sorted.
    let shards: [String]
    /// Their total size in bytes.
    let shardBytes: Int
    /// Sanitized tensor name to its shard and shape.
    let tensors: [String: Entry]
    /// Sanitized tensor name to the shard the index places it in.
    let indexed: [String: String]
    /// The tensors sanitize dropped.
    let droppedCount: Int

    init(directory: URL) throws {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        var indexed: [String: String] = [:]
        var shardNames: [String]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            struct Index: Decodable {
                let weightMap: [String: String]
                enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
            }
            let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: indexURL))
            for (name, shard) in index.weightMap {
                if let sanitized = DiffusionGemmaModel.sanitizedName(name) {
                    indexed[sanitized] = shard
                }
            }
            shardNames = Array(Set(index.weightMap.values)).sorted()
        } else {
            let contents =
                (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
            shardNames = contents.filter {
                $0.hasPrefix("model") && $0.hasSuffix(".safetensors")
            }.sorted()
        }
        guard !shardNames.isEmpty else {
            throw WeightLoadingError.noWeightFiles(directory: directory.path)
        }

        var tensors: [String: Entry] = [:]
        var bytes = 0
        var dropped = 0
        for shard in shardNames {
            let url = directory.appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            // A Hugging Face cache snapshot holds symlinks to the blobs: measure the blob.
            let attributes = try FileManager.default.attributesOfItem(
                atPath: url.resolvingSymlinksInPath().path)
            bytes += (attributes[.size] as? NSNumber)?.intValue ?? 0
            for (name, shape) in try Self.header(of: url, shard: shard) {
                if let sanitized = DiffusionGemmaModel.sanitizedName(name) {
                    tensors[sanitized] = Entry(shard: shard, shape: shape)
                } else {
                    dropped += 1
                }
            }
        }
        shards = shardNames
        shardBytes = bytes
        self.tensors = tensors
        self.indexed = indexed
        droppedCount = dropped
    }

    /// The names and shapes in a safetensors header: an 8-byte little-endian length, then that
    /// many bytes of JSON mapping each name to its `dtype`, `shape` and `data_offsets`.
    static func header(of url: URL, shard: String) throws -> [(String, [Int])] {
        func fail(_ reason: String) -> WeightLoadingError {
            .unreadableShard(shard, reason: reason)
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw fail("shorter than 8 bytes")
        }
        let length = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard length > 0, length <= 512 * 1024 * 1024 else {
            throw fail("header length \(length)")
        }
        guard let data = try handle.read(upToCount: Int(length)), data.count == Int(length),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw fail("the header is not a JSON object")
        }
        var out: [(String, [Int])] = []
        for (name, value) in object where name != "__metadata__" {
            guard let entry = value as? [String: Any], let shape = entry["shape"] as? [NSNumber]
            else {
                throw fail("\(name) has no shape")
            }
            out.append((name, shape.map(\.intValue)))
        }
        return out
    }

    /// Requires the tree's parameters to be exactly these tensors, with the same shapes.
    func verify(against model: DiffusionGemmaModel) throws {
        let parameters = Dictionary(
            model.parameters().flattened().map { ($0.0, $0.1.shape) },
            uniquingKeysWith: { first, _ in first })
        let missing = parameters.keys.filter { tensors[$0] == nil }.sorted().map {
            WeightLoadingError.Tensor(name: $0, shard: indexed[$0])
        }
        let unexpected = tensors.keys.filter { parameters[$0] == nil }.sorted().map {
            WeightLoadingError.Tensor(name: $0, shard: tensors[$0]?.shard)
        }
        let mismatched = parameters.keys.sorted().compactMap {
            name -> WeightLoadingError.ShapeMismatch? in
            guard let expected = parameters[name], let entry = tensors[name],
                entry.shape != expected
            else { return nil }
            return .init(name: name, shard: entry.shard, expected: expected, found: entry.shape)
        }
        if !missing.isEmpty || !unexpected.isEmpty || !mismatched.isEmpty {
            throw WeightLoadingError.coverage(
                missing: missing, unexpected: unexpected, mismatched: mismatched)
        }
    }
}
