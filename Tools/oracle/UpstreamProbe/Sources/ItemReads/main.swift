// The port's reads of JevBench and TypeSafe items, recorded for Tools/oracle/item_reads.py compare,
// which holds them against upstream's bit for bit.
//
//     swift run --package-path Tools/oracle/UpstreamProbe -c release ItemReads [--metallib PATH]
//         [--oracle-rope] [--work DIR] [--out PATH] [--cache-limit-gb 4] [--no-engine] [--no-replay]
//
// Engine: each body in WORK/bodies.json (item_reads.py bodies) goes through OpenJevCore's
// JSONParser, RequestValidator and DecisionEngine with EngineConfiguration.default (upstream's
// defaults, the route's seed) over a backend that reads as DiffusionGemmaRuntime does: one prefill
// per prompt, cached, then the model's read with the full projection and the top 20, and
// ReadOutput.readResult. Every CanvasRead is recorded with the model's maps, and every layer of
// each prompt's prefill cache is digested.
// Replay: every read in WORK/upstream.json (item_reads.py upstream) runs through the model with
// upstream's own prompt ids, canvas, slots and steps.
//
// --metallib loads the Metal kernels from that file, for D-014's exact tier the mlx-metal wheel's
// mlx.metallib, instead of this build's default.metallib, which is the Swift server's. It is set
// before any MLX call. --oracle-rope installs Fixtures/oracle/reads.json's RoPE table in the
// full-attention layers. Without --out the result goes to WORK/port_<configuration>.json:
// port_exact.json with both the oracle's metallib (by SHA-256) and its table, port_native.json
// with neither, port_wheel_metallib.json or port_oracle_rope.json with one. Run from the
// repository root, one model process at a time.

import CryptoKit
import Darwin
import Foundation
import MLX
import OpenJevCore
import OpenJevDiffusionGemma

setvbuf(stdout, nil, _IOLBF, 0)

// MARK: - Options

let environment = ProcessInfo.processInfo.environment
var workPath =
    environment["OPENJEV_ITEM_READS"]
    ?? (NSHomeDirectory() + "/Library/Caches/OpenJevSwift/item-reads")
var metallibPath: String?
var oracleRope = false
var cacheLimitGB = 4.0
var outPath: String?
var runEngine = true
var runReplay = true
var index = 1
let arguments = CommandLine.arguments
func operand() -> String {
    index += 1
    guard index < arguments.count else { fatalError("\(arguments[index - 1]) needs a value") }
    return arguments[index]
}
while index < arguments.count {
    switch arguments[index] {
    case "--metallib": metallibPath = operand()
    case "--oracle-rope": oracleRope = true
    case "--work": workPath = operand()
    case "--out": outPath = operand()
    case "--cache-limit-gb":
        guard let gb = Double(operand()), gb.isFinite, gb >= 0 else {
            fatalError("--cache-limit-gb takes a number of GB, 0 or more")
        }
        cacheLimitGB = gb
    case "--no-engine": runEngine = false
    case "--no-replay": runReplay = false
    default: fatalError("unknown argument \(arguments[index])")
    }
    index += 1
}
let work = URL(fileURLWithPath: (workPath as NSString).expandingTildeInPath)
let hub = environment["HF_HUB_CACHE"] ?? (NSHomeDirectory() + "/.cache/huggingface/hub")
let snapshot = URL(
    fileURLWithPath: environment["OPENJEV_TEST_MODEL"]
        ?? (hub + "/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots/"
            + "a7a81407613811e8ba63af92ac0d852b809e191f"))

/// The default.metallib that Swift Build copies next to this executable, compiled from the same
/// mlx-swift 0.32.3 sources by the same toolchain as the Swift server's.
func bundledMetallib() -> URL {
    let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
    let directory = executable.resolvingSymlinksInPath().deletingLastPathComponent()
    for path in [
        "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
        "mlx-swift_Cmlx.bundle/default.metallib",
    ] {
        let url = directory.appending(path: path)
        if FileManager.default.fileExists(atPath: url.path) { return url }
    }
    fatalError("no default.metallib next to \(directory.path); pass --metallib")
}

// The kernels, before any MLX call.
let metallib = metallibPath.map { URL(fileURLWithPath: $0) } ?? bundledMetallib()
GPU.metallib = metallib

func hex(_ digest: some Sequence<UInt8>) -> String {
    digest.map { String(format: "%02x", $0) }.joined()
}

/// A path as the result records it: relative to the working directory when it lies inside it,
/// otherwise only its last component, never an absolute home path.
func shown(_ url: URL) -> String {
    let path = url.standardizedFileURL.path
    let root =
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath).standardizedFileURL.path
        + "/"
    return path.hasPrefix(root) ? String(path.dropFirst(root.count)) : url.lastPathComponent
}

func processMemory() -> JSONValue {
    var info = rusage_info_v4()
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
        }
    }
    var out: JSONObject = [
        "mlx_active_bytes": JSONValue(Memory.activeMemory),
        "mlx_cache_bytes": JSONValue(Memory.cacheMemory),
        "mlx_peak_bytes": JSONValue(Memory.peakMemory),
    ]
    if status == 0 {
        out["phys_footprint_bytes"] = JSONValue(info.ri_phys_footprint)
        out["lifetime_max_phys_footprint_bytes"] = JSONValue(info.ri_lifetime_max_phys_footprint)
    }
    return .object(out)
}

// MARK: - Load

let clock = ContinuousClock()
let loadStart = clock.now
let tokenizer = try await SwiftTransformersTokenizer.load(
    from: TokenizerFiles(directory: snapshot))
let model = try await DiffusionGemmaModel.load(from: snapshot).model
Memory.cacheLimit = Int(cacheLimitGB * 1024 * 1024 * 1024)
let loadTime = clock.now - loadStart
let metallibSHA256 = hex(SHA256.hash(data: try Data(contentsOf: metallib)))
print("loaded in \(loadTime); metallib \(shown(metallib)) (SHA-256 \(metallibSHA256.prefix(16)))")

struct OracleFixture: Decodable {
    struct Generator: Decodable {
        let metallibSHA256: String
        enum CodingKeys: String, CodingKey { case metallibSHA256 = "metallib_sha256" }
    }
    struct Rope: Decodable {
        let float32Bits: [UInt32]
        enum CodingKeys: String, CodingKey { case float32Bits = "float32_bits" }
    }
    let generator: Generator
    let rope: Rope
}

let oracle = try JSONDecoder().decode(
    OracleFixture.self, from: Data(contentsOf: URL(fileURLWithPath: "Fixtures/oracle/reads.json")))
// The oracle's kernels: the metallib its generator ran, the mlx-metal wheel's.
let wheelKernels = metallibSHA256 == oracle.generator.metallibSHA256
// The proportional RoPE table: the port's own, computed with MLX pow at init, or the oracle's.
let oracleBits = oracle.rope.float32Bits
let fullLayers = model.decoder.layers.filter { $0.layerType == .fullAttention }
guard let ownTable = fullLayers.first?.selfAttention.fullAttentionFrequencies else {
    fatalError("the model has no full-attention RoPE table")
}
let ownBits = ownTable.asType(.float32).asArray(Float.self).map(\.bitPattern)
let ownDiffering = zip(ownBits, oracleBits).filter { $0 != $1 }.count
if oracleRope {
    let table = MLXArray(oracleBits.map { Float(bitPattern: $0) })
    for layer in fullLayers {
        layer.selfAttention.fullAttentionFrequencies = table
    }
}
print(
    "RoPE table: \(oracleRope ? "the oracle's" : "the port's own") in \(fullLayers.count) layers; "
        + "the port's own differs from the oracle's in \(ownDiffering) of \(ownBits.count) entries")
let memoryAfterLoad = processMemory()

// MARK: - Backend

/// One read the engine made, with the model's maps.
struct ReadRecord: Sendable {
    var read: CanvasRead
    var output: ReadOutput
    var prefillCached: Bool
}

/// The digests of one layer of a prefill cache.
struct LayerDigest: Sendable {
    var layer: Int
    var fullAttention: Bool
    var offset: Int
    var keys: TensorDigest
    var values: TensorDigest
    /// A sliding layer's decoder view: the last `window − 1` positions, from `start`.
    var view: (start: Int, keys: TensorDigest, values: TensorDigest)?
}

/// DiffusionGemmaRuntime's read path without its prefill cache's bounds (every prompt stays
/// cached), recording what it reads.
actor ModelBackend: DecisionBackend {
    nonisolated let tokenizer: any DecisionTokenizer
    nonisolated let maxPromptTokens = DiffusionGemmaRuntime.Configuration.default.maxPromptTokens
    nonisolated let capabilities = BackendCapabilities(
        steps: true, samples: true, think: false, sequential: true, images: false)
    nonisolated let modelName = ServedModels.diffusionGemmaVersion
    private let model: DiffusionGemmaModel
    private var prefills: [[Int]: PromptCache] = [:]
    private var records: [ReadRecord] = []

    init(tokenizer: any DecisionTokenizer, model: DiffusionGemmaModel) {
        self.tokenizer = tokenizer
        self.model = model
    }

    private func prefill(_ ids: [Int]) throws -> (cache: PromptCache, cached: Bool) {
        if let cache = prefills[ids] { return (cache, true) }
        let cache = try model.prefill(promptIDs: ids)
        prefills[ids] = cache
        return (cache, false)
    }

    /// DiffusionGemmaRuntime.read: the prompt cap, the prefill or a cached one, the model's read
    /// with the top 20, and each slot's map through slot_distribution.
    func read(_ read: CanvasRead) async throws -> ReadResult {
        guard case .tokens(let ids) = read.prompt else {
            throw DiffusionGemmaRuntimeError.unsupported("images")
        }
        if ids.count > maxPromptTokens {
            throw SchemaError("the request is \(ids.count) tokens; the limit is \(maxPromptTokens)")
        }
        let slots = read.slots.map { SlotRequest(position: $0.position, labelIDs: $0.labelIDs) }
        let (cache, cached) = try prefill(ids)
        let output = try model.read(
            canvas: read.canvas.tokens, slots: slots, cache: cache, steps: read.steps,
            topK: DiffusionGemmaRuntime.topK)
        records.append(ReadRecord(read: read, output: output, prefillCached: cached))
        return output.readResult(for: slots)
    }

    func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws -> ThoughtGeneration {
        throw DiffusionGemmaRuntimeError.unsupported("think")
    }

    /// A read with upstream's recorded inputs, through the same prefill cache.
    func replay(promptIDs: [Int], canvas: [Int], slots: [SlotRequest], steps: Int) throws
        -> (output: ReadOutput, prefillCached: Bool)
    {
        let (cache, cached) = try prefill(promptIDs)
        let output = try model.read(
            canvas: canvas, slots: slots, cache: cache, steps: steps,
            topK: DiffusionGemmaRuntime.topK)
        return (output, cached)
    }

    func takeRecords() -> [ReadRecord] {
        defer { records = [] }
        return records
    }

    /// Every layer of the prompt's prefill cache, as item_reads.py digests upstream's.
    func digests(promptIDs: [Int], slidingWindow: Int) throws -> [LayerDigest] {
        let cache = try prefill(promptIDs).cache
        return cache.layers.enumerated().map { layer, entry in
            guard let digest = entry.digest, let keys = entry.keys else {
                fatalError("layer \(layer) of the prefill cache is empty")
            }
            var view: (start: Int, keys: TensorDigest, values: TensorDigest)?
            if !entry.isFullAttention,
                let decoder = entry.decoderViewDigest(slidingWindow: slidingWindow)
            {
                view = (max(keys.dim(2) - (slidingWindow - 1), 0), decoder.keys, decoder.values)
            }
            return LayerDigest(
                layer: layer, fullAttention: entry.isFullAttention, offset: entry.offset,
                keys: digest.keys, values: digest.values, view: view)
        }
    }
}

// MARK: - JSON

func json(_ ids: [Int]) -> JSONValue { .array(ids.map(JSONValue.init)) }

/// The SHA-256 of `json.dumps(ids)`, as item_reads.py digests a prompt.
func idsDigest(_ ids: [Int]) throws -> String {
    hex(SHA256.hash(data: try PythonJSONWriter().bytes(json(ids))))
}

func json(_ digest: TensorDigest) -> JSONValue {
    [
        "dtype": .string(digest.dtype), "shape": json(digest.shape),
        "sha256": .string(digest.sha256), "sum": .float(digest.sum),
        "sum_of_squares": .float(digest.sumOfSquares),
    ]
}

func json(_ layers: [LayerDigest]) -> JSONValue {
    .array(
        layers.map { layer in
            var row: JSONObject = [
                "layer": JSONValue(layer.layer),
                "kind": .string(layer.fullAttention ? "full_attention" : "sliding_attention"),
                "offset": JSONValue(layer.offset), "keys": json(layer.keys),
                "values": json(layer.values),
            ]
            if let view = layer.view {
                row["decoder_view"] = [
                    "start": JSONValue(view.start), "keys": json(view.keys),
                    "values": json(view.values),
                ]
            }
            return .object(row)
        })
}

/// Each slot's map: `[token id, logprob, the logprob's float32 bits]`, in token id order.
func json(_ output: ReadOutput) -> JSONObject {
    [
        "prompt_tokens": JSONValue(output.promptTokens),
        "written": .array(output.written.map { json($0) }),
        "logprobs": .array(
            output.slots.map { map in
                .array(
                    map.map {
                        JSONValue.array([
                            JSONValue($0.tokenID), .float($0.logprob),
                            JSONValue(Float($0.logprob).bitPattern),
                        ])
                    })
            }),
    ]
}

func json(_ slots: [SlotRequest]) -> JSONValue {
    .array(
        slots.map {
            JSONValue.object(["pos": JSONValue($0.position), "label_ids": json($0.labelIDs)])
        })
}

// MARK: - Run

struct Body: Decodable {
    let id: String
    let dataset: String
    let bodyText: String
    enum CodingKeys: String, CodingKey {
        case id, dataset
        case bodyText = "body_text"
    }
}

struct UpstreamRun: Decodable {
    struct Slot: Decodable {
        let pos: Int
        let labelIDs: [Int]
        enum CodingKeys: String, CodingKey {
            case pos
            case labelIDs = "label_ids"
        }
    }
    struct Read: Decodable {
        let promptIDs: [Int]
        let canvas: [Int]
        let slots: [Slot]
        let steps: Int
        enum CodingKeys: String, CodingKey {
            case canvas, slots, steps
            case promptIDs = "prompt_ids"
        }
    }
    struct Item: Decodable {
        let id: String
        let reads: [Read]
    }
    let items: [Item]
}

let backend = ModelBackend(tokenizer: tokenizer, model: model)
let window = model.configuration.slidingWindow
var items: [JSONValue] = []
var replays: [JSONValue] = []

func cachesJSON(_ prompts: [[Int]]) async throws -> JSONValue {
    var rows: [JSONValue] = []
    for ids in prompts {
        let layers = try await backend.digests(promptIDs: ids, slidingWindow: window)
        rows.append(["prompt_ids_sha256": .string(try idsDigest(ids)), "layers": json(layers)])
    }
    return .array(rows)
}

/// The distinct prompts, in first-use order.
func distinct(_ prompts: [[Int]]) -> [[Int]] {
    var seen: Set<[Int]> = []
    return prompts.filter { seen.insert($0).inserted }
}

if runEngine {
    let bodies = try JSONDecoder().decode(
        [Body].self, from: Data(contentsOf: work.appending(path: "bodies.json")))
    let engine = try DecisionEngine(backend: backend, configuration: .default)
    for body in bodies {
        let request = try RequestValidator().validate(try JSONParser().parse(body.bodyText))
        let started = clock.now
        let decision = try await engine.decide(request)
        let records = await backend.takeRecords()
        var answers = JSONObject()
        for (key, answer) in decision.answers {
            answers[key] = answer.json
        }
        let prompts = distinct(
            records.map { record -> [Int] in
                guard case .tokens(let ids) = record.read.prompt else { return [] }
                return ids
            })
        let reads: [JSONValue] = records.map { record in
            var row = json(record.output)
            if case .tokens(let ids) = record.read.prompt { row["prompt_ids"] = json(ids) }
            row["seed"] = JSONValue(record.read.seed)
            row["canvas"] = json(record.read.canvas.tokens)
            row["slots"] = json(
                record.read.slots.map { SlotRequest(position: $0.position, labelIDs: $0.labelIDs) })
            row["steps"] = JSONValue(record.read.steps)
            row["prefill_cached"] = .bool(record.prefillCached)
            return .object(row)
        }
        print(
            "\(body.id): \(records.count) reads, prompt tokens "
                + "\(Set(records.map(\.output.promptTokens)).sorted()), \(clock.now - started), "
                + "answers \(try PythonJSONWriter().string(.object(answers)).prefix(200))")
        items.append([
            "id": .string(body.id), "dataset": .string(body.dataset), "answers": .object(answers),
            "usage": [
                "input_tokens": JSONValue(decision.inputTokens),
                "output_tokens": JSONValue(decision.outputTokens),
            ],
            "reads": .array(reads), "caches": try await cachesJSON(prompts),
        ])
    }
}

if runReplay {
    let upstream = try JSONDecoder().decode(
        UpstreamRun.self, from: Data(contentsOf: work.appending(path: "upstream.json")))
    for item in upstream.items {
        var reads: [JSONValue] = []
        for read in item.reads {
            let (output, cached) = try await backend.replay(
                promptIDs: read.promptIDs, canvas: read.canvas,
                slots: read.slots.map { SlotRequest(position: $0.pos, labelIDs: $0.labelIDs) },
                steps: read.steps)
            var row = json(output)
            row["prefill_cached"] = .bool(cached)
            reads.append(.object(row))
        }
        replays.append([
            "id": .string(item.id), "reads": .array(reads),
            "caches": try await cachesJSON(distinct(item.reads.map(\.promptIDs))),
        ])
        print("\(item.id): replayed \(item.reads.count) of upstream's reads")
    }
}

let configuration =
    switch (wheelKernels, oracleRope) {
    case (true, true): "exact"
    case (true, false): "wheel_metallib"
    case (false, true): "oracle_rope"
    case (false, false): "native"
    }
let dateFormatter = ISO8601DateFormatter()
dateFormatter.formatOptions = [.withFullDate]
dateFormatter.timeZone = .current
let result: JSONValue = [
    "run_on": .string(dateFormatter.string(from: Date())),
    "configuration": .string(configuration),
    "metallib": ["path": .string(shown(metallib)), "sha256": .string(metallibSHA256)],
    "rope_table": .string(oracleRope ? "oracle" : "own"),
    "own_rope_entries_differing_from_oracle": JSONValue(ownDiffering),
    "cache_limit_gb": .float(cacheLimitGB),
    "model": .string(shown(snapshot)),
    "load_seconds": .float(
        Double(loadTime.components.seconds) + Double(loadTime.components.attoseconds) / 1e18),
    "memory": ["after_load": memoryAfterLoad, "at_end": processMemory()],
    "items": .array(items), "replays": .array(replays),
]
let out =
    outPath.map { URL(fileURLWithPath: $0) } ?? work.appending(path: "port_\(configuration).json")
try FileManager.default.createDirectory(
    at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
try Data(try PythonJSONWriter().bytes(result)).write(to: out)
print("wrote \(out.path)")
