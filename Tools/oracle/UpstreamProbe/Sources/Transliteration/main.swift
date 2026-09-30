// Runs every read of Fixtures/oracle/reads.json through the transliteration in Model.swift, the
// way upstream's MlxRuntime.read does (mlx_backend.py:185-208), and compares the result with the
// oracle: bit for bit first, then with the Swift probe's metrics. Also compares the prefill cache
// of the first and last layer with the oracle's digests.
//
//     swift run --package-path Tools/oracle/UpstreamProbe -c release Transliteration [--cache-limit-gb 4]
//
// Without --out the result goes to a file under Tools/oracle/results named after the options
// (see defaultResultPath), never over another configuration's file.

import CryptoKit
import Darwin
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import OpenJevCore

struct TensorDigest: Decodable {
    let sha256: String
    let sumOfSquares: Double
    enum CodingKeys: String, CodingKey {
        case sha256
        case sumOfSquares = "sum_of_squares"
    }
}

struct CacheView: Decodable {
    let keys: TensorDigest
    let values: TensorDigest
}

struct CacheDigest: Decodable {
    let layer: Int
    let keys: TensorDigest
    let values: TensorDigest
}

struct Prompt: Decodable {
    let ids: [Int]
    let cache: [CacheDigest]?
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
    let canvas: [Int]
    let slots: [Slot]
    let steps: Int
    let written: [[Int]]
    let logprobs: [[[Double]]]
    let distributions: [Distribution]
}

struct RopeTable: Decodable {
    let float32Bits: [UInt32]
    enum CodingKeys: String, CodingKey { case float32Bits = "float32_bits" }
}

struct Oracle: Decodable {
    let prompts: [String: Prompt]
    let reads: [Read]
    let rope: RopeTable?
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

func processMemory() -> [String: Int] {
    var info = rusage_info_v4()
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
        }
    }
    var out: [String: Int] = [
        "mlx_active_bytes": Memory.activeMemory, "mlx_cache_bytes": Memory.cacheMemory,
        "mlx_peak_bytes": Memory.peakMemory,
    ]
    if status == 0 {
        out["resident_bytes"] = Int(info.ri_resident_size)
        out["phys_footprint_bytes"] = Int(info.ri_phys_footprint)
        out["lifetime_max_phys_footprint_bytes"] = Int(info.ri_lifetime_max_phys_footprint)
    }
    return out
}

func sha256(_ array: MLXArray) -> (String, Double) {
    let bits = array.view(dtype: .uint16).asArray(UInt16.self)
    let digest = bits.withUnsafeBufferPointer { SHA256.hash(data: UnsafeRawBufferPointer($0)) }
    let squares = array.asType(.float32).asArray(Float.self).reduce(0.0) { $0 + Double($1) * Double($1) }
    return (digest.map { String(format: "%02x", $0) }.joined(), squares)
}

let topK = 20

/// mlx_backend.py:203-207.
func slotLogprobs(_ logits: MLXArray, _ slot: Slot) -> [[Double]] {
    let row = logits[0, slot.pos].asType(.float32)
    let logprobs = row - logSumExp(row)
    let top = argPartition(-logprobs, kth: topK)[0 ..< topK].asType(.int32).asArray(Int32.self)
    let keep = Array(Set(top.map(Int.init)).union(slot.labelIDs)).sorted()
    let values = logprobs[MLXArray(keep.map(Int32.init))].asArray(Float.self)
    return zip(keep, values).map { [Double($0), Double($1)] }
}

// MARK: - Options

var oraclePath = "Fixtures/oracle/reads.json"
var cacheLimitGB: Double?
var outPath: String?
var metallibOverride: String?
var stagesFile: String?
var freqsFrom: String?
var oracleRope = false
var summaryOnly = false
var stagesPrompt: String?
var index = 1
let arguments = CommandLine.arguments
while index < arguments.count {
    switch arguments[index] {
    case "--oracle": index += 1; oraclePath = arguments[index]
    case "--cache-limit-gb": index += 1; cacheLimitGB = Double(arguments[index])
    case "--out": index += 1; outPath = arguments[index]
    case "--stages":
        // --stages FILE PROMPT: compare this prompt's prefill stage by stage with FILE, written
        // by Tools/oracle/stage_dump.py, then stop.
        stagesFile = arguments[index + 1]
        stagesPrompt = arguments[index + 2]
        index += 2
    case "--summary-only":
        // Leave the per-read slot maps and written argmaxes out of the result file (the exact
        // tier's equal the oracle's; a cache-limited run's equal its unlimited run's).
        summaryOnly = true
    case "--oracle-rope":
        // Use the proportional RoPE table recorded in the oracle (reads.json "rope").
        oracleRope = true
    case "--freqs-from":
        // Use the proportional RoPE frequencies mlx-vlm computed (a5.freqs in a stage dump
        // written by Tools/oracle/stage_dump.py) instead of this process's own pow.
        index += 1
        freqsFrom = arguments[index]
    case "--metallib":
        // Load the Metal kernels from this file (for example the Python mlx-metal wheel's
        // mlx.metallib) instead of the library SwiftPM compiled. Must precede the first MLX call.
        index += 1
        GPU.metallib = URL(fileURLWithPath: arguments[index])
        metallibOverride = arguments[index]
    default: fatalError("unknown argument \(arguments[index])")
    }
    index += 1
}
let environment = ProcessInfo.processInfo.environment
let hub = environment["HF_HUB_CACHE"] ?? (NSHomeDirectory() + "/.cache/huggingface/hub")
let directory = URL(
    fileURLWithPath: environment["OPENJEV_TEST_MODEL"]
        ?? (hub + "/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots/"
            + "a7a81407613811e8ba63af92ac0d852b809e191f"))

// MARK: - Load

let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: URL(fileURLWithPath: oraclePath)))
var memory: [String: [String: Int]] = ["before_load": processMemory()]
let loadStart = now()
let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
let root = try JSONDecoder().decode(RootConfiguration.self, from: configData)
let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
let model = DiffusionGemmaReference(root.textConfig)
try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
model.train(false)
if oracleRope {
    guard let bits = oracle.rope?.float32Bits else { fatalError("the oracle has no rope table") }
    let table = MLXArray(bits.map { Float(bitPattern: $0) })
    for layer in model.decoder.layers where layer.layerType == "full_attention" {
        layer.attention.freqs!.value = table
    }
    print("proportional RoPE frequencies taken from the oracle")
}
if let freqsFrom {
    let table = try loadArrays(url: URL(fileURLWithPath: freqsFrom))["a5.freqs"]!
    for layer in model.decoder.layers where layer.layerType == "full_attention" {
        layer.attention.freqs!.value = table
    }
    print("proportional RoPE frequencies taken from \(freqsFrom)")
}
let loadSeconds = now() - loadStart
if let cacheLimitGB { Memory.cacheLimit = Int(cacheLimitGB * 1024 * 1024 * 1024) }
memory["after_load"] = processMemory()
print(String(format: "loaded in %.2f s, embedding %@", loadSeconds, String(describing: type(of: model.decoder.embedTokens))))

// MARK: - Stage comparison

func reference(_ path: String) -> [String: MLXArray] {
    (try? loadArrays(url: URL(fileURLWithPath: path))) ?? [:]
}

if let stagesFile, let stagesPrompt {
    var recorded: [(String, MLXArray)] = []
    stageRecorder = { recorded.append(($0, $1)) }
    let ids = oracle.prompts[stagesPrompt]!.ids
    let caches = model.prefill(MLXArray(ids.map(Int32.init)).reshaped(1, ids.count))
    stageRecorder = nil
    recorded.append(("a5.freqs", model.decoder.layers[5].attention.freqs!.value))
    do {
        let exponents = MLXArray(stride(from: 0, to: 128, by: 2)).asType(.float32) / Float(512)
        let gpu = pow(MLXArray(Float(1_000_000)), exponents)
        let cpu = pow(MLXArray(Float(1_000_000)), exponents, stream: .cpu)
        let pythonGPU = reference(stagesFile)["a5.freqs"]!
        func bits(_ a: MLXArray) -> [UInt32] { a.asArray(Float.self).prefix(8).map(\.bitPattern) }
        print("freqs in the model  ", bits(model.decoder.layers[5].attention.freqs!.value))
        print("freqs recomputed gpu", bits(gpu), "cpu", bits(cpu))
        print("freqs python gpu    ", bits(pythonGPU))
        print("exponents dtype \(exponents.dtype), default device \(Device.defaultDevice())")
        let mine = model.decoder.layers[5].attention.freqs!.value
        print("shapes", mine.shape, pythonGPU.shape, "dtypes", mine.dtype, pythonGPU.dtype)
        let a = mine.asArray(Float.self)
        let b = pythonGPU.asArray(Float.self)
        for i in 0 ..< min(a.count, b.count) where a[i].bitPattern != b[i].bitPattern {
            print("  index \(i): swift \(a[i]) (\(a[i].bitPattern)) python \(b[i]) (\(b[i].bitPattern))")
        }
    }
    for i in [0, caches.count - 1] {
        recorded.append(("cache.\(i).keys", caches[i].keys!))
        recorded.append(("cache.\(i).values", caches[i].values!))
    }
    eval(recorded.map(\.1))
    let reference = try loadArrays(url: URL(fileURLWithPath: stagesFile))
    var firstMismatch: String?
    var exactCount = 0
    for (name, got) in recorded {
        guard let want = reference[name] else { print("missing in reference: \(name)"); continue }
        let same = got.shape == want.shape && got.dtype == want.dtype && arrayEqual(got, want).item(Bool.self)
        if same { exactCount += 1 } else if firstMismatch == nil { firstMismatch = name }
        if name.hasPrefix("a5.") || name.hasPrefix("a0.") { print("  \(name): \(same ? "identical" : "DIFFERS")") }
        if !same && (name.hasSuffix(".0") || name.contains(".0.") || name == firstMismatch) {
            let difference = abs(got.asType(.float32) - want.asType(.float32))
            print("  \(name) differs: shape \(got.shape) vs \(want.shape), dtype \(got.dtype) vs \(want.dtype), max |d| \(difference.max().item(Float.self)), elements differing \((difference .> 0).sum().item(Int.self)) of \(got.size)")
        }
    }
    print("stages bit-identical: \(exactCount)/\(recorded.count); first mismatch in execution order: \(firstMismatch ?? "none")")
    exit(0)
}

// MARK: - Reads

struct Result {
    var logprobs: [[[Double]]]
    var written: [[Int]]
    var seconds: Double
    var prefillSeconds: Double
    var cached: Bool
}

var prefills: [String: [LayerCache]] = [:]
var cacheChecks: [[String: Any]] = []

func run(_ read: Read, checkCache: Bool) -> Result {
    let prompt = oracle.prompts[read.prompt]!
    let started = now()
    var prefillSeconds = 0.0
    var cached = true
    let caches: [LayerCache]
    if let hit = prefills[read.prompt] {
        caches = hit
    } else {
        cached = false
        caches = model.prefill(MLXArray(prompt.ids.map(Int32.init)).reshaped(1, prompt.ids.count))
        eval(caches.flatMap { [$0.keys!, $0.values!] })
        prefillSeconds = now() - started
        prefills[read.prompt] = caches
        if checkCache, let digests = prompt.cache {
            for want in digests {
                let (keyHash, keySquares) = sha256(caches[want.layer].keys!)
                let (valueHash, valueSquares) = sha256(caches[want.layer].values!)
                cacheChecks.append([
                    "prompt": read.prompt, "layer": want.layer,
                    "keys_sha256_equal": keyHash == want.keys.sha256,
                    "values_sha256_equal": valueHash == want.values.sha256,
                    "keys_relative_sum_of_squares_difference":
                        abs(keySquares - want.keys.sumOfSquares) / want.keys.sumOfSquares,
                    "values_relative_sum_of_squares_difference":
                        abs(valueSquares - want.values.sumOfSquares) / want.values.sumOfSquares,
                ])
            }
        }
    }
    var canvas = MLXArray(read.canvas.map(Int32.init)).reshaped(1, read.canvas.count)
    let masks = model.decoderMasks(canvasLength: read.canvas.count, caches: caches)
    let positions = MLXArray(read.slots.map { Int32($0.pos) })
    var conditioning: MLXArray?
    var logits = MLXArray(0)
    var written: [[Int]] = []
    for step in 0 ..< read.steps {
        logits = model.decoderLogits(canvas, caches: caches, selfConditioningLogits: conditioning, masks: masks)
        if step + 1 == read.steps { break }
        canvas[0, positions] = argMax(logits[0, positions], axis: -1).asType(canvas.dtype)
        conditioning = logits  // diffusion_self_conditioning with a quantized embedding
        eval(canvas, logits)
        written.append(read.slots.map { canvas[0, $0.pos].item(Int.self) })
    }
    let maps = read.slots.map { slotLogprobs(logits, $0) }
    return Result(
        logprobs: maps, written: written, seconds: now() - started, prefillSeconds: prefillSeconds,
        cached: cached)
}

// Warm-up: kernels compile on the first read.
_ = run(oracle.reads[0], checkCache: false)
prefills = [:]

var results: [[String: Result]] = []
for pass in 0 ..< 2 {
    prefills = [:]
    var out: [String: Result] = [:]
    let order = pass == 0 ? Array(oracle.reads.indices) : Array(oracle.reads.indices.reversed())
    for i in order { out[oracle.reads[i].id] = run(oracle.reads[i], checkCache: pass == 0) }
    results.append(out)
    memory["after_pass_\(pass + 1)"] = processMemory()
}

// MARK: - Compare

var exact = 0
var rows: [[String: Any]] = []
var maxProbability = 0.0
var maxEntropy = 0.0
var agree = 0
var slots = 0
var deterministic = true
print("")
print("read                                 bit-exact  max|dp|    top    max|dH|   total ms")
for read in oracle.reads {
    let got = results[0][read.id]!
    if got.logprobs != results[1][read.id]!.logprobs || got.written != results[1][read.id]!.written {
        deterministic = false
    }
    let bitExact = got.logprobs == read.logprobs && got.written == read.written
    exact += bitExact ? 1 : 0
    var readProbability = 0.0
    var readEntropy = 0.0
    var readAgree = 0
    for (index, slot) in read.slots.enumerated() {
        let pairs = got.logprobs[index].map { (tokenID: Int($0[0]), logprob: $0[1]) }
        let mine = SlotDistribution.compute(top: pairs, labelIDs: slot.labelIDs)
        let theirs = read.distributions[index]
        for (a, b) in zip(mine.probabilities, theirs.probs) { readProbability = max(readProbability, abs(a - b)) }
        readEntropy = max(readEntropy, abs(mine.entropy - theirs.entropy))
        let top = mine.probabilities.indices.max { mine.probabilities[$0] < mine.probabilities[$1] }!
        let theirTop = theirs.probs.indices.max { theirs.probs[$0] < theirs.probs[$1] }!
        readAgree += top == theirTop ? 1 : 0
    }
    maxProbability = max(maxProbability, readProbability)
    maxEntropy = max(maxEntropy, readEntropy)
    agree += readAgree
    slots += read.slots.count
    let other = results[1][read.id]!
    var row: [String: Any] = [
        "id": read.id, "bit_exact": bitExact, "max_probability_difference": readProbability,
        "max_entropy_difference": readEntropy, "top_label_agreement": readAgree, "slots": read.slots.count,
        "written_equal": got.written == read.written,
        "timing_pass_1": ["seconds": got.seconds, "prefill_seconds": got.prefillSeconds, "prefill_cached": got.cached],
        "timing_pass_2": ["seconds": other.seconds, "prefill_seconds": other.prefillSeconds, "prefill_cached": other.cached],
    ]
    if !summaryOnly {
        row["logprobs"] = got.logprobs
        row["written"] = got.written
    }
    rows.append(row)
    print(
        read.id.padding(toLength: 36, withPad: " ", startingAt: 0),
        bitExact ? "yes      " : "NO       ",
        String(format: "%.2e  %2d/%-2d  %.2e  %8.1f", readProbability, readAgree, read.slots.count, readEntropy, got.seconds * 1000))
}
print("")
print("bit-exact reads: \(exact)/\(oracle.reads.count); top label \(agree)/\(slots); max |dp| \(maxProbability); max |dH| \(maxEntropy)")
print("deterministic across two passes: \(deterministic)")
let hashesEqual = cacheChecks.filter { ($0["keys_sha256_equal"] as! Bool) && ($0["values_sha256_equal"] as! Bool) }.count
print("prefill cache digests equal (layers 0 and 29): \(hashesEqual)/\(cacheChecks.count)")
for check in cacheChecks where !((check["keys_sha256_equal"] as! Bool) && (check["values_sha256_equal"] as! Bool)) {
    print("  differs:", check)
}
let gib = Double(1 << 30)
for (key, value) in memory.sorted(by: { $0.key < $1.key }) {
    print(key.padding(toLength: 13, withPad: " ", startingAt: 0),
        String(format: "footprint %.2f GiB, mlx active %.2f, cache %.2f, peak %.2f GiB",
            Double(value["phys_footprint_bytes"] ?? 0) / gib, Double(value["mlx_active_bytes"] ?? 0) / gib,
            Double(value["mlx_cache_bytes"] ?? 0) / gib, Double(value["mlx_peak_bytes"] ?? 0) / gib))
}
let summary: [String: Any] = [
    "stack": "ml-explore/mlx-swift 0.32.2, ml-explore/mlx-swift-lm c043fb3",
    "cache_limit_gb": cacheLimitGB as Any, "metallib": metallibRecord() as Any,
    "rope_frequencies_from": oracleRope ? "oracle" : (freqsFrom as Any),
    "load_seconds": loadSeconds, "memory": memory,
    "bit_exact_reads": exact, "reads": oracle.reads.count, "deterministic": deterministic,
    "top_label_agreement": agree, "slots": slots, "max_probability_difference": maxProbability,
    "max_entropy_difference": maxEntropy, "cache_checks": cacheChecks, "per_read": rows,
]
/// The --metallib file as the result records it: its path relative to the working directory when
/// it lies inside it (never an absolute home path), and its SHA-256, which the oracle's generator
/// also records.
func metallibRecord() -> [String: String]? {
    guard let metallibOverride else { return nil }
    let url = URL(fileURLWithPath: metallibOverride).standardizedFileURL
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).standardizedFileURL.path + "/"
    let shown = url.path.hasPrefix(root) ? String(url.path.dropFirst(root.count)) : url.lastPathComponent
    let hash = (try? Data(contentsOf: url)).map { data in
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    return ["path": shown, "sha256": hash ?? "unreadable"]
}

/// The default result file names the configuration, so that no run overwrites another's: only a
/// run on mlx-swift's own kernels with nothing overridden writes transliteration_run.json, the
/// native baseline tolerance_stats.py reads, and only the wheel's metallib together with the
/// oracle's RoPE table writes transliteration_run_exact.json.
func defaultResultPath() -> String {
    let table = oracleRope || freqsFrom != nil
    var name = "transliteration_run"
    switch (metallibOverride != nil, table) {
    case (true, true): name += "_exact"
    case (true, false): name += "_wheel_metallib"
    case (false, true): name += "_oracle_rope"
    case (false, false): break
    }
    if !plantedBug.isEmpty { name += "_planted_bug_\(plantedBug)" }
    if let cacheLimitGB {
        let size = cacheLimitGB.rounded() == cacheLimitGB ? String(Int(cacheLimitGB)) : String(cacheLimitGB)
        name += "_cache_limit_\(size)gb"
    }
    return "Tools/oracle/results/\(name).json"
}

let path = outPath ?? defaultResultPath()
try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
    .write(to: URL(fileURLWithPath: path))
print("wrote \(path)")
