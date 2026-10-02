// Spike #22: the read-only pass in Swift through the Layr-Labs fork, compared with the mlx-vlm
// oracle in Fixtures/oracle/reads.json.
//
// For every oracle read: encode(tokenIds:cache:) with the recorded prompt ids, then
// denoise(canvasIds:cache:selfConditioningLogits:) with the recorded canvas; float32
// log-softmax at each slot, the top 20 plus every label kept as upstream's MlxRuntime.read
// keeps them, and the label probabilities and entropy from OpenJevCore's SlotDistribution (the
// fixture-tested port of upstream's slot_distribution). For steps above 1 the loop is
// upstream's: argmax written back at the slot positions only, the previous logits passed as
// self-conditioning.
//
// Run from the repository root, after the oracle exists:
//
//     swift run --package-path Tools/oracle/Probe -c release Probe
//     swift run --package-path Tools/oracle/Probe -c release Probe --cache-limit-gb 4
//     swift run --package-path Tools/oracle/Probe -c release Probe --cache-limit-gb 4 --maps FILE

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import OpenJevCore

// MARK: - Options

struct Options {
    var oracle = "Fixtures/oracle/reads.json"
    var model = Options.defaultModel()
    var cacheLimitGB: Double?
    var out: String?
    var passes = 2
    /// With --dump DIR, the prefill caches and Swift slot maps of --dump-reads go to DIR, for
    /// Tools/oracle/crossfeed.py to run mlx-vlm's decoder on the fork's encoder output.
    var dumpDirectory: String?
    var dumpReads: [String] = []
    /// With --maps FILE, every read's slot maps and written argmaxes from the first pass go to
    /// FILE, as swift_reads.json holds them, for Tools/oracle/tolerance_stats.py; no cache is dumped.
    var maps: String?

    static func defaultModel() -> String {
        let environment = ProcessInfo.processInfo.environment
        let hub =
            environment["HF_HUB_CACHE"]
            ?? (environment["HF_HOME"].map { $0 + "/hub" })
            ?? (NSHomeDirectory() + "/.cache/huggingface/hub")
        return hub
            + "/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots/"
            + "a7a81407613811e8ba63af92ac0d852b809e191f"
    }

    static func parse(_ arguments: [String]) -> Options {
        var options = Options()
        var index = 1
        func value() -> String {
            index += 1
            guard index < arguments.count else { fatalError("\(arguments[index - 1]) needs a value") }
            return arguments[index]
        }
        while index < arguments.count {
            switch arguments[index] {
            case "--oracle": options.oracle = value()
            case "--model": options.model = value()
            case "--cache-limit-gb": options.cacheLimitGB = Double(value())
            case "--out": options.out = value()
            case "--passes": options.passes = Int(value()) ?? 2
            case "--dump": options.dumpDirectory = value()
            case "--dump-reads": options.dumpReads = value().split(separator: ",").map(String.init)
            case "--maps": options.maps = value()
            default: fatalError("unknown argument \(arguments[index])")
            }
            index += 1
        }
        return options
    }
}

// MARK: - Loading

/// The factory wants a tokenizer; a read never tokenizes (the prompt ids come from the oracle),
/// so this one only answers the factory's load. See the spike report for why the main
/// package's SwiftTransformersTokenizer cannot be linked here.
struct UnusedTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}

struct UnusedTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { UnusedTokenizer() }
}

// MARK: - One read

struct ProbeRead {
    /// Per slot, (token id, logprob) for the top 20 plus every label, in token id order.
    var slots: [[(tokenID: Int, logprob: Double)]]
    /// Per slot, the logprobs of the oracle's token ids, for an entropy over the same set.
    var onOracleSet: [[Double]]
    var written: [[Int]]
    var encodeSeconds: Double
    var stepSeconds: [Double]
    var totalSeconds: Double
    var prefillCached: Bool
}

let topK = 20

/// Upstream's per-slot extraction (mlx_backend.py:203-207): float32 row, log-softmax by
/// logsumexp, the argpartition top 20 united with the label ids, sorted by token id.
func slotLogprobs(_ logits: MLXArray, slot: OracleSlot, oracleIDs: [Int])
    -> ([(tokenID: Int, logprob: Double)], [Double])
{
    let row = logits[0, slot.pos].asType(.float32)
    let logprobs = row - logSumExp(row)
    let top = argPartition(-logprobs, kth: topK)[0 ..< topK].asType(.int32).asArray(Int32.self)
    let keep = Array(Set(top.map(Int.init)).union(slot.labelIDs)).sorted()
    let values = logprobs[MLXArray(keep.map(Int32.init))].asArray(Float.self)
    let same = logprobs[MLXArray(oracleIDs.map(Int32.init))].asArray(Float.self)
    return (zip(keep, values).map { (tokenID: $0, logprob: Double($1)) }, same.map(Double.init))
}

final class Reader {
    let model: DiffusionGemma
    let oracle: OracleFile
    var prefills: [String: DiffusionGemmaRequestCache] = [:]
    var digests: [String: [[String: Any]]] = [:]

    init(model: DiffusionGemma, oracle: OracleFile) {
        self.model = model
        self.oracle = oracle
    }

    func read(_ read: OracleRead, digest: Bool) throws -> ProbeRead {
        let prompt = oracle.prompts[read.prompt]!
        let started = now()
        var encodeSeconds = 0.0
        let cache: DiffusionGemmaRequestCache
        let cached: Bool
        if let hit = prefills[read.prompt] {
            cache = hit
            cached = true
        } else {
            let ids = MLXArray(prompt.ids.map(Int32.init)).reshaped(1, prompt.ids.count)
            cache = try model.makeCache(expectedPromptLength: prompt.ids.count)
            _ = try model.encode(tokenIds: ids, cache: cache)
            eval(cache.stateArrays())
            encodeSeconds = now() - started
            prefills[read.prompt] = cache
            cached = false
            if digest, digests[read.prompt] == nil {
                digests[read.prompt] = compareCache(cache, prompt: prompt)
            }
        }

        var canvas = MLXArray(read.canvas.map(Int32.init)).reshaped(1, read.canvas.count)
        let positions = MLXArray(read.slots.map { Int32($0.pos) })
        var conditioning: MLXArray?
        var logits = MLXArray(0)
        var written: [[Int]] = []
        var stepSeconds: [Double] = []
        var stepStart = now()
        for step in 0 ..< read.steps {
            stepStart = now()
            logits = try model.denoise(
                canvasIds: canvas, cache: cache, selfConditioningLogits: conditioning)
            if step + 1 == read.steps { break }
            // upstream: ids[0, pos] = argmax(logits[0, pos]); sc = logits (quantized embedding)
            canvas[0, positions] = argMax(logits[0, positions], axis: -1).asType(.int32)
            conditioning = logits
            eval(canvas, logits)
            stepSeconds.append(now() - stepStart)
            written.append(canvas[0, positions].asArray(Int32.self).map(Int.init))
        }
        var slots: [[(tokenID: Int, logprob: Double)]] = []
        var onOracleSet: [[Double]] = []
        for (index, slot) in read.slots.enumerated() {
            let oracleIDs = read.logprobs[index].map { Int($0[0]) }
            let (pairs, same) = slotLogprobs(logits, slot: slot, oracleIDs: oracleIDs)
            slots.append(pairs)
            onOracleSet.append(same)
        }
        let finished = now()
        stepSeconds.append(finished - stepStart)
        return ProbeRead(
            slots: slots, onOracleSet: onOracleSet, written: written,
            encodeSeconds: encodeSeconds, stepSeconds: stepSeconds,
            totalSeconds: finished - started, prefillCached: cached)
    }

    /// The first and last layer of the fork's request cache against the oracle's digests of
    /// mlx-vlm's prefill cache. The fork's sliding rows hold a ring of 1,024 positions, so a
    /// sliding layer is compared on the decoder's view (the last 1,023) when the prompt is longer.
    func compareCache(_ cache: DiffusionGemmaRequestCache, prompt: OraclePrompt) -> [[String: Any]] {
        guard let expected = prompt.cache else { return [] }
        let snapshots = cache.snapshots()
        var out: [[String: Any]] = []
        for want in expected {
            let snapshot = snapshots[want.layer]
            var keys = snapshot.keys
            var values = snapshot.values
            var wantKeys = want.keys
            var wantValues = want.values
            var positions = want.positions
            if let view = want.decoderView {
                let count = view.positions[1] - view.positions[0]
                let start = keys.dim(2) - count
                keys = keys[.ellipsis, start..., 0...]
                values = values[.ellipsis, start..., 0...]
                wantKeys = view.keys
                wantValues = view.values
                positions = view.positions
            }
            let gotKeys = tensorDigest(keys)
            let gotValues = tensorDigest(values)
            func compare(_ got: TensorDigest, _ want: TensorDigest) -> [String: Any] {
                [
                    "shape_equal": got.shape == want.shape,
                    "sha256_equal": got.sha256 == want.sha256,
                    "sum": got.sum, "oracle_sum": want.sum,
                    "sum_of_squares": got.sumOfSquares, "oracle_sum_of_squares": want.sumOfSquares,
                    "relative_sum_of_squares_difference":
                        abs(got.sumOfSquares - want.sumOfSquares) / max(want.sumOfSquares, 1e-30),
                    "max_abs": got.maxAbs, "oracle_max_abs": want.maxAbs,
                ]
            }
            out.append([
                "layer": want.layer, "kind": want.kind, "positions": positions,
                "fork_retained": snapshot.keys.dim(2), "offset": snapshot.offset,
                "keys": compare(gotKeys, wantKeys), "values": compare(gotValues, wantValues),
            ])
        }
        return out
    }
}

// MARK: - Comparison

struct SlotComparison {
    var maxProbabilityDifference: Double
    var topLabelAgrees: Bool
    var entropyDifference: Double
    var entropyDifferenceOnOracleSet: Double
    var maxLabelLogprobDifference: Double
    var top20Overlap: Int
}

func compareSlot(
    swift pairs: [(tokenID: Int, logprob: Double)], onOracleSet: [Double],
    oracle: [(tokenID: Int, logprob: Double)], distribution: OracleDistribution, labels: [Int]
) -> SlotComparison {
    let mine = SlotDistribution.compute(top: pairs, labelIDs: labels)
    let theirs = distribution.probs
    var maxDifference = 0.0
    for (a, b) in zip(mine.probabilities, theirs) { maxDifference = max(maxDifference, abs(a - b)) }
    func argmax(_ values: [Double]) -> Int { values.indices.max { values[$0] < values[$1] }! }
    let sameSet = SlotDistribution.compute(
        top: zip(oracle.map(\.tokenID), onOracleSet).map { (tokenID: $0, logprob: $1) },
        labelIDs: labels)
    let mineByID = Dictionary(uniqueKeysWithValues: pairs.map { ($0.tokenID, $0.logprob) })
    let theirsByID = Dictionary(uniqueKeysWithValues: oracle.map { ($0.tokenID, $0.logprob) })
    var labelDifference = 0.0
    for label in labels {
        labelDifference = max(labelDifference, abs(mineByID[label]! - theirsByID[label]!))
    }
    func top20(_ values: [(tokenID: Int, logprob: Double)]) -> Set<Int> {
        Set(values.sorted { $0.logprob > $1.logprob }.prefix(topK).map(\.tokenID))
    }
    return SlotComparison(
        maxProbabilityDifference: maxDifference,
        topLabelAgrees: argmax(mine.probabilities) == argmax(theirs),
        entropyDifference: abs(mine.entropy - distribution.entropy),
        entropyDifferenceOnOracleSet: abs(sameSet.entropy - distribution.entropy),
        maxLabelLogprobDifference: labelDifference,
        top20Overlap: top20(pairs).intersection(top20(oracle)).count)
}

func identical(_ a: ProbeRead, _ b: ProbeRead) -> Bool {
    guard a.written == b.written, a.slots.count == b.slots.count else { return false }
    for (x, y) in zip(a.slots, b.slots) {
        guard x.count == y.count else { return false }
        for (p, q) in zip(x, y) where p.tokenID != q.tokenID || p.logprob != q.logprob {
            return false
        }
    }
    return true
}

// MARK: - Main

let options = Options.parse(CommandLine.arguments)
let oracle = try JSONDecoder().decode(
    OracleFile.self, from: Data(contentsOf: URL(fileURLWithPath: options.oracle)))
print("oracle: \(oracle.reads.count) reads, mlx \(oracle.generator["mlx"]!), mlx-vlm \(oracle.generator["mlx_vlm"]!)")

var memory: [String: [String: Int]] = ["before_load": processMemory()]
let loadStart = now()
let context = try await DiffusionGemmaModelFactory.shared.load(
    from: URL(fileURLWithPath: options.model), using: UnusedTokenizerLoader())
let loadSeconds = now() - loadStart
let model = context.model
if let gigabytes = options.cacheLimitGB {
    // What MLX.GPU.set(cacheLimit:) does; that spelling is deprecated in this mlx-swift.
    Memory.cacheLimit = Int(gigabytes * 1024 * 1024 * 1024)
}
memory["after_load"] = processMemory()
print(String(format: "loaded in %.2f s", loadSeconds))

let reader = Reader(model: model, oracle: oracle)
// One unrecorded read compiles the kernels; the prefill cache is emptied afterwards.
let warmStart = now()
_ = try reader.read(oracle.reads[0], digest: false)
let warmupSeconds = now() - warmStart
reader.prefills = [:]

/// The fork's request cache for one prompt, every layer in temporal order, as safetensors.
func dumpCache(_ cache: DiffusionGemmaRequestCache, prompt: String, promptTokens: Int, to directory: URL)
    throws
{
    var arrays: [String: MLXArray] = [:]
    var metadata: [String: String] = ["prompt_tokens": String(promptTokens)]
    for (layer, snapshot) in cache.snapshots().enumerated() {
        arrays["layers.\(layer).keys"] = snapshot.keys
        arrays["layers.\(layer).values"] = snapshot.values
        metadata["layers.\(layer).offset"] = String(snapshot.offset)
    }
    let name = prompt.replacingOccurrences(of: "/", with: "_") + ".safetensors"
    try save(arrays: arrays, metadata: metadata, url: directory.appendingPathComponent(name))
}

var dumped: [String: Any] = [:]
var passes: [[String: ProbeRead]] = []
for pass in 0 ..< options.passes {
    reader.prefills = [:]
    let order = pass % 2 == 0 ? Array(oracle.reads.indices) : Array(oracle.reads.indices.reversed())
    var results: [String: ProbeRead] = [:]
    for index in order {
        let read = oracle.reads[index]
        results[read.id] = try reader.read(read, digest: pass == 0)
        if pass == 0, let directory = options.dumpDirectory, options.dumpReads.contains(read.id) {
            let url = URL(fileURLWithPath: directory)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try dumpCache(
                reader.prefills[read.prompt]!, prompt: read.prompt,
                promptTokens: oracle.prompts[read.prompt]!.ids.count, to: url)
            let got = results[read.id]!
            dumped[read.id] = [
                "logprobs": got.slots.map { $0.map { [Double($0.tokenID), $0.logprob] } },
                "written": got.written,
            ]
        }
    }
    passes.append(results)
    memory["after_pass_\(pass + 1)"] = processMemory()
}
let differing = oracle.reads.map(\.id).filter { id in
    passes.dropFirst().contains { !identical($0[id]!, passes[0][id]!) }
}
let deterministic = differing.isEmpty

var rows: [[String: Any]] = []
var overallProbability = 0.0
var overallEntropy = 0.0
var overallEntropySameSet = 0.0
var overallLabelLogprob = 0.0
var slotsTotal = 0
var slotsAgreeing = 0
var writtenMismatches: [String] = []
var worstRead = ""
print("")
print("read                                 slots  max|dp|     top   max|dH|   max|dlp|  enc ms  step ms")
for read in oracle.reads {
    let got = passes[0][read.id]!
    let oraclePairs = read.pairs
    var comparisons: [SlotComparison] = []
    for (index, slot) in read.slots.enumerated() {
        comparisons.append(
            compareSlot(
                swift: got.slots[index], onOracleSet: got.onOracleSet[index],
                oracle: oraclePairs[index], distribution: read.distributions[index],
                labels: slot.labelIDs))
    }
    let maxProbability = comparisons.map(\.maxProbabilityDifference).max()!
    let agreeing = comparisons.filter(\.topLabelAgrees).count
    let maxEntropy = comparisons.map(\.entropyDifference).max()!
    let maxEntropySameSet = comparisons.map(\.entropyDifferenceOnOracleSet).max()!
    let maxLabel = comparisons.map(\.maxLabelLogprobDifference).max()!
    if maxProbability > overallProbability { worstRead = read.id }
    overallProbability = max(overallProbability, maxProbability)
    overallEntropy = max(overallEntropy, maxEntropy)
    overallEntropySameSet = max(overallEntropySameSet, maxEntropySameSet)
    overallLabelLogprob = max(overallLabelLogprob, maxLabel)
    slotsTotal += comparisons.count
    slotsAgreeing += agreeing
    let writtenEqual = got.written == read.written
    if !writtenEqual { writtenMismatches.append(read.id) }
    let other = passes.count > 1 ? passes[1][read.id]! : got
    rows.append([
        "id": read.id, "slots": comparisons.count, "steps": read.steps,
        "prompt_tokens": read.promptTokens,
        "max_probability_difference": maxProbability,
        "top_label_agreement": agreeing,
        "max_entropy_difference": maxEntropy,
        "max_entropy_difference_on_oracle_set": maxEntropySameSet,
        "max_label_logprob_difference": maxLabel,
        "min_top20_overlap": comparisons.map(\.top20Overlap).min()!,
        "written_equal": writtenEqual, "written": got.written,
        "per_slot": comparisons.map {
            [
                "max_probability_difference": $0.maxProbabilityDifference,
                "top_label_agrees": $0.topLabelAgrees,
                "entropy_difference": $0.entropyDifference,
                "top20_overlap": $0.top20Overlap,
            ] as [String: Any]
        },
        "timing_pass_1": [
            "prefill_cached": got.prefillCached, "encode_seconds": got.encodeSeconds,
            "step_seconds": got.stepSeconds, "total_seconds": got.totalSeconds,
        ] as [String: Any],
        "timing_pass_2": [
            "prefill_cached": other.prefillCached, "encode_seconds": other.encodeSeconds,
            "step_seconds": other.stepSeconds, "total_seconds": other.totalSeconds,
        ] as [String: Any],
    ])
    print(
        read.id.padding(toLength: 36, withPad: " ", startingAt: 0),
        String(format: "%5d  %.2e  %2d/%-2d  %.2e  %.2e  %6.1f  %7.1f",
            comparisons.count, maxProbability, agreeing, comparisons.count, maxEntropy, maxLabel,
            got.encodeSeconds * 1000, got.stepSeconds.reduce(0, +) * 1000),
        writtenEqual ? "" : "WRITTEN DIFFERS")
}
print("")
print(String(format: "overall: max |dp| %.3e (%@), top label %d/%d, max |dH| %.3e (same set %.3e), max |dlogprob| %.3e",
    overallProbability, worstRead, slotsAgreeing, slotsTotal, overallEntropy, overallEntropySameSet,
    overallLabelLogprob))
print("steps>1 argmaxes equal: \(writtenMismatches.isEmpty) \(writtenMismatches)")
print("deterministic across \(options.passes) passes: \(deterministic) \(differing)")
for (key, rows) in reader.digests.sorted(by: { $0.key < $1.key }) {
    for row in rows {
        let keys = row["keys"] as! [String: Any]
        let values = row["values"] as! [String: Any]
        print(
            "cache \(key) layer \(row["layer"]!) hash equal keys \(keys["sha256_equal"]!) values \(values["sha256_equal"]!); rel d(sum sq) keys",
            String(format: "%.2e values %.2e",
                keys["relative_sum_of_squares_difference"] as! Double,
                values["relative_sum_of_squares_difference"] as! Double))
    }
}

let gib = Double(1 << 30)
for (key, value) in memory.sorted(by: { $0.key < $1.key }) {
    print(
        key.padding(toLength: 13, withPad: " ", startingAt: 0),
        String(format: "footprint %.2f GiB, resident %.2f GiB, mlx active %.2f, cache %.2f, peak %.2f GiB",
            Double(value["phys_footprint_bytes"] ?? 0) / gib, Double(value["resident_bytes"] ?? 0) / gib,
            Double(value["mlx_active_bytes"] ?? 0) / gib, Double(value["mlx_cache_bytes"] ?? 0) / gib,
            Double(value["mlx_peak_bytes"] ?? 0) / gib))
}

let summary: [String: Any] = [
    "oracle": options.oracle,
    "oracle_generator": oracle.generator.mapValues { $0.description },
    "fork": "Layr-Labs/mlx-swift-lm eeba2afaf059a153ff909c9e01aa5e65b7bcad67",
    "cache_limit_gb": options.cacheLimitGB as Any,
    "load_seconds": loadSeconds,
    "warmup_read_seconds": warmupSeconds,
    "memory": memory,
    "deterministic": deterministic,
    "reads_differing_between_passes": differing,
    "overall": [
        "max_probability_difference": overallProbability,
        "worst_read": worstRead,
        "top_label_agreement": slotsAgreeing, "slots": slotsTotal,
        "max_entropy_difference": overallEntropy,
        "max_entropy_difference_on_oracle_set": overallEntropySameSet,
        "max_label_logprob_difference": overallLabelLogprob,
        "written_mismatches": writtenMismatches,
    ] as [String: Any],
    "cache_digests": reader.digests,
    "reads": rows,
]
if let directory = options.dumpDirectory {
    let data = try JSONSerialization.data(withJSONObject: dumped, options: [.sortedKeys])
    try data.write(to: URL(fileURLWithPath: directory).appendingPathComponent("swift_reads.json"))
    print("dumped \(dumped.count) reads to \(directory)")
}
if let path = options.maps {
    let maps = Dictionary(uniqueKeysWithValues: passes[0].map { id, read in
        (id, ["logprobs": read.slots.map { $0.map { [Double($0.tokenID), $0.logprob] } }, "written": read.written] as [String: Any])
    })
    try JSONSerialization.data(withJSONObject: maps, options: [.sortedKeys]).write(to: URL(fileURLWithPath: path))
    print("wrote the maps of \(maps.count) reads to \(path)")
}

/// The default result file names the configuration, so that no run overwrites another's: only
/// the default run (the fork's own settings, two passes, no dump, no cache limit) writes
/// probe_run.json.
func defaultResultPath() -> String {
    var name = "probe_run"
    if !DiffusionGemmaExpertUnsortSwitch.enabled { name += "_expert_unsort_off" }
    if options.passes != 2 { name += "_passes_\(options.passes)" }
    if options.dumpDirectory != nil { name += "_dump" }
    if let gigabytes = options.cacheLimitGB {
        let size = gigabytes.rounded() == gigabytes ? String(Int(gigabytes)) : String(gigabytes)
        name += "_cache_limit_\(size)gb"
    }
    return "Tools/oracle/results/\(name).json"
}

/// The fork reads DARKBLOOM_DIFFUSION_EXPERT_UNSORT once (DiffusionGemmaExpertReduction.enabled,
/// which is internal); this repeats its rule for the file name only.
enum DiffusionGemmaExpertUnsortSwitch {
    static var enabled: Bool {
        guard let value = ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_EXPERT_UNSORT"] else {
            return true
        }
        return ["1", "true", "yes", "on"].contains(value.lowercased())
    }
}

let outPath = options.out ?? defaultResultPath()
let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
try FileManager.default.createDirectory(
    at: URL(fileURLWithPath: outPath).deletingLastPathComponent(), withIntermediateDirectories: true)
try data.write(to: URL(fileURLWithPath: outPath))
print("wrote \(outPath)")
