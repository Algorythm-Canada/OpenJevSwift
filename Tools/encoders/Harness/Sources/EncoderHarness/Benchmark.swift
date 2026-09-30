import CoreML
import Foundation

/// How closely a package's answers follow the PyTorch reference.
public struct ParityResult: Codable, Sendable {
    public let questions: Int
    /// Verdict: upstream's calibrated probabilities. Laya: the probabilities before rounding.
    public let maxAbsProbabilityDifference: Double
    public let meanAbsProbabilityDifference: Double
    public let maxAbsLogitDifference: Double
    public let topLabelAgreement: Int
    public let topLabelChanges: [String]
    public let nonFinite: Int
}

/// One package measured with one set of compute units.
public struct BenchmarkResult: Codable, Sendable {
    public let package: String
    public let model: String
    public let kind: PackageKind
    public let units: ComputeUnitsName
    public let device: DeviceInfo
    public let fixtureRevisions: [String: String]
    public let packageBytes: UInt64
    public let compileSeconds: Double
    /// The first and second load of the package (an enumerated one) or of its b1_s128 function.
    public let firstLoadSeconds: Double
    public let secondLoadSeconds: Double
    /// How long each function took to load when the runs first needed it.
    public let functionLoadSeconds: [String: Double]
    public let computePlan: ComputePlanSummary?
    public let tokenizerLoadSeconds: Double
    public let tokenizePerQuestion: LatencyStats
    public let tokenizationMatches: String
    public let warmupSeconds: [String: Double]
    /// One question per call; the latency of each call.
    public let batch1: LatencyStats
    public let batch1ByLength: [String: LatencyStats]
    /// The corpus in order, 16 questions per call (the last call has 8), grouped by padded shape.
    public let batch16PerCall: LatencyStats
    public let batch16PerQuestion: LatencyStats
    public let batch16ByLength: [String: LatencyStats]
    public let parityBatch1: ParityResult
    /// Empty for a package without batch-16 functions.
    public let parityBatch16: ParityResult?
    public let footprintBeforeLoadBytes: UInt64
    public let footprintAfterLoadBytes: UInt64
    /// The footprint after each shape's first call, with at most one function loaded.
    public let footprintAfterWarmupBytes: [String: UInt64]
    /// The largest footprint sampled from the first warmup call to the end of the reads; compiling,
    /// the test loads and the compute-plan query come before it.
    public let peakFootprintDuringRunBytes: UInt64
    /// The process's peak footprint so far, earlier configurations of the same launch included.
    public let lifetimePeakFootprintBytes: UInt64
    public let thermalStart: String
    public let thermalEnd: String
    public let wallSeconds: Double
}

/// One question's model input and what the reference says about it.
struct Question {
    let key: String
    let planes: [[Int32]]
    let referenceIds: [Int]
    let ids: [Int]
    let referenceMarkers: [Int]
    let markers: [Int]
    let k: Int
    let temperature: Double
    let referenceLogits: [Double]
    let referenceProbabilities: [Double]
}

/// The corpus as model inputs, with the padding id and how long each tokenization took.
struct QuestionSet {
    let questions: [Question]
    let padId: Int
    let tokenizeSeconds: [Double]
}

func verdictQuestions(locations: HarnessLocations, tokenizer: EncoderTokenizer) throws
    -> QuestionSet
{
    let reference = try FixtureFile.decode(VerdictReference.self, from: locations.verdictFixture)
    guard let cls = tokenizer.id(of: "[CLS]"), let sep = tokenizer.id(of: "[SEP]") else {
        throw HarnessError.missing("[CLS] or [SEP] in the Verdict tokenizer")
    }
    var seconds: [Double] = []
    let questions = try reference.reads.map { read in
        let started = now()
        let ids = VerdictInput.ids(
            prompt: read.prompt, tokenizer: tokenizer, cls: cls, sep: sep,
            maxLength: reference.maxLength)
        seconds.append(now() - started)
        let temperature = VerdictCalibration.temperature(
            k: read.k, calibrator: reference.calibrator)
        guard temperature == read.temperature else {
            throw HarnessError.mismatch("temperature for \(read.request)/\(read.key)")
        }
        return Question(
            key: "\(read.request)/\(read.key)",
            planes: [ids.map(Int32.init), Array(repeating: 1, count: ids.count)],
            referenceIds: read.inputIds, ids: ids, referenceMarkers: [], markers: [], k: read.k,
            temperature: temperature, referenceLogits: read.logits,
            referenceProbabilities: read.probabilities)
    }
    return QuestionSet(questions: questions, padId: reference.padTokenId, tokenizeSeconds: seconds)
}

func layaQuestions(locations: HarnessLocations, tokenizer: EncoderTokenizer) throws -> QuestionSet {
    let reference = try FixtureFile.decode(LayaReference.self, from: locations.layaFixture)
    var seconds: [Double] = []
    let questions = try reference.reads.map { read in
        guard let state = reference.stateTexts[read.request] else {
            throw HarnessError.missing("state text of \(read.request)")
        }
        let started = now()
        let (ids, markers) = LayaSequence.build(
            head: read.texts.head, options: read.texts.options, state: state, tokenizer: tokenizer,
            special: reference.specialTokens, maxLen: reference.maxLen,
            headMaxLen: reference.headMaxLen)
        seconds.append(now() - started)
        let temperature = LayaCalibration.temperature(
            qtype: read.qtype, k: read.markers.count, calibration: reference.calibration,
            qtypes: reference.qtypes)
        guard abs(temperature - read.temperature) < 1e-12 else {
            throw HarnessError.mismatch("temperature for \(read.request)/\(read.key)")
        }
        return Question(
            key: "\(read.request)/\(read.key)",
            planes: [
                ids.map(Int32.init), Array(repeating: 1, count: ids.count),
                Array(repeating: Int32(read.qtype), count: ids.count),
            ],
            referenceIds: read.ids, ids: ids, referenceMarkers: read.markers, markers: markers,
            k: read.markers.count, temperature: temperature, referenceLogits: read.logits,
            referenceProbabilities: read.probabilitiesUnrounded)
    }
    return QuestionSet(
        questions: questions, padId: reference.specialTokens.pad, tokenizeSeconds: seconds)
}

/// A question's logits and probabilities from one output row.
func answer(_ q: Question, spec: PackageSpec, row: [Float]) -> (
    logits: [Float], probabilities: [Double]
) {
    if spec.model == "verdict" {
        let logits = Array(row.prefix(q.k))
        return (
            logits,
            VerdictCalibration.probabilities(logits: logits[...], temperature: q.temperature)
        )
    }
    let logits = q.markers.map { row[$0] }
    return (logits, LayaCalibration.probabilities(logits: logits, temperature: q.temperature))
}

func parity(_ questions: [Question], _ answers: [(logits: [Float], probabilities: [Double])])
    -> ParityResult
{
    var maxP = 0.0
    var sumP = 0.0
    var countP = 0
    var maxL = 0.0
    var agree = 0
    var changes: [String] = []
    var nonFinite = 0
    for (q, a) in zip(questions, answers) {
        if !a.probabilities.allSatisfy(\.isFinite) || !a.logits.allSatisfy(\.isFinite) {
            nonFinite += 1
            changes.append(q.key)
            continue
        }
        for (r, p) in zip(q.referenceProbabilities, a.probabilities) {
            maxP = max(maxP, abs(r - p))
            sumP += abs(r - p)
            countP += 1
        }
        for (r, l) in zip(q.referenceLogits, a.logits) { maxL = max(maxL, abs(r - Double(l))) }
        let top = a.probabilities.indices.max { a.probabilities[$0] < a.probabilities[$1] }
        let want = q.referenceProbabilities.indices.max {
            q.referenceProbabilities[$0] < q.referenceProbabilities[$1]
        }
        if top == want { agree += 1 } else { changes.append(q.key) }
    }
    return ParityResult(
        questions: questions.count, maxAbsProbabilityDifference: maxP,
        meanAbsProbabilityDifference: countP > 0 ? sumP / Double(countP) : 0,
        maxAbsLogitDifference: maxL,
        topLabelAgreement: agree, topLabelChanges: changes, nonFinite: nonFinite)
}

/// Measures one package with one set of compute units: compile, a first and a second load, the
/// compute plan, every question alone, then the corpus in calls of 16, with parity against the
/// reference and the physical footprint sampled throughout.
///
/// Questions run grouped by the shape they pad to, shortest first, so that a multifunction package
/// loads each function once and holds one at a time. The first call of each shape is a warmup and
/// is not timed with the others.
public func runBenchmark(
    spec: PackageSpec, units: ComputeUnitsName, locations: HarnessLocations, passes16: Int = 3,
    log: (String) -> Void = { print($0) }
) async throws -> BenchmarkResult {
    let started = now()
    let thermalStart = thermalStateName()
    let footprintBefore = physicalFootprint().current
    let packageURL = locations.package(named: spec.name)
    guard FileManager.default.fileExists(atPath: packageURL.path) else {
        throw HarnessError.missing(packageURL.path)
    }
    let packageBytes = directorySize(packageURL)
    let label = "\(spec.name) \(units.rawValue)"

    var t = now()
    let tokenizerFolder =
        spec.model == "verdict" ? locations.verdictTokenizer : locations.layaTokenizer
    let tokenizer = try await EncoderTokenizer.load(folder: tokenizerFolder)
    let tokenizerLoadSeconds = now() - t
    let set =
        try spec.model == "verdict"
        ? verdictQuestions(locations: locations, tokenizer: tokenizer)
        : layaQuestions(locations: locations, tokenizer: tokenizer)
    let matches = set.questions.filter {
        $0.ids == $0.referenceIds && $0.markers == $0.referenceMarkers
    }.count
    log("\(label): tokenized \(set.questions.count) questions, \(matches) match the reference")
    // A package whose shapes cannot hold a question (a fixed-shape one) is measured on the
    // questions that fit; tokenization is checked on all of them.
    let questions = set.questions.filter { spec.shape(rows: 1, length: $0.ids.count) != nil }
    if questions.count < set.questions.count {
        log(
            "\(label): measuring the \(questions.count) questions that fit \(spec.lengths.max() ?? 0) tokens"
        )
    }

    t = now()
    let compiled = try await MLModel.compileModel(at: packageURL)
    let compileSeconds = now() - t
    defer { try? FileManager.default.removeItem(at: compiled) }
    // The first load after compiling pays for any device-specific compilation (the Neural Engine
    // compiles on first load); the second shows what a later launch costs.
    let firstFunction =
        spec.kind == .enumerated ? nil : spec.functionName(batch: 1, length: spec.lengths[0])
    t = now()
    do {
        _ = try EncoderModel.load(compiled: compiled, units: units.units, function: firstFunction)
            .modelDescription
    }
    let firstLoadSeconds = now() - t
    t = now()
    do {
        _ = try EncoderModel.load(compiled: compiled, units: units.units, function: firstFunction)
            .modelDescription
    }
    let secondLoadSeconds = now() - t
    let footprintAfterLoad = physicalFootprint().current
    log(
        "\(label): compiled in \(fmt(compileSeconds)) s, loads \(fmt(firstLoadSeconds)) s and \(fmt(secondLoadSeconds)) s"
    )
    let plan = try? await computePlanSummary(compiled: compiled, spec: spec, units: units.units)
    let model = try EncoderModel(
        spec: spec, compiled: compiled, units: units.units, padId: set.padId)

    let sampler = FootprintSampler()
    sampler.start()
    defer { _ = sampler.stop() }
    var warmup: [String: Double] = [:]
    var footprintByShape: [String: UInt64] = [:]
    func shapeName(rows: Int, length: Int) throws -> String {
        guard let s = spec.shape(rows: rows, length: length) else {
            throw HarnessError.shape("\(rows) rows of \(length) tokens for \(spec.name)")
        }
        return spec.functionName(batch: s.batch, length: s.length)
    }
    /// Groups in the order of the package's shapes, so each function is loaded once.
    func ordered<T>(_ groups: [String: [T]]) -> [(String, [T])] {
        let order = spec.batches.flatMap { b in
            spec.lengths.map { spec.functionName(batch: b, length: $0) }
        }
        return order.compactMap { name in groups[name].map { (name, $0) } }
    }
    func warm(_ name: String, _ rows: [[[Int32]]]) throws {
        let loadedBefore = model.functionLoadSeconds[name]
        t = now()
        _ = try model.run(rows)
        let load = loadedBefore == nil ? (model.functionLoadSeconds[name] ?? 0) : 0
        warmup[name] = now() - t - load
        footprintByShape[name] = physicalFootprint().current
    }

    var batch1: [Double] = []
    var batch1ByLength: [String: [Double]] = [:]
    var answers1 = Array(
        repeating: (logits: [Float](), probabilities: [Double]()), count: questions.count)
    let groups1 = try Dictionary(grouping: questions.indices) {
        try shapeName(rows: 1, length: questions[$0].ids.count)
    }
    for (name, indices) in ordered(groups1) {
        try warm(name, [questions[indices[0]].planes])
        for i in indices {
            t = now()
            let out = try model.run([questions[i].planes])
            let seconds = now() - t
            batch1.append(seconds)
            batch1ByLength["s\(out.length)", default: []].append(seconds)
            answers1[i] = answer(questions[i], spec: spec, row: out.outputs[0])
        }
    }
    log("\(label): batch 1 median \(fmt(LatencyStats(seconds: batch1).medianMs)) ms")

    var perCall: [Double] = []
    var perQuestion: [Double] = []
    var byLength16: [String: [Double]] = [:]
    var answers16 = Array(
        repeating: (logits: [Float](), probabilities: [Double]()), count: questions.count)
    let chunks = stride(from: 0, to: questions.count, by: 16).map {
        Array($0..<min($0 + 16, questions.count))
    }
    // A package without batch-16 functions skips this phase; its batch-16 figures stay empty.
    let batch16 = spec.batches.contains(16)
    let groups16 =
        batch16
        ? try Dictionary(grouping: chunks) { chunk in
            try shapeName(
                rows: chunk.count, length: chunk.map { questions[$0].ids.count }.max() ?? 0)
        } : [:]
    for (name, group) in ordered(groups16) {
        try warm(name, group[0].map { questions[$0].planes })
        for pass in 0..<max(1, passes16) {
            for chunk in group {
                t = now()
                let out = try model.run(chunk.map { questions[$0].planes })
                let seconds = now() - t
                perCall.append(seconds)
                perQuestion.append(seconds / Double(chunk.count))
                byLength16[name, default: []].append(seconds)
                if pass == 0 {
                    for (i, row) in zip(chunk, out.outputs) {
                        answers16[i] = answer(questions[i], spec: spec, row: row)
                    }
                }
            }
        }
    }
    let peak = sampler.stop()
    log("\(label): batch 16 median \(fmt(LatencyStats(seconds: perCall).medianMs)) ms per call")

    let revisions = (try? FixtureFile.generator(of: locations.verdictFixture)).map {
        [
            "verdict": $0.verdictRevision, "laya": $0.layaRevision, "script": $0.script,
            "version": String($0.version),
        ]
    }
    return BenchmarkResult(
        package: spec.name, model: spec.model, kind: spec.kind, units: units, device: .current(),
        fixtureRevisions: revisions ?? [:], packageBytes: packageBytes,
        compileSeconds: compileSeconds,
        firstLoadSeconds: firstLoadSeconds, secondLoadSeconds: secondLoadSeconds,
        functionLoadSeconds: model.functionLoadSeconds, computePlan: plan,
        tokenizerLoadSeconds: tokenizerLoadSeconds,
        tokenizePerQuestion: LatencyStats(seconds: set.tokenizeSeconds),
        tokenizationMatches: "\(matches)/\(set.questions.count)", warmupSeconds: warmup,
        batch1: LatencyStats(seconds: batch1),
        batch1ByLength: batch1ByLength.mapValues { LatencyStats(seconds: $0) },
        batch16PerCall: LatencyStats(seconds: perCall),
        batch16PerQuestion: LatencyStats(seconds: perQuestion),
        batch16ByLength: byLength16.mapValues { LatencyStats(seconds: $0) },
        parityBatch1: parity(questions, answers1),
        parityBatch16: batch16 ? parity(questions, answers16) : nil,
        footprintBeforeLoadBytes: footprintBefore, footprintAfterLoadBytes: footprintAfterLoad,
        footprintAfterWarmupBytes: footprintByShape, peakFootprintDuringRunBytes: peak,
        lifetimePeakFootprintBytes: physicalFootprint().peak,
        thermalStart: thermalStart, thermalEnd: thermalStateName(), wallSeconds: now() - started)
}

func fmt(_ value: Double) -> String {
    String(format: "%.3f", value)
}

func directorySize(_ url: URL) -> UInt64 {
    guard
        let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey])
    else { return 0 }
    var total: UInt64 = 0
    for case let file as URL in enumerator {
        total += UInt64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }
    return total
}

/// The JSON a run writes: sorted keys, so reruns diff cleanly.
public func encodeResult(_ result: BenchmarkResult) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try encoder.encode(result)
}
