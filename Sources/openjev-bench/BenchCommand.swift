// openjev-bench (issue #32): the performance and memory baseline of DiffusionGemma reads.
// docs/benchmarks.md holds the tables it produced; docs/development.md says how to run it.

import ArgumentParser
import Foundation
import MLX
import Metal
import OpenJevCore
import OpenJevDiffusionGemma

@main
struct BenchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "openjev-bench",
        abstract: "Latency, throughput, memory and prefill figures for DiffusionGemma reads.",
        discussion: """
            reads and concurrency run on the engine in this process, or with --url on any \
            /v1/systemone server. memory, prefill and profile run in this process only. \
            Markdown tables go to standard output; --json also appends the run to \
            Tools/bench/results/<date>-<machine>.json.
            """,
        subcommands: [
            Reads.self, Concurrency.self, MemoryMode.self, Prefill.self, Profile.self,
        ])
}

/// The options every mode takes.
struct CommonOptions: ParsableArguments {
    @Option(
        help: """
            The checkpoint directory. Default: OPENJEV_TEST_MODEL, else the Hugging Face cache \
            snapshot of the pinned 4-bit checkpoint.
            """)
    var model: String?

    @Flag(help: "Also append the run to <output-dir>/<date>-<machine>.json.")
    var json = false

    @Option(help: "Where --json writes.")
    var outputDir = "Tools/bench/results"

    @Option(help: "Timed requests per row, after the warm-up.")
    var runs = 50

    @Option(help: "Untimed requests before each row.")
    var warmup = 5

    @Option(
        help: """
            A Metal library for MLX instead of the one built with the package, such as the \
            mlx-metal wheel's mlx.metallib (D-014's exact tier).
            """)
    var metallib: String?

    /// Points MLX at ``metallib`` when it is set; call before the first MLX call.
    func applyMetallib() {
        if let metallib {
            GPU.metallib = URL(fileURLWithPath: (metallib as NSString).expandingTildeInPath)
        }
    }

    /// What every run records beside its own settings: the Metal library and the thermal state
    /// when it starts.
    var recordedSettings: [String: String] {
        ["metallib": metallib ?? "the package's", "thermal state at start": Bench.thermalState()]
    }

    /// The checkpoint directory, found as the live tests find it.
    func modelDirectory(environment: [String: String] = ProcessInfo.processInfo.environment)
        -> URL
    {
        if let model, !model.isEmpty {
            return URL(fileURLWithPath: (model as NSString).expandingTildeInPath)
        }
        if let path = environment["OPENJEV_TEST_MODEL"], !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        guard case .hub(let repository, let revision?) = ModelSource.fourBit else {
            preconditionFailure("the 4-bit preset is a pinned Hub source")
        }
        return HubCacheLocation(environment: environment).snapshotDirectory(
            repository, commit: revision)
    }
}

/// The options of the modes that can run over HTTP.
struct HTTPOptions: ParsableArguments {
    @Option(help: "A /v1/systemone server's base URL, such as http://127.0.0.1:8000.")
    var url: String?

    @Option(help: "A label for the server in the tables and the result file.")
    var server: String?

    @Option(help: "The environment variable holding the server's API key, if it needs one.")
    var apiKeyEnv: String?
}

// MARK: Running

/// What every mode shares: the machine, the target, the output.
enum Bench {
    /// A tag that makes this invocation's states unique, so a server that outlives one run does
    /// not serve the next run's prompts from its prefill cache.
    static let tag = String(UUID().uuidString.prefix(8)).lowercased()

    static func sysctl(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(
            decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    static func memoryGB() -> Int {
        var bytes: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &bytes, &size, nil, 0)
        return Int(bytes >> 30)
    }

    /// Whether `pmset -g batt` says the Mac draws from AC power.
    static func onACPower() -> Bool? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
        process.arguments = ["-g", "batt"]
        let pipe = Pipe()
        process.standardOutput = pipe
        guard (try? process.run()) != nil else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let text = String(decoding: data, as: UTF8.self)
        if text.contains("'AC Power'") { return true }
        if text.contains("'Battery Power'") { return false }
        return nil
    }

    static func machine() -> Machine {
        Machine(
            model: sysctl("hw.model") ?? "unknown",
            chip: sysctl("machdep.cpu.brand_string") ?? MTLCreateSystemDefaultDevice()?.name
                ?? "unknown",
            memoryGB: memoryGB(), macOS: ProcessInfo.processInfo.operatingSystemVersionString,
            onACPower: onACPower())
    }

    /// `ProcessInfo.thermalState`: nominal, fair, serious or critical.
    static func thermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func now() -> String {
        ISO8601DateFormatter().string(from: Date())
    }

    static func today() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: Date())
    }

    /// Loads the runtime from `directory`, printing what it is doing on standard error.
    static func loadRuntime(_ directory: URL, cacheLimitGB: Double? = nil) async throws
        -> DiffusionGemmaRuntime
    {
        FileHandle.standardError.write(Data("loading \(directory.path)\n".utf8))
        let runtime = try await DiffusionGemmaRuntime.load(
            .directory(directory), configuration: .init(cacheLimitGB: cacheLimitGB, warmUp: true))
        if let report = await runtime.loadReport {
            FileHandle.standardError.write(
                Data(
                    ("loaded: weights \(milliseconds(report.modelMetrics.wallTime)) ms, warm-up "
                        + "\(report.warmUpTime.map { milliseconds($0) } ?? 0) ms, \(report.memory)\n")
                        .utf8))
        }
        return runtime
    }

    /// The target `http` names, or the engine over the checkpoint.
    static func target(_ common: CommonOptions, _ http: HTTPOptions) async throws -> (
        any DecisionTarget, BenchRun
    ) {
        if let url = http.url {
            guard let base = URL(string: url), base.scheme != nil else {
                throw ValidationError("--url \(url) is not a URL")
            }
            let key = http.apiKeyEnv.flatMap { ProcessInfo.processInfo.environment[$0] }
            let run = BenchRun(
                mode: "", target: "http", url: url, server: http.server, modelDirectory: nil,
                started: now(), settings: [:])
            return (HTTPTarget(baseURL: base, apiKey: key), run)
        }
        let directory = common.modelDirectory()
        common.applyMetallib()
        let runtime = try await loadRuntime(directory)
        let engine = try DecisionEngine(backend: runtime, configuration: .default)
        let run = BenchRun(
            mode: "", target: "engine", url: nil, server: http.server ?? "swift",
            modelDirectory: directory.path, started: now(), settings: [:])
        return (EngineTarget(engine: engine), run)
    }

    /// Prints the run's tables and, with `--json`, appends it to the result file.
    static func finish(_ run: BenchRun, _ common: CommonOptions) throws {
        var run = run
        run.settings["thermal state at end"] = thermalState()
        if run.target == "http" {
            // The server's kernels are its own; this process runs none.
            run.settings["metallib"] = nil
        }
        let machine = machine()
        print(Markdown.render(run, machine: machine))
        if common.json {
            let url = URL(fileURLWithPath: common.outputDir).appendingPathComponent(
                BenchResultFile.fileName(date: today(), machine: machine))
            try BenchResultFile.append(run, machine: machine, to: url)
            FileHandle.standardError.write(Data("appended the run to \(url.path)\n".utf8))
        }
    }

    /// `runs` requests one after another after `warmup` untimed ones, each with its own state.
    static func serial(
        _ target: any DecisionTarget, questions: String, label: String, runs: Int, warmup: Int
    ) async throws -> LatencyRow {
        for index in 0..<warmup {
            _ = try await target.decide(
                body: Workload.body(
                    state: Workload.uniqueState(tag: "\(tag)-\(label)-w", index: index),
                    questions: questions))
        }
        var samples: [Sample] = []
        for index in 0..<runs {
            samples.append(
                try await target.decide(
                    body: Workload.body(
                        state: Workload.uniqueState(tag: "\(tag)-\(label)", index: index),
                        questions: questions)))
        }
        return row(label, samples)
    }

    static func row(_ label: String, _ samples: [Sample], throughput: Double? = nil)
        -> LatencyRow
    {
        let models = samples.compactMap(\.modelMilliseconds)
        return LatencyRow(
            label: label, latency: LatencySummary(milliseconds: samples.map(\.milliseconds)),
            model: models.count == samples.count ? LatencySummary(milliseconds: models) : nil,
            throughput: throughput, inputTokens: samples.first?.inputTokens)
    }
}

/// Hands out request indexes to concurrent workers.
actor Counter {
    private var next = 0
    let limit: Int

    init(limit: Int) {
        self.limit = limit
    }

    func take() -> Int? {
        guard next < limit else { return nil }
        defer { next += 1 }
        return next
    }
}

// MARK: Modes

struct Reads: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "p50 and p95 latency of 1, 3 and 12 questions, each request a new state.")

    @OptionGroup var common: CommonOptions
    @OptionGroup var http: HTTPOptions

    func run() async throws {
        let (target, started) = try await Bench.target(common, http)
        var run = started
        run.mode = "reads"
        run.settings = [
            "runs": "\(common.runs)", "warmup": "\(common.warmup)", "samples": "1",
            "states": "unique per request",
        ].merging(common.recordedSettings) { mode, _ in mode }
        var rows: [LatencyRow] = []
        for count in [1, 3, 12] {
            rows.append(
                try await Bench.serial(
                    target, questions: Workload.questions(count),
                    label: count == 1 ? "1 question" : "\(count) questions", runs: common.runs,
                    warmup: common.warmup))
        }
        run.latency = rows
        try Bench.finish(run, common)
    }
}

struct Concurrency: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Throughput and latency of 3-question reads at 1, 4 and 16 concurrent requests.")

    @OptionGroup var common: CommonOptions
    @OptionGroup var http: HTTPOptions

    @Option(help: "Requests per concurrency level.")
    var requests = 48

    @Option(parsing: .upToNextOption, help: "The concurrency levels.")
    var levels: [Int] = [1, 4, 16]

    func run() async throws {
        let (target, started) = try await Bench.target(common, http)
        var run = started
        run.mode = "concurrency"
        run.settings = [
            "requests per level": "\(requests)", "warmup": "\(common.warmup)", "samples": "1",
            "questions": "3", "states": "unique per request",
        ].merging(common.recordedSettings) { mode, _ in mode }
        let questions = Workload.questions(3)
        for index in 0..<common.warmup {
            _ = try await target.decide(
                body: Workload.body(
                    state: Workload.uniqueState(tag: "\(Bench.tag)-cw", index: index),
                    questions: questions))
        }
        var rows: [LatencyRow] = []
        for level in levels {
            let counter = Counter(limit: requests)
            let clock = ContinuousClock()
            let start = clock.now
            let samples = try await withThrowingTaskGroup(of: [Sample].self) { group in
                for _ in 0..<level {
                    group.addTask {
                        var samples: [Sample] = []
                        while let index = await counter.take() {
                            samples.append(
                                try await target.decide(
                                    body: Workload.body(
                                        state: Workload.uniqueState(
                                            tag: "\(Bench.tag)-c\(level)", index: index),
                                        questions: questions)))
                        }
                        return samples
                    }
                }
                var all: [Sample] = []
                for try await part in group {
                    all += part
                }
                return all
            }
            let wall = milliseconds(clock.now - start) / 1000
            rows.append(
                Bench.row(
                    "\(level) concurrent", samples, throughput: Double(samples.count) / wall))
        }
        run.latency = rows
        try Bench.finish(run, common)
    }
}

struct MemoryMode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "memory",
        abstract: "MemoryReport after load, after 20 reads and after 200 unique prompts.")

    @OptionGroup var common: CommonOptions

    @Option(help: "cacheLimitGB, MLX's buffer pool limit; unset leaves MLX alone.")
    var cacheLimitGB: Double?

    func run() async throws {
        let directory = common.modelDirectory()
        common.applyMetallib()
        let runtime = try await Bench.loadRuntime(directory, cacheLimitGB: cacheLimitGB)
        let engine = EngineTarget(
            engine: try DecisionEngine(backend: runtime, configuration: .default))
        var run = BenchRun(
            mode: "memory", target: "engine", url: nil, server: "swift",
            modelDirectory: directory.path, started: Bench.now(),
            settings: [
                "cacheLimitGB": cacheLimitGB.map { "\($0)" } ?? "unset", "questions": "3",
                "states": "unique per request",
            ].merging(common.recordedSettings) { mode, _ in mode })
        func snapshot(_ label: String) async -> MemoryRow {
            let report = await runtime.memoryReport()
            let gib = { (bytes: Int) in Double(bytes) / Double(1 << 30) }
            return MemoryRow(
                label: label, activeGiB: gib(report.activeBytes), cacheGiB: gib(report.cacheBytes),
                peakGiB: gib(report.peakBytes), residentGiB: gib(report.residentBytes),
                peakResidentGiB: gib(report.peakResidentBytes),
                cachedPrefills: await runtime.statistics().cachedPrefills)
        }
        var rows = [await snapshot("after load")]
        let questions = Workload.questions(3)
        for index in 0..<200 {
            _ = try await engine.decide(
                body: Workload.body(
                    state: Workload.uniqueState(tag: "\(Bench.tag)-m", index: index),
                    questions: questions))
            if index == 19 {
                rows.append(await snapshot("after 20 reads"))
            }
        }
        rows.append(await snapshot("after 200 unique prompts"))
        run.memory = rows
        try Bench.finish(run, common)
    }
}

struct Prefill: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Prefill tokens per second for about 1,000 and 10,000 token states.")

    @OptionGroup var common: CommonOptions

    @Option(parsing: .upToNextOption, help: "The approximate state sizes in tokens.")
    var tokens: [Int] = [1_000, 10_000]

    @Option(help: "Cold and cached reads per size, after one warm-up.")
    var prefillRuns = 5

    func run() async throws {
        let directory = common.modelDirectory()
        common.applyMetallib()
        let runtime = try await Bench.loadRuntime(directory)
        let engine = try DecisionEngine(backend: runtime, configuration: .default)
        var run = BenchRun(
            mode: "prefill", target: "engine", url: nil, server: "swift",
            modelDirectory: directory.path, started: Bench.now(),
            settings: ["runs": "\(prefillRuns)", "warmup": "1", "questions": "1"].merging(
                common.recordedSettings
            ) { mode, _ in mode })
        // The read the engine would make for one noul question, built from its own parts so the
        // timing is the runtime's alone.
        let request = try SystemOneRequest(
            json: JSONParser().parse(Workload.body(state: "", questions: Workload.questions(1))))
        let schema = try engine.schemaBuilder.build(request.questions)
        let resolved = try engine.resolver.resolve(schema.questions, format: schema.format)
        let system = SystemText.render(schema.questions, format: schema.format, chunked: false)
        let canvas = CanvasBuilder.build(
            template: resolved.template, slots: resolved.slots, seed: 0,
            geometry: engine.configuration.geometry)
        let clock = ContinuousClock()
        var rows: [PrefillRow] = []
        for size in tokens {
            var cold: [Double] = []
            var warm: [Double] = []
            var promptTokens = 0
            for index in 0...prefillRuns {
                let state = Workload.longState(tokens: size, tag: Bench.tag, index: index)
                let ids = try runtime.tokenizer.chatPromptIDs(
                    system: system, user: state, thinking: false)
                let read = CanvasRead(
                    prompt: .tokens(ids), systemText: system, stateText: state,
                    template: resolved.template, slots: resolved.slots, canvas: canvas, steps: 1,
                    seed: 0)
                var start = clock.now
                let result = try await runtime.read(read)
                let coldTime = milliseconds(clock.now - start)
                start = clock.now
                _ = try await runtime.read(read)
                let warmTime = milliseconds(clock.now - start)
                promptTokens = result.promptTokens
                // The first pair warms the kernels for this length.
                if index > 0 {
                    cold.append(coldTime)
                    warm.append(warmTime)
                }
            }
            let coldSummary = LatencySummary(milliseconds: cold)
            let warmSummary = LatencySummary(milliseconds: warm)
            let prefill = coldSummary.p50 - warmSummary.p50
            rows.append(
                PrefillRow(
                    label: "about \(size) tokens", promptTokens: promptTokens, cold: coldSummary,
                    warm: warmSummary, prefillMilliseconds: prefill,
                    tokensPerSecond: Double(promptTokens) / (prefill / 1000)))
        }
        run.prefill = rows
        try Bench.finish(run, common)
    }
}
