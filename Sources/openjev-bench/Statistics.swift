// The model-free part of openjev-bench (issue #32): percentiles, the server-timing header, the
// markdown tables and the result file. OpenJevBenchTests covers it without a model.

import Foundation

/// Latency figures over a set of samples, in milliseconds.
struct LatencySummary: Codable, Equatable, Sendable {
    var count: Int
    var p50: Double
    var p95: Double
    var mean: Double
    var min: Double
    var max: Double

    /// The summary of `samples`, which must not be empty.
    init(milliseconds samples: [Double]) {
        precondition(!samples.isEmpty, "a summary needs at least one sample")
        let sorted = samples.sorted()
        count = sorted.count
        p50 = Self.percentile(sorted, 50)
        p95 = Self.percentile(sorted, 95)
        mean = sorted.reduce(0, +) / Double(sorted.count)
        min = sorted[0]
        max = sorted[sorted.count - 1]
    }

    /// The `p`th percentile of ascending `sorted` values by linear interpolation between the
    /// closest ranks, NumPy's default (`numpy.percentile(..., method="linear")`), so a table
    /// computed from a result file in Python gives the same figure.
    static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        precondition(!sorted.isEmpty && (0...100).contains(p))
        let rank = p / 100 * Double(sorted.count - 1)
        let lower = Int(rank.rounded(.down))
        let upper = Swift.min(lower + 1, sorted.count - 1)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - Double(lower))
    }
}

/// The milliseconds of a `Duration`.
func milliseconds(_ duration: Duration) -> Double {
    let (seconds, attoseconds) = duration.components
    return Double(seconds) * 1e3 + Double(attoseconds) / 1e15
}

/// Upstream's and the Swift server's `server-timing` header, `model;dur=A, server;dur=B,
/// total;dur=C`.
enum ServerTiming {
    /// Each metric's `dur` in milliseconds, by name; metrics without a `dur` are left out.
    static func parse(_ header: String) -> [String: Double] {
        var metrics: [String: Double] = [:]
        for metric in header.split(separator: ",") {
            let parts = metric.split(separator: ";").map {
                $0.trimmingCharacters(in: .whitespaces)
            }
            guard let name = parts.first, !name.isEmpty else { continue }
            for parameter in parts.dropFirst() where parameter.hasPrefix("dur=") {
                if let value = Double(parameter.dropFirst(4)) {
                    metrics[name] = value
                }
            }
        }
        return metrics
    }
}

/// The machine a run measured, recorded with every table.
struct Machine: Codable, Equatable, Sendable {
    /// `hw.model`, such as `Mac15,8`.
    var model: String
    /// `machdep.cpu.brand_string`, such as `Apple M3 Max`.
    var chip: String
    /// `hw.memsize` in GiB.
    var memoryGB: Int
    /// `ProcessInfo.operatingSystemVersionString`.
    var macOS: String
    /// Whether the Mac ran on AC power when the run started, from `pmset -g batt`; nil when
    /// unknown.
    var onACPower: Bool?

    /// One line for a table's caption.
    var caption: String {
        let power = onACPower.map { $0 ? ", on AC power" : ", on battery" } ?? ""
        return "\(chip) (\(model)), \(memoryGB) GB, macOS \(macOS)\(power)"
    }

    /// The machine part of a result file's name: the chip and memory, lower case, letters and
    /// digits joined by `-`, such as `apple-m3-max-128gb`.
    var slug: String {
        let text = "\(chip) \(memoryGB)gb".lowercased()
        return text.split { !($0.isLetter || $0.isNumber) }.joined(separator: "-")
    }
}

/// One row of a latency table.
struct LatencyRow: Codable, Equatable, Sendable {
    /// What was measured, such as `3 questions` or `16 concurrent`.
    var label: String
    /// The caller's latency: the engine call, or the HTTP exchange.
    var latency: LatencySummary
    /// The `model` metric of `server-timing` (HTTP) or the decision's model time (engine).
    var model: LatencySummary?
    /// Requests per second over the row's wall time, for the concurrency rows.
    var throughput: Double?
    /// The billed input tokens of the row's first request.
    var inputTokens: Int?
}

/// One row of the memory table, in GiB.
struct MemoryRow: Codable, Equatable, Sendable {
    var label: String
    var activeGiB: Double
    var cacheGiB: Double
    var peakGiB: Double
    var residentGiB: Double
    var peakResidentGiB: Double
    /// The prefills cached when it was taken.
    var cachedPrefills: Int
}

/// One row of the prefill table.
struct PrefillRow: Codable, Equatable, Sendable {
    var label: String
    var promptTokens: Int
    /// A read that prefills, a read of the same prompt from the cache, and their difference.
    var cold: LatencySummary
    var warm: LatencySummary
    var prefillMilliseconds: Double
    var tokensPerSecond: Double
}

/// One stage of the profile.
struct StageRow: Codable, Equatable, Sendable {
    /// `prefill` or `decoder pass`.
    var phase: String
    var stage: String
    /// The stage's mean time per read.
    var meanMilliseconds: Double
    /// Its share of the phase's staged total; nil for the unstaged total.
    var share: Double?
}

/// One invocation of a mode.
struct BenchRun: Codable, Equatable, Sendable {
    /// `reads`, `concurrency`, `memory`, `prefill` or `profile`.
    var mode: String
    /// `engine` (in process) or `http`.
    var target: String
    /// The server's URL for an HTTP run.
    var url: String?
    /// A free label, such as `swift` or `upstream`.
    var server: String?
    /// The checkpoint directory of an engine run.
    var modelDirectory: String?
    /// When it started, ISO 8601.
    var started: String
    /// The settings that shape the figures: runs, warm-up, cache limit.
    var settings: [String: String]
    var latency: [LatencyRow]?
    var memory: [MemoryRow]?
    var prefill: [PrefillRow]?
    var stages: [StageRow]?
}

/// `Tools/bench/results/<date>-<machine>.json`: every run of one machine on one day.
struct BenchResultFile: Codable, Equatable, Sendable {
    var schema = "openjevswift-bench/1"
    var machine: Machine
    var runs: [BenchRun]

    /// The file's name for `machine` on `date` (`yyyy-MM-dd`).
    static func fileName(date: String, machine: Machine) -> String {
        "\(date)-\(machine.slug).json"
    }

    /// Adds `run` to the file at `url`, creating it, and writes it back with sorted keys.
    static func append(_ run: BenchRun, machine: Machine, to url: URL) throws {
        var file = BenchResultFile(machine: machine, runs: [])
        if let data = try? Data(contentsOf: url) {
            file = try JSONDecoder().decode(BenchResultFile.self, from: data)
        }
        file.machine = machine
        file.runs.append(run)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(file)
        data.append(0x0A)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
}

/// The markdown the default output prints.
enum Markdown {
    /// A table with leading and trailing pipes on every row.
    static func table(_ header: [String], _ rows: [[String]]) -> String {
        let lines = [header, header.map { _ in "---" }] + rows
        return lines.map { "| " + $0.joined(separator: " | ") + " |" }.joined(separator: "\n")
    }

    static func ms(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    static func gib(_ value: Double) -> String {
        String(format: "%.2f", value)
    }

    /// The tables of one run under a caption naming the machine.
    static func render(_ run: BenchRun, machine: Machine) -> String {
        var target = run.target
        if let url = run.url {
            target += " \(url)"
        }
        if let server = run.server {
            target += " (\(server))"
        }
        let settings = run.settings.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }
            .joined(separator: ", ")
        var parts = ["**\(run.mode)**, \(target); \(settings). \(machine.caption)."]
        if let rows = run.latency {
            let timed = rows.contains { $0.model != nil }
            let throughput = rows.contains { $0.throughput != nil }
            var header = ["Requests", "n", "p50 ms", "p95 ms", "mean ms"]
            if timed { header += ["model p50 ms", "model p95 ms"] }
            if throughput { header += ["requests/s"] }
            header += ["input tokens"]
            parts.append(
                table(
                    header,
                    rows.map { row in
                        var cells = [
                            row.label, "\(row.latency.count)", ms(row.latency.p50),
                            ms(row.latency.p95), ms(row.latency.mean),
                        ]
                        if timed {
                            cells += [
                                row.model.map { ms($0.p50) } ?? "",
                                row.model.map { ms($0.p95) } ?? "",
                            ]
                        }
                        if throughput {
                            cells.append(row.throughput.map { String(format: "%.2f", $0) } ?? "")
                        }
                        cells.append(row.inputTokens.map(String.init) ?? "")
                        return cells
                    }))
        }
        if let rows = run.memory {
            parts.append(
                table(
                    [
                        "When", "MLX active GiB", "MLX cache GiB", "MLX peak GiB", "resident GiB",
                        "peak resident GiB", "cached prefills",
                    ],
                    rows.map {
                        [
                            $0.label, gib($0.activeGiB), gib($0.cacheGiB), gib($0.peakGiB),
                            gib($0.residentGiB), gib($0.peakResidentGiB), "\($0.cachedPrefills)",
                        ]
                    }))
        }
        if let rows = run.prefill {
            parts.append(
                table(
                    [
                        "State", "prompt tokens", "cold p50 ms", "cached p50 ms", "prefill ms",
                        "tokens/s",
                    ],
                    rows.map {
                        [
                            $0.label, "\($0.promptTokens)", ms($0.cold.p50), ms($0.warm.p50),
                            ms($0.prefillMilliseconds), String(format: "%.0f", $0.tokensPerSecond),
                        ]
                    }))
        }
        if let rows = run.stages {
            parts.append(
                table(
                    ["Phase", "Stage", "mean ms", "share"],
                    rows.map {
                        [
                            $0.phase, $0.stage, ms($0.meanMilliseconds),
                            $0.share.map { String(format: "%.1f%%", $0 * 100) } ?? "",
                        ]
                    }))
        }
        return parts.joined(separator: "\n\n")
    }
}
