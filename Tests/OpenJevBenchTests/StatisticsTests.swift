import ArgumentParser
import Foundation
import Testing

@testable import openjev_bench

/// openjev-bench's model-free parts (issue #32): the percentiles, the server-timing header, the
/// request sets, the tables and the result file.
@Suite("openjev-bench statistics")
struct StatisticsTests {
    @Test("Percentiles interpolate between ranks as NumPy's default does")
    func percentiles() {
        // numpy.percentile([1, 2, 3, 4], [50, 95]) is [2.5, 3.85].
        let summary = LatencySummary(milliseconds: [4, 1, 3, 2])
        #expect(summary.p50 == 2.5)
        #expect(abs(summary.p95 - 3.85) < 1e-12)
        #expect(summary.mean == 2.5)
        #expect(summary.min == 1 && summary.max == 4 && summary.count == 4)
        // numpy.percentile(range(1, 51), 95) is 47.55.
        let fifty = LatencySummary(milliseconds: (1...50).map(Double.init))
        #expect(abs(fifty.p95 - 47.55) < 1e-9)
        #expect(fifty.p50 == 25.5)
        let one = LatencySummary(milliseconds: [7])
        #expect(one.p50 == 7 && one.p95 == 7)
    }

    @Test("A Duration in milliseconds")
    func durationMilliseconds() {
        #expect(milliseconds(.seconds(1.5)) == 1500)
        #expect(abs(milliseconds(.microseconds(250)) - 0.25) < 1e-12)
    }

    @Test("server-timing: model, server and total in milliseconds")
    func serverTiming() {
        let metrics = ServerTiming.parse("model;dur=212.4, server;dur=3.1, total;dur=215.5")
        #expect(metrics == ["model": 212.4, "server": 3.1, "total": 215.5])
        #expect(ServerTiming.parse("cache;desc=hit, db;dur=5") == ["db": 5])
        #expect(ServerTiming.parse("").isEmpty)
    }

    @Test("The request sets: 1, 3 and 12 questions, samples 1, valid JSON")
    func requestSets() throws {
        for (count, keys) in [(1, 1), (3, 3), (12, 12)] {
            let body = Workload.body(
                state: Workload.uniqueState(tag: "t", index: 3),
                questions: Workload.questions(count))
            let object = try #require(
                JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
            #expect((object["questions"] as? [String: Any])?.count == keys)
            #expect(object["samples"] as? Int == 1)
            #expect((object["state"] as? String)?.hasPrefix("Ticket t-3: ") == true)
        }
        #expect(
            Workload.uniqueState(tag: "a", index: 1) != Workload.uniqueState(tag: "a", index: 2))
        let long = Workload.longState(tokens: 1_000, tag: "a", index: 0)
        #expect(long.count > 4_000)
    }

    @Test("The machine's caption and file name")
    func machine() {
        let machine = Machine(
            model: "Mac15,8", chip: "Apple M3 Max", memoryGB: 128,
            macOS: "Version 27.0.1 (Build 26A434)", onACPower: true)
        #expect(machine.slug == "apple-m3-max-128gb")
        #expect(
            BenchResultFile.fileName(date: "2026-10-01", machine: machine)
                == "2026-10-01-apple-m3-max-128gb.json")
        #expect(
            machine.caption
                == "Apple M3 Max (Mac15,8), 128 GB, macOS Version 27.0.1 (Build 26A434), on AC power"
        )
    }

    @Test("Markdown tables have leading and trailing pipes on every row")
    func markdown() {
        let table = Markdown.table(["a", "b"], [["1", "2"]])
        #expect(table == "| a | b |\n| --- | --- |\n| 1 | 2 |")
        let run = BenchRun(
            mode: "reads", target: "http", url: "http://127.0.0.1:1", server: "upstream",
            modelDirectory: nil, started: "2026-10-01T00:00:00Z", settings: ["runs": "2"],
            latency: [
                LatencyRow(
                    label: "1 question", latency: LatencySummary(milliseconds: [100, 200]),
                    model: LatencySummary(milliseconds: [90, 180]), throughput: nil,
                    inputTokens: 120)
            ])
        let machine = Machine(
            model: "Mac15,8", chip: "Apple M3 Max", memoryGB: 128, macOS: "27", onACPower: nil)
        let text = Markdown.render(run, machine: machine)
        #expect(text.contains("| 1 question | 2 | 150.0 | 195.0 | 150.0 | 135.0 | 175.5 | 120 |"))
        #expect(text.contains("Apple M3 Max (Mac15,8), 128 GB, macOS 27"))
        for line in text.split(separator: "\n") where line.hasPrefix("|") {
            #expect(line.hasSuffix("|"))
        }
    }

    @Test("Profile stages: observer names map to stages, rows render with shares")
    func profileStages() {
        #expect(profileStage("attn.3") == "attention")
        #expect(profileStage("mlp.0") == "dense MLP")
        #expect(profileStage("router.29.weights") == "router")
        #expect(profileStage("experts.12") == "experts (gathered quantized matmuls)")
        #expect(profileStage("layer.5") == "norms, residuals, layer scalar")
        #expect(profileStage("final norm") == "final norm")
        let run = BenchRun(
            mode: "profile", target: "engine", url: nil, server: nil, modelDirectory: nil,
            started: "s", settings: [:],
            stages: [
                StageRow(
                    phase: "prefill", stage: "attention", meanMilliseconds: 12.34, share: 0.25),
                StageRow(phase: "prefill", stage: "unstaged", meanMilliseconds: 40, share: nil),
            ])
        let text = Markdown.render(
            run, machine: Machine(model: "m", chip: "c", memoryGB: 1, macOS: "x", onACPower: nil))
        #expect(text.contains("| prefill | attention | 12.3 | 25.0% |"))
        #expect(text.contains("| prefill | unstaged | 40.0 |  |"))
    }

    @Test("Runs are appended to the day's result file")
    func resultFile() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "openjev-bench-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("results/day.json")
        let machine = Machine(
            model: "m", chip: "c", memoryGB: 1, macOS: "x", onACPower: true)
        let run = BenchRun(
            mode: "prefill", target: "engine", url: nil, server: nil, modelDirectory: "/m",
            started: "s", settings: [:],
            prefill: [
                PrefillRow(
                    label: "about 1000 tokens", promptTokens: 1_020,
                    cold: LatencySummary(milliseconds: [900]),
                    warm: LatencySummary(milliseconds: [100]), prefillMilliseconds: 800,
                    tokensPerSecond: 1_275)
            ])
        try BenchResultFile.append(run, machine: machine, to: url)
        try BenchResultFile.append(run, machine: machine, to: url)
        let file = try JSONDecoder().decode(BenchResultFile.self, from: Data(contentsOf: url))
        #expect(file.runs.count == 2)
        #expect(file.schema == "openjevswift-bench/1")
        #expect(file.runs[0] == run)
    }

    @Test("The model directory: --model, then OPENJEV_TEST_MODEL, then the cache snapshot")
    func modelDirectory() throws {
        var options = try CommonOptions.parse(["--model", "/tmp/model"])
        #expect(options.modelDirectory(environment: [:]).path == "/tmp/model")
        options = try CommonOptions.parse([])
        #expect(
            options.modelDirectory(environment: ["OPENJEV_TEST_MODEL": "/m"]).path == "/m")
        #expect(
            options.modelDirectory(environment: ["HF_HUB_CACHE": "/cache"]).path
                == "/cache/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots/"
                + "a7a81407613811e8ba63af92ac0d852b809e191f")
        #expect(options.runs == 50 && options.warmup == 5 && !options.json)
    }

    @Test("Every mode parses; memory, prefill and profile take no --url")
    func parsing() throws {
        let reads = try BenchCommand.parseAsRoot(["reads", "--url", "http://127.0.0.1:8000"])
        #expect((reads as? Reads)?.http.url == "http://127.0.0.1:8000")
        let concurrency = try BenchCommand.parseAsRoot(["concurrency", "--levels", "1", "2"])
        #expect((concurrency as? Concurrency)?.levels == [1, 2])
        let memory = try BenchCommand.parseAsRoot(["memory", "--cache-limit-gb", "4"])
        #expect((memory as? MemoryMode)?.cacheLimitGB == 4)
        #expect(throws: (any Error).self) {
            try BenchCommand.parseAsRoot(["memory", "--url", "http://x"])
        }
        let prefill = try BenchCommand.parseAsRoot(["prefill", "--tokens", "500"])
        #expect((prefill as? Prefill)?.tokens == [500])
        let profile = try BenchCommand.parseAsRoot(["profile", "--runs", "20"])
        #expect((profile as? Profile)?.common.runs == 20)
        #expect((profile as? Profile)?.stateTokens == nil)
        let long = try BenchCommand.parseAsRoot(["profile", "--state-tokens", "10000"])
        #expect((long as? Profile)?.stateTokens == 10_000)
        let wheel = try BenchCommand.parseAsRoot(["reads", "--metallib", "/tmp/mlx.metallib"])
        #expect((wheel as? Reads)?.common.metallib == "/tmp/mlx.metallib")
    }
}
