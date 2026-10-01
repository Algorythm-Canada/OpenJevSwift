import Foundation
import MLX
import OpenJevCore
import OpenJevTestSupport
import Testing

@testable import OpenJevDiffusionGemma

/// The oracle's prompts and reads, the parts the runtime tests use.
private struct OracleFixture: Decodable {
    struct Prompt: Decodable {
        let system: String
        let user: String
        let ids: [Int]
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
        let promptTokens: Int
        let distributions: [Distribution]
        enum CodingKeys: String, CodingKey {
            case id, prompt, canvas, slots, steps, distributions
            case promptTokens = "prompt_tokens"
        }
    }
    let prompts: [String: Prompt]
    let reads: [Read]

    static func load() throws -> OracleFixture {
        let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent("oracle/reads.json")
        return try JSONDecoder().decode(OracleFixture.self, from: Data(contentsOf: url))
    }
}

/// A request body as JSON text.
private func request(_ text: String) throws -> SystemOneRequest {
    try SystemOneRequest(json: JSONParser().parse(text))
}

/// Upstream's tests/test_live.py `QUESTIONS`, the README example.
private let readmeQuestions = """
    {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
     "team": {"type": "choice", "instructions": "Which team should handle it?",
              "criteria": {"outage": "service down", "billing": "charges, refunds",
                           "feature": "requests, how-to"}},
     "tone": {"type": "score", "instructions": "How upset is the customer?",
              "criteria": ["calm", "annoyed", "furious"]}}
    """

/// The README example over `state`.
private func readmeRequest(state: String) throws -> SystemOneRequest {
    let quoted = String(decoding: try JSONEncoder().encode(state), as: UTF8.self)
    return try request(
        #"{"model": "openjev-latest", "state": \#(quoted), "questions": \#(readmeQuestions)}"#)
}

/// The README quickstart request of Fixtures/wire/cases.json.
private func quickstartRequest() throws -> SystemOneRequest {
    let recorded = try WireFixtures.recordedCase(named: "quickstart")
    let body = try #require(recorded["request"]?["body_text"]?.stringValue)
    return try request(body)
}

private func mean(_ values: [Double]) -> Double {
    values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
}

private func gib(_ bytes: Int) -> String {
    String(format: "%.2f GiB", Double(bytes) / Double(1 << 30))
}

private func seconds(_ duration: Duration) -> String {
    let parts = duration.components
    return String(
        format: "%.2f s", Double(parts.seconds) + Double(parts.attoseconds) / 1e18)
}

extension MLXTests {
    /// The runtime on the pinned checkpoint, loaded with ``DiffusionGemmaRuntime/load(_:configuration:cache:token:resolver:progress:)``
    /// from ``ModelFixtures/checkpointDirectory`` as a ``ModelSource/directory(_:)``.
    @Suite(
        "DiffusionGemma runtime on the checkpoint",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage))
    struct RuntimeLiveTests {
        @Test("The load report: resolve, tokenizer, weights, warm-up and memory")
        func loadReport() async throws {
            let live = try await LiveCheckpoint.shared()
            let report = live.report
            print(
                """
                runtime load report:
                  directory \(report.directory.path)
                  resolve \(seconds(report.resolveTime)), downloaded \(report.downloadedFiles) files, \(report.downloadedBytes) bytes
                  tokenizer \(seconds(report.tokenizerMetrics.wallTime))
                  weights \(seconds(report.modelMetrics.wallTime)), \(gib(report.modelMetrics.mappedBytes)) of shards, MLX active \(gib(report.modelMetrics.mlxActiveBytes))
                  warm-up \(report.warmUpTime.map(seconds) ?? "off")
                  memory \(report.memory)
                """)
            #expect(report.downloadedFiles == 0 && report.downloadedBytes == 0)
            #expect(report.warmUpTime != nil)
            #expect(report.modelMetrics.shardCount == 4)
            #expect(report.memory.activeBytes > 10 << 30)
        }

        @Test("upstream's README example: urgent, team outage, tone annoyed or furious")
        func readmeExample() async throws {
            let live = try await LiveCheckpoint.shared()
            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            let decision = try await engine.decide(
                readmeRequest(
                    state: "Everything is down and we have a demo with our biggest client at noon.")
            )
            print("README example answers: \(decision.answers)")
            #expect(decision.answers.count == 3)
            guard case .noul(let urgent)? = decision.answers["urgent"],
                case .choice(let team, let teams, _)? = decision.answers["team"],
                case .score(let score, _, let levels, _)? = decision.answers["tone"]
            else {
                Issue.record("unexpected answer types: \(decision.answers)")
                return
            }
            // upstream's test_readme_example.
            #expect(urgent > 0.5)
            #expect(team == "outage")
            #expect(score > 1.0)
            #expect(abs(teams.values.reduce(0, +) - 1) < 1e-6)
            // The most probable tone is annoyed or furious; on this checkpoint it is furious.
            let mostProbable = levels.indices.max { levels[$0] < levels[$1] }
            #expect(mostProbable == 1 || mostProbable == 2, "tone levels \(levels)")
        }

        @Test("The runtime's reads meet D-014's native bounds over the 27 oracle reads")
        func oracleBounds() async throws {
            let live = try await LiveCheckpoint.shared()
            let oracle = try OracleFixture.load()
            var labels: [Double] = []
            var longLabels: [Double] = []
            var entropies: [Double] = []
            var longEntropies: [Double] = []
            var slots = 0
            var topAgree = 0
            var confident = 0
            var confidentAgree = 0
            var quickstart: [String] = []
            let top = { (p: [Double]) in p.indices.max { p[$0] < p[$1] } ?? 0 }
            for read in oracle.reads {
                let prompt = try #require(oracle.prompts[read.prompt])
                let canvasRead = CanvasRead(
                    prompt: .tokens(prompt.ids), systemText: prompt.system,
                    stateText: prompt.user, template: [],
                    slots: read.slots.map { .init(position: $0.pos, labelIDs: $0.labelIDs) },
                    canvas: SeededCanvas(tokens: read.canvas, noise: []), steps: read.steps,
                    seed: 0)
                let result = try await live.runtime.read(canvasRead)
                #expect(result.promptTokens == read.promptTokens)
                let long = read.promptTokens > 1024
                var readDifferences: [Double] = []
                for (ours, theirs) in zip(result.slots, read.distributions) {
                    slots += 1
                    let differences = zip(ours.probabilities, theirs.probs).map { abs($0 - $1) }
                    readDifferences += differences
                    labels += differences
                    entropies.append(abs(ours.entropy - theirs.entropy))
                    if long {
                        longLabels += differences
                        longEntropies.append(abs(ours.entropy - theirs.entropy))
                    }
                    let agree = top(ours.probabilities) == top(theirs.probs)
                    if agree { topAgree += 1 }
                    let sorted = theirs.probs.sorted(by: >)
                    if sorted.count < 2 || sorted[0] - sorted[1] >= 0.5 {
                        confident += 1
                        if agree { confidentAgree += 1 }
                    }
                }
                if read.prompt == "quickstart/g0" {
                    let ours = result.slots.map {
                        $0.probabilities.map { String(format: "%.4f", $0) }
                    }
                    let theirs = read.distributions.map {
                        $0.probs.map { String(format: "%.4f", $0) }
                    }
                    quickstart.append(
                        "\(read.id): ours \(ours), oracle \(theirs), mean |dp| "
                            + String(format: "%.4f", mean(readDifferences)))
                }
            }
            let meanP = mean(labels)
            let longMeanP = mean(longLabels)
            let meanH = mean(entropies)
            let longMeanH = mean(longEntropies)
            let topShare = Double(topAgree) / Double(slots)
            let confidentShare = Double(confidentAgree) / Double(confident)
            print(
                """
                runtime against the oracle, native tier, \(oracle.reads.count) reads, \(slots) slots:
                  mean |dp| all labels \(String(format: "%.4f", meanP)) (bound 0.02)
                  mean |dp| long prompts \(String(format: "%.4f", longMeanP)) (bound 0.01)
                  mean |dH| all slots \(String(format: "%.4f", meanH)) (bound 0.2)
                  mean |dH| long prompts \(String(format: "%.4f", longMeanH)) (bound 0.2)
                  top label \(topAgree)/\(slots) \(String(format: "%.1f%%", topShare * 100)) (bound 90%)
                  top label, margin >= 0.5: \(confidentAgree)/\(confident) \(String(format: "%.1f%%", confidentShare * 100)) (bound 97%)
                quickstart/g0, reported, not bounded:
                  \(quickstart.joined(separator: "\n  "))
                """)
            #expect(oracle.reads.count == 27 && slots == 156)
            #expect(meanP <= 0.02)
            #expect(longMeanP <= 0.01)
            #expect(meanH <= 0.2)
            #expect(longMeanH <= 0.2)
            #expect(topShare >= 0.9)
            #expect(confidentShare >= 0.97)
        }

        @Test("The runtime's read is the model's read, bit for bit")
        func runtimeIsTheModel() async throws {
            let live = try await LiveCheckpoint.shared()
            let oracle = try OracleFixture.load()
            let model = live.loaded.model
            for read in oracle.reads where read.prompt == "quickstart/g0" {
                let prompt = try #require(oracle.prompts[read.prompt])
                let requests = read.slots.map {
                    SlotRequest(position: $0.pos, labelIDs: $0.labelIDs)
                }
                let canvasRead = CanvasRead(
                    prompt: .tokens(prompt.ids), systemText: prompt.system,
                    stateText: prompt.user, template: [],
                    slots: read.slots.map { .init(position: $0.pos, labelIDs: $0.labelIDs) },
                    canvas: SeededCanvas(tokens: read.canvas, noise: []), steps: read.steps,
                    seed: 0)
                let ours = try await live.runtime.modelRead(canvasRead).output
                let direct = try model.read(
                    canvas: read.canvas, slots: requests,
                    cache: model.prefill(promptIDs: prompt.ids), steps: read.steps)
                for (a, b) in zip(ours.slots, direct.slots) {
                    #expect(a.map(\.tokenID) == b.map(\.tokenID), "\(read.id)")
                    #expect(
                        a.map { Float($0.logprob).bitPattern }
                            == b.map { Float($0.logprob).bitPattern }, "\(read.id)")
                }
            }
        }

        @Test("The wire quickstart through the engine: three answers, department technical")
        func quickstart() async throws {
            let live = try await LiveCheckpoint.shared()
            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            let decision = try await engine.decide(quickstartRequest())
            print("quickstart answers: \(decision.answers)")
            #expect(decision.answers.keys == ["department", "frustration", "is_urgent"])
            guard case .choice(let department, _, _)? = decision.answers["department"],
                case .score(_, _, let frustration, _)? = decision.answers["frustration"],
                case .noul(let urgent)? = decision.answers["is_urgent"]
            else {
                Issue.record("unexpected answer types: \(decision.answers)")
                return
            }
            // The oracle's top labels where its margin is above 0.8 on every read; is_urgent's
            // reads sit near 0.5 (0.57, 0.94, 0.52) and are reported, not asserted (D-014).
            #expect(department == "technical")
            #expect(frustration.indices.max { frustration[$0] < frustration[$1] } == 0)
            #expect((0...1).contains(urgent))
            #expect(decision.inputTokens % 182 == 0)
        }

        @Test("The same request twice answers the same; a cold and a cached read are bit-identical")
        func determinism() async throws {
            let live = try await LiveCheckpoint.shared()
            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            let request = try quickstartRequest()
            let first = try await engine.decide(request)
            let second = try await engine.decide(request)
            #expect(first.answers == second.answers)

            let oracle = try OracleFixture.load()
            let read = try #require(oracle.reads.first { $0.id == "quickstart/g0/c0/steps1" })
            let prompt = try #require(oracle.prompts[read.prompt])
            let canvasRead = CanvasRead(
                prompt: .tokens(prompt.ids), systemText: prompt.system, stateText: prompt.user,
                template: [],
                slots: read.slots.map { .init(position: $0.pos, labelIDs: $0.labelIDs) },
                canvas: SeededCanvas(tokens: read.canvas, noise: []), steps: 1, seed: 0)
            await live.runtime.removeCachedPrefills()
            let before = await live.runtime.statistics()
            let cold = try await live.runtime.modelRead(canvasRead).output
            let cached = try await live.runtime.modelRead(canvasRead).output
            let after = await live.runtime.statistics()
            #expect(after.prefillMisses == before.prefillMisses + 1)
            #expect(after.prefillHits == before.prefillHits + 1)
            #expect(cold.slots.count == cached.slots.count)
            for (a, b) in zip(cold.slots, cached.slots) {
                #expect(a.map(\.tokenID) == b.map(\.tokenID))
                #expect(
                    a.map { Float($0.logprob).bitPattern } == b.map { Float($0.logprob).bitPattern }
                )
            }
        }

        @Test("20 concurrent requests answer as the same requests one at a time")
        func concurrentCallers() async throws {
            let live = try await LiveCheckpoint.shared()
            let oracle = try OracleFixture.load()
            // The oracle's twelve states under the README questions, then eight of them under
            // the quickstart's: 20 different requests.
            let states = oracle.prompts.keys.sorted().compactMap { oracle.prompts[$0]?.user }
            let quickstart = try quickstartRequest()
            var requests = try states.map { try readmeRequest(state: $0) }
            for state in states.prefix(20 - requests.count) {
                var request = quickstart
                request.state = .string(state)
                requests.append(request)
            }
            #expect(requests.count == 20)

            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            var serial: [OrderedMap<Answer>] = []
            for request in requests {
                serial.append(try await engine.decide(request).answers)
            }
            await live.runtime.removeCachedPrefills()
            let concurrent = try await withThrowingTaskGroup(
                of: (Int, OrderedMap<Answer>).self
            ) { group in
                for (index, request) in requests.enumerated() {
                    group.addTask { (index, try await engine.decide(request).answers) }
                }
                var answers = [OrderedMap<Answer>?](repeating: nil, count: requests.count)
                for try await (index, value) in group {
                    answers[index] = value
                }
                return answers
            }
            for index in requests.indices {
                #expect(concurrent[index] == serial[index], "request \(index)")
            }
        }

        @Test("Memory over 100 unique prompts, with a 4 GB cache limit and without")
        func memory() async throws {
            let live = try await LiveCheckpoint.shared()
            let runtime = live.runtime
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let originalLimit = Memory.cacheLimit
            defer { Memory.cacheLimit = originalLimit }

            func run(_ label: String, limitGB: Double?) async throws -> (
                DiffusionGemmaRuntime.MemoryReport, DiffusionGemmaRuntime.MemoryReport, Int
            ) {
                await runtime.removeCachedPrefills()
                Memory.cacheLimit = originalLimit
                try await runtime.setCacheLimit(gb: limitGB)
                Memory.clearCache()
                Memory.peakMemory = 0
                let before = await runtime.memoryReport()
                let reads = await runtime.statistics().reads
                for index in 0..<100 {
                    _ = try await engine.decide(
                        readmeRequest(
                            state: "Ticket \(label)-\(index): everything is down and we have a "
                                + "demo with client \(index) at noon."))
                }
                let after = await runtime.memoryReport()
                let count = await runtime.statistics().reads - reads
                print(
                    """
                    memory, cache limit \(limitGB.map { "\($0) GB" } ?? "unset"), \(count) reads of 100 prompts:
                      before \(before)
                      after  \(after)
                    """)
                return (before, after, count)
            }

            let (_, limited, limitedReads) = try await run("limited", limitGB: 4)
            let (_, unlimited, _) = try await run("unlimited", limitGB: nil)
            #expect(limitedReads >= 100)
            // MLX trims the pool on the next allocation, so it can pass the limit by one buffer.
            #expect(limited.cacheBytes <= (4 << 30) + (512 << 20))
            #expect(limited.activeBytes + limited.cacheBytes < 26 << 30)
            // The resident bound R4 records: 15.1 to 15.5 GiB were measured on an M3 Max.
            #expect(limited.residentBytes < 20 << 30)
            #expect(unlimited.residentBytes < 20 << 30)
            #expect(await runtime.statistics().cachedPrefills <= 12)
        }
    }
}
