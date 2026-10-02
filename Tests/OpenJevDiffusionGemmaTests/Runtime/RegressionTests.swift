import Foundation
import Metal
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// Fixtures/regression/reads.json: the Swift port's own answers for a fixed request set, with
/// the pins they were recorded under (Fixtures/regression/README.md).
struct RegressionFile: Codable, Equatable {
    /// The pins the file was recorded under, in the `generator` object every fixture file
    /// starts with (FixturePinTests).
    struct Pins: Codable, Equatable {
        /// The test that wrote the file.
        var script = "Tests/OpenJevDiffusionGemmaTests/Runtime/RegressionTests.swift"
        var modelRepo = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
        /// The checkpoint directory's name: the snapshot's commit for a Hugging Face cache.
        var modelRevision: String
        /// The mlx-swift version Package.resolved pins.
        var mlxSwift: String
        var macOS: String
        var gpu: String
        /// The day it was recorded, `yyyy-MM-dd`.
        var date: String
        /// The file's format.
        var version = 1

        enum CodingKeys: String, CodingKey {
            case script, version, date, gpu
            case modelRepo = "model_repo"
            case modelRevision = "model_revision"
            case mlxSwift = "mlx_swift"
            case macOS = "macos"
        }

        /// Whether the figures can be expected to repeat exactly: the same checkpoint, MLX,
        /// macOS and GPU. The date does not matter.
        func sameMachine(as other: Pins) -> Bool {
            modelRevision == other.modelRevision && mlxSwift == other.mlxSwift
                && macOS == other.macOS && gpu == other.gpu
        }
    }
    struct Slot: Codable, Equatable {
        /// One probability per label, in label order.
        var probabilities: [Double]
        var entropy: Double
        /// The index of the first largest probability, and its label's token id.
        var topLabel: Int
        var topLabelID: Int
    }
    struct Read: Codable, Equatable {
        var steps: Int
        var promptTokens: Int
        var slots: [Slot]
    }
    struct AnswerProbabilities: Codable, Equatable {
        var key: String
        /// A noul's yes probability, a choice's or a score's distribution.
        var probabilities: [Double]
    }
    struct Entry: Codable, Equatable {
        /// An oracle read's id, or `engine/quickstart` and `engine/readme`.
        var id: String
        /// The engine requests' billed input tokens; nil for an oracle read.
        var inputTokens: Int?
        /// The engine requests' answers in request order; nil for an oracle read.
        var answers: [AnswerProbabilities]?
        /// One read for an oracle read; for an engine request every read it made, sorted by
        /// seed, steps and canvas so concurrent reads record in one order.
        var reads: [Read]
    }
    var generator: Pins
    var entries: [Entry]
}

/// A backend that passes every read to the runtime and keeps what it returned.
private actor RecordingBackend: DecisionBackend {
    nonisolated let runtime: DiffusionGemmaRuntime
    private(set) var reads: [(CanvasRead, ReadResult)] = []

    init(runtime: DiffusionGemmaRuntime) {
        self.runtime = runtime
    }

    nonisolated var tokenizer: any DecisionTokenizer { runtime.tokenizer }
    nonisolated var maxPromptTokens: Int { runtime.maxPromptTokens }
    nonisolated var capabilities: BackendCapabilities { runtime.capabilities }
    nonisolated var modelName: String { runtime.modelName }

    func read(_ read: CanvasRead) async throws -> ReadResult {
        let result = try await runtime.read(read)
        reads.append((read, result))
        return result
    }

    func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws -> ThoughtGeneration {
        try await runtime.think(prompt: prompt, budget: budget, stopIDs: stopIDs)
    }
}

extension MLXTests {
    /// The regression file of issue #31: the port's answers for the 27 oracle reads, the wire
    /// quickstart and upstream's README example through the engine, against the recorded ones.
    @Suite(
        "DiffusionGemma regression file",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage))
    struct RegressionTests {
        /// The largest change allowed in any recorded probability or entropy when the pins match.
        /// Five runs of the suite on the M3 Max reproduced every value bit for bit (D-043), so
        /// any change is a change in what the port computes.
        static let tolerance = 0.0

        static let fileURL = TokenizerFixtures.fixturesDirectory.appendingPathComponent(
            "regression/reads.json")

        /// The variable that records the file instead of comparing with it.
        static let recordVariable = "OPENJEV_RECORD_REGRESSION"

        static func read(_ steps: Int, _ result: ReadResult, labelIDs: [[Int]])
            -> RegressionFile.Read
        {
            RegressionFile.Read(
                steps: steps, promptTokens: result.promptTokens,
                slots: zip(result.slots, labelIDs).map { slot, ids in
                    let top = ReadDivergence.firstLargest(slot.probabilities)
                    return .init(
                        probabilities: slot.probabilities, entropy: slot.entropy, topLabel: top,
                        topLabelID: ids[top])
                })
        }

        static func answers(_ decision: Decision) -> [RegressionFile.AnswerProbabilities] {
            decision.answers.map { key, answer in
                let probabilities: [Double]
                switch answer {
                case .noul(let p): probabilities = [p]
                case .choice(_, let map, _): probabilities = map.map(\.value)
                case .score(_, _, let levels, _): probabilities = levels
                }
                return .init(key: key, probabilities: probabilities)
            }
        }

        /// The pins of this process.
        static func currentPins() throws -> RegressionFile.Pins {
            let resolved = ModelFixtures.repositoryRoot.appendingPathComponent("Package.resolved")
            struct Resolved: Decodable {
                struct Pin: Decodable {
                    struct State: Decodable { let version: String? }
                    let identity: String
                    let state: State
                }
                let pins: [Pin]
            }
            let pins = try JSONDecoder().decode(Resolved.self, from: Data(contentsOf: resolved))
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = "yyyy-MM-dd"
            return .init(
                modelRevision: ModelFixtures.checkpointDirectory.lastPathComponent,
                mlxSwift: pins.pins.first { $0.identity == "mlx-swift" }?.state.version
                    ?? "unknown",
                macOS: ProcessInfo.processInfo.operatingSystemVersionString,
                gpu: MTLCreateSystemDefaultDevice()?.name ?? "unknown",
                date: formatter.string(from: Date()))
        }

        /// The answers of this build.
        static func produce(_ live: LiveCheckpoint) async throws -> [RegressionFile.Entry] {
            var entries: [RegressionFile.Entry] = []
            let oracle = try OracleFixture.load()
            for read in oracle.reads {
                let prompt = try #require(oracle.prompts[read.prompt])
                let result = try await live.runtime.read(
                    CanvasRead(
                        prompt: .tokens(prompt.ids), systemText: prompt.system,
                        stateText: prompt.user, template: [],
                        slots: read.slots.map { .init(position: $0.pos, labelIDs: $0.labelIDs) },
                        canvas: SeededCanvas(tokens: read.canvas, noise: []), steps: read.steps,
                        seed: 0))
                entries.append(
                    .init(
                        id: read.id, inputTokens: nil, answers: nil,
                        reads: [Self.read(read.steps, result, labelIDs: read.slots.map(\.labelIDs))]
                    ))
            }
            let requests = [
                ("engine/quickstart", try quickstartRequest()),
                (
                    "engine/readme",
                    try readmeRequest(
                        state:
                            "Everything is down and we have a demo with our biggest client at noon."
                    )
                ),
            ]
            for (id, request) in requests {
                let backend = RecordingBackend(runtime: live.runtime)
                let engine = try DecisionEngine(backend: backend, configuration: .default)
                let decision = try await engine.decide(request)
                let reads = await backend.reads.sorted { a, b in
                    (a.0.seed, a.0.steps) != (b.0.seed, b.0.steps)
                        ? (a.0.seed, a.0.steps) < (b.0.seed, b.0.steps)
                        : a.0.canvas.tokens.lexicographicallyPrecedes(b.0.canvas.tokens)
                }
                entries.append(
                    .init(
                        id: id, inputTokens: decision.inputTokens, answers: answers(decision),
                        reads: reads.map {
                            Self.read($0.0.steps, $0.1, labelIDs: $0.0.slots.map(\.labelIDs))
                        }))
            }
            return entries
        }

        @Test("The port's answers match Fixtures/regression/reads.json")
        func regression() async throws {
            let live = try await LiveCheckpoint.shared()
            let pins = try Self.currentPins()
            let entries = try await Self.produce(live)

            if ProcessInfo.processInfo.environment[Self.recordVariable] == "1" {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                var data = try encoder.encode(RegressionFile(generator: pins, entries: entries))
                data.append(0x0A)
                try data.write(to: Self.fileURL)
                print("recorded \(entries.count) entries to \(Self.fileURL.path) under \(pins)")
                return
            }

            let recorded = try JSONDecoder().decode(
                RegressionFile.self, from: Data(contentsOf: Self.fileURL))
            #expect(recorded.entries.map(\.id) == entries.map(\.id))
            let exact = recorded.generator.sameMachine(as: pins)
            var largest = 0.0
            var differences: [Double] = []
            var slots = 0
            var topAgree = 0
            for (want, got) in zip(recorded.entries, entries) {
                #expect(got.inputTokens == want.inputTokens, "\(want.id)")
                #expect(got.reads.count == want.reads.count, "\(want.id)")
                for (wantRead, gotRead) in zip(want.reads, got.reads) {
                    #expect(gotRead.promptTokens == wantRead.promptTokens, "\(want.id)")
                    #expect(gotRead.steps == wantRead.steps, "\(want.id)")
                    for (index, (a, b)) in zip(wantRead.slots, gotRead.slots).enumerated() {
                        slots += 1
                        topAgree += a.topLabel == b.topLabel ? 1 : 0
                        let moved =
                            zip(a.probabilities, b.probabilities).map { abs($0 - $1) } + [
                                abs(a.entropy - b.entropy)
                            ]
                        differences += moved.dropLast()
                        largest = max(largest, moved.max() ?? 0)
                        if exact {
                            #expect(
                                (moved.max() ?? 0) <= Self.tolerance && a.topLabel == b.topLabel,
                                "\(want.id) slot \(index): recorded \(a), now \(b)")
                        }
                    }
                }
                if exact, let wantAnswers = want.answers, let gotAnswers = got.answers {
                    for (a, b) in zip(wantAnswers, gotAnswers) {
                        let moved = zip(a.probabilities, b.probabilities).map { abs($0 - $1) }
                        largest = max(largest, moved.max() ?? 0)
                        #expect(
                            a.key == b.key && (moved.max() ?? 0) <= Self.tolerance,
                            "\(want.id) \(a.key): recorded \(a.probabilities), now \(b.probabilities)"
                        )
                    }
                }
            }
            let mean =
                differences.isEmpty ? 0 : differences.reduce(0, +) / Double(differences.count)
            print(
                """
                regression file: \(entries.count) entries, \(slots) slots, recorded under \(recorded.generator), \
                now \(pins); \(exact ? "same machine, tolerance \(Self.tolerance)" : "another machine, D-014's bounds"): \
                largest change \(largest), mean |dp| \(mean), top label \(topAgree)/\(slots)
                """)
            if !exact {
                // Another GPU, macOS or MLX rounds differently; hold the port to D-014's
                // aggregate bounds there and say how to record this machine's own file.
                #expect(mean <= 0.02, "re-record with \(Self.recordVariable)=1 on this machine")
                #expect(Double(topAgree) >= 0.9 * Double(slots))
            }
        }
    }
}
