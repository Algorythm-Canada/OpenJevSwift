import Foundation
import OpenJevCore
import OpenJevEncoders
import Testing

/// ``VerdictBackend`` over models that replay recorded logits: upstream's own read test, the
/// whole reference corpus through ``EncoderDecisionEngine`` with both batch policies, and the
/// errors a model can cause. No Core ML and no tokenizer are needed.
@Suite("Verdict backend")
struct VerdictBackendTests {
    /// A tokenizer that makes every prompt four tokens, as upstream's test tokenizer does.
    private struct FourTokens: VerdictTokenizing {
        func inputIDs(for prompt: String) -> [Int] { [1, 1, 1, 1] }
    }

    @Test("Matches upstream's test_verdict_read_calibrates_and_drops_abstention")
    func upstreamRead() async throws {
        var row0 = [Float](repeating: -100, count: 25)
        row0.replaceSubrange(0..<3, with: [2, 0, 5])  // noul: true, false, abstain
        var row1 = [Float](repeating: -100, count: 25)
        row1.replaceSubrange(0..<4, with: [1, 3, 0, 0])
        let model = FixedModel(rows: [row0, row1])
        let backend = VerdictBackend(
            model: model, tokenizer: FourTokens(),
            calibration: VerdictCalibration(temperature: 2, perK: [3: 1]), maxBatchRows: 16)
        let questions = try EncoderQuestionSchemaBuilder(maxChoices: 24).build([
            "n": .noul(instructions: .string("x"), criteria: nil),
            "c": .choice(
                instructions: .string("y"), criteria: ["a": .null, "b": .null, "c": .null]),
        ]).questions
        let result = try await backend.readBatch(
            state: .string("state"), stateText: "state", questions: questions)
        #expect(result.inputTokens == 8)
        // Upstream compares with pytest.approx, a relative 1e-6, which float32 arithmetic meets.
        // k = 3 has its own temperature (1.0); the abstention's mass is renormalised away.
        let noul = [1 / (1 + exp(-2.0)), 1 / (1 + exp(2.0))]
        #expect(zip(result.probabilities[0], noul).allSatisfy { abs($0 - $1) <= 1e-6 * $1 })
        // k = 4 has no entry, so the global temperature (2.0) applies.
        let e = [1.0, 3.0, 0.0].map { exp($0 / 2) }
        let choice = e.map { $0 / e.reduce(0, +) }
        #expect(zip(result.probabilities[1], choice).allSatisfy { abs($0 - $1) <= 1e-6 * $1 })
        #expect(await model.rowCounts == [2])
    }

    @Test(
        "The corpus through the engine gives the recorded answers and billing",
        .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage),
        arguments: [1, 16])
    func corpusThroughTheEngine(maxBatchRows: Int) async throws {
        let reference = try VerdictFixtures.reference()
        let readsByRequest = try VerdictFixtures.readsByRequest()
        var questionsRead = 0
        for corpus in try VerdictFixtures.corpus() {
            let reads = try #require(readsByRequest[corpus.name])
            let model = ReplayModel(reads)
            let backend = VerdictBackend(
                model: model, tokenizer: ReplayTokenizer(reads),
                calibration: reference.calibrator, maxBatchRows: maxBatchRows)
            let engine = EncoderDecisionEngine(
                backend: backend,
                configuration: EncoderEngineConfiguration(batchSize: reference.encoderBatch))
            let decision = try await engine.decide(corpus.request)

            // Billing: the rows' lengths, upstream's attention-mask sum for the request.
            #expect(
                decision.inputTokens == reads.last?.requestInputTokens,
                "\(corpus.name): input tokens")
            #expect(decision.inputTokens == reads.reduce(0) { $0 + $1.inputIDs.count })

            // Batching: the engine's batches of 16, each cut into calls of maxBatchRows, in
            // question order, every row with its full attention mask.
            let calls = await model.calls
            #expect(calls.flatMap(\.ids) == reads.map(\.inputIDs), "\(corpus.name): row order")
            #expect(calls.allSatisfy { $0.maskSums == $0.ids.map(\.count) })
            var expectedCalls: [Int] = []
            for batch in stride(from: 0, to: reads.count, by: reference.encoderBatch) {
                let size = min(reference.encoderBatch, reads.count - batch)
                for call in stride(from: 0, to: size, by: maxBatchRows) {
                    expectedCalls.append(min(maxBatchRows, size - call))
                }
            }
            #expect(calls.map(\.ids.count) == expectedCalls, "\(corpus.name): calls")

            // Answers in the caller's option order.
            for read in reads {
                let answer = try #require(decision.answers[read.key], "\(read.name)")
                let probabilities: [Double]
                switch answer {
                case .noul(let yes):
                    probabilities = [yes, 1 - yes]
                case .choice(_, let byOption, _):
                    probabilities = byOption.map(\.value)
                case .score(_, _, let byLevel, _):
                    probabilities = byLevel
                }
                #expect(probabilities.count == read.options, "\(read.name)")
                let worst = zip(probabilities, read.probabilities).map { abs($0 - $1) }.max() ?? 0
                #expect(worst < 1e-6, "\(read.name): \(probabilities) \(read.probabilities)")
                questionsRead += 1
            }
        }
        #expect(questionsRead == 200)
    }

    @Test(
        "A batch returns each distribution in option order, a noul's as true then false",
        .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage))
    func optionOrder() async throws {
        let reference = try VerdictFixtures.reference()
        let corpus = try #require(try VerdictFixtures.corpus().first { $0.name == "quickstart" })
        let reads = try #require(try VerdictFixtures.readsByRequest()["quickstart"])
        #expect(reads.map(\.type) == ["choice", "score", "noul"])
        let backend = VerdictBackend(
            model: ReplayModel(reads), tokenizer: ReplayTokenizer(reads),
            calibration: reference.calibrator, maxBatchRows: 16)
        let result = try await backend.readBatch(
            state: corpus.request.state, stateText: StateText.render(corpus.request.state),
            questions: try VerdictFixtures.questions(of: corpus))
        #expect(result.probabilities.count == 3)
        for (probabilities, read) in zip(result.probabilities, reads) {
            #expect(probabilities.count == read.options)
            #expect(zip(probabilities, read.probabilities).allSatisfy { abs($0 - $1) < 1e-6 })
        }
        // The noul's first value is P(true), which the engine answers as the noul.
        #expect(abs(result.probabilities[2][0] - reads[2].probabilities[0]) < 1e-6)
        #expect(result.probabilities[2][0] > result.probabilities[2][1])
    }

    @Test("Serves verdict-1.4 with 24 options, and refuses what upstream's Verdict refuses")
    func contract() async throws {
        let backend = VerdictBackend(
            model: FixedModel(rows: []), tokenizer: FourTokens(),
            calibration: VerdictCalibration(temperature: 1, perK: [:]))
        #expect(backend.modelInfo == KnownEncoderModels.verdict)
        #expect(backend.maxChoices == 24)
        #expect(backend.maxPromptTokens == nil)
        let engine = EncoderDecisionEngine(backend: backend)
        let base = SystemOneRequest(
            model: "verdict-1.4", state: .string("s"),
            questions: ["n": .noul(instructions: .string("x"), criteria: nil)])
        var withImages = base
        withImages.images = [.dataURL("data:image/png;base64,AAAA")]
        var withSteps = base
        withSteps.steps = 2
        var withSamples = base
        withSamples.samples = 2
        var withThink = base
        withThink.think = 16
        var withSequential = base
        withSequential.sequential = true
        let refused = [
            ("images", withImages), ("steps", withSteps), ("samples", withSamples),
            ("think", withThink), ("sequential", withSequential),
        ]
        for (field, request) in refused {
            let error = await #expect(throws: SchemaError.self) { try await engine.decide(request) }
            #expect(error?.message == "verdict-1.4 does not support \(field)")
        }
        var tooMany = base
        tooMany.questions = [
            "c": .choice(
                instructions: .string("x"),
                criteria: JSONObject(
                    uniqueKeysWithValues: (0..<25).map { ("o\($0)", JSONValue.null) }))
        ]
        let error = await #expect(throws: SchemaError.self) { try await engine.decide(tooMany) }
        #expect(error?.message == "Too many choices. Must have at most 24 choices.")
    }

    @Test("A score built in code with no levels is refused as a broken contract, not a crash")
    func scoreWithoutLevels() async throws {
        // The wire refuses an empty score; a request built in code can still hold one.
        let backend = VerdictBackend(
            model: FixedModel(rows: [[Float](repeating: 0, count: 25)]), tokenizer: FourTokens(),
            calibration: VerdictCalibration(temperature: 1, perK: [:]), maxBatchRows: 16)
        let request = SystemOneRequest(
            model: "verdict-1.4", state: .string("s"),
            questions: ["s": .score(instructions: .string("x"), criteria: [])])
        await #expect(throws: BackendContractError.self) {
            try await EncoderDecisionEngine(backend: backend).decide(request)
        }
    }

    @Test("A model that returns too few rows or too few logits is an error, not an answer")
    func modelErrors() async throws {
        let questions = try EncoderQuestionSchemaBuilder(maxChoices: 24).build([
            "a": .noul(instructions: .string("x"), criteria: nil),
            "b": .noul(instructions: .string("y"), criteria: nil),
        ]).questions
        let calibration = VerdictCalibration(temperature: 1, perK: [:])
        let short = VerdictBackend(
            model: FixedModel(rows: [[1, 2, 3]]), tokenizer: FourTokens(),
            calibration: calibration, maxBatchRows: 16)
        await #expect(throws: EncoderModelError.self) {
            try await short.readBatch(state: .string("s"), stateText: "s", questions: questions)
        }
        let narrow = VerdictBackend(
            model: FixedModel(rows: [[1, 2], [1, 2]]), tokenizer: FourTokens(),
            calibration: calibration, maxBatchRows: 16)
        await #expect(throws: EncoderModelError.self) {
            try await narrow.readBatch(state: .string("s"), stateText: "s", questions: questions)
        }
    }
}
