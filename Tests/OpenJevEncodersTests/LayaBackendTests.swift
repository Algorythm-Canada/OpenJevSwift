import Foundation
import OpenJevCore
import OpenJevEncoders
import Testing

/// ``LayaBackend`` over models that replay the recorded scores: the whole reference corpus
/// through ``EncoderDecisionEngine`` with both batch policies, the option order and the noul's
/// swap, the refusal of options that overflow the head, and the errors a model can cause. No
/// Core ML and no tokenizer are needed.
@Suite("Laya backend")
struct LayaBackendTests {
    /// A tokenizer with one token per Unicode scalar, its value.
    private struct ScalarTokenizer: LayaTokenizing {
        let specialTokens = LayaSpecialTokens(
            classToken: 50_281, separator: 50_282, mask: 50_284, padding: 50_283)
        func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    }

    private let calibration = LayaCalibration(
        temperatures: [1, 1, 1], temperaturesByOptions: [:], maxLength: 1024, headMaxLength: 256)

    @Test(
        "The corpus through the engine gives upstream's published answers and billing",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage),
        arguments: [1, 16])
    func corpusThroughTheEngine(maxBatchRows: Int) async throws {
        let reference = try LayaFixtures.reference()
        let readsByRequest = try LayaFixtures.readsByRequest()
        let tokenizer = LayaReplayTokenizer(reference)
        var questionsRead = 0
        for corpus in try LayaFixtures.corpus() {
            let reads = try #require(readsByRequest[corpus.name])
            let model = ReplayModel(laya: reads, filler: .nan)
            let backend = LayaBackend(
                model: model, tokenizer: tokenizer, calibration: reference.calibration,
                maxBatchRows: maxBatchRows)
            let engine = EncoderDecisionEngine(
                backend: backend,
                configuration: EncoderEngineConfiguration(batchSize: reference.encoderBatch))
            let decision = try await engine.decide(corpus.request)

            // Billing: the rows' lengths, laya's attention-mask sum for the request.
            #expect(
                decision.inputTokens == reads.last?.requestInputTokens,
                "\(corpus.name): input tokens")
            #expect(decision.inputTokens == reads.reduce(0) { $0 + $1.ids.count })

            // Batching: the engine's batches of 16, each cut into calls of maxBatchRows, in
            // question order; each row has its full attention mask and its question type in
            // every position.
            let calls = await model.calls
            #expect(calls.flatMap(\.ids) == reads.map(\.ids), "\(corpus.name): row order")
            #expect(calls.allSatisfy { $0.maskSums == $0.ids.map(\.count) })
            let types = calls.flatMap { $0.planes.map { $0[2] } }
            for (plane, read) in zip(types, reads) {
                #expect(plane == [Int32](repeating: Int32(read.qtype), count: read.ids.count))
            }
            var expectedCalls: [Int] = []
            for batch in stride(from: 0, to: reads.count, by: reference.encoderBatch) {
                let size = min(reference.encoderBatch, reads.count - batch)
                for call in stride(from: 0, to: size, by: maxBatchRows) {
                    expectedCalls.append(min(maxBatchRows, size - call))
                }
            }
            #expect(calls.map(\.ids.count) == expectedCalls, "\(corpus.name): calls")

            // Answers in the caller's option order, exactly as upstream published them: laya's
            // float32 arithmetic, its rounding and upstream's renormalisation are reproduced
            // (within one rounding step off Apple silicon, LayaFixtures.arithmeticIsLayas).
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
                #expect(
                    LayaFixtures.matchesPublished(probabilities, read.probabilities),
                    "\(read.name): \(probabilities)")
                questionsRead += 1
            }
        }
        #expect(questionsRead == 200)
    }

    @Test(
        "A batch returns each distribution in option order, a noul's as true then false",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func optionOrder() async throws {
        let reference = try LayaFixtures.reference()
        let corpus = try #require(try LayaFixtures.corpus().first { $0.name == "quickstart" })
        let reads = try #require(try LayaFixtures.readsByRequest()["quickstart"])
        #expect(reads.map(\.kind) == [.choice, .score, .noul])
        let backend = LayaBackend(
            model: ReplayModel(laya: reads), tokenizer: LayaReplayTokenizer(reference),
            calibration: reference.calibration, maxBatchRows: 16)
        let result = try await backend.readBatch(
            state: corpus.request.state, stateText: StateText.render(corpus.request.state),
            questions: try LayaFixtures.questions(of: corpus))
        #expect(result.probabilities.count == 3)
        for (probabilities, read) in zip(result.probabilities, reads) {
            #expect(LayaFixtures.matchesPublished(probabilities, read.probabilities))
        }
        // laya's markers are false then true: the engine's first value is P(true), laya's
        // second probability, rounded.
        let noul = reads[2]
        #expect(noul.optionTexts.first?.hasPrefix(" false: ") == true)
        let yes = LayaCalibration.roundedToFourPlaces(noul.probabilitiesUnrounded[1])
        #expect(LayaFixtures.matchesPublished(result.probabilities[2], [yes, 1 - yes]))
        #expect(LayaFixtures.matchesPublished([result.probabilities[2][0]], try noul.answerValues))
        // A choice keeps the criteria's order.
        let choice = LayaCalibration.published(
            reads[0].probabilitiesUnrounded.map { Float($0) }, kind: .choice)
        #expect(LayaFixtures.matchesPublished(result.probabilities[0], choice))
    }

    @Test("Options that overflow the head's budget are refused before anything is read")
    func overflowRefused() async throws {
        // 255 options keep 4 tokens each, 1,020 of them: with the head, the last markers fall
        // past 1,024 tokens, which laya raises as ValueError and upstream refuses.
        let criteria = JSONObject(
            uniqueKeysWithValues: (0..<255).map {
                ("option \($0)", JSONValue.string("a description long enough to be cut"))
            })
        let request = SystemOneRequest(
            model: "laya-1.0", state: .string("s"),
            questions: [
                "fine": .noul(instructions: .string("x"), criteria: nil),
                "wide": .choice(instructions: .string("Pick"), criteria: criteria),
            ])
        let model = FixedModel(rows: [])
        let backend = LayaBackend(
            model: model, tokenizer: ScalarTokenizer(), calibration: calibration,
            maxBatchRows: 16)
        let error = await #expect(throws: SchemaError.self) {
            try await EncoderDecisionEngine(backend: backend).decide(request)
        }
        #expect(
            error?.message
                == "Too many choices for laya-1.0: a question's options must fit in 256 tokens.")
        #expect(await model.rowCounts.isEmpty)
        // 60 options of a few tokens fit.
        let fitting = JSONObject(
            uniqueKeysWithValues: (0..<60).map { ("o\($0)", JSONValue.null) })
        let fine = SystemOneRequest(
            model: "laya-1.0", state: .string("s"),
            questions: ["c": .choice(instructions: .string("x"), criteria: fitting)])
        let echo = EchoScores()
        let decision = try await EncoderDecisionEngine(
            backend: LayaBackend(
                model: echo, tokenizer: ScalarTokenizer(), calibration: calibration,
                maxBatchRows: 16)
        ).decide(fine)
        guard case .choice(_, let byOption, _) = decision.answers["c"] else {
            Issue.record("no choice answer: \(decision.answers)")
            return
        }
        #expect(byOption.count == 60)
    }

    @Test("Serves laya-1.0 with 255 options, and refuses what upstream's Laya refuses")
    func contract() async throws {
        let backend = LayaBackend(
            model: FixedModel(rows: []), tokenizer: ScalarTokenizer(), calibration: calibration)
        #expect(backend.modelInfo == KnownEncoderModels.laya)
        #expect(backend.maxChoices == 255)
        #expect(backend.maxPromptTokens == nil)
        let engine = EncoderDecisionEngine(backend: backend)
        let base = SystemOneRequest(
            model: "laya-1.0", state: .string("s"),
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
            #expect(error?.message == "laya-1.0 does not support \(field)")
        }
        var tooMany = base
        tooMany.questions = [
            "c": .choice(
                instructions: .string("x"),
                criteria: JSONObject(
                    uniqueKeysWithValues: (0..<256).map { ("o\($0)", JSONValue.null) }))
        ]
        let error = await #expect(throws: SchemaError.self) { try await engine.decide(tooMany) }
        #expect(error?.message == "Too many choices. Must have at most 255 choices.")
        // Nothing to prefetch without per-length packages.
        try await backend.prefetch(lengths: [128, 1024])
    }

    @Test("A model that returns too few rows or too few scores is an error, not an answer")
    func modelErrors() async throws {
        let questions = try EncoderQuestionSchemaBuilder(maxChoices: 255).build([
            "a": .noul(instructions: .string("x"), criteria: nil),
            "b": .noul(instructions: .string("y"), criteria: nil),
        ]).questions
        let short = LayaBackend(
            model: FixedModel(rows: [[Float](repeating: 0, count: 200)]),
            tokenizer: ScalarTokenizer(), calibration: calibration, maxBatchRows: 16)
        await #expect(throws: EncoderModelError.self) {
            try await short.readBatch(state: .string("s"), stateText: "s", questions: questions)
        }
        let narrow = LayaBackend(
            model: FixedModel(rows: [[1, 2], [1, 2]]), tokenizer: ScalarTokenizer(),
            calibration: calibration, maxBatchRows: 16)
        await #expect(throws: EncoderModelError.self) {
            try await narrow.readBatch(state: .string("s"), stateText: "s", questions: questions)
        }
    }

    @Test("A score built in code with no levels is refused as a broken contract, not a crash")
    func scoreWithoutLevels() async throws {
        let backend = LayaBackend(
            model: EchoScores(), tokenizer: ScalarTokenizer(), calibration: calibration,
            maxBatchRows: 16)
        let request = SystemOneRequest(
            model: "laya-1.0", state: .string("s"),
            questions: ["s": .score(instructions: .string("x"), criteria: [])])
        await #expect(throws: BackendContractError.self) {
            try await EncoderDecisionEngine(backend: backend).decide(request)
        }
    }

    @Test("The state is tokenized once for every question of a batch")
    func stateTokenizedOnce() async throws {
        let tokenizer = CountingTokenizer()
        let backend = LayaBackend(
            model: EchoScores(), tokenizer: tokenizer, calibration: calibration,
            maxBatchRows: 16)
        let questions = try EncoderQuestionSchemaBuilder(maxChoices: 255).build([
            "a": .noul(instructions: .string("x"), criteria: nil),
            "b": .noul(instructions: .string("y"), criteria: nil),
            "c": .choice(instructions: .string("z"), criteria: ["p": .null, "q": .null]),
        ]).questions
        let state = "the state, only once"
        let result = try await backend.readBatch(
            state: .string(state), stateText: state, questions: questions)
        #expect(result.probabilities.count == 3)
        #expect(tokenizer.count(of: state) == 1)
    }
}

/// A model that returns, for each row, its token ids as scores: deterministic, and every row as
/// long as its sequence.
actor EchoScores: EncoderModelRunner {
    func run(_ rows: [[[Int32]]]) throws -> [[Float]] {
        rows.map { $0[0].map { Float($0 % 7) } }
    }
}

/// A tokenizer with one token per Unicode scalar that counts how often it sees each text.
final class CountingTokenizer: LayaTokenizing, @unchecked Sendable {
    let specialTokens = LayaSpecialTokens(
        classToken: 50_281, separator: 50_282, mask: 50_284, padding: 50_283)
    private let lock = NSLock()
    private var counts: [String: Int] = [:]

    func encode(_ text: String) -> [Int] {
        lock.withLock { counts[text, default: 0] += 1 }
        return text.unicodeScalars.map { Int($0.value) }
    }

    func count(of text: String) -> Int {
        lock.withLock { counts[text] ?? 0 }
    }
}
