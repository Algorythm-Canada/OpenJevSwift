import Foundation
import OpenJevCore
import Testing

/// Checks of ``EncoderDecisionEngine`` over ``StubQuestionReadBackend``: the generic tests at the
/// top of upstream's `tests/test_encoders.py`, plus the contract checks the issue adds.
@Suite("Encoder decision engine")
struct EncoderDecisionEngineTests {
    /// Upstream's `REQUEST`: a billing complaint with a choice, a score and a noul.
    static let request = SystemOneRequest(
        model: "laya-1.0",
        state: .string("I was charged twice this month."),
        questions: [
            "team": .choice(
                instructions: .string("Which team should handle it?"),
                criteria: [
                    "outage": .string("service down"), "billing": .string("charges, refunds"),
                    "feature": .null,
                ]),
            "tone": .score(
                instructions: .string("How upset is the customer?"),
                criteria: [.string("calm"), .string("annoyed"), .string("furious")]),
            "urgent": .noul(
                instructions: .string("Does the customer need a reply within the hour?"),
                criteria: nil),
        ])

    /// `request` with the read options and images set as given.
    private func request(
        images: [ImageInput]? = nil, steps: Int? = nil, samples: Int? = nil, think: Int? = nil,
        sequential: Bool? = nil
    ) -> SystemOneRequest {
        var request = Self.request
        request.images = images
        request.steps = steps
        request.samples = samples
        request.think = think
        request.sequential = sequential
        return request
    }

    /// `request` with other questions.
    private func request(questions: OrderedMap<Question>) -> SystemOneRequest {
        var request = Self.request
        request.questions = questions
        return request
    }

    /// `n` noul questions named `n0` to `n{n-1}`.
    private func nouls(_ n: Int, prefix: String = "n") -> [(String, Question)] {
        (0..<n).map { ("\(prefix)\($0)", .noul(instructions: .string("x"), criteria: nil)) }
    }

    private func engine(
        _ stub: StubQuestionReadBackend, configuration: EncoderEngineConfiguration = .default
    ) -> EncoderDecisionEngine {
        EncoderDecisionEngine(backend: stub, configuration: configuration)
    }

    /// A response body for a decision, as the server would send it.
    private func body(of decision: Decision, model: String) throws -> String {
        try WireEncoder().string(
            SystemOneResponse(
                model: model, answers: decision.answers,
                usage: Usage(
                    inputTokens: decision.inputTokens, outputTokens: decision.outputTokens)))
    }

    @Test("Answers come in Jev's shapes, as upstream's test_answers_in_jevs_shapes checks")
    func answersInJevsShapes() async throws {
        let stub = StubQuestionReadBackend()
        let decision = try await engine(stub).decide(Self.request)
        #expect(decision.answers.keys == ["team", "tone", "urgent"])
        #expect(decision.inputTokens == 99)
        #expect(decision.outputTokens == 0)
        guard case .choice(let choice, let probabilities, _) = decision.answers["team"] else {
            Issue.record("team is not a choice answer: \(String(describing: decision.answers))")
            return
        }
        #expect(choice == "billing")
        #expect(probabilities.keys == ["outage", "billing", "feature"])
        #expect(probabilities["billing"] == 0.7)
        guard case .score(let score, let legend, _, _) = decision.answers["tone"] else {
            Issue.record("tone is not a score answer")
            return
        }
        #expect(abs(score - (0.15 * 0 + 0.7 * 1 + 0.15 * 2)) < 1e-12)
        #expect(legend == [.string("calm"), .string("annoyed"), .string("furious")])
        // P(yes) is the first option, which the stub gives 0.3.
        #expect(decision.answers["urgent"] == .noul(0.3))
        // The whole body, byte for byte, as FastAPI would render it.
        #expect(
            try body(of: decision, model: "laya-1.0") == """
                {"model":"laya-1.0","answers":{"team":{"type":"choice","choice":"billing",\
                "probabilities":{"outage":0.15,"billing":0.7,"feature":0.15},\
                "confidence":\(Confidence.compute([0.15, 0.7, 0.15]))},\
                "tone":{"type":"score","score":1.0,"legend":{"0":"calm","1":"annoyed",\
                "2":"furious"},"probabilities":{"0":0.15,"1":0.7,"2":0.15},\
                "confidence":\(Confidence.compute([0.15, 0.7, 0.15]))},\
                "urgent":{"type":"noul","noul":0.3}},\
                "usage":{"input_tokens":99,"output_tokens":0}}
                """)
        // One batch, with the state as sent and as rendered.
        let call = try #require(stub.calls.first)
        #expect(stub.calls.count == 1)
        #expect(call.keys == ["team", "tone", "urgent"])
        #expect(call.state == .string("I was charged twice this month."))
        #expect(call.stateText == "I was charged twice this month.")
    }

    @Test(
        "Every recorded answer of Fixtures/wire/answers.json comes out of the engine byte for byte",
        .enabled(if: UpstreamFixtures.exists("wire/answers.json"), UpstreamFixtures.missingMessage))
    func recordedAnswers() async throws {
        let rows = try #require(UpstreamFixtures.load("wire/answers.json")["answers"]?.arrayValue)
        var compared = 0
        for row in rows {
            let name = try #require(row["name"]?.stringValue)
            let question = try Question(json: try #require(row["question"]))
            let probabilities = try #require(row["probabilities"]?.arrayValue).map {
                try #require($0.doubleValue)
            }
            let stub = StubQuestionReadBackend(scripted: ["q": probabilities])
            let engine = engine(stub)
            let request = request(questions: ["q": question])
            // The layout rows probe number rendering with vectors that are not distributions;
            // the encoder engine refuses those as a backend bug, so only proper distributions
            // pass through it. Fixtures/read tests cover the rendering itself.
            guard abs(probabilities.reduce(0, +) - 1) <= 1e-6 else {
                await #expect(throws: BackendContractError.self, "row \(name)") {
                    try await engine.decide(request)
                }
                continue
            }
            let decision = try await engine.decide(request)
            let answer = try #require(decision.answers["q"])
            #expect(
                try WireEncoder().string(answer) == row["body_text"]?.stringValue,
                "row \(name)")
            compared += 1
        }
        #expect(compared >= 10)
    }

    @Test(
        "The recorded full response, forced answers among read ones, comes out in request order",
        .enabled(if: UpstreamFixtures.exists("wire/answers.json"), UpstreamFixtures.missingMessage))
    func recordedFullResponse() async throws {
        let row = try #require(UpstreamFixtures.load("wire/answers.json")["response"])
        let questions = try #require(row["questions"]?.objectValue)
        var decoded = OrderedMap<Question>()
        for (key, value) in questions {
            decoded.updateValue(try Question(json: value), forKey: key)
        }
        var scripted: [String: [Double]] = [:]
        for (key, value) in try #require(row["reads"]?.objectValue) {
            scripted[key] = try #require(value.arrayValue).map { try #require($0.doubleValue) }
        }
        let stub = StubQuestionReadBackend(inputTokensPerBatch: 123, scripted: scripted)
        let decision = try await engine(stub).decide(request(questions: decoded))
        #expect(try body(of: decision, model: "openjev-0.1") == row["body_text"]?.stringValue)
        // Only the read questions reach the backend.
        #expect(stub.calls.map(\.keys) == [["department", "frustration", "is_urgent"]])
    }

    @Test("Each unsupported option is refused with upstream's message and location")
    func unsupportedOptions() async throws {
        let image = ImageInput.dataURL("data:image/png;base64,iVBORw0KGgo=")
        let cases: [(SystemOneRequest, String)] = [
            (request(images: [image]), "images"),
            (request(steps: 2), "steps"),
            (request(samples: 4), "samples"),
            (request(think: 64), "think"),
            (request(sequential: true), "sequential"),
        ]
        for (request, field) in cases {
            let stub = StubQuestionReadBackend()
            let error = await #expect(throws: SchemaError.self, "\(field)") {
                try await engine(stub).decide(request)
            }
            #expect(
                error
                    == SchemaError(
                        "laya-1.0 does not support \(field)", loc: ["body", .key(field)]))
            #expect(stub.calls.isEmpty, "\(field) was refused after a read")
        }
        // Verdict's name appears in its own refusals.
        let verdict = StubQuestionReadBackend(modelInfo: KnownEncoderModels.verdict, maxChoices: 24)
        let error = await #expect(throws: SchemaError.self) {
            try await engine(verdict).decide(request(think: 1))
        }
        #expect(error?.message == "verdict-1.4 does not support think")
    }

    @Test("Several unsupported options are refused in upstream's order")
    func unsupportedOptionsOrder() async throws {
        let image = ImageInput.dataURL("data:image/png;base64,iVBORw0KGgo=")
        var request = request(
            images: [image], steps: 2, samples: 4, think: 64, sequential: true)
        let expected = ["images", "steps", "samples", "think", "sequential"]
        for field in expected {
            let error = await #expect(throws: SchemaError.self) {
                try await engine(StubQuestionReadBackend()).decide(request)
            }
            #expect(error?.message == "laya-1.0 does not support \(field)")
            #expect(error?.loc == ["body", .key(field)])
            switch field {
            case "images": request.images = nil
            case "steps": request.steps = nil
            case "samples": request.samples = nil
            case "think": request.think = nil
            default: request.sequential = nil
            }
        }
        _ = try await engine(StubQuestionReadBackend()).decide(request)
    }

    @Test("Options left at their defaults are accepted")
    func defaultsAccepted() async throws {
        for request in [
            request(images: [], steps: 1, samples: 1, think: 0, sequential: false),
            request(images: nil),
            request(steps: 1),
            request(samples: 1),
        ] {
            let stub = StubQuestionReadBackend()
            let decision = try await engine(stub).decide(request)
            #expect(decision.answers.count == 3)
            #expect(stub.calls.count == 1)
        }
    }

    @Test("One-option choices and one-level scores are answered without a read")
    func forcedAnswersNeedNoRead() async throws {
        let stub = StubQuestionReadBackend(delay: .milliseconds(50))
        let decision = try await engine(stub).decide(
            request(questions: [
                "only": .choice(instructions: nil, criteria: ["yes": .null]),
                "lvl": .score(instructions: nil, criteria: [.string("one")]),
            ]))
        #expect(stub.calls.isEmpty)
        #expect(decision.inputTokens == 0)
        #expect(decision.outputTokens == 0)
        #expect(decision.modelTime == .zero)
        #expect(
            decision.answers["only"]
                == .choice(choice: "yes", probabilities: ["yes": 1.0], confidence: 1.0))
        #expect(
            decision.answers["lvl"]
                == .score(
                    score: 0.0, legend: [.string("one")], probabilities: [1.0], confidence: 1.0))
    }

    @Test("40 questions are read in batches of 16, 16 and 8, in order, and answered in order")
    func batches() async throws {
        let keys = nouls(40).map(\.0)
        let stub = StubQuestionReadBackend()
        let decision = try await engine(stub).decide(
            request(questions: OrderedMap(uniqueKeysWithValues: nouls(40))))
        #expect(decision.answers.keys == keys)
        #expect(stub.calls.map { $0.questions.count } == [16, 16, 8])
        #expect(stub.calls.flatMap(\.keys) == keys)
        #expect(decision.inputTokens == 3 * 99)
        #expect(stub.calls.allSatisfy { $0.stateText == "I was charged twice this month." })
    }

    @Test("Forced questions interleaved with read ones keep their positions")
    func forcedInterleaved() async throws {
        var questions = OrderedMap<Question>()
        var expectedRead: [String] = []
        for index in 0..<20 {
            if index % 3 == 0 {
                questions.updateValue(
                    .choice(instructions: nil, criteria: ["only\(index)": .null]),
                    forKey: "f\(index)")
            } else {
                questions.updateValue(
                    .noul(instructions: .string("x"), criteria: nil), forKey: "r\(index)")
                expectedRead.append("r\(index)")
            }
        }
        let stub = StubQuestionReadBackend()
        let decision = try await engine(
            stub, configuration: EncoderEngineConfiguration(batchSize: 5)
        ).decide(request(questions: questions))
        #expect(decision.answers.keys == questions.keys)
        #expect(stub.calls.flatMap(\.keys) == expectedRead)
        #expect(stub.calls.map { $0.questions.count } == [5, 5, 3])
        for (key, answer) in decision.answers {
            if key.hasPrefix("f") {
                #expect(answer.type == "choice", Comment(rawValue: key))
            } else {
                #expect(answer == .noul(0.3), Comment(rawValue: key))
            }
        }
    }

    @Test("The backend's option limit and the level limit are upstream's messages")
    func limits() async throws {
        func choice(_ n: Int) -> OrderedMap<Question> {
            [
                "c": .choice(
                    instructions: nil,
                    criteria: JSONObject(
                        uniqueKeysWithValues: (0..<n).map { ("o\($0)", JSONValue.null) }))
            ]
        }
        let verdict = engine(
            StubQuestionReadBackend(modelInfo: KnownEncoderModels.verdict, maxChoices: 24))
        var error = await #expect(throws: SchemaError.self) {
            try await verdict.decide(request(questions: choice(25)))
        }
        #expect(
            error
                == SchemaError(
                    "Too many choices. Must have at most 24 choices.",
                    loc: ["body", "questions", "c", "criteria"]))
        _ = try await verdict.decide(request(questions: choice(24)))

        let laya = engine(StubQuestionReadBackend())
        error = await #expect(throws: SchemaError.self) {
            try await laya.decide(request(questions: choice(256)))
        }
        #expect(error?.message == "Too many choices. Must have at most 255 choices.")
        _ = try await laya.decide(request(questions: choice(255)))

        error = await #expect(throws: SchemaError.self) {
            try await laya.decide(
                request(questions: [
                    "s": .score(
                        instructions: nil, criteria: (0..<11).map { .string("level \($0)") })
                ]))
        }
        #expect(
            error
                == SchemaError(
                    "Too many score levels. Must have at most 10 levels.",
                    loc: ["body", "questions", "s", "criteria"]))

        error = await #expect(throws: SchemaError.self) {
            try await laya.decide(request(questions: choice(0)))
        }
        #expect(error?.message == "Choice question must have at least one choice: c")
    }

    @Test("warmUp makes one read of upstream's WARMUP_QUESTIONS against the state warmup")
    func warmUp() async throws {
        let stub = StubQuestionReadBackend()
        try await engine(stub).warmUp()
        let call = try #require(stub.calls.first)
        #expect(stub.calls.count == 1)
        #expect(call.state == .string("warmup"))
        #expect(call.stateText == "warmup")
        #expect(call.keys == ["c", "s", "n"])
        #expect(call.questions.map(\.kind) == [.choice, .score, .noul])
        #expect(call.questions.map(\.instructions) == ["x", "x", "x"])
        #expect(call.questions[0].choices.map(\.name) == ["a", "b"])
        #expect(call.questions[0].choices.map(\.description) == ["", ""])
        #expect(call.questions[1].choices.map(\.description) == ["low", "high"])
        #expect(call.questions[1].legend == [.string("low"), .string("high")])
        #expect(call.questions[2].choices.map(\.name) == ["yes", "no"])
        #expect(EncoderDecisionEngine.warmUpQuestions.keys == ["c", "s", "n"])

        let quiet = StubQuestionReadBackend()
        try await engine(quiet, configuration: EncoderEngineConfiguration(warmUp: false)).warmUp()
        #expect(quiet.calls.isEmpty)
    }

    @Test("The queue bound refuses the request that would exceed it, naming the model")
    func queueBound() async throws {
        let stub = StubQuestionReadBackend(delay: .milliseconds(300))
        let engine = engine(stub, configuration: EncoderEngineConfiguration(maxQueue: 1))
        let first = Task { try await engine.decide(Self.request) }
        // Let the first request pass the bound before the second arrives.
        while stub.calls.isEmpty {
            try await Task.sleep(for: .milliseconds(5))
        }
        let error = await #expect(throws: OverloadedError.self) {
            try await engine.decide(Self.request)
        }
        #expect(error?.message == "laya-1.0 is at capacity. Retry shortly.")
        #expect(error?.description == "laya-1.0 is at capacity. Retry shortly.")
        _ = try await first.value
        // The slot is free again once the first request has finished.
        _ = try await engine.decide(Self.request)

        // A bound of zero refuses every request, as upstream's `waiting >= max_queue` does.
        let closed = self.engine(
            StubQuestionReadBackend(), configuration: EncoderEngineConfiguration(maxQueue: 0))
        await #expect(throws: OverloadedError.self) { try await closed.decide(Self.request) }
    }

    @Test("A distribution that breaks the contract is an internal error, not a request error")
    func contractViolations() async throws {
        let failures: [StubQuestionReadBackend.Failure] = [
            .wrongDistributionCount, .wrongProbabilityCount, .notFinite, .sumFarFromOne,
        ]
        for failure in failures {
            let stub = StubQuestionReadBackend(failure: failure)
            let error = await #expect(throws: BackendContractError.self, "\(failure)") {
                try await engine(stub).decide(Self.request)
            }
            #expect(error?.message.hasPrefix("laya-1.0 ") == true, "\(failure)")
        }
        // The backend's own error passes through unchanged.
        let thrown = await #expect(throws: StubQuestionReadBackend.StubError.self) {
            try await engine(StubQuestionReadBackend(failure: .throwing("boom"))).decide(
                Self.request)
        }
        #expect(thrown?.message == "boom")
    }

    @Test("modelTime covers every batch, wait included")
    func modelTime() async throws {
        let stub = StubQuestionReadBackend(delay: .milliseconds(20))
        let decision = try await engine(stub).decide(
            request(questions: OrderedMap(uniqueKeysWithValues: nouls(40))))
        #expect(stub.calls.count == 3)
        #expect(decision.modelTime >= .milliseconds(60))
    }

    @Test("The engine serves its backend's model under SystemOneService")
    func service() async throws {
        let service: any SystemOneService = engine(
            StubQuestionReadBackend(modelInfo: KnownEncoderModels.verdict, maxChoices: 24))
        #expect(service.servedModels == .encoder(KnownEncoderModels.verdict))
        #expect(service.servedModels.version == "verdict-1.4")
        let decision = try await service.decide(Self.request)
        #expect(decision.answers.keys == ["team", "tone", "urgent"])
    }
}
