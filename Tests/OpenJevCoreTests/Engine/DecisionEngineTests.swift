import Foundation
import OpenJevCore
import OpenJevTestSupport
import Testing

/// Hand-written checks of ``DecisionEngine`` over ``StubBackend``. The requests come from the
/// recorded cases, because ``FixtureTokenizer`` only knows the prompts upstream rendered.
@Suite("Decision engine", .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
struct DecisionEngineTests {
    /// The quickstart request of the fixtures, with `options` applied.
    private func quickstart(
        steps: Int? = nil, samples: Int? = nil, think: Int? = nil, sequential: Bool? = nil
    ) throws -> SystemOneRequest {
        var request = try PolicyFixtures.request(named: "plain")
        request.steps = steps
        request.samples = samples
        request.think = think
        request.sequential = sequential
        return request
    }

    @Test("A plain request and one with the defaults spelled out make the same calls and body")
    func defaults() async throws {
        let plainStub = StubBackend()
        let plain = try await DecisionEngine(backend: plainStub).decide(quickstart())
        let explicitStub = StubBackend()
        let explicit = try await DecisionEngine(backend: explicitStub).decide(
            quickstart(steps: 1, think: 0, sequential: false))
        #expect(plainStub.reads == explicitStub.reads)
        #expect(plainStub.reads.count == 1)
        #expect(plainStub.thinks.isEmpty && explicitStub.thinks.isEmpty)
        #expect(try PolicyFixtures.body(of: plain) == PolicyFixtures.body(of: explicit))
        #expect(plain.inputTokens == 123 && plain.outputTokens == 0)
    }

    @Test("ReadOptions fill every unset field with upstream's default")
    func readOptions() throws {
        #expect(ReadOptions(try quickstart()) == .default)
        #expect(
            ReadOptions(try quickstart(steps: 4, samples: 2, think: 8, sequential: true))
                == ReadOptions(steps: 4, samples: 2, think: 8, sequential: true))
        #expect(
            ReadOptions.default
                == ReadOptions(steps: 1, samples: nil, think: 0, sequential: false))
    }

    @Test("samples 4 averages the flipping noul to 0.5 and bills every read")
    func samples() async throws {
        let stub = StubBackend()
        let decision = try await DecisionEngine(backend: stub).decide(quickstart(samples: 4))
        #expect(stub.reads.count == 4)
        #expect(decision.inputTokens == 4 * 123)
        #expect(decision.answers["is_urgent"] == .noul(0.5))
        let base: UInt64 = 1_788_574_486
        #expect(Set(stub.reads.map(\.seed)) == Set((0..<4).map { base + 7919 * UInt64($0) }))
    }

    @Test("think bills the input twice, the thought as output, and the read continues the thought")
    func think() async throws {
        let stub = StubBackend()
        let engine = try DecisionEngine(backend: stub)
        let decision = try await engine.decide(quickstart(think: 256))
        #expect(decision.inputTokens == 100 + 123)
        #expect(decision.outputTokens == 3)
        let call = try #require(stub.thinks.first)
        #expect(stub.thinks.count == 1)
        #expect(call.budget == 256)
        #expect(call.stopIDs == engine.tokens.thoughtClose)
        let open = engine.tokens.thoughtOpen
        #expect(Array(call.prompt.suffix(open.count)) == open)
        let read = try #require(stub.reads.first)
        guard case .tokens(let prefix) = read.prompt else {
            Issue.record("the read did not continue a token prefix")
            return
        }
        let tail = [7, 8, 9] + engine.tokens.thoughtClose
        #expect(Array(prefix.suffix(tail.count)) == tail)
        #expect(prefix == call.prompt + tail)
        // After a thought the canvas has no scaffold: the template starts at the answer text.
        #expect(!read.template.starts(with: engine.tokens.scaffold))
    }

    @Test("A thought is cut at the first close id, whatever follows it")
    func thoughtCut() async throws {
        let stub = StubBackend()
        let close = try #require(try EngineTokens(tokenizer: stub.tokenizer).thoughtClose.first)
        let cutting = StubBackend(thought: [7, 8, close, 9, 9])
        let engine = try DecisionEngine(backend: cutting)
        let decision = try await engine.decide(quickstart(think: 32))
        #expect(decision.outputTokens == 2)
        guard case .tokens(let prefix) = try #require(cutting.reads.first).prompt else {
            Issue.record("the read did not continue a token prefix")
            return
        }
        let tail = [7, 8] + engine.tokens.thoughtClose
        #expect(Array(prefix.suffix(tail.count)) == tail)
    }

    @Test("sequential with 24 nouls reads twice and the second read continues the first answers")
    func sequential() async throws {
        let stub = StubBackend()
        let engine = try DecisionEngine(backend: stub)
        let request = try PolicyFixtures.request(named: "sequential_24_nouls")
        let decision = try await engine.decide(request)
        #expect(stub.reads.count > 1)
        let first = try #require(stub.reads.first)
        let second = try #require(stub.reads.dropFirst().first)
        guard case .tokens(let prefix) = second.prompt else {
            Issue.record("the second read did not continue a token prefix")
            return
        }
        #expect(prefix.count > first.template.count)
        // The first read's prompt and scaffold open the second's prefix.
        guard case .tokens(let firstPrompt) = first.prompt else {
            Issue.record("the first read did not use a token prompt")
            return
        }
        #expect(prefix.starts(with: firstPrompt + engine.tokens.scaffold))
        #expect(first.template.starts(with: engine.tokens.scaffold))
        #expect(!second.template.starts(with: engine.tokens.scaffold))
        #expect(decision.answers.keys == request.questions.keys)
        #expect(decision.inputTokens == 2 * 123)
    }

    @Test("Forced answers keep the request's order among the read ones")
    func forcedOrder() async throws {
        let stub = StubBackend()
        let request = try PolicyFixtures.request(named: "forced_among_read")
        let decision = try await DecisionEngine(backend: stub).decide(request)
        #expect(decision.answers.keys == request.questions.keys)
        #expect(decision.answers.keys.count > stub.reads.first?.slots.count ?? 0)
        let recorded = try PolicyFixtures.policyCase(named: "forced_among_read")
        #expect(
            try PolicyFixtures.body(of: decision)
                == recorded["response"]?["body_text"]?.stringValue)
    }

    @Test("Images cannot be combined with think or sequential")
    func imagesNeedText() async throws {
        let engine = try DecisionEngine(backend: StubBackend())
        var request = try PolicyFixtures.request(named: "images_ahead_of_state")
        request.think = 16
        let think = await #expect(throws: SchemaError.self) { try await engine.decide(request) }
        #expect(think?.message == "think needs a text state; send images without it")
        #expect(think?.loc == ["body", "think"])
        request.think = nil
        request.sequential = true
        let sequential = await #expect(throws: SchemaError.self) {
            try await engine.decide(request)
        }
        #expect(sequential?.message == "sequential needs a text state; send images without it")
        #expect(sequential?.loc == ["body", "sequential"])
        // think wins when both are set.
        request.think = 16
        let both = await #expect(throws: SchemaError.self) { try await engine.decide(request) }
        #expect(both?.loc == ["body", "think"])
    }

    @Test("A reads-only backend refuses every extension in upstream's field order")
    func capabilities() async throws {
        let engine = try DecisionEngine(
            backend: StubBackend(capabilities: .readsOnly, modelName: "verdict"))
        let think = await #expect(throws: SchemaError.self) {
            try await engine.decide(quickstart(think: 8))
        }
        #expect(think?.message == "verdict does not support think")
        #expect(think?.loc == ["body", "think"])
        let steps = await #expect(throws: SchemaError.self) {
            try await engine.decide(quickstart(steps: 2, samples: 3))
        }
        #expect(steps?.message == "verdict does not support steps")
        let samples = await #expect(throws: SchemaError.self) {
            try await engine.decide(quickstart(samples: 3, sequential: true))
        }
        #expect(samples?.message == "verdict does not support samples")
        let sequential = await #expect(throws: SchemaError.self) {
            try await engine.decide(quickstart(sequential: true))
        }
        #expect(sequential?.message == "verdict does not support sequential")
        let images = await #expect(throws: SchemaError.self) {
            try await engine.decide(PolicyFixtures.request(named: "images_ahead_of_state"))
        }
        #expect(images?.message == "verdict does not support images")
        #expect(images?.loc == ["body", "images"])
        // The values upstream treats as defaults pass.
        let decision = try await engine.decide(
            quickstart(steps: 1, samples: 1, think: 0, sequential: false))
        #expect(decision.inputTokens == 123)
    }

    /// Upstream's `asyncio.gather` starts the groups in order, and each resolves its template
    /// before its first `await`, so a canvas too small for every group is refused with the first
    /// group's size. Fixtures/errors/cases.json records this request as `canvas_8_quickstart`.
    /// At that canvas each question is a group of its own; the first and third are slowed down,
    /// so that the second group's larger template is refused first.
    @Test("A refusal before any read names the first group's problem, whichever group runs first")
    func firstGroupRefusal() async throws {
        let tokenizer = DelayingTokenizer(prefixes: ["q1:", "q3:"])
        let engine = try DecisionEngine(
            backend: StubBackend(tokenizer: tokenizer),
            configuration: EngineConfiguration(geometry: CanvasGeometry(canvas: 8, step: 16)))
        let request = try quickstart()
        tokenizer.isDelaying = true
        for _ in 0..<5 {
            let error = await #expect(throws: SchemaError.self) { try await engine.decide(request) }
            #expect(error?.message == "answer template is 8 tokens; the canvas holds 7")
        }
    }

    @Test("The queue bound refuses the request that would exceed it")
    func queueBound() async throws {
        let stub = StubBackend(delay: .milliseconds(300))
        let engine = try DecisionEngine(
            backend: stub, configuration: EngineConfiguration(maxQueue: 1))
        let request = try quickstart()
        let first = Task { try await engine.decide(request) }
        // Let the first request pass the bound before the second arrives.
        while stub.reads.isEmpty {
            try await Task.sleep(for: .milliseconds(5))
        }
        let error = await #expect(throws: OverloadedError.self) { try await engine.decide(request) }
        #expect(error?.message == "OpenJev is at capacity. Retry shortly.")
        #expect(error?.description == "OpenJev is at capacity. Retry shortly.")
        let decision = try await first.value
        #expect(decision.inputTokens == 123)
        // The slot is free again once the first request has finished.
        _ = try await engine.decide(request)

        // A bound of zero refuses every request, as upstream's `waiting >= max_queue` does.
        let closed = try DecisionEngine(
            backend: StubBackend(), configuration: EngineConfiguration(maxQueue: 0))
        await #expect(throws: OverloadedError.self) { try await closed.decide(request) }
    }

    @Test(
        "More label ids than one read allows is refused with upstream's message",
        .enabled(
            if: UpstreamFixtures.exists("templates/errors.json"), UpstreamFixtures.missingMessage))
    func labelLimit() throws {
        let row = try #require(
            UpstreamFixtures.cases("templates/errors.json").first {
                $0["name"]?.stringValue == "label_ids_over_read_limit"
            })
        let slots = try PolicyFixtures.slots(row["slots"])
        let read = CanvasRead(
            prompt: .tokens([]), systemText: "", stateText: "",
            template: try PolicyFixtures.ints(row["template"]), slots: slots,
            canvas: SeededCanvas(tokens: [], noise: []), steps: 1, seed: 0)
        #expect(read.labelIDs.count == 513)
        let error = #expect(throws: SchemaError.self) { try DecisionEngine.checkLabelLimit(slots) }
        #expect(error?.message == row["error"]?["message"]?.stringValue)
        #expect(error?.loc == ["body"])
        #expect(DecisionEngine.maxLabelIDs == 512)
        #expect(throws: Never.self) { try DecisionEngine.checkLabelLimit(Array(slots.prefix(1))) }
    }

    @Test("A prompt over the backend's limit is refused with the MLX engine's message")
    func promptLimit() async throws {
        let engine = try DecisionEngine(backend: StubBackend(maxPromptTokens: 10))
        let error = await #expect(throws: SchemaError.self) {
            try await engine.decide(quickstart())
        }
        let message = try #require(error?.message)
        #expect(message.hasPrefix("the request is "))
        #expect(message.hasSuffix(" tokens; the limit is 10"))
        let thinking = await #expect(throws: SchemaError.self) {
            try await engine.decide(quickstart(think: 8))
        }
        #expect(thinking?.message.hasSuffix("; the limit is 10") == true)
    }

    @Test("Model time sums the time inside backend calls over concurrent reads")
    func modelTime() async throws {
        let delay = Duration.milliseconds(20)
        let stub = StubBackend(delay: delay)
        let decision = try await DecisionEngine(backend: stub).decide(quickstart(samples: 4))
        #expect(stub.reads.count == 4)
        // The four reads ran at once, so the model time counts each read's delay although the
        // wall time was about one delay.
        #expect(decision.modelTime >= delay * 4)

        // With one slot in flight the reads run in series.
        let clock = ContinuousClock()
        let serialStub = StubBackend(delay: delay)
        let serialStart = clock.now
        let serial = try await DecisionEngine(
            backend: serialStub, configuration: EngineConfiguration(maxInflight: 1)
        ).decide(quickstart(samples: 4))
        #expect(clock.now - serialStart >= delay * 4)
        #expect(serial.modelTime >= delay * 4)
    }

    @Test("A request cancelled while waiting for a slot leaves the slots to the others")
    func cancelledWaiter() async throws {
        let delay = Duration.milliseconds(150)
        let stub = StubBackend(delay: delay)
        let engine = try DecisionEngine(
            backend: stub, configuration: EngineConfiguration(maxInflight: 1))
        let request = try quickstart(samples: 4)
        let first = Task { try await engine.decide(request) }
        while stub.reads.isEmpty {
            try await Task.sleep(for: .milliseconds(5))
        }
        // The second request's reads queue behind the first's; cancel it while it waits.
        let second = Task { try await engine.decide(request) }
        try await Task.sleep(for: .milliseconds(20))
        second.cancel()
        await #expect(throws: CancellationError.self) { try await second.value }
        #expect(try await first.value.inputTokens == 4 * 123)
        // The permit was not leaked to the cancelled waiters: a third request runs through.
        let third = try await engine.decide(quickstart())
        #expect(third.inputTokens == 123)
    }

    @Test("Concurrent requests on one engine get their own answers")
    func concurrentRequests() async throws {
        let stub = StubBackend(delay: .milliseconds(2))
        let engine = try DecisionEngine(backend: stub)
        // One engine serves every case recorded at the default settings.
        let cases = try UpstreamFixtures.cases(PolicyFixtures.policies).filter {
            $0["settings"]?.objectValue?.isEmpty == true
        }
        #expect(cases.count >= 20)
        let bodies = try await withThrowingTaskGroup(of: (Int, String).self) { group in
            for (index, row) in cases.enumerated() {
                let request = try RequestValidator().validate(row["request"])
                group.addTask {
                    (index, try PolicyFixtures.body(of: try await engine.decide(request)))
                }
            }
            var out = [String?](repeating: nil, count: cases.count)
            for try await (index, body) in group {
                out[index] = body
            }
            return out
        }
        for (row, body) in zip(cases, bodies) {
            let name = row["name"]?.stringValue ?? "?"
            #expect(body == row["response"]?["body_text"]?.stringValue, "\(name)")
        }
    }

    @Test("A backend's own error passes through unchanged")
    func backendErrors() async throws {
        struct Failing: DecisionBackend {
            struct Boom: Error, Equatable {}
            var tokenizer: any DecisionTokenizer { FixtureTokenizer.shared }
            var maxPromptTokens: Int { 32768 }
            var capabilities: BackendCapabilities { .all }
            var modelName: String { "failing" }
            func read(_ read: CanvasRead) async throws -> ReadResult { throw Boom() }
            func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws
                -> ThoughtGeneration
            {
                throw Boom()
            }
        }
        let engine = try DecisionEngine(backend: Failing())
        await #expect(throws: Failing.Boom.self) { try await engine.decide(quickstart()) }
        await #expect(throws: Failing.Boom.self) { try await engine.decide(quickstart(think: 4)) }
    }

    @Test("ReadResult's raw initializer computes upstream's slot distributions")
    func rawReadResult() {
        let tops: [[(tokenID: Int, logprob: Double)]] = [
            [(1, log(0.5)), (2, log(0.25)), (3, log(0.25))],
            [(7, 0.0)],
        ]
        let result = ReadResult(tops: tops, labelIDs: [[1, 2], [7, 8]], promptTokens: 9)
        #expect(result.promptTokens == 9)
        #expect(result.slots.count == 2)
        let first = SlotDistribution.compute(top: tops[0], labelIDs: [1, 2])
        #expect(result.slots[0].probabilities == first.probabilities)
        #expect(result.slots[0].entropy == first.entropy)
        #expect(result.slots[1].probabilities[0] > 0.99)
    }

    @Test("Capability sets and the configuration carry upstream's defaults")
    func configuration() {
        let configuration = EngineConfiguration.default
        #expect(configuration.geometry.canvas == 64 && configuration.geometry.step == 16)
        #expect(configuration.autoThreshold == 0.1 && configuration.autoMax == 4)
        #expect(configuration.maxInflight == 64 && configuration.maxQueue == 512)
        #expect(configuration.templateCacheLimit == 4096)
        #expect(configuration.servedModelVersion == "openjev-0.1")
        #expect(configuration.imageLimits == ImageLimits())
        #expect(BackendCapabilities.all.think && BackendCapabilities.all.images)
        #expect(!BackendCapabilities.readsOnly.steps && !BackendCapabilities.readsOnly.samples)
        #expect(!BackendCapabilities.readsOnly.think && !BackendCapabilities.readsOnly.sequential)
        #expect(!BackendCapabilities.readsOnly.images)
    }
}

/// ``FixtureTokenizer``, with every encoding of a text that starts with one of `prefixes`
/// delayed by 50 milliseconds while ``isDelaying`` is set, so that the groups those texts belong
/// to finish last.
final class DelayingTokenizer: DecisionTokenizer, @unchecked Sendable {
    let prefixes: [String]
    private let lock = NSLock()
    private var delaying = false

    init(prefixes: [String]) {
        self.prefixes = prefixes
    }

    /// Whether encodings are delayed. Off while the engine discovers its labels.
    var isDelaying: Bool {
        get { lock.withLock { delaying } }
        set { lock.withLock { delaying = newValue } }
    }

    func encode(_ text: String, addSpecialTokens: Bool) throws -> [Int] {
        if isDelaying && prefixes.contains(where: text.hasPrefix) {
            Thread.sleep(forTimeInterval: 0.05)
        }
        return try FixtureTokenizer.shared.encode(text, addSpecialTokens: addSpecialTokens)
    }

    func decode(_ ids: [Int], skipSpecialTokens: Bool) throws -> String {
        try FixtureTokenizer.shared.decode(ids, skipSpecialTokens: skipSpecialTokens)
    }

    func chatPromptIDs(system: String, user: String, thinking: Bool) throws -> [Int] {
        try FixtureTokenizer.shared.chatPromptIDs(system: system, user: user, thinking: thinking)
    }
}
