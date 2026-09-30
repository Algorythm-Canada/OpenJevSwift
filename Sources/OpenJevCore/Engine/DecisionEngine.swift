// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, methods
// `Engine.__init__`, `Engine.decide`, `Engine.read_group`, `Engine.think`, `Engine._sequential`
// and the label-id check of `Engine.one_read`, with the prompt-length checks of
// `openjev/mlx_backend.py`, `MlxEngine.one_read` and `MlxEngine.think`. The capability check of
// `openjev/encoders.py`, `EncoderEngine.decide`, and the queue bound are in
// RequestAdmission.swift. Apache-2.0. See THIRD_PARTY.md.

/// Answers Jev requests by reading canvases through a ``DecisionBackend``.
///
/// The engine owns everything upstream's `Engine` owns apart from the model: the marker tokens,
/// the choice labels, the schema builder, the template resolver and its cache, the read policies
/// (automatic re-reads, `samples`, `steps`, `think`, `sequential`), the seeds, the billing and the
/// two capacity bounds. A request goes through ``decide(_:seed:)``.
///
/// Errors: a ``SchemaError`` is a request the model cannot answer as asked (a 400), an
/// ``OverloadedError`` is the queue bound (a 529), and a backend's or tokenizer's own error
/// passes through unchanged.
public actor DecisionEngine {
    /// The most distinct label ids one read may ask for, upstream's `MAX_LABEL_IDS`: vLLM's
    /// `logprob_token_ids` cap per request.
    public static let maxLabelIDs = 512

    /// The model the engine reads through.
    public nonisolated let backend: any DecisionBackend
    /// The settings.
    public nonisolated let configuration: EngineConfiguration
    /// The marker sequences encoded with the backend's tokenizer.
    public nonisolated let tokens: EngineTokens
    /// The single-token choice labels the tokenizer allows.
    public nonisolated let choiceLabels: LabelSet
    /// The schema builder over ``choiceLabels``.
    public nonisolated let schemaBuilder: QuestionSchemaBuilder
    /// The template resolver over the configured canvas, with the engine's template cache.
    public nonisolated let resolver: TemplateResolver

    private nonisolated let reader: GroupReader
    /// Requests inside ``decide(_:seed:)`` right now, upstream's `waiting`, and its bound.
    private var queue: RequestQueue

    /// Creates an engine, as `Engine.__init__` does: encodes the markers, discovers the choice
    /// labels and prepares the resolver and its cache.
    ///
    /// - Throws: Whatever the backend's tokenizer throws.
    public init(
        backend: any DecisionBackend, configuration: EngineConfiguration = .default
    ) throws {
        let tokenizer = backend.tokenizer
        let tokens = try EngineTokens(tokenizer: tokenizer)
        let labels = try LabelDiscovery.choiceLabels(using: tokenizer)
        let resolver = TemplateResolver(
            tokenizer: tokenizer, tokens: tokens, canvas: configuration.geometry.canvas,
            cacheLimit: configuration.templateCacheLimit)
        self.backend = backend
        self.configuration = configuration
        self.tokens = tokens
        self.choiceLabels = labels
        self.schemaBuilder = QuestionSchemaBuilder(choiceLabels: labels.labels)
        self.resolver = resolver
        self.reader = GroupReader(
            backend: backend, configuration: configuration, tokens: tokens, resolver: resolver,
            slots: AsyncSemaphore(permits: configuration.maxInflight))
        self.queue = RequestQueue(limit: configuration.maxQueue)
    }

    /// Answers a request, as upstream's `decide` does.
    ///
    /// In order: the options with their defaults; the backend's capabilities; the images; the
    /// rule that `think` and `sequential` need a text state; the request seed; the queue bound;
    /// the schema, the state text and the groups; the reads, in series when `sequential` is set
    /// and there is more than one group, otherwise every group at once; and the answers in the
    /// request's order, forced answers included.
    ///
    /// `seed` is the request seed. Upstream's route derives it from the request
    /// (``SeedDerivation``), which `nil` does here; a value replays a recording or a past
    /// request, as upstream's `Engine.decide(questions, state, seed, ...)` takes it.
    ///
    /// - Throws: ``SchemaError`` for an unsupported option (`"{model} does not support
    ///   {field}"`), a refused image, images with `think` or `sequential` (`"{field} needs a
    ///   text state; send images without it"`), a schema the model cannot answer, a prompt over
    ///   the backend's limit or a read over ``maxLabelIDs``; ``OverloadedError`` when
    ///   ``EngineConfiguration/maxQueue`` requests are already inside; and the backend's or the
    ///   tokenizer's own errors.
    public func decide(_ request: SystemOneRequest, seed: UInt64? = nil) async throws -> Decision {
        let options = ReadOptions(request)
        let sentImages = request.images ?? []
        try UnsupportedOptions.check(
            options, hasImages: !sentImages.isEmpty, capabilities: backend.capabilities,
            modelName: backend.modelName)
        let images = try ImageValidation.parts(sentImages, limits: configuration.imageLimits)
        if !images.isEmpty && (options.think != 0 || options.sequential) {
            let field = options.think != 0 ? "think" : "sequential"
            throw SchemaError(
                "\(field) needs a text state; send images without it", loc: ["body", .key(field)])
        }
        let seed =
            try seed
            ?? SeedDerivation.seed(
                for: SeedDerivation.seedKey(
                    state: request.state, questions: request.questions, images: images))
        try queue.admit(refusing: OverloadedError().message)
        defer { queue.leave() }

        let schema = try schemaBuilder.build(request.questions)
        let format = schema.format
        let stateText = StateText.render(request.state)
        let groups =
            schema.questions.isEmpty
            ? []
            : try ReadGrouping.groups(
                schema.questions, format: format, geometry: configuration.geometry,
                scaffold: tokens.scaffold, tokenizer: backend.tokenizer)
        let results: [GroupResult]
        if groups.isEmpty {
            results = []
        } else if options.sequential && groups.count > 1 {
            results = try await reader.sequential(
                groups, format: format, allQuestions: schema.questions, stateText: stateText,
                seed: seed, options: options)
        } else {
            results = try await reader.readGroups(
                groups, format: format, stateText: stateText, images: images, seed: seed,
                options: options)
        }

        var answers = schema.forced
        var billed = 0
        var thoughtTokens = 0
        var modelTime = Duration.zero
        for (group, result) in zip(groups, results) {
            billed += result.billed
            thoughtTokens += result.thoughtTokens
            modelTime += result.modelTime
            for (question, mean) in zip(group, result.means) {
                guard let asked = request.questions[question.key] else {
                    preconditionFailure("the schema names a question the request lacks")
                }
                answers.updateValue(
                    Answer.make(for: asked, probabilities: mean), forKey: question.key)
            }
        }
        return Decision(
            answers: answers.ordered(as: request.questions.keys), inputTokens: billed,
            outputTokens: thoughtTokens, modelTime: modelTime)
    }

    /// Refuses a read whose slots need more than ``maxLabelIDs`` distinct label ids, as
    /// upstream's `one_read` does. `ReadGrouping` splits ahead of this, and a read that still
    /// asks for more would get silently truncated evidence.
    ///
    /// - Throws: ``SchemaError`` with `"the questions of one read need {n} label tokens; a read
    ///   allows 512. Ask them in separate requests."`.
    public static func checkLabelLimit(_ slots: [ResolvedTemplate.Slot]) throws(SchemaError) {
        let count = Set(slots.lazy.flatMap(\.labelIDs)).count
        if count > maxLabelIDs {
            throw SchemaError(
                "the questions of one read need \(count) label tokens; a read allows "
                    + "\(maxLabelIDs). Ask them in separate requests.")
        }
    }
}

/// What `read_group` returns for one group: the averaged distributions and the cost.
struct GroupResult: Sendable {
    /// One mean label distribution per question of the group.
    var means: [[Double]]
    /// The billed input tokens: the billed reads plus the thought's prompt tokens.
    var billed: Int
    /// The thought tokens generated for this group.
    var thoughtTokens: Int
    /// The time spent in backend calls for this group.
    var modelTime: Duration
}

/// The read policies of `read_group`, `think` and `_sequential`, outside the actor so that
/// concurrent groups and reads run as plain tasks and only the backend serialises them.
struct GroupReader: Sendable {
    let backend: any DecisionBackend
    let configuration: EngineConfiguration
    let tokens: EngineTokens
    let resolver: TemplateResolver
    /// Upstream's `self.slots`, the `max_inflight` semaphore around every backend call.
    let slots: AsyncSemaphore

    private var tokenizer: any DecisionTokenizer { backend.tokenizer }

    /// A thought as `Engine.think` returns it, with the time it took.
    struct Thought: Sendable {
        var prefix: [Int]
        var thoughtTokens: Int
        var promptTokens: Int
        var modelTime: Duration
    }

    /// Reads every group at once, group `k` at `groupSeed(seed, k)` with chunked system text
    /// when there is more than one group. The results are in group order whatever the
    /// scheduling.
    ///
    /// Without a thought, upstream's `read_group` resolves its template and builds its prompt
    /// before its first `await`, and `asyncio.gather` starts the groups in order, so a request
    /// refused there is refused with the first group's error. Every group is prepared before
    /// any read, and the first group's error is thrown, so the answer does not depend on which
    /// group happens to finish first.
    func readGroups(
        _ groups: [[ReadQuestion]], format: AnswerFormat, stateText: String,
        images: [ImagePart], seed: UInt64, options: ReadOptions
    ) async throws -> [GroupResult] {
        let chunked = groups.count > 1
        let systemTexts = groups.map { SystemText.render($0, format: format, chunked: chunked) }
        var prepared = [PreparedGroup?](repeating: nil, count: groups.count)
        if options.think == 0 {
            prepared = try await inGroupOrder(groups.indices) { k in
                try self.prepare(
                    groups[k], format: format, systemText: systemTexts[k], stateText: stateText,
                    images: images, prefix: nil, lead: "")
            }
        }
        return try await concurrently(groups.indices) { [prepared] k in
            try await self.readGroup(
                groups[k], format: format, systemText: systemTexts[k], stateText: stateText,
                images: images, seed: SeedDerivation.groupSeed(seed, k), options: options,
                prefix: nil, lead: "", prepared: prepared[k])
        }
    }

    /// What a group's reads start from: its resolved template and its prompt.
    struct PreparedGroup: Sendable {
        var resolved: ResolvedTemplate
        var prompt: ReadPrompt
    }

    /// The template and prompt of a group, checked as upstream checks them before reading: the
    /// canvas, the backend's prompt limit and the label limit, in that order.
    func prepare(
        _ questions: [ReadQuestion], format: AnswerFormat, systemText: String,
        stateText: String, images: [ImagePart], prefix: [Int]?, lead: String
    ) throws -> PreparedGroup {
        let resolved = try resolver.resolve(
            questions, format: format, head: prefix == nil ? nil : [], lead: lead)
        let prompt = try readPrompt(
            prefix: prefix, systemText: systemText, stateText: stateText, images: images)
        try DecisionEngine.checkLabelLimit(resolved.slots)
        return PreparedGroup(resolved: resolved, prompt: prompt)
    }

    /// Upstream's `read_group`: the reads for one group of questions, averaged. `prepared` is
    /// the group's template and prompt when they were made ahead; they are made here otherwise,
    /// after the thought when there is one.
    func readGroup(
        _ questions: [ReadQuestion], format: AnswerFormat, systemText: String,
        stateText: String, images: [ImagePart], seed: UInt64, options: ReadOptions,
        prefix: [Int]?, lead: String, prepared: PreparedGroup? = nil
    ) async throws -> GroupResult {
        var prefix = prefix
        var thoughtTokens = 0
        var thinkInput = 0
        var modelTime = Duration.zero
        if prefix == nil && options.think != 0 {
            let thought = try await think(
                systemText: systemText, stateText: stateText, budget: options.think)
            prefix = thought.prefix
            thoughtTokens = thought.thoughtTokens
            thinkInput = thought.promptTokens
            modelTime += thought.modelTime
        }
        let group =
            try prepared
            ?? prepare(
                questions, format: format, systemText: systemText, stateText: stateText,
                images: images, prefix: prefix, lead: lead)
        let resolved = group.resolved
        let prompt = group.prompt
        let steps = options.steps
        let read: @Sendable (Int) async throws -> (result: ReadResult, time: Duration) = { k in
            let sampleSeed = SeedDerivation.sampleSeed(seed, k)
            let canvas = CanvasBuilder.build(
                template: resolved.template, slots: resolved.slots, seed: sampleSeed,
                geometry: self.configuration.geometry)
            let canvasRead = CanvasRead(
                prompt: prompt, systemText: systemText, stateText: stateText,
                template: resolved.template, slots: resolved.slots, canvas: canvas, steps: steps,
                seed: sampleSeed)
            let timed = try await self.timed { try await self.backend.read(canvasRead) }
            precondition(
                timed.value.slots.count == resolved.slots.count,
                "the backend returned \(timed.value.slots.count) slot reads for "
                    + "\(resolved.slots.count) slots")
            return (timed.value, timed.time)
        }

        var reads: [ReadResult] = []
        var billed = 0
        if let samples = options.samples, samples > 0 {
            let results = try await concurrently(0..<samples, read)
            reads = results.map(\.result)
            billed = results.reduce(0) { $0 + $1.result.promptTokens }
            modelTime += results.reduce(Duration.zero) { $0 + $1.time }
        } else {
            let first = try await read(0)
            reads = [first.result]
            billed = first.result.promptTokens
            modelTime += first.time
            // Re-reads are the server's own policy and are not billed.
            let entropy = first.result.slots.lazy.map(\.entropy).max() ?? -.infinity
            if configuration.autoMax > 1 && entropy > configuration.autoThreshold {
                let more = try await concurrently(1..<configuration.autoMax, read)
                reads += more.map(\.result)
                modelTime += more.reduce(Duration.zero) { $0 + $1.time }
            }
        }
        let means = questions.indices.map { qi in
            ReadAveraging.mean(reads.map { $0.slots[qi].probabilities })
        }
        return GroupResult(
            means: means, billed: billed + thinkInput, thoughtTokens: thoughtTokens,
            modelTime: modelTime)
    }

    /// Upstream's `_sequential`: the groups continue one answer in order under the full
    /// question list, each group's chosen labels written into the prompt before the next read.
    func sequential(
        _ groups: [[ReadQuestion]], format: AnswerFormat, allQuestions: [ReadQuestion],
        stateText: String, seed: UInt64, options: ReadOptions
    ) async throws -> [GroupResult] {
        let systemText = SystemText.render(allQuestions, format: format, chunked: false)
        var thoughtTokens = 0
        var thinkInput = 0
        var thinkTime = Duration.zero
        let base: [Int]
        if options.think != 0 {
            let thought = try await think(
                systemText: systemText, stateText: stateText, budget: options.think)
            base = thought.prefix
            thoughtTokens = thought.thoughtTokens
            thinkInput = thought.promptTokens
            thinkTime = thought.modelTime
        } else {
            base =
                try tokenizer.chatPromptIDs(system: systemText, user: stateText, thinking: false)
                + tokens.scaffold
        }
        let join = format.join
        var groupOptions = options
        groupOptions.think = 0
        var lines: [String] = []
        var results: [GroupResult] = []
        for (k, group) in groups.enumerated() {
            let prefix: [Int]?
            let lead: String
            if !lines.isEmpty {
                prefix =
                    base
                    + (try tokenizer.encode(
                        lines.joined(separator: join), addSpecialTokens: false))
                lead = join
            } else {
                prefix = options.think != 0 ? base : nil
                lead = ""
            }
            let result = try await readGroup(
                group, format: format, systemText: systemText, stateText: stateText, images: [],
                seed: SeedDerivation.groupSeed(seed, k), options: groupOptions, prefix: prefix,
                lead: lead)
            // The thought's cost is attributed to the first group only.
            let first = k == 0
            results.append(
                GroupResult(
                    means: result.means, billed: result.billed + (first ? thinkInput : 0),
                    thoughtTokens: first ? thoughtTokens : 0,
                    modelTime: result.modelTime + (first ? thinkTime : .zero)))
            let chosen = result.means.map(Self.firstMaximum)
            lines.append(AnswerText.render(group, labelIndices: chosen, format: format))
        }
        return results
    }

    /// Upstream's `think`: the chat prompt with thinking on and the open marker, the backend's
    /// generation cut at the first close id, then the close marker.
    func think(systemText: String, stateText: String, budget: Int) async throws -> Thought {
        let prompt =
            try tokenizer.chatPromptIDs(system: systemText, user: stateText, thinking: true)
            + tokens.thoughtOpen
        try checkPromptLength(prompt)
        let timed = try await timed {
            try await backend.think(prompt: prompt, budget: budget, stopIDs: tokens.thoughtClose)
        }
        var ids = timed.value.generated
        if let close = tokens.thoughtClose.first, let index = ids.firstIndex(of: close) {
            ids = Array(ids[..<index])
        }
        return Thought(
            prefix: prompt + ids + tokens.thoughtClose, thoughtTokens: ids.count,
            promptTokens: timed.value.promptTokens, modelTime: timed.time)
    }

    /// The prompt of a read, as `MlxEngine.one_read` builds it: an image prompt for images,
    /// otherwise the prefix or the chat prompt ids, checked against the backend's limit.
    private func readPrompt(
        prefix: [Int]?, systemText: String, stateText: String, images: [ImagePart]
    ) throws -> ReadPrompt {
        if !images.isEmpty {
            // Images and think or sequential are mutually exclusive, so there is no prefix here.
            return .image(systemText: systemText, stateText: stateText, images: images)
        }
        let ids =
            try prefix
            ?? tokenizer.chatPromptIDs(system: systemText, user: stateText, thinking: false)
        try checkPromptLength(ids)
        return .tokens(ids)
    }

    /// `MlxEngine`'s prompt bound: `"the request is {n} tokens; the limit is {max}"`.
    private func checkPromptLength(_ ids: [Int]) throws(SchemaError) {
        if ids.count > backend.maxPromptTokens {
            throw SchemaError(
                "the request is \(ids.count) tokens; the limit is \(backend.maxPromptTokens)")
        }
    }

    /// Runs one backend call under the in-flight semaphore and measures it, wait included.
    private func timed<T: Sendable>(
        _ call: @Sendable () async throws -> T
    ) async throws -> (value: T, time: Duration) {
        let clock = ContinuousClock()
        let started = clock.now
        let value = try await slots.withPermit(call)
        return (value, clock.now - started)
    }

    /// Runs `body` for every index at once and returns the results in index order. When some
    /// calls fail, it waits for every call and throws the error of the lowest index.
    private func inGroupOrder<T: Sendable>(
        _ indices: Range<Int>, _ body: @Sendable @escaping (Int) throws -> T
    ) async throws -> [T] {
        let outcomes = await withTaskGroup(of: (Int, Result<T, any Error>).self) { group in
            for k in indices {
                group.addTask { (k, Result { try body(k) }) }
            }
            var outcomes = [Result<T, any Error>?](repeating: nil, count: indices.count)
            for await (k, outcome) in group {
                outcomes[k - indices.lowerBound] = outcome
            }
            return outcomes
        }
        return try outcomes.map { outcome in
            guard let outcome else { preconditionFailure("a task returned no result") }
            return try outcome.get()
        }
    }

    /// Runs `body` for every index at once and returns the results in index order.
    private func concurrently<T: Sendable>(
        _ indices: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T] {
        try await withThrowingTaskGroup(of: (Int, T).self) { group in
            for k in indices {
                group.addTask { (k, try await body(k)) }
            }
            var results = [T?](repeating: nil, count: indices.count)
            for try await (k, value) in group {
                results[k - indices.lowerBound] = value
            }
            let ordered = results.compactMap { $0 }
            precondition(ordered.count == indices.count, "a task returned no result")
            return ordered
        }
    }

    /// The index of the first largest value, Python's `max(range(len(m)), key=m.__getitem__)`.
    static func firstMaximum(_ values: [Double]) -> Int {
        var top = 0
        for index in values.indices where values[index] > values[top] {
            top = index
        }
        return top
    }
}
