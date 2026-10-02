// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`,
// `WARMUP_QUESTIONS` and the methods `EncoderEngine.__init__` (the warm-up read),
// `EncoderEngine.read` and `EncoderEngine.decide`. Apache-2.0. See THIRD_PARTY.md.

/// Answers Jev requests by reading questions in batches through a ``QuestionReadBackend``.
///
/// The engine owns what upstream's `EncoderEngine` owns apart from the model: the schema builder
/// with the backend's option limit, the refusal of the options no encoder honours, the queue
/// bound, the batching, the billing and the answer shapes. A request goes through
/// ``decide(_:)``; the CLI and the server call ``warmUp()`` once after the model has loaded.
///
/// Errors: a ``SchemaError`` is a request the model cannot answer as asked (a 400), an
/// ``OverloadedError`` is the queue bound (a 529), a ``BackendContractError`` is a distribution
/// the backend should never have returned (the server's 503, as for any backend failure), and the
/// backend's own errors pass through unchanged.
public actor EncoderDecisionEngine {
    /// Upstream's `WARMUP_QUESTIONS`: a two-option choice, a two-level score and a noul, each
    /// with the instructions `x`, read against the state `warmup` before the first user.
    public static let warmUpQuestions: OrderedMap<Question> = [
        "c": .choice(instructions: .string("x"), criteria: ["a": .null, "b": .null]),
        "s": .score(instructions: .string("x"), criteria: [.string("low"), .string("high")]),
        "n": .noul(instructions: .string("x"), criteria: nil),
    ]

    /// The state the warm-up read is about.
    public static let warmUpState: JSONValue = .string("warmup")

    /// The model the engine reads through.
    public nonisolated let backend: any QuestionReadBackend
    /// The settings.
    public nonisolated let configuration: EncoderEngineConfiguration
    /// The schema builder over the backend's option limit.
    public nonisolated let schemaBuilder: EncoderQuestionSchemaBuilder

    /// Upstream's one model thread: the `maxInflight` semaphore around every backend call.
    private nonisolated let permits: AsyncSemaphore
    /// Requests inside ``decide(_:)`` right now, upstream's `waiting`, and its bound.
    private var queue: RequestQueue

    /// Creates an engine over a loaded backend. Nothing is read here; call ``warmUp()`` for
    /// upstream's warm-up read.
    public init(
        backend: any QuestionReadBackend,
        configuration: EncoderEngineConfiguration = .default
    ) {
        self.backend = backend
        self.configuration = configuration
        self.schemaBuilder = EncoderQuestionSchemaBuilder(maxChoices: backend.maxChoices)
        self.permits = AsyncSemaphore(permits: configuration.maxInflight)
        self.queue = RequestQueue(limit: configuration.maxQueue)
    }

    /// The served model's name, which every refusal names.
    private nonisolated var modelName: String { backend.modelInfo.name }

    /// Answers a request, as upstream's `EncoderEngine.decide` does.
    ///
    /// In order: the options with their defaults; the refusal of `images`, `steps` above 1,
    /// `samples` above 1, `think` and `sequential`, in that order and before any read; the queue
    /// bound; the schema, whose forced answers and limits are ``EncoderQuestionSchemaBuilder``'s;
    /// the reads, in batches of ``EncoderEngineConfiguration/batchSize`` in request order, each
    /// one backend call under the in-flight bound; and the answers in the request's order,
    /// forced answers included.
    ///
    /// The request's seed is not used: a read is deterministic, so there is no seed parameter
    /// and ``SeedDerivation`` is never run. `outputTokens` is always 0.
    ///
    /// - Throws: ``SchemaError`` for an unsupported option (`"{model} does not support
    ///   {field}"`, located at `["body", field]`) or a schema the model cannot answer;
    ///   ``OverloadedError`` (`"{model} is at capacity. Retry shortly."`) when
    ///   ``EncoderEngineConfiguration/maxQueue`` requests are already inside;
    ///   ``BackendContractError`` for a distribution that breaks the backend's contract; and the
    ///   backend's own errors.
    public func decide(_ request: SystemOneRequest) async throws -> Decision {
        let options = ReadOptions(request)
        try UnsupportedOptions.check(
            options, hasImages: !(request.images ?? []).isEmpty, capabilities: .readsOnly,
            modelName: modelName)
        try queue.admit(refusing: "\(modelName) is at capacity. Retry shortly.")
        defer { queue.leave() }

        let schema = try schemaBuilder.build(request.questions)
        let read = try await read(schema.questions, state: request.state)

        var answers = schema.forced
        for (question, probabilities) in zip(schema.questions, read.probabilities) {
            answers.updateValue(
                Answer.make(for: question.question, probabilities: probabilities),
                forKey: question.key)
        }
        return Decision(
            answers: answers.ordered(as: request.questions.keys), inputTokens: read.inputTokens,
            outputTokens: 0, modelTime: read.modelTime)
    }

    /// Reads upstream's ``warmUpQuestions`` once, so the first user does not pay for kernel
    /// compilation, as `EncoderEngine.__init__` does when `OPENJEV_WARMUP` is set.
    ///
    /// Does nothing when ``EncoderEngineConfiguration/warmUp`` is false. The read is not counted
    /// against the queue bound.
    ///
    /// - Throws: The backend's own errors, or ``BackendContractError``.
    public func warmUp() async throws {
        guard configuration.warmUp else { return }
        let schema = try schemaBuilder.build(Self.warmUpQuestions)
        _ = try await read(schema.questions, state: Self.warmUpState)
    }

    /// What `EncoderEngine.read` returns, with the time spent in the backend.
    private struct ReadOutcome: Sendable {
        var probabilities: [[Double]]
        var inputTokens: Int
        var modelTime: Duration
    }

    /// Upstream's `EncoderEngine.read`: the questions in batches of `batchSize`, in order, each
    /// batch one backend call under the in-flight bound, the distributions concatenated and the
    /// tokens summed. No questions means no call. Nonisolated, so the actor is free for other
    /// requests while a batch runs.
    ///
    /// Each call's time, wait included, also goes to ``ModelTimeRecorder/current`` when the call
    /// ends, whether it returned, threw or was cancelled, so a request whose later batch fails
    /// still reports the earlier ones. A cancelled request starts no further batch: the permit
    /// refuses a cancelled task.
    private nonisolated func read(_ questions: [EncoderQuestion], state: JSONValue) async throws
        -> ReadOutcome
    {
        var outcome = ReadOutcome(probabilities: [], inputTokens: 0, modelTime: .zero)
        guard !questions.isEmpty else { return outcome }
        let stateText = StateText.render(state)
        let clock = ContinuousClock()
        var start = questions.startIndex
        while start < questions.endIndex {
            let end = min(start + configuration.batchSize, questions.endIndex)
            let batch = Array(questions[start..<end])
            let began = clock.now
            let result: BatchReadResult
            do {
                result = try await permits.withPermit {
                    try await backend.readBatch(
                        state: state, stateText: stateText, questions: batch)
                }
            } catch {
                ModelTimeRecorder.record(clock.now - began)
                throw error
            }
            let time = clock.now - began
            ModelTimeRecorder.record(time)
            outcome.modelTime += time
            try validate(result, for: batch)
            outcome.probabilities += result.probabilities
            outcome.inputTokens += result.inputTokens
            start = end
        }
        return outcome
    }

    /// Checks one batch's result against the contract: one distribution per question, each with
    /// one value in `[0, 1]` per option (which rules out NaN and the infinities), summing to 1
    /// within 1e-6.
    ///
    /// - Throws: ``BackendContractError`` naming the model and the question.
    private nonisolated func validate(_ result: BatchReadResult, for batch: [EncoderQuestion])
        throws(BackendContractError)
    {
        guard result.probabilities.count == batch.count else {
            throw BackendContractError(
                "\(modelName) returned \(result.probabilities.count) distributions for a batch "
                    + "of \(batch.count) questions")
        }
        for (question, probabilities) in zip(batch, result.probabilities) {
            let expected = question.choices.count
            guard probabilities.count == expected else {
                throw BackendContractError(
                    "\(modelName) returned \(probabilities.count) probabilities for question "
                        + "\(question.key.pythonRepr), which has \(expected) options")
            }
            // A NaN fails both comparisons, so this also rejects values that are not finite.
            guard probabilities.allSatisfy({ $0 >= 0 && $0 <= 1 }) else {
                throw BackendContractError(
                    "\(modelName) returned a probability outside [0, 1] for question "
                        + "\(question.key.pythonRepr): \(probabilities)")
            }
            let total = probabilities.reduce(0, +)
            guard abs(total - 1) <= 1e-6 else {
                throw BackendContractError(
                    "\(modelName) returned probabilities summing to \(total) for question "
                        + "\(question.key.pythonRepr); they must sum to 1")
            }
        }
    }
}
