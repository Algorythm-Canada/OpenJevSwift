// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/chat.py`: `MlxGenerator`
// (`prompt_ids`, `stop_ids`, `generate`, `complete`), the capacity bound and slots `Generator`
// holds, and the order of the `chat_completions` route. Apache-2.0. See THIRD_PARTY.md.

import Foundation

/// The settings of text generation, upstream's `gen_*` settings.
public struct ChatCompletionsConfiguration: Sendable, Hashable {
    /// Generations running at once, `OPENJEV_GEN_MAX_INFLIGHT`. The model runs one at a time, so
    /// above 1 this only queues, as upstream's does.
    public var maxInflight: Int
    /// Requests waiting for a generation before a 529, `OPENJEV_GEN_MAX_QUEUE`.
    public var maxQueue: Int
    /// The longest reply in tokens, `OPENJEV_GEN_MAX_TOKENS`.
    public var maxTokens: Int

    /// Creates settings; every argument defaults to upstream's value.
    ///
    /// - Precondition: `maxInflight` and `maxTokens` are at least 1 and `maxQueue` is not
    ///   negative, which the server's settings check at startup.
    public init(maxInflight: Int = 8, maxQueue: Int = 32, maxTokens: Int = 8192) {
        precondition(maxInflight >= 1, "maxInflight must be at least 1, got \(maxInflight)")
        precondition(maxQueue >= 0, "maxQueue must not be negative, got \(maxQueue)")
        precondition(maxTokens >= 1, "maxTokens must be at least 1, got \(maxTokens)")
        self.maxInflight = maxInflight
        self.maxQueue = maxQueue
        self.maxTokens = maxTokens
    }
}

/// A chat completion request ready to generate: checked, normalized, its prompt rendered within
/// the limit and its stop ids found, all before an answer starts.
public struct PreparedChatCompletion: Sendable, Hashable {
    /// The normalized request.
    public var request: ChatCompletionRequest
    /// The prompt ids, upstream's `prompt_ids`, scaffold included.
    public var prompt: [Int]
    /// The ids of the request's single-token `stop` strings, upstream's `stop_ids`.
    public var stopIDs: [Int]
}

/// `POST /v1/chat/completions` over a ``TextGenerator``: upstream's `MlxGenerator`, whose
/// normalization, capacity bound and response shapes it shares with the vLLM proxy this port does
/// not have.
///
/// A request is answered in three steps, each before the next: ``prepare(_:)`` refuses it with
/// upstream's answers before anything is sent; ``complete(_:)`` generates a whole reply, or
/// ``stream(_:)`` admits a streamed one and waits for its turn, before its answer starts; the
/// stream is then drained by ``ChatCompletionStream/run(_:)``.
///
/// Capacity is upstream's: a request counts from its admission until its generation has stopped,
/// at most `OPENJEV_GEN_MAX_INFLIGHT` of them generate at once and the rest wait, and a request that
/// finds `OPENJEV_GEN_MAX_INFLIGHT` plus `OPENJEV_GEN_MAX_QUEUE` counted is the 529. A request
/// cancelled while it waits gives its place back.
public final class ChatCompletions: Sendable {
    /// The model.
    public let generator: any TextGenerator
    /// The settings.
    public let configuration: ChatCompletionsConfiguration
    /// The capacity bound and the slots.
    let capacity: GenerationCapacity
    /// Where each reply's id and creation time come from: ``ChatCompletionIdentity/random()``,
    /// or a fixed one in tests.
    let identity: @Sendable () -> ChatCompletionIdentity

    /// Creates the service over a generator.
    public convenience init(
        generator: any TextGenerator, configuration: ChatCompletionsConfiguration = .init()
    ) {
        self.init(generator: generator, configuration: configuration, identity: { .random() })
    }

    /// Creates the service with the source of each reply's id and creation time.
    public init(
        generator: any TextGenerator, configuration: ChatCompletionsConfiguration,
        identity: @escaping @Sendable () -> ChatCompletionIdentity
    ) {
        self.generator = generator
        self.configuration = configuration
        self.capacity = GenerationCapacity(
            inflight: configuration.maxInflight, queue: configuration.maxQueue)
        self.identity = identity
    }

    /// The requests counted against the capacity bound now: admitted and not yet stopped.
    public var running: Int {
        capacity.running
    }

    /// The slots free now, for tests: `OPENJEV_GEN_MAX_INFLIGHT` when nothing generates.
    public var freeSlots: Int {
        capacity.freeSlots
    }

    /// Checks and normalizes a parsed body and renders its prompt, in upstream's order: the
    /// body's `messages` and `model` (``ChatCompletionRequest/checked(_:)``), the capacity bound,
    /// `normalize` (``ChatCompletionRequest/init(normalizing:maxTokensCap:)``), the prompt and its
    /// limit, then the stop strings.
    ///
    /// - Throws: A ``ChatCompletionError``: upstream's 400s and 404, the 529 when the bound is
    ///   reached, the 400 for a prompt over ``TextGenerator/maxPromptTokens``, and this port's 400s
    ///   for the requests upstream crashes on, a template that cannot render the messages
    ///   included; and the 503 for a `stop` string the generator cannot encode.
    public func prepare(
        _ body: JSONValue
    ) async throws(ChatCompletionError) -> PreparedChatCompletion {
        let object = try ChatCompletionRequest.checked(body)
        if capacity.isFull {
            throw .overloaded
        }
        let request = try ChatCompletionRequest(
            normalizing: object, maxTokensCap: configuration.maxTokens)
        let prompt: [Int]
        do {
            prompt = try await generator.generationPromptIDs(
                messages: request.messages, thinking: request.thinking)
        } catch {
            throw .invalidRequest(
                "The messages could not be rendered with the model's chat template: \(error)")
        }
        if prompt.count > generator.maxPromptTokens {
            throw .promptTooLong(tokens: prompt.count, limit: generator.maxPromptTokens)
        }
        var stopIDs: [Int] = []
        for text in try request.stopStrings() {
            let ids: [Int]
            do {
                ids = try generator.encode(text)
            } catch {
                throw .backendUnavailable(String(describing: type(of: error)))
            }
            // Only a stop that is one token can end a denoised block; the rest are dropped.
            if ids.count == 1 {
                stopIDs.append(ids[0])
            }
        }
        return PreparedChatCompletion(request: request, prompt: prompt, stopIDs: stopIDs)
    }

    /// Generates a whole reply, upstream's `MlxGenerator.complete`: admitted, then in its turn,
    /// every emitted text joined, and in JSON mode reduced to its first JSON object or array.
    /// Cancelling the task stops the generation at its next block.
    ///
    /// - Throws: ``ChatCompletionError/overloaded`` when the bound was reached since
    ///   ``prepare(_:)``, `CancellationError` when the task was cancelled while it waited, and
    ///   whatever the generator throws.
    public func complete(_ prepared: PreparedChatCompletion) async throws -> ChatCompletion {
        try capacity.admit()
        defer { capacity.leave() }
        try await capacity.slots.wait()
        defer { capacity.slots.signal() }
        let parts = TextParts()
        let generation = try await generator.generate(
            prompt: prepared.prompt, maxTokens: prepared.request.maxTokens,
            stopIDs: prepared.stopIDs, skipSpecialTokenIDs: generator.thoughtChannelMarkerIDs,
            emit: { text, _ in
                parts.append(text)
                return true
            })
        let text = parts.joined()
        return ChatCompletion(
            identity: identity(),
            content: prepared.request.jsonMode ? ExtractJSON.extract(text) : text,
            finishReason: generation.finishReason,
            usage: ChatCompletionUsage(
                promptTokens: generation.promptTokens,
                completionTokens: generation.generated.count))
    }

    /// Admits a streamed reply and waits for its turn, upstream's `MlxGenerator.stream` before
    /// its response starts. The returned stream holds the place until it has been run and its
    /// generation has stopped, or until it is discarded without being run.
    ///
    /// - Throws: ``ChatCompletionError/overloaded`` when the bound was reached since
    ///   ``prepare(_:)``, and `CancellationError` when the task was cancelled while it waited,
    ///   which gives the place back.
    public func stream(_ prepared: PreparedChatCompletion) async throws -> ChatCompletionStream {
        try capacity.admit()
        let lease = GenerationLease(capacity: capacity)
        try await capacity.slots.wait()
        lease.holdSlot()
        return ChatCompletionStream(
            prepared: prepared, generator: generator, identity: identity(), lease: lease)
    }
}

/// Upstream's `running` count and `slots` semaphore: requests counted from admission until their
/// generation stops, and the generations allowed to run at once.
final class GenerationCapacity: @unchecked Sendable {
    /// `OPENJEV_GEN_MAX_INFLIGHT` plus `OPENJEV_GEN_MAX_QUEUE`.
    let limit: Int
    /// `asyncio.Semaphore(gen_max_inflight)`.
    let slots: AsyncSemaphore

    // Guarded by `lock`.
    private let lock = NSLock()
    private var counted = 0

    init(inflight: Int, queue: Int) {
        limit = inflight + queue
        slots = AsyncSemaphore(permits: inflight)
    }

    /// The requests counted now.
    var running: Int {
        lock.withLock { counted }
    }

    /// The slots free now.
    var freeSlots: Int {
        slots.availablePermits
    }

    /// Whether a request now would be refused, the route's `running >= max_inflight +
    /// max_queue`.
    var isFull: Bool {
        lock.withLock { counted >= limit }
    }

    /// Counts a request in, or refuses it when the bound is reached.
    func admit() throws(ChatCompletionError) {
        let admitted = lock.withLock { () -> Bool in
            guard counted < limit else { return false }
            counted += 1
            return true
        }
        if !admitted {
            throw .overloaded
        }
    }

    /// Counts a request out.
    func leave() {
        lock.withLock {
            precondition(counted > 0, "leave() without a matching admit()")
            counted -= 1
        }
    }
}

/// A streamed request's place: counted in from ``ChatCompletions/stream(_:)`` and, once its turn
/// came, holding a slot. ``release()`` gives both back once; a lease that is discarded first, as
/// the stream of an answer whose head could not be written is, gives them back as it goes.
final class GenerationLease: @unchecked Sendable {
    private let capacity: GenerationCapacity

    // Guarded by `lock`.
    private let lock = NSLock()
    private var slotHeld = false
    private var released = false

    init(capacity: GenerationCapacity) {
        self.capacity = capacity
    }

    /// Records that the slot was taken.
    func holdSlot() {
        lock.withLock { slotHeld = true }
    }

    /// Gives the slot, if taken, and the place back; later calls do nothing.
    func release() {
        let slot = lock.withLock { () -> Bool? in
            guard !released else { return nil }
            released = true
            return slotHeld
        }
        guard let slot else { return }
        if slot {
            capacity.slots.signal()
        }
        capacity.leave()
    }

    deinit {
        release()
    }
}

/// The texts a generation emitted, in order, from whichever executor emits them.
final class TextParts: @unchecked Sendable {
    private let lock = NSLock()
    private var parts: [String] = []

    func append(_ text: String) {
        lock.withLock { parts.append(text) }
    }

    func joined() -> String {
        lock.withLock { parts.joined() }
    }
}
