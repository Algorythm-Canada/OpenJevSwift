// A port of upstream OpenJev (razorback16/openjev at dcd2094), `MlxGenerator.stream` in
// `openjev/chat.py`: the generation beside a bounded queue of 64 chunks, the role chunk first, the
// finish and usage chunks and `[DONE]` last, and a reply that ends early, rather than one that
// loses a chunk, when its reader falls behind or goes away. Apache-2.0. See THIRD_PARTY.md.

import Foundation

/// A streamed reply, admitted and holding its slot (``ChatCompletions/stream(_:)``), ready to run.
///
/// ``run(_:)`` starts the generation and writes the reply's server-sent events as the generation
/// emits its text: the role, each non-empty piece, the finish reason, the usage and `[DONE]`. The
/// generation hands its pieces over through a queue of ``bufferCapacity`` entries, so it never
/// waits for the reader and cannot run far ahead of one. A piece that finds the queue full means
/// the reader is gone or hopelessly behind: the generation is asked to stop at its next block, the
/// pieces already queued are still written, and the stream ends there without a finish chunk or
/// `[DONE]`, so a client never reads a reply with a piece missing as if it were whole. Cancelling
/// the task that runs it, as the server does when the client goes away, stops the generation the
/// same way and writes nothing more.
///
/// The slot and the place are given back once the generation has stopped, however the run ends,
/// or when a stream that never runs is discarded.
public final class ChatCompletionStream: Sendable {
    /// Upstream's `asyncio.Queue(maxsize=64)`: the pieces, empty ones included, that may wait for
    /// the reader.
    public static let bufferCapacity = 64

    /// The generator failed after the stream started; ``error`` is what it threw.
    public struct GenerationFailed: Error, @unchecked Sendable {
        /// The generator's error.
        public let error: any Error
    }

    /// How a run ended.
    public enum Ending: Sendable, Hashable {
        /// Every event was written, `[DONE]` last.
        case completed(TextGeneration)
        /// The reader fell ``ChatCompletionStream/bufferCapacity`` pieces behind: the generation
        /// was stopped and the stream ended after the pieces already queued.
        case readerFellBehind
    }

    /// The request.
    public let prepared: PreparedChatCompletion
    /// The id and creation time every event carries.
    public let identity: ChatCompletionIdentity
    let generator: any TextGenerator
    let lease: GenerationLease
    private let buffer = ChunkBuffer(capacity: ChatCompletionStream.bufferCapacity)
    private let started = StartFlag()

    init(
        prepared: PreparedChatCompletion, generator: any TextGenerator,
        identity: ChatCompletionIdentity, lease: GenerationLease
    ) {
        self.prepared = prepared
        self.generator = generator
        self.identity = identity
        self.lease = lease
    }

    /// Stops the stream from any task, as the server does when the client goes away: the
    /// generation is asked to stop at its next block and ``run(_:)`` writes nothing more and
    /// throws `CancellationError`. Before the run, the run stops at once.
    public func cancel() {
        buffer.cancel()
    }

    /// Runs the generation and writes the reply's events with `write`, each a complete event
    /// (`data: ...` and a blank line), returning once the generation has stopped.
    ///
    /// - Precondition: The stream has not run before.
    /// - Throws: Whatever `write` throws, after stopping the generation; ``GenerationFailed``
    ///   when the generation failed, after the pieces it emitted were written, with no finish
    ///   chunk; and `CancellationError` when the stream or the task was cancelled.
    public func run(_ write: (String) async throws -> Void) async throws -> Ending {
        precondition(started.start(), "a ChatCompletionStream runs once")
        // Runs after the task group, so once the generation has stopped.
        defer { lease.release() }
        let buffer = buffer
        if buffer.isCancelled {
            throw CancellationError()
        }
        let generator = generator
        let prepared = prepared
        return try await withThrowingTaskGroup(of: Void.self) { group in
            // The outcome goes through the buffer, never a child's result, and only after every
            // piece the generation emitted is queued.
            group.addTask {
                do {
                    let generation = try await generator.generate(
                        prompt: prepared.prompt, maxTokens: prepared.request.maxTokens,
                        stopIDs: prepared.stopIDs,
                        skipSpecialTokenIDs: generator.thoughtChannelMarkerIDs,
                        emit: { text, _ in buffer.offer(text) })
                    buffer.finish(.success(generation))
                } catch {
                    buffer.finish(.failure(error))
                }
            }
            do {
                let ending = try await withTaskCancellationHandler {
                    try await drain(buffer, write)
                } onCancel: {
                    buffer.cancel()
                }
                try await group.waitForAll()
                return ending
            } catch {
                // Nothing more can be written: stop the generation, which the group then awaits.
                buffer.cancel()
                group.cancelAll()
                throw error
            }
        }
    }

    /// Writes the events as the pieces arrive.
    private func drain(
        _ buffer: ChunkBuffer, _ write: (String) async throws -> Void
    ) async throws -> Ending {
        try await write(ChatCompletionEvent.role(identity))
        while true {
            switch await buffer.next() {
            case .piece(let text):
                if !text.isEmpty {
                    try await write(ChatCompletionEvent.content(identity, text))
                }
            case .finished(.success(let generation)):
                try await write(ChatCompletionEvent.finish(identity, generation.finishReason))
                if prepared.request.includeUsage {
                    try await write(
                        ChatCompletionEvent.usage(
                            identity,
                            ChatCompletionUsage(
                                promptTokens: generation.promptTokens,
                                completionTokens: generation.generated.count)))
                }
                try await write(ChatCompletionEvent.done)
                return .completed(generation)
            case .finished(.failure(let error)):
                if error is CancellationError {
                    throw error
                }
                throw GenerationFailed(error: error)
            case .readerFellBehind:
                return .readerFellBehind
            case .cancelled:
                throw CancellationError()
            }
        }
    }
}

/// Whether a stream has run.
private final class StartFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false

    /// Marks the stream started; false when it already was.
    func start() -> Bool {
        lock.withLock {
            defer { started = true }
            return !started
        }
    }
}

/// The queue between a generation's `emit` and a stream's reader, upstream's `asyncio.Queue(64)`
/// with its `END` marker and `cancel` flag.
///
/// `offer` never blocks. The reader takes the pieces in order and, once none is left, how the
/// stream ends: the generation's outcome, or that the reader fell behind. Cancelling ends the
/// reader's wait at once, queued pieces or not, since nobody reads them.
final class ChunkBuffer: @unchecked Sendable {
    /// What the reader gets next.
    enum Next {
        /// A piece the generation emitted, possibly empty.
        case piece(String)
        /// The generation ended, and every piece it emitted has been taken.
        case finished(Result<TextGeneration, any Error>)
        /// A piece found the queue full; every piece queued before it has been taken.
        case readerFellBehind
        /// The stream was cancelled.
        case cancelled
    }

    private typealias Waiter = CheckedContinuation<Void, Never>

    let capacity: Int

    // Guarded by `lock`.
    private let lock = NSLock()
    private var pieces: [String] = []
    private var head = 0
    private var outcome: Result<TextGeneration, any Error>?
    private var fellBehind = false
    private var cancelled = false
    private var waiter: Waiter?

    init(capacity: Int) {
        self.capacity = capacity
    }

    /// Queues a piece, from the generation's executor. False when the reply is no longer wanted:
    /// the stream was cancelled, the reader fell behind before, or this piece finds the queue full,
    /// in which case it is not queued and the reader ends after the pieces before it.
    func offer(_ text: String) -> Bool {
        let (accepted, woken) = lock.withLock { () -> (Bool, Waiter?) in
            guard !cancelled, !fellBehind else { return (false, nil) }
            if pieces.count - head >= capacity {
                fellBehind = true
                return (false, takeWaiter())
            }
            pieces.append(text)
            return (true, takeWaiter())
        }
        woken?.resume()
        return accepted
    }

    /// Records the generation's outcome, after its last piece.
    func finish(_ result: Result<TextGeneration, any Error>) {
        let woken = lock.withLock { () -> Waiter? in
            outcome = result
            return takeWaiter()
        }
        woken?.resume()
    }

    /// Whether the stream was cancelled.
    var isCancelled: Bool {
        lock.withLock { cancelled }
    }

    /// Cancels the stream: the reader's wait ends and `offer` refuses from now on.
    func cancel() {
        let woken = lock.withLock { () -> Waiter? in
            cancelled = true
            return takeWaiter()
        }
        woken?.resume()
    }

    /// The next piece, or how the stream ended once none is left.
    func next() async -> Next {
        while true {
            if let next = lock.withLock({ ready() }) {
                return next
            }
            await withTaskCancellationHandler {
                await withCheckedContinuation { (continuation: Waiter) in
                    let now = lock.withLock { () -> Bool in
                        if ready(peeking: true) != nil {
                            return true
                        }
                        waiter = continuation
                        return false
                    }
                    if now {
                        continuation.resume()
                    }
                }
            } onCancel: {
                cancel()
            }
        }
    }

    /// What the reader gets now, or nil when it must wait. Called with the lock held; takes the
    /// piece unless `peeking`.
    private func ready(peeking: Bool = false) -> Next? {
        if cancelled {
            return .cancelled
        }
        if head < pieces.count {
            let piece = pieces[head]
            if !peeking {
                head += 1
                // Drop the pieces taken once there are as many as the queue holds, so a reader
                // that never quite catches up does not keep every piece of a long reply.
                if head == pieces.count {
                    pieces.removeAll(keepingCapacity: true)
                    head = 0
                } else if head >= capacity {
                    pieces.removeFirst(head)
                    head = 0
                }
            }
            return .piece(piece)
        }
        if fellBehind {
            return .readerFellBehind
        }
        if let outcome {
            return .finished(outcome)
        }
        return nil
    }

    /// The waiting reader, removed. Called with the lock held.
    private func takeWaiter() -> Waiter? {
        defer { waiter = nil }
        return waiter
    }
}
