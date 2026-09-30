import Foundation

/// A counting semaphore for tasks, standing in for upstream's `asyncio.Semaphore(max_inflight)`.
///
/// `wait()` suspends until a permit is free and `signal()` hands the permit to the longest
/// waiting task, or returns it. A waiting task that is cancelled leaves the queue and throws
/// `CancellationError`. The state is behind a lock so the semaphore can be shared by the engine
/// actor and the tasks it spawns for concurrent reads.
///
/// The queue is first in, first out in O(1) per operation, as `asyncio`'s deque is: waiter ids
/// sit in an array read from a moving head, and the continuations live in a dictionary keyed by
/// id. Cancelling removes the continuation and leaves a stale id that `signal()` skips; the array
/// is compacted once the head has passed half of it. With `max_queue` requests each waiting on
/// up to 32 sample reads, an array shifted on every hand-off would make a drain quadratic.
final class AsyncSemaphore: @unchecked Sendable {
    private typealias Continuation = CheckedContinuation<Void, any Error>

    private let lock = NSLock()
    private var permits: Int
    /// Waiter ids in arrival order, live from ``head`` on.
    private var queue: [UInt64] = []
    /// The index of the oldest entry of ``queue`` that has not been handed a permit.
    private var head = 0
    /// The continuation of every waiter that is still waiting, by id.
    private var continuations: [UInt64: Continuation] = [:]
    private var nextID: UInt64 = 0

    /// Creates a semaphore with `permits` free permits.
    ///
    /// - Precondition: `permits` is at least 1, as upstream's settings require of
    ///   `max_inflight`.
    init(permits: Int) {
        precondition(permits >= 1, "a semaphore needs at least one permit, got \(permits)")
        self.permits = permits
    }

    /// Takes a permit, suspending until one is free.
    ///
    /// - Throws: `CancellationError` when the task is cancelled while waiting.
    func wait() async throws {
        let id = lock.withLock {
            nextID += 1
            return nextID
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: Continuation) in
                let outcome: Result<Void, any Error>? = lock.withLock {
                    if Task.isCancelled {
                        return .failure(CancellationError())
                    }
                    if permits > 0 {
                        permits -= 1
                        return .success(())
                    }
                    continuations[id] = continuation
                    queue.append(id)
                    return nil
                }
                if let outcome {
                    continuation.resume(with: outcome)
                }
            }
        } onCancel: {
            let waiting = lock.withLock { continuations.removeValue(forKey: id) }
            waiting?.resume(throwing: CancellationError())
        }
    }

    /// Returns a permit, waking the oldest waiter still waiting if there is one.
    func signal() {
        let waiting = lock.withLock { () -> Continuation? in
            while head < queue.count {
                let id = queue[head]
                head += 1
                if let continuation = continuations.removeValue(forKey: id) {
                    compactIfNeeded()
                    return continuation
                }
            }
            // Every queued id was cancelled: the queue is stale and the permit is free.
            queue.removeAll(keepingCapacity: true)
            head = 0
            permits += 1
            return nil
        }
        waiting?.resume()
    }

    /// Runs `body` holding a permit.
    func withPermit<T>(_ body: () async throws -> T) async throws -> T {
        try await wait()
        defer { signal() }
        return try await body()
    }

    /// Drops the consumed prefix of ``queue`` once it is at least half of the array, so the
    /// shift costs O(1) amortised per hand-off. Called with the lock held.
    private func compactIfNeeded() {
        if head >= 64 && head * 2 >= queue.count {
            queue.removeFirst(head)
            head = 0
        }
    }
}
