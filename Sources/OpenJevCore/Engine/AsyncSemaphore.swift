import Foundation

/// A counting semaphore for tasks, standing in for upstream's `asyncio.Semaphore(max_inflight)`.
///
/// `wait()` suspends until a permit is free and `signal()` hands the permit to the longest
/// waiting task, or returns it. A waiting task that is cancelled leaves the queue and throws
/// `CancellationError`. The state is behind a lock so the semaphore can be shared by the engine
/// actor and the tasks it spawns for concurrent reads.
final class AsyncSemaphore: @unchecked Sendable {
    private struct Waiter {
        var id: UInt64
        var continuation: CheckedContinuation<Void, any Error>
    }

    private let lock = NSLock()
    private var permits: Int
    private var waiters: [Waiter] = []
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
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                let outcome: Result<Void, any Error>? = lock.withLock {
                    if Task.isCancelled {
                        return .failure(CancellationError())
                    }
                    if permits > 0 {
                        permits -= 1
                        return .success(())
                    }
                    waiters.append(Waiter(id: id, continuation: continuation))
                    return nil
                }
                if let outcome {
                    continuation.resume(with: outcome)
                }
            }
        } onCancel: {
            let waiter = lock.withLock { () -> Waiter? in
                guard let index = waiters.firstIndex(where: { $0.id == id }) else { return nil }
                return waiters.remove(at: index)
            }
            waiter?.continuation.resume(throwing: CancellationError())
        }
    }

    /// Returns a permit, waking the first waiter if there is one.
    func signal() {
        let waiter = lock.withLock { () -> Waiter? in
            if waiters.isEmpty {
                permits += 1
                return nil
            }
            return waiters.removeFirst()
        }
        waiter?.continuation.resume()
    }

    /// Runs `body` holding a permit.
    func withPermit<T>(_ body: () async throws -> T) async throws -> T {
        try await wait()
        defer { signal() }
        return try await body()
    }
}
