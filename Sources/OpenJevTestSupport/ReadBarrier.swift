import Foundation

/// Holds a stub backend's calls until ``parties`` of them are waiting, then lets that round
/// through together, for the timing tests: the calls of a round are inside the backend at once
/// however late the scheduler starts each of them, which a test that sums their times needs.
///
/// The barrier is cyclic: the call after a full round starts the next one, so each request of a
/// test passes the same way when it makes ``parties`` calls. A call whose task is cancelled while
/// it waits throws `CancellationError` and leaves its round, as a ``ReadGate`` call does; nothing
/// else releases a round that never fills, so a test whose calls could run one at a time needs a
/// time limit.
public final class ReadBarrier: @unchecked Sendable {
    private typealias Waiter = CheckedContinuation<Void, any Error>

    /// The calls that make a round.
    public let parties: Int

    // Guarded by `lock`.
    private let lock = NSLock()
    private var nextID = 0
    private var waiters: [Int: Waiter] = [:]

    /// Creates a barrier whose rounds are `parties` calls.
    ///
    /// - Precondition: `parties` is at least 1.
    public init(parties: Int) {
        precondition(parties >= 1, "a round needs at least one call, got \(parties)")
        self.parties = parties
    }

    /// What a stub calls: returns once ``parties`` calls are waiting, this one included.
    ///
    /// - Throws: `CancellationError` when the task is cancelled before its round fills, or was
    ///   cancelled already when it called.
    public func arrive() async throws {
        let id = lock.withLock {
            nextID += 1
            return nextID
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: Waiter) in
                let round = lock.withLock { () -> [Waiter]? in
                    if Task.isCancelled {
                        return nil
                    }
                    waiters[id] = continuation
                    guard waiters.count == parties else { return [] }
                    defer { waiters.removeAll() }
                    return Array(waiters.values)
                }
                guard let round else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                for waiter in round {
                    waiter.resume()
                }
            }
        } onCancel: {
            let waiter = lock.withLock { waiters.removeValue(forKey: id) }
            waiter?.resume(throwing: CancellationError())
        }
    }
}
