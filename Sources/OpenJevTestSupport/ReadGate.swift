import Foundation

/// Holds a stub backend's calls until a test opens it, for the capacity, cancellation and shutdown
/// tests: each call records its arrival and then waits, cancellably, until ``open()``.
///
/// ``waitForArrivals(_:)`` lets a test act once a request has reached the backend, without
/// sleeping for a guessed time. A call whose task is cancelled while it waits throws
/// `CancellationError` and is counted in ``cancellations``.
public final class ReadGate: @unchecked Sendable {
    private typealias Waiter = CheckedContinuation<Void, any Error>

    // Guarded by `lock`.
    private let lock = NSLock()
    private var isOpen = false
    private var arrived = 0
    private var cancelled = 0
    private var nextID = 0
    private var waiters: [Int: Waiter] = [:]
    private var arrivalWaiters: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []

    /// Creates a closed gate.
    public init() {}

    /// The calls that have reached the gate so far.
    public var arrivals: Int {
        lock.withLock { arrived }
    }

    /// The calls that threw because their task was cancelled, while they waited or before.
    public var cancellations: Int {
        lock.withLock { cancelled }
    }

    /// Lets every waiting call through, and every later one at once unless its task is already
    /// cancelled.
    public func open() {
        let released = lock.withLock { () -> [Waiter] in
            isOpen = true
            defer { waiters.removeAll() }
            return Array(waiters.values)
        }
        for waiter in released {
            waiter.resume()
        }
    }

    /// Returns once `count` calls have arrived.
    public func waitForArrivals(_ count: Int) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let ready = lock.withLock { () -> Bool in
                if arrived >= count {
                    return true
                }
                arrivalWaiters.append((count, continuation))
                return false
            }
            if ready {
                continuation.resume()
            }
        }
    }

    /// What a stub calls: counts the arrival, then returns once the gate is open.
    ///
    /// - Throws: `CancellationError` when the task is cancelled before the gate opens, or was
    ///   cancelled already when it called, open or not.
    public func pass() async throws {
        let (id, reached) = lock.withLock { () -> (Int, [CheckedContinuation<Void, Never>]) in
            arrived += 1
            nextID += 1
            let reached = arrivalWaiters.filter { $0.count <= arrived }.map(\.continuation)
            arrivalWaiters.removeAll { $0.count <= arrived }
            return (nextID, reached)
        }
        for waiter in reached {
            waiter.resume()
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: Waiter) in
                let outcome = lock.withLock { () -> Result<Void, any Error>? in
                    if Task.isCancelled {
                        cancelled += 1
                        return .failure(CancellationError())
                    }
                    if isOpen {
                        return .success(())
                    }
                    waiters[id] = continuation
                    return nil
                }
                if let outcome {
                    continuation.resume(with: outcome)
                }
            }
        } onCancel: {
            let waiter = lock.withLock { () -> Waiter? in
                guard let waiter = waiters.removeValue(forKey: id) else { return nil }
                cancelled += 1
                return waiter
            }
            waiter?.resume(throwing: CancellationError())
        }
    }
}
