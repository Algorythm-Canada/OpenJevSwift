// A port of upstream OpenJev (razorback16/openjev at dcd2094), the `model_ns` context variable of
// `openjev/engine.py` and the timing in the `finally` of `Engine._post`. Apache-2.0. See
// THIRD_PARTY.md.

import Foundation

/// The model time of one request, upstream's `model_ns`: the time spent inside backend calls,
/// the wait for a free slot included, summed over the calls.
///
/// The server installs one recorder per request as ``current`` and writes its ``total`` into
/// `server-timing`. ``DecisionEngine`` and ``EncoderDecisionEngine`` add each backend call when it
/// ends, whether it returned, threw or was cancelled, as upstream's `_post` adds in its `finally`.
/// A request the engine refuses after a read or a thought therefore reports that time, and so does
/// one whose backend failed. Reads of one request run at once, so the total can exceed the
/// request's wall time: it is model time spent, not model time elapsed.
///
/// A child task inherits the task-local value and adds to the same recorder, which is why this is
/// a reference type, as upstream's accumulator is a one-element list for the same reason.
public final class ModelTimeRecorder: @unchecked Sendable {
    /// The recorder of the request the current task serves, or `nil` outside one.
    @TaskLocal public static var current: ModelTimeRecorder?

    // Guarded by `lock`: concurrent reads of one request add from their own tasks.
    private let lock = NSLock()
    private var sum = Duration.zero

    /// Creates a recorder at zero.
    public init() {}

    /// The model time recorded so far.
    public var total: Duration {
        lock.withLock { sum }
    }

    /// Adds model time.
    public func add(_ duration: Duration) {
        lock.withLock { sum += duration }
    }

    /// Adds model time to the current request's recorder, if there is one.
    public static func record(_ duration: Duration) {
        current?.add(duration)
    }
}
