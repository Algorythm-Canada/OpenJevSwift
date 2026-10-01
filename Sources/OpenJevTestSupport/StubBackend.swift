import Foundation
import OpenJevCore

/// A ``DecisionBackend`` over ``FixtureTokenizer`` that answers as upstream's `tests/test_api.py`
/// stubs `Engine.one_read` and `Engine.think`, which is how `Fixtures/policies/` was recorded.
///
/// - A read gives every slot 0.7 on its first label and splits 0.3 over the rest; a slot with
///   two labels flips to `[0.3, 0.7]` when the read's seed is odd, so averaging over samples
///   shows. Every slot reports ``entropy`` (0.05 by default; 0.5 puts the automatic re-reads
///   above the threshold, as `auto_rereads.json` was recorded) and every read bills
///   ``readPromptTokens`` (123).
/// - A thought is ``thought``, the ids 7, 8 and 9 by default, and bills 100 prompt tokens.
/// - ``scripted`` replaces the stub read for the seeds it names with the recorded raw
///   log-probability maps of `Fixtures/distributions/`, through ``ReadResult``'s raw initializer.
///
/// Every ``CanvasRead`` and every think call is recorded in call order, with the time each call
/// took inside the stub (``callTimes``). ``gate`` holds each call until the test opens it and
/// ``delay`` makes each one take at least that long, for the capacity, timing and shutdown tests;
/// both let cancellation through. ``failure`` makes each call after the first
/// ``succeedingCalls`` throw, for the error contract tests. ``close()`` is counted.
public final class StubBackend: DecisionBackend, ModelReleasing, @unchecked Sendable {
    /// One recorded think call.
    public struct ThinkCall: Equatable, Sendable {
        /// The prompt ids.
        public var prompt: [Int]
        /// The token budget.
        public var budget: Int
        /// The ids that end the thought.
        public var stopIDs: [Int]

        /// Creates a recorded call.
        public init(prompt: [Int], budget: Int, stopIDs: [Int]) {
            self.prompt = prompt
            self.budget = budget
            self.stopIDs = stopIDs
        }
    }

    public let tokenizer: any DecisionTokenizer
    public let maxPromptTokens: Int
    public let capabilities: BackendCapabilities
    public let modelName: String
    /// The entropy every slot reports.
    public let entropy: Double
    /// The prompt tokens every stub read bills.
    public let readPromptTokens: Int
    /// How long every backend call takes at least.
    public let delay: Duration?
    /// The ids every think call generates.
    public let thought: [Int]
    /// Recorded raw maps by read seed; a read whose seed is named answers from these with
    /// `scriptedPromptTokens` billed, instead of the stub distribution.
    public let scripted: [UInt64: [[(tokenID: Int, logprob: Double)]]]
    /// The prompt tokens a scripted read bills.
    public let scriptedPromptTokens: Int
    /// The error the reads and think calls after the first ``succeedingCalls`` throw, after
    /// recording the call, instead of answering; `nil` answers every call.
    public let failure: (any Error)?
    /// How many calls answer before ``failure`` applies to the rest; 0 fails every call.
    public let succeedingCalls: Int
    /// Holds every call until it is opened; `nil` lets calls through.
    public let gate: ReadGate?

    private let lock = NSLock()
    private var recordedReads: [CanvasRead] = []
    private var recordedThinks: [ThinkCall] = []
    private var recordedTimes: [Duration] = []
    private var calls = 0
    private var closed = 0

    /// Creates a stub; every argument defaults to what the recordings used.
    public init(
        tokenizer: any DecisionTokenizer = FixtureTokenizer.shared,
        maxPromptTokens: Int = 32768,
        capabilities: BackendCapabilities = .all,
        modelName: String = "stub",
        entropy: Double = 0.05,
        readPromptTokens: Int = 123,
        delay: Duration? = nil,
        thought: [Int] = [7, 8, 9],
        scripted: [UInt64: [[(tokenID: Int, logprob: Double)]]] = [:],
        scriptedPromptTokens: Int = 100,
        failure: (any Error)? = nil,
        succeedingCalls: Int = 0,
        gate: ReadGate? = nil
    ) {
        self.tokenizer = tokenizer
        self.maxPromptTokens = maxPromptTokens
        self.capabilities = capabilities
        self.modelName = modelName
        self.entropy = entropy
        self.readPromptTokens = readPromptTokens
        self.delay = delay
        self.thought = thought
        self.scripted = scripted
        self.scriptedPromptTokens = scriptedPromptTokens
        self.failure = failure
        self.succeedingCalls = succeedingCalls
        self.gate = gate
    }

    /// Every read so far, in call order.
    public var reads: [CanvasRead] {
        lock.withLock { recordedReads }
    }

    /// Every think call so far, in call order.
    public var thinks: [ThinkCall] {
        lock.withLock { recordedThinks }
    }

    /// The time each finished call spent inside the stub, gate and delay included, in the order
    /// the calls finished; a call that was cancelled or failed counts too.
    public var callTimes: [Duration] {
        lock.withLock { recordedTimes }
    }

    /// How many times ``close()`` was called.
    public var closeCount: Int {
        lock.withLock { closed }
    }

    public func close() async {
        lock.withLock { closed += 1 }
    }

    /// Waits for the gate and the delay, counting the call, timing it and throwing ``failure``
    /// once ``succeedingCalls`` calls have answered.
    private func simulateCall() async throws {
        let clock = ContinuousClock()
        let started = clock.now
        let index = lock.withLock {
            calls += 1
            return calls
        }
        defer {
            let time = clock.now - started
            lock.withLock { recordedTimes.append(time) }
        }
        try await gate?.pass()
        if let delay {
            try await Task.sleep(for: delay)
        }
        if let failure, index > succeedingCalls {
            throw failure
        }
    }

    public func read(_ read: CanvasRead) async throws -> ReadResult {
        lock.withLock { recordedReads.append(read) }
        try await simulateCall()
        if let tops = scripted[read.seed] {
            return ReadResult(
                tops: tops, labelIDs: read.slots.map(\.labelIDs),
                promptTokens: scriptedPromptTokens)
        }
        let slots = read.slots.map { slot in
            let n = slot.labelIDs.count
            var probabilities = [0.7] + [Double](repeating: 0.3 / Double(n - 1), count: n - 1)
            if n == 2 && read.seed % 2 == 1 {
                probabilities = [0.3, 0.7]
            }
            return SlotRead(probabilities: probabilities, entropy: entropy)
        }
        return ReadResult(slots: slots, promptTokens: readPromptTokens)
    }

    public func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws -> ThoughtGeneration
    {
        lock.withLock {
            recordedThinks.append(ThinkCall(prompt: prompt, budget: budget, stopIDs: stopIDs))
        }
        try await simulateCall()
        return ThoughtGeneration(generated: thought, promptTokens: 100)
    }
}
