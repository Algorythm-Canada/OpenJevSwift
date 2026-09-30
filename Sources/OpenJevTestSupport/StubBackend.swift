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
/// Every ``CanvasRead`` and every think call is recorded in call order. ``delay`` makes each
/// backend call take at least that long, for the concurrency and timing tests.
public final class StubBackend: DecisionBackend, @unchecked Sendable {
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

    private let lock = NSLock()
    private var recordedReads: [CanvasRead] = []
    private var recordedThinks: [ThinkCall] = []

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
        scriptedPromptTokens: Int = 100
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
    }

    /// Every read so far, in call order.
    public var reads: [CanvasRead] {
        lock.withLock { recordedReads }
    }

    /// Every think call so far, in call order.
    public var thinks: [ThinkCall] {
        lock.withLock { recordedThinks }
    }

    public func read(_ read: CanvasRead) async throws -> ReadResult {
        lock.withLock { recordedReads.append(read) }
        if let delay {
            try await Task.sleep(for: delay)
        }
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
        if let delay {
            try await Task.sleep(for: delay)
        }
        return ThoughtGeneration(generated: thought, promptTokens: 100)
    }
}
