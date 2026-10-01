import Foundation
import OpenJevCore

/// A ``QuestionReadBackend`` that answers as upstream's `tests/test_encoders.py` `FakeEngine`
/// does: the second option gets 0.7 and the rest share 0.3, and every batch bills
/// ``inputTokensPerBatch`` (99).
///
/// - ``scripted`` replaces the stub distribution for the question keys it names, so a test can
///   feed the probability vectors of `Fixtures/wire/answers.json` through the engine.
/// - ``failure`` makes every batch after the first ``succeedingBatches`` break the contract in
///   one way, or throw.
/// - ``gate`` holds every batch until the test opens it, and ``delay`` makes every batch take at
///   least that long, for the queue, timing and shutdown tests; both let cancellation through.
///
/// Every batch is recorded in call order, with the time it took inside the stub
/// (``callTimes``), and ``close()`` is counted.
public final class StubQuestionReadBackend: QuestionReadBackend, ModelReleasing,
    @unchecked Sendable
{
    /// One recorded `readBatch` call.
    public struct Call: Sendable {
        /// The state as sent.
        public var state: JSONValue
        /// The state as rendered for the model.
        public var stateText: String
        /// The questions of the batch, in order.
        public var questions: [EncoderQuestion]

        /// The keys of the questions in the batch, in order.
        public var keys: [String] { questions.map(\.key) }
    }

    /// How a batch breaks the backend's contract.
    public enum Failure: Sendable {
        /// One distribution too few.
        case wrongDistributionCount
        /// One probability too many for the first question.
        case wrongProbabilityCount
        /// A NaN in the first question's distribution.
        case notFinite
        /// 0.5 moved from the first question's first value to its second (a noul becomes
        /// `[-0.2, 1.2]`), so the sum is unchanged but two values leave [0, 1].
        case outOfRange
        /// Every probability doubled, so the sum is 2.
        case sumFarFromOne
        /// The batch throws the given error instead of answering.
        case throwing(String)
    }

    /// The error ``Failure/throwing(_:)`` raises.
    public struct StubError: Error, Equatable, Sendable {
        /// The message the failure was configured with.
        public var message: String

        /// Creates the error.
        public init(message: String) {
            self.message = message
        }
    }

    public let modelInfo: ModelInfo
    public let maxChoices: Int
    public let maxPromptTokens: Int?
    /// The input tokens every batch bills.
    public let inputTokensPerBatch: Int
    /// How long every batch takes at least.
    public let delay: Duration?
    /// Distributions by question key, used instead of the stub's for the keys named.
    public let scripted: [String: [Double]]
    /// The contract violation the batches after the first ``succeedingBatches`` commit, if any.
    public let failure: Failure?
    /// How many batches answer before ``failure`` applies to the rest; 0 fails every batch.
    public let succeedingBatches: Int
    /// Holds every batch until it is opened; `nil` lets batches through.
    public let gate: ReadGate?

    private let lock = NSLock()
    private var recorded: [Call] = []
    private var recordedTimes: [Duration] = []
    private var closed = 0

    /// Creates a stub; by default it serves Laya and answers every batch as upstream's fake does.
    public init(
        modelInfo: ModelInfo = KnownEncoderModels.laya,
        maxChoices: Int = 255,
        maxPromptTokens: Int? = nil,
        inputTokensPerBatch: Int = 99,
        delay: Duration? = nil,
        scripted: [String: [Double]] = [:],
        failure: Failure? = nil,
        succeedingBatches: Int = 0,
        gate: ReadGate? = nil
    ) {
        self.modelInfo = modelInfo
        self.maxChoices = maxChoices
        self.maxPromptTokens = maxPromptTokens
        self.inputTokensPerBatch = inputTokensPerBatch
        self.delay = delay
        self.scripted = scripted
        self.failure = failure
        self.succeedingBatches = succeedingBatches
        self.gate = gate
    }

    /// Every batch so far, in call order.
    public var calls: [Call] {
        lock.withLock { recorded }
    }

    /// The time each finished batch spent inside the stub, gate and delay included, in the order
    /// the batches finished; a batch that was cancelled or failed counts too.
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

    /// Upstream's stub distribution over `n` options: the second option 70%, the rest share 30%.
    public static func distribution(options n: Int) -> [Double] {
        var probabilities = [Double](repeating: 0.3 / Double(n - 1), count: n)
        probabilities[1] = 0.7
        return probabilities
    }

    public func readBatch(
        state: JSONValue, stateText: String, questions: [EncoderQuestion]
    ) async throws -> BatchReadResult {
        let clock = ContinuousClock()
        let started = clock.now
        let index = lock.withLock {
            recorded.append(Call(state: state, stateText: stateText, questions: questions))
            return recorded.count
        }
        defer {
            let time = clock.now - started
            lock.withLock { recordedTimes.append(time) }
        }
        try await gate?.pass()
        if let delay {
            try await Task.sleep(for: delay)
        }
        var probabilities = questions.map { question in
            scripted[question.key] ?? Self.distribution(options: question.choices.count)
        }
        switch index > succeedingBatches ? failure : nil {
        case nil:
            break
        case .wrongDistributionCount:
            probabilities.removeLast()
        case .wrongProbabilityCount:
            probabilities[0].append(0.0)
        case .notFinite:
            probabilities[0][0] = .nan
        case .outOfRange:
            probabilities[0][0] -= 0.5
            probabilities[0][1] += 0.5
        case .sumFarFromOne:
            probabilities = probabilities.map { $0.map { $0 * 2 } }
        case .throwing(let message):
            throw StubError(message: message)
        }
        return BatchReadResult(probabilities: probabilities, inputTokens: inputTokensPerBatch)
    }
}
