import Foundation
import OpenJevCore

/// A ``QuestionReadBackend`` that answers as upstream's `tests/test_encoders.py` `FakeEngine`
/// does: the second option gets 0.7 and the rest share 0.3, and every batch bills
/// ``inputTokensPerBatch`` (99).
///
/// - ``scripted`` replaces the stub distribution for the question keys it names, so a test can
///   feed the probability vectors of `Fixtures/wire/answers.json` through the engine.
/// - ``failure`` makes every batch break the contract in one way, or throw.
/// - ``delay`` makes every batch take at least that long, for the queue and timing tests.
///
/// Every batch is recorded in call order.
final class StubQuestionReadBackend: QuestionReadBackend, @unchecked Sendable {
    /// One recorded `readBatch` call.
    struct Call {
        var state: JSONValue
        var stateText: String
        var questions: [EncoderQuestion]

        /// The keys of the questions in the batch, in order.
        var keys: [String] { questions.map(\.key) }
    }

    /// How a batch breaks the backend's contract.
    enum Failure {
        /// One distribution too few.
        case wrongDistributionCount
        /// One probability too many for the first question.
        case wrongProbabilityCount
        /// A NaN in the first question's distribution.
        case notFinite
        /// Every probability doubled, so the sum is 2.
        case sumFarFromOne
        /// The batch throws the given error instead of answering.
        case throwing(String)
    }

    /// The error ``Failure/throwing(_:)`` raises.
    struct StubError: Error, Equatable {
        var message: String
    }

    let modelInfo: ModelInfo
    let maxChoices: Int
    let maxPromptTokens: Int?
    /// The input tokens every batch bills.
    let inputTokensPerBatch: Int
    /// How long every batch takes at least.
    let delay: Duration?
    /// Distributions by question key, used instead of the stub's for the keys named.
    let scripted: [String: [Double]]
    /// The contract violation every batch commits, if any.
    let failure: Failure?

    private let lock = NSLock()
    private var recorded: [Call] = []

    init(
        modelInfo: ModelInfo = KnownEncoderModels.laya,
        maxChoices: Int = 255,
        maxPromptTokens: Int? = nil,
        inputTokensPerBatch: Int = 99,
        delay: Duration? = nil,
        scripted: [String: [Double]] = [:],
        failure: Failure? = nil
    ) {
        self.modelInfo = modelInfo
        self.maxChoices = maxChoices
        self.maxPromptTokens = maxPromptTokens
        self.inputTokensPerBatch = inputTokensPerBatch
        self.delay = delay
        self.scripted = scripted
        self.failure = failure
    }

    /// Every batch so far, in call order.
    var calls: [Call] {
        lock.withLock { recorded }
    }

    /// Upstream's stub distribution over `n` options: the second option 70%, the rest share 30%.
    static func distribution(options n: Int) -> [Double] {
        var probabilities = [Double](repeating: 0.3 / Double(n - 1), count: n)
        probabilities[1] = 0.7
        return probabilities
    }

    func readBatch(
        state: JSONValue, stateText: String, questions: [EncoderQuestion]
    ) async throws -> BatchReadResult {
        lock.withLock {
            recorded.append(Call(state: state, stateText: stateText, questions: questions))
        }
        if let delay {
            try await Task.sleep(for: delay)
        }
        var probabilities = questions.map { question in
            scripted[question.key] ?? Self.distribution(options: question.choices.count)
        }
        switch failure {
        case nil:
            break
        case .wrongDistributionCount:
            probabilities.removeLast()
        case .wrongProbabilityCount:
            probabilities[0].append(0.0)
        case .notFinite:
            probabilities[0][0] = .nan
        case .sumFarFromOne:
            probabilities = probabilities.map { $0.map { $0 * 2 } }
        case .throwing(let message):
            throw StubError(message: message)
        }
        return BatchReadResult(probabilities: probabilities, inputTokens: inputTokensPerBatch)
    }
}
