// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, the
// contract of `EncoderEngine.read_batch` and the class attributes `model_name` and `max_choices`
// that the Laya, Verdict, CLM and JevK5 engines set. Apache-2.0. See THIRD_PARTY.md.

/// What one batched read returns: one distribution per question and the prompt tokens billed.
public struct BatchReadResult: Sendable, Hashable {
    /// One probability distribution per question of the batch, in the batch's order. Each is in
    /// the question's option order: a noul is `[P(true), 1 - P(true)]`, a choice follows the
    /// criteria order and a score follows the levels.
    public var probabilities: [[Double]]
    /// The input tokens the batch processed, upstream's `usage.input_tokens` share of the batch.
    public var inputTokens: Int

    /// Creates a result.
    public init(probabilities: [[Double]], inputTokens: Int) {
        self.probabilities = probabilities
        self.inputTokens = inputTokens
    }
}

/// A model that answers Jev's questions directly, without a canvas: upstream's `EncoderEngine`
/// contract, which Verdict, Laya, JevK5 and CLM implement.
///
/// This is the sibling of ``DecisionBackend`` that decision D-005 names. The engine around it,
/// ``EncoderDecisionEngine``, owns the schema, the refusals, the queue bound, the batching and
/// the answer shapes; a backend owns the prompt format, the model and the calibration, and
/// returns a distribution over the caller's options for every question it is handed.
public protocol QuestionReadBackend: Sendable {
    /// The served model: its name (`verdict-1.4`, `laya-1.0`, `clm-v0.1` or `jevk5-0.2`), its
    /// description and its release date, as ``KnownEncoderModels`` gives them. The name goes into
    /// every refusal (`"{model} does not support {field}"`) and into the response's `model`.
    var modelInfo: ModelInfo { get }
    /// The most options one choice may have: 24 for Verdict, 255 for the others.
    var maxChoices: Int { get }
    /// The most tokens one question's sequence may carry, when the backend refuses a longer one
    /// as CLM and JevK5 do. `nil` when it truncates instead, as Verdict and Laya do. The engine
    /// does not count tokens; the backend applies its own limit inside ``readBatch``.
    var maxPromptTokens: Int? { get }

    /// Reads one batch of questions against the state, upstream's `read_batch`.
    ///
    /// `state` is the request's state as sent and `stateText` is ``StateText/render(_:)`` of it.
    /// Verdict takes the text; Laya and CLM take the raw value and render it themselves; JevK5
    /// embeds the raw value in its JSON prompt.
    ///
    /// - Returns: One distribution per question in the batch's order, each in the question's
    ///   option order (``EncoderQuestion/choices``), and the input tokens processed.
    func readBatch(
        state: JSONValue, stateText: String, questions: [EncoderQuestion]
    ) async throws -> BatchReadResult
}

/// A backend returned something its contract forbids: a distribution with the wrong number of
/// entries, a value that is not finite or a sum far from 1.
///
/// This is a bug in the backend, not in the request, so it is neither a ``SchemaError`` nor an
/// ``OverloadedError``; the server should answer it as an internal error.
public struct BackendContractError: Error, Sendable, Hashable, CustomStringConvertible {
    /// What was wrong, naming the model and the question.
    public var message: String

    /// Creates the error.
    public init(_ message: String) {
        self.message = message
    }

    /// The message.
    public var description: String { message }
}
