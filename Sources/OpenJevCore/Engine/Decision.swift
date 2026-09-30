// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, the return
// value of `Engine.decide`, the class `Overloaded` and the model time of `model_ns`. Apache-2.0.
// See THIRD_PARTY.md.

/// What ``DecisionEngine/decide(_:)`` returns: the answers and what the request cost.
public struct Decision: Sendable, Hashable {
    /// One answer per question, in the request's order, forced answers included.
    public var answers: OrderedMap<Answer>
    /// The billed input tokens: every billed read's prompt tokens plus each thought's.
    public var inputTokens: Int
    /// The thought tokens generated, upstream's `usage.output_tokens`; 0 without `think`.
    public var outputTokens: Int
    /// The time spent inside backend calls, including the wait for a free slot, summed over the
    /// calls. Reads of one request run concurrently, so this can exceed the request's wall time:
    /// it is model time spent, not elapsed, upstream's Server-Timing `model` value.
    public var modelTime: Duration

    /// Creates a decision.
    public init(
        answers: OrderedMap<Answer>, inputTokens: Int, outputTokens: Int, modelTime: Duration
    ) {
        self.answers = answers
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.modelTime = modelTime
    }
}

/// Too many requests are already inside the engine; the server answers 529 with `retry-after`.
public struct OverloadedError: Error, Sendable, Hashable, CustomStringConvertible {
    /// Upstream's message.
    public var message: String

    /// Creates the error with upstream's message.
    public init(message: String = "OpenJev is at capacity. Retry shortly.") {
        self.message = message
    }

    /// The message.
    public var description: String { message }
}
