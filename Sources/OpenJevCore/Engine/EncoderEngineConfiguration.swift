// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/config.py`, the settings
// `EncoderEngine` reads (`encoder_batch`, `max_queue`, `warmup`) and `openjev/encoders.py`, the
// class attribute `workers`. Apache-2.0. See THIRD_PARTY.md.

/// The settings of an ``EncoderDecisionEngine``, with upstream's names and defaults.
public struct EncoderEngineConfiguration: Sendable, Hashable {
    /// The most questions one ``QuestionReadBackend/readBatch(state:stateText:questions:)`` call
    /// receives, `OPENJEV_ENCODER_BATCH` (16). Every question is a full-length sequence, so a
    /// batch bounds the memory of one forward pass.
    public var batchSize: Int
    /// The most requests inside `decide` at once, `OPENJEV_MAX_QUEUE` (512); one more is refused
    /// with ``OverloadedError``.
    public var maxQueue: Int
    /// The most backend calls in flight at once, upstream's `workers` (1): the model lives on one
    /// thread. CLM and JevK5, which call a vLLM server, raise it.
    public var maxInflight: Int
    /// Whether ``EncoderDecisionEngine/warmUp()`` reads upstream's warm-up questions,
    /// `OPENJEV_WARMUP` (true).
    public var warmUp: Bool

    /// Creates a configuration; every argument defaults to upstream's value.
    ///
    /// - Precondition: `batchSize` and `maxInflight` are at least 1 and `maxQueue` is not
    ///   negative, as upstream's settings require.
    public init(batchSize: Int = 16, maxQueue: Int = 512, maxInflight: Int = 1, warmUp: Bool = true)
    {
        precondition(batchSize >= 1, "batchSize must be at least 1, got \(batchSize)")
        precondition(maxInflight >= 1, "maxInflight must be at least 1, got \(maxInflight)")
        precondition(maxQueue >= 0, "maxQueue must not be negative, got \(maxQueue)")
        self.batchSize = batchSize
        self.maxQueue = maxQueue
        self.maxInflight = maxInflight
        self.warmUp = warmUp
    }

    /// Upstream's defaults.
    public static let `default` = EncoderEngineConfiguration()
}
