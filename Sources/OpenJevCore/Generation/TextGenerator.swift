// A port of upstream OpenJev (razorback16/openjev at dcd2094): the contract of `MlxRuntime.generate`
// in `openjev/mlx_backend.py` and what `MlxGenerator` in `openjev/chat.py` asks of its engine:
// `prompt_ids`, `stop_ids` through `Engine.enc`, the thought-channel markers and
// `OPENJEV_MLX_MAX_PROMPT`. Apache-2.0. See THIRD_PARTY.md.

/// What a model generated for a prompt: upstream's `(ids, prompt_tokens, finish)`.
public struct TextGeneration: Sendable, Hashable {
    /// Why a generation ended.
    public enum FinishReason: String, Sendable, Hashable {
        /// The model produced its end of turn or a stop id. The stop token is not in
        /// ``TextGeneration/generated``.
        case stop
        /// The generation reached its `maxTokens`.
        case length
        /// `emit` returned false, or the calling task was cancelled, and the generation stopped
        /// at the next block boundary.
        case cancelled
    }

    /// The generated token ids, without the stop token.
    public var generated: [Int]
    /// The prompt tokens the generation processed, which a reply bills as `prompt_tokens`.
    public var promptTokens: Int
    /// Why the generation ended.
    public var finishReason: FinishReason

    /// Creates a generation.
    public init(generated: [Int], promptTokens: Int, finishReason: FinishReason) {
        self.generated = generated
        self.promptTokens = promptTokens
        self.finishReason = finishReason
    }
}

/// A model that generates text, which `POST /v1/chat/completions` serves (issue #53).
///
/// This is upstream's `MlxRuntime.generate` (`mlx_backend.py`, lines 210 to 257) together with what
/// `MlxGenerator` reads from its engine. The server registers the chat routes only for a service
/// whose model conforms (``SystemOneService/textGenerator``); upstream's encoder containers have no
/// chat routes either. The core never renders a prompt itself: the chat template and the tokenizer
/// are the model's.
///
/// The prompt, encoding and limit requirements are used on the request's task, before it waits for
/// its turn to generate, so a conformance that is an actor implements them `nonisolated`: a request
/// whose prompt is too long is refused without waiting for the generation in flight. The chat
/// route counts a request against its capacity bound before rendering its prompt, so no more
/// prompts render at once than the bound allows.
public protocol TextGenerator: Sendable {
    /// The most prompt tokens a generation may carry, upstream's `OPENJEV_MLX_MAX_PROMPT`. The
    /// chat route compares it with ``generationPromptIDs(messages:thinking:)``, scaffold included,
    /// and answers a longer prompt with its 400 before a response starts.
    var maxPromptTokens: Int { get }

    /// The ids of the thought-channel markers, upstream's `engine.thought_open +
    /// engine.thought_close`: `enc("<|channel>thought\n") + enc("<channel|>")`, which for the
    /// DiffusionGemma tokenizer are `[100, 45518, 107, 101]`.
    ///
    /// The chat route passes them to ``generate(prompt:maxTokens:stopIDs:skipSpecialTokenIDs:emit:)``
    /// as `skipSpecialTokenIDs`: the model opens a thought channel of its own accord on some
    /// replies, and a chat client asked for the reply, not the markers.
    var thoughtChannelMarkerIDs: [Int] { get }

    /// The prompt of a chat request, upstream's `MlxGenerator.prompt_ids`: the tokenizer's chat
    /// template over `messages` with `add_generation_prompt` and `enable_thinking` set to
    /// `thinking`, tokenized without special tokens, then the ids of the empty thought scaffold
    /// `<|channel>thought\n<channel|>` (``EngineTokens/scaffoldText``).
    ///
    /// `messages` are the request's messages after normalization: JSON objects with a string
    /// `role`, whose `content` is absent, null, a string or an array of parts, and whatever other
    /// keys the client sent, which the template reads as it reads them upstream.
    ///
    /// It is asynchronous so that a conformance can render where the template's recursion has room:
    /// the chat template recurses into tool call arguments, and a request decides how deep they go.
    ///
    /// - Throws: Any error the template or the tokenizer raises; the chat route answers it with a
    ///   400 (upstream's route crashes with a 500 on those requests).
    func generationPromptIDs(messages: [JSONValue], thinking: Bool) async throws -> [Int]

    /// The ids of `text` without special tokens, upstream's `Engine.enc`, which the chat route
    /// encodes each `stop` string with: a stop that is one token becomes a stop id.
    func encode(_ text: String) throws -> [Int]

    /// Generates greedily from `prompt`, upstream's `MlxRuntime.generate`.
    ///
    /// The generation stops at the model's end of turn, at a token of `stopIDs` (which is not in
    /// the result) or after `maxTokens` tokens. The ids of `skipSpecialTokenIDs` are dropped before
    /// they reach the detokenizer, so their text never reaches `emit`.
    ///
    /// `emit` is called once per committed token with the detokenizer's text for it, which may be
    /// empty, and once more at the end with the detokenizer's final buffered segment and a `nil`
    /// token; the chat route takes an empty text as no chunk, so that last call may be skipped when
    /// the segment is empty, as upstream's runtime skips it, and its return value is ignored.
    /// Returning false asks the generation to stop: it ends at the next block boundary with
    /// ``TextGeneration/FinishReason/cancelled``, as it does when the calling task is cancelled.
    /// `emit` may be called on the generator's own executor and must not block.
    ///
    /// - Throws: Whatever the model throws. A generation the calling task cancelled may throw
    ///   `CancellationError` instead of returning a cancelled result.
    func generate(
        prompt: [Int], maxTokens: Int, stopIDs: [Int], skipSpecialTokenIDs: [Int],
        emit: @Sendable (_ text: String, _ token: Int?) -> Bool
    ) async throws -> TextGeneration
}
