/// The tokenizer operations the decision engine needs, kept behind a protocol so `OpenJevCore`
/// never links a tokenizer library.
///
/// Upstream's engine is tokenizer-shaped: choice labels must stay one token after `"q1: "`, the
/// answer template is re-tokenized to find the slots, and a read's prompt is the chat template
/// rendered to ids. The production conformance wraps swift-transformers (decision D-008); the
/// core's tests replay recorded tokenizations instead, so they also run on Linux.
///
/// Every method throws so a conformance can report an input it cannot handle, such as a text a
/// replay tokenizer never recorded, instead of returning wrong ids.
public protocol DecisionTokenizer: Sendable {
    /// The token ids of `text`.
    ///
    /// `encode(text, addSpecialTokens: false)` is upstream's `Engine.enc`,
    /// `tokenizer.encode(text, add_special_tokens=False)`, which is how the engine tokenizes
    /// label candidates, answer templates and markers. With `addSpecialTokens: true` the
    /// tokenizer's post-processor may add tokens such as `<bos>`; the DiffusionGemma tokenizer
    /// adds none.
    func encode(_ text: String, addSpecialTokens: Bool) throws -> [Int]

    /// The text of `ids`, as `tokenizer.decode(ids, skip_special_tokens=...)` gives it.
    func decode(_ ids: [Int], skipSpecialTokens: Bool) throws -> String

    /// The ids of the prompt a read starts from, upstream's `Engine.chat_prompt_ids`.
    ///
    /// That is `apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
    /// enable_thinking=thinking)` with `messages` the system message `system` followed by the
    /// user message `user`, taking `input_ids` when the result is a dictionary. The ids end after
    /// the model turn marker; the engine puts the empty thought block at the head of the canvas.
    func chatPromptIDs(system: String, user: String, thinking: Bool) throws -> [Int]
}

/// Why a ``DecisionTokenizer`` could not answer.
public struct TokenizerError: Error, Sendable, Hashable, CustomStringConvertible {
    /// A readable account of the failure. A conformance names the input it could not handle, so
    /// a missing recording or an unsupported text is diagnosable from the message alone.
    public var message: String

    /// Creates an error with a message.
    public init(_ message: String) {
        self.message = message
    }

    /// The message.
    public var description: String { message }
}
