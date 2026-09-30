// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, the
// constants `VOCAB`, `TURN_CLOSE`, `PAD` and `SCAFFOLD_TEXT` and the token sequences
// `Engine.__init__` encodes. Apache-2.0. See THIRD_PARTY.md.

/// The fixed token ids and marker sequences the engine builds canvases and prompts from.
public struct EngineTokens: Sendable, Hashable {
    /// The vocabulary size, upstream's `VOCAB`. Noise tokens are drawn from `0..<vocabularySize`.
    public static let vocabularySize = 262_144

    /// `<turn|>`, which closes a turn in this vocabulary and follows the answer template on the
    /// canvas. Upstream calls it `TURN_CLOSE`; its documentation names it after `<end_of_turn>`,
    /// which is not a token of this vocabulary.
    public static let turnClose = 106

    /// `<pad>`, which fills the canvas after the turn close, upstream's `PAD`.
    public static let pad = 0

    /// The empty thought block the chat template leaves to the model, upstream's `SCAFFOLD_TEXT`.
    public static let scaffoldText = "<|channel>thought\n<channel|>"

    /// The text that opens a thought.
    public static let thoughtOpenText = "<|channel>thought\n"

    /// The text that closes a thought.
    public static let thoughtCloseText = "<channel|>"

    /// `enc(scaffoldText)`, which heads every canvas.
    public var scaffold: [Int]
    /// `enc(thoughtOpenText)`.
    public var thoughtOpen: [Int]
    /// `enc(thoughtCloseText)`.
    public var thoughtClose: [Int]

    /// Encodes the markers with `tokenizer`, without special tokens, as `Engine.__init__` does.
    ///
    /// - Throws: Whatever the tokenizer throws.
    public init(tokenizer: any DecisionTokenizer) throws {
        scaffold = try tokenizer.encode(Self.scaffoldText, addSpecialTokens: false)
        thoughtOpen = try tokenizer.encode(Self.thoughtOpenText, addSpecialTokens: false)
        thoughtClose = try tokenizer.encode(Self.thoughtCloseText, addSpecialTokens: false)
    }
}
