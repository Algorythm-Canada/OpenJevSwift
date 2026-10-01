// A port of laya 0.3.6 (NandhaKishorM/laya), `laya/agent.py`: the tokenizer `Agent.__init__` loads
// from the checkpoint's `tokenizer/` folder, and the calls `build_sequence` makes of it
// (`tok(text, add_special_tokens=False)`, `cls_token_id`, `sep_token_id`, `mask_token_id`,
// `pad_token_id`). Apache-2.0. See THIRD_PARTY.md.

import Foundation
import Tokenizers

/// The ids of the special tokens Laya's sequences use.
public struct LayaSpecialTokens: Sendable, Hashable {
    /// `[CLS]`, which starts every sequence.
    public var classToken: Int
    /// `[SEP]`, which ends the head, the options and the state.
    public var separator: Int
    /// `[MASK]`, the marker in front of each option.
    public var mask: Int
    /// `[PAD]`, which pads the token ids of a shorter row.
    public var padding: Int

    /// Creates the ids.
    public init(classToken: Int, separator: Int, mask: Int, padding: Int) {
        self.classToken = classToken
        self.separator = separator
        self.mask = mask
        self.padding = padding
    }
}

/// Turns Laya's texts into token ids.
///
/// ``LayaBackend`` tokenizes through this protocol, so that a test can replay recorded ids
/// instead of loading the tokenizer. ``LayaTokenizer`` is the real one.
public protocol LayaTokenizing: Sendable {
    /// The ids of a text without special tokens, as Hugging Face's
    /// `tok(text, add_special_tokens=False)["input_ids"]`.
    func encode(_ text: String) -> [Int]
    /// The ids of `[CLS]`, `[SEP]`, `[MASK]` and `[PAD]`.
    var specialTokens: LayaSpecialTokens { get }
}

/// Laya's tokenizer: ModernBERT-large's byte-level BPE, loaded by swift-transformers from the
/// checkpoint's `tokenizer/` folder (tokenizer.json and tokenizer_config.json).
///
/// swift-transformers reproduces the ids laya's Hugging Face tokenizer gives for every head,
/// option and state of the reference corpus (spike #56), and the special tokens are looked up by
/// name, as laya's tokenizer configuration names them.
public struct LayaTokenizer: LayaTokenizing {
    /// The files the folder must hold.
    public static let fileNames = TokenizerFolder.fileNames

    private let tokenizer: any Tokenizer
    /// The ids of `[CLS]`, `[SEP]`, `[MASK]` and `[PAD]`.
    public let specialTokens: LayaSpecialTokens

    /// Loads the tokenizer from a folder holding tokenizer.json and tokenizer_config.json.
    ///
    /// - Throws: ``EncoderLoadError/missingFile(_:)`` for a missing file,
    ///   ``EncoderLoadError/missingToken(_:)`` when the vocabulary lacks `[CLS]`, `[SEP]`,
    ///   `[MASK]` or `[PAD]`, and swift-transformers' errors for a file it cannot read.
    public static func load(directory: URL) async throws -> LayaTokenizer {
        let tokenizer = try await TokenizerFolder.load(directory)
        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token) else {
                throw EncoderLoadError.missingToken(token)
            }
            return id
        }
        return LayaTokenizer(
            tokenizer: tokenizer,
            specialTokens: LayaSpecialTokens(
                classToken: try id("[CLS]"), separator: try id("[SEP]"), mask: try id("[MASK]"),
                padding: try id("[PAD]")))
    }

    /// The ids of a text without special tokens.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// The id of a token, or `nil` when the vocabulary lacks it.
    public func tokenID(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }
}
