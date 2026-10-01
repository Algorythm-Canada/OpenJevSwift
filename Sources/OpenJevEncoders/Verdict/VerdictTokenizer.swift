// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, the
// tokenizer call of `VerdictEngine.read_batch` (`truncation=True, max_length=512`) and its
// billing (`batch["attention_mask"].sum()`). Apache-2.0. See THIRD_PARTY.md.

import Foundation
import Tokenizers

/// Turns a Verdict prompt into the token ids the model reads.
///
/// ``VerdictBackend`` tokenizes through this protocol, so that a test can replay recorded ids
/// instead of loading the tokenizer. ``VerdictTokenizer`` is the real one.
public protocol VerdictTokenizing: Sendable {
    /// The model input of one prompt: `[CLS]`, the prompt's tokens cut to fit, and `[SEP]`.
    ///
    /// The row's length is its attention-mask sum, which is what upstream bills as input tokens.
    func inputIDs(for prompt: String) -> [Int]
}

/// Verdict's tokenizer: ModernBERT's byte-level BPE, loaded by swift-transformers from a folder
/// that holds the checkpoint's tokenizer.json and tokenizer_config.json.
///
/// Upstream calls the Hugging Face tokenizer with `truncation=True, max_length=512`. Its
/// post-processor wraps the text in `[CLS]` and `[SEP]`, so a long prompt keeps its first 510
/// tokens between them. swift-transformers reproduces upstream's ids for every prompt of the
/// reference corpus, the `<<LABEL>>` and `<<SEP>>` markers, non-ASCII text and the truncation
/// included (spike #56).
public struct VerdictTokenizer: VerdictTokenizing {
    /// Upstream's `VERDICT_MAX_LEN`: the longest row, `[CLS]` and `[SEP]` included.
    public static let defaultMaxLength = 512

    /// The files the folder must hold.
    public static let fileNames = TokenizerFolder.fileNames

    private let tokenizer: any Tokenizer
    /// The id of `[CLS]`, which starts every row.
    public let classTokenID: Int
    /// The id of `[SEP]`, which ends every row.
    public let separatorTokenID: Int
    /// The longest row, `[CLS]` and `[SEP]` included.
    public let maxLength: Int

    /// Loads the tokenizer from a folder holding tokenizer.json and tokenizer_config.json.
    ///
    /// - Throws: ``EncoderLoadError/missingFile(_:)`` for a missing file,
    ///   ``EncoderLoadError/missingToken(_:)`` when the vocabulary lacks `[CLS]` or `[SEP]`, and
    ///   swift-transformers' errors for a file it cannot read.
    public static func load(directory: URL, maxLength: Int = defaultMaxLength) async throws
        -> VerdictTokenizer
    {
        precondition(maxLength > 2, "a row holds [CLS], at least one token and [SEP]")
        let tokenizer = try await TokenizerFolder.load(directory)
        guard let classTokenID = tokenizer.convertTokenToId("[CLS]") else {
            throw EncoderLoadError.missingToken("[CLS]")
        }
        guard let separatorTokenID = tokenizer.convertTokenToId("[SEP]") else {
            throw EncoderLoadError.missingToken("[SEP]")
        }
        return VerdictTokenizer(
            tokenizer: tokenizer, classTokenID: classTokenID, separatorTokenID: separatorTokenID,
            maxLength: maxLength)
    }

    /// The token ids of a text without special tokens, as Hugging Face's
    /// `tok(text, add_special_tokens=False)`.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// The id of a token, such as `<<LABEL>>` or `[PAD]`.
    public func tokenID(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    /// `[CLS]`, the prompt's first `maxLength - 2` tokens and `[SEP]`.
    public func inputIDs(for prompt: String) -> [Int] {
        [classTokenID] + encode(prompt).prefix(maxLength - 2) + [separatorTokenID]
    }
}
