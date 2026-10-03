import Foundation
import Hub
import Tokenizers

/// JevK5's tokenizer (Qwen3.5's byte-level BPE, `tokenizer.json` and `tokenizer_config.json`)
/// through swift-transformers, as mlx-swift-lm loads a Qwen tokenizer.
///
/// `encode` adds no special tokens, as upstream asks vLLM (`add_special_tokens: false`); the
/// template's `<|im_start|>` and `<|im_end|>` are added tokens matched in the text. The value is
/// immutable after loading and safe to use from any task.
public struct JevK5Tokenizer: LetterReadoutTokenizing {
    /// The vocabulary's longest entry in Unicode scalars: 128 for JevK5.
    public let maxCharactersPerToken: Int

    private let tokenizer: any Tokenizers.Tokenizer

    /// Loads the tokenizer from a checkpoint folder holding `tokenizer.json` and
    /// `tokenizer_config.json`.
    ///
    /// - Throws: ``JevK5LoadError/missingFiles(_:in:)`` for an absent file, and swift-transformers'
    ///   and the file system's errors.
    public static func load(directory: URL) async throws -> JevK5Tokenizer {
        let names = ["tokenizer.json", "tokenizer_config.json"]
        let missing = names.filter {
            !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        guard missing.isEmpty else {
            throw JevK5LoadError.missingFiles(missing, in: directory)
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let longest = try longestEntry(in: directory.appendingPathComponent("tokenizer.json"))
        return JevK5Tokenizer(maxCharactersPerToken: longest, tokenizer: tokenizer)
    }

    /// The ids of `text`, without special tokens added.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// vLLM's `max_chars_per_token`: the most Unicode scalars of any entry of
    /// `tokenizer.get_vocab()`, the model's vocabulary and the added tokens.
    static func longestEntry(in tokenizerJSON: URL) throws -> Int {
        let data = try Data(contentsOf: tokenizerJSON)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let vocabulary = model["vocab"] as? [String: Any]
        else {
            throw JevK5LoadError.unsupportedModel(
                "\(tokenizerJSON.path) holds no model vocabulary")
        }
        let added = (root["added_tokens"] as? [[String: Any]] ?? []).compactMap {
            $0["content"] as? String
        }
        return (vocabulary.keys + added).map(\.unicodeScalars.count).max() ?? 0
    }
}
