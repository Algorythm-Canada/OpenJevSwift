import Foundation
import Tokenizers

/// ModernBERT's byte-level BPE tokenizer through swift-transformers, loaded from a folder that
/// holds tokenizer.json and tokenizer_config.json.
public struct EncoderTokenizer: Sendable {
    let tokenizer: any Tokenizer

    public static func load(folder: URL) async throws -> EncoderTokenizer {
        EncoderTokenizer(tokenizer: try await AutoTokenizer.from(modelFolder: folder))
    }

    /// Token ids without special tokens, as Hugging Face's `tok(text, add_special_tokens=False)`.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func id(of token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }
}

/// Verdict's model input for one prompt.
///
/// Upstream calls the tokenizer with `truncation=True, max_length=512`. Its post-processor wraps a
/// sequence in [CLS] and [SEP], so a long prompt keeps its first 510 tokens between them.
public enum VerdictInput {
    public static func ids(
        prompt: String, tokenizer: EncoderTokenizer, cls: Int, sep: Int, maxLength: Int
    )
        -> [Int]
    {
        [cls] + tokenizer.encode(prompt).prefix(maxLength - 2) + [sep]
    }
}

/// Laya's `build_sequence` (laya 0.3.6, laya/common.py) over the texts it tokenizes.
///
/// The texts are the head (`"<type> question: <instructions>"`), each option with its leading
/// space, and the state, each with any "[MASK]" already replaced by a space. Rendering them from a
/// question (render_options, render_criterion, serialize_state) is Python-side JSON formatting that
/// issue #58 ports with the core's JSON model; the fixture records the rendered strings.
public enum LayaSequence {
    /// The sequence `[CLS] head [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]` and the
    /// positions of its [MASK] markers.
    public static func build(
        head: String, options: [String], state: String, tokenizer: EncoderTokenizer,
        special: LayaReference.SpecialTokens, maxLen: Int, headMaxLen: Int
    ) -> (ids: [Int], markers: [Int]) {
        var headIds = tokenizer.encode(head)
        var optionIds = options.map { [special.mask] + tokenizer.encode($0).prefix(48) }
        var budget = headMaxLen - optionIds.reduce(0) { $0 + $1.count }
        if budget < 16 {
            // Python's floor division; both operands are positive here.
            let per = max(4, (headMaxLen - 16) / max(1, optionIds.count))
            optionIds = optionIds.map { Array($0.prefix(per)) }
            budget = headMaxLen - optionIds.reduce(0) { $0 + $1.count }
        }
        headIds = Array(headIds.prefix(max(8, budget)))
        var ids = [special.cls] + headIds + [special.sep]
        var markers: [Int] = []
        for option in optionIds {
            markers.append(ids.count)
            ids += option
        }
        ids.append(special.sep)
        let room = max(0, maxLen - ids.count - 1)
        ids += tokenizer.encode(state).prefix(room)
        ids.append(special.sep)
        return (Array(ids.prefix(maxLen)), markers.filter { $0 < maxLen })
    }
}
