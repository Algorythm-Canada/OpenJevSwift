// The context limit of upstream OpenJev's JevK5 server (razorback16/openjev at dcd2094,
// `docker/Dockerfile.jevk5` and `OPENJEV_MAX_MODEL_LEN=16384` for `jevk5`): vLLM's checks of a
// completion's prompt at the commit that image pins (vllm-project/vllm at 1b3b88e,
// `vllm/renderers/params.py`, `TokenizeParams._text_len_check` and `_token_len_check`, and the
// `str()` of `vllm/exceptions.py`'s `VLLMValidationError`), which upstream's `JevK5Engine._post`
// turns into its 400. Apache-2.0. See THIRD_PARTY.md.

import OpenJevCore

/// The longest prompt one JevK5 pass may read, and the refusal of a longer one, as upstream's
/// vLLM server gives it.
///
/// Upstream asks vLLM for one token per pass (`max_tokens: 1`) from a model served with a context
/// of 16,384 tokens, so a pass holds at most 16,383 prompt tokens. vLLM checks a text prompt
/// twice: its length in characters against the bound that many tokens could cover (16,383 times
/// the vocabulary's longest entry), before tokenizing, then its tokens. Either refusal reaches
/// the client as upstream's 400, `the model rejected this request: <vLLM's message>`, through
/// ``/OpenJevCore/BackendRefusal``. A prompt is never truncated.
public struct JevK5PromptLimit: Sendable, Hashable {
    /// vLLM's context, prompt and output together: 16,384.
    public var maxModelLength: Int
    /// The tokens a pass asks vLLM for: 1.
    public var outputTokens: Int
    /// The vocabulary's longest entry in Unicode scalars, Python's `len` of each key of
    /// `tokenizer.get_vocab()`, which vLLM takes as the most characters one token covers: 128 for
    /// JevK5's tokenizer.
    public var maxCharactersPerToken: Int

    /// Creates a limit; the defaults are upstream's.
    public init(maxModelLength: Int = 16_384, outputTokens: Int = 1, maxCharactersPerToken: Int) {
        self.maxModelLength = maxModelLength
        self.outputTokens = outputTokens
        self.maxCharactersPerToken = maxCharactersPerToken
    }

    /// The most prompt tokens a pass holds: 16,383.
    public var maxInputTokens: Int { maxModelLength - outputTokens }

    /// The most characters a prompt may have before vLLM refuses it untokenized.
    public var maxInputCharacters: Int { maxInputTokens * maxCharactersPerToken }

    /// vLLM's refusal of a prompt of `characters` Unicode scalars, more than
    /// ``maxInputCharacters``, or `nil` when the prompt is within the bound.
    public func characterRefusal(characters: Int) -> BackendRefusal? {
        guard characters > maxInputCharacters else { return nil }
        return BackendRefusal(
            reason: "This model's maximum context length is \(maxModelLength) tokens. However, "
                + "you requested \(outputTokens) output tokens and your prompt contains "
                + "\(characters) characters (more than \(maxInputCharacters) characters, which is "
                + "the upper bound for \(maxInputTokens) input tokens). Please reduce the length "
                + "of the input prompt or the number of requested output tokens. "
                + "(parameter=input_text, value=\(characters))")
    }

    /// vLLM's refusal of a prompt of `tokens` tokens, more than ``maxInputTokens``, or `nil` when
    /// the prompt fits.
    ///
    /// vLLM tokenizes with truncation at one token past the bound, so a prompt over it is counted
    /// as that many tokens and the message says "at least": every refused prompt gets the same
    /// text.
    public func tokenRefusal(tokens: Int) -> BackendRefusal? {
        guard tokens > maxInputTokens else { return nil }
        let counted = min(tokens, maxInputTokens + 1)
        let qualifier = counted == maxInputTokens + 1 ? "at least " : ""
        return BackendRefusal(
            reason: "This model's maximum context length is \(maxModelLength) tokens. However, "
                + "you requested \(outputTokens) output tokens and your prompt contains "
                + "\(qualifier)\(counted) input tokens, for a total of "
                + "\(qualifier)\(counted + outputTokens) tokens. Please reduce the length of the "
                + "input prompt or the number of requested output tokens. "
                + "(parameter=input_tokens, value=\(counted))")
    }
}
