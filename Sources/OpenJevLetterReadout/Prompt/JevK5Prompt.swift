// A port of the `jevk5` package by Alibi Serikbay (github.com/allebee/jevk5 at v0.2.2, 0571ef3),
// `jevk5/prompt.py`: `LETTERS`, `SYSTEM`, `CHAT_TEMPLATE`, `messages`, `prompt_text` and
// `decision_options`, which upstream OpenJev's `JevK5Engine.read_question` sends to the model
// (razorback16/openjev at dcd2094, `openjev/encoders.py`). Apache-2.0. See THIRD_PARTY.md.

import OpenJevCore

/// JevK5's prompt: a fixed system instruction, the decision as JSON (the evidence, the criterion
/// and up to 16 lettered options) and Qwen3.5's chat template with thinking off, pinned as text.
///
/// The text is byte for byte the `jevk5` package's `prompt_text`, which upstream sends as is:
///
/// ```
/// <|im_start|>system
/// {system}<|im_end|>
/// <|im_start|>user
/// {user}<|im_end|>
/// <|im_start|>assistant
/// <think>
///
/// </think>
///
/// ```
///
/// where `{user}` is `json.dumps({"evidence": state, "criterion": instructions, "options":
/// [{"letter": "A", "description": text}, ...]}, ensure_ascii=False)`. The state and the
/// instructions go in as sent: a string stays a string and is not stripped, an object stays an
/// object, absent instructions are `null`. Python fills the template with `str.format`, which
/// inserts the two texts verbatim, braces included.
public enum JevK5Prompt {
    /// The answer letters, `A` to `P`: one forward pass reads at most 16 options.
    public static let letters: [String] = "ABCDEFGHIJKLMNOP".map { String($0) }

    /// The system instruction.
    public static let system =
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
        + "Respond with only its uppercase letter, with no explanation or reasoning."

    /// The template's text before the system instruction.
    static let systemPrefix = "<|im_start|>system\n"
    /// The template's text between the system instruction and the user message.
    static let userPrefix = "<|im_end|>\n<|im_start|>user\n"
    /// The template's text after the user message: the generation prompt with an empty think
    /// block, so the next token is the answer letter.
    static let assistantPrefix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    /// The user message, `json.dumps(payload, ensure_ascii=False)`: the evidence, the criterion
    /// and the lettered options, in that order.
    ///
    /// - Precondition: At most 16 options, and values the request parser produced (no infinite
    ///   or NaN float, D-016).
    public static func user(state: JSONValue, criterion: Described, options: [String]) -> String {
        precondition(options.count <= letters.count, "one pass reads at most 16 options")
        var payload = JSONObject()
        payload["evidence"] = state
        payload["criterion"] = criterion ?? .null
        payload["options"] = .array(
            options.enumerated().map { index, description in
                .object(["letter": .string(letters[index]), "description": .string(description)])
            })
        do {
            return try PythonJSONWriter.modelText(.object(payload))
        } catch {
            preconditionFailure("a request's values cannot be written as JSON: \(error)")
        }
    }

    /// The full prompt, the chat template around ``system`` and ``user(state:criterion:options:)``,
    /// as `prompt_text(state, criterion, options)` writes it.
    public static func text(state: JSONValue, criterion: Described, options: [String]) -> String {
        systemPrefix + system + userPrefix
            + user(state: state, criterion: criterion, options: options)
            + assistantPrefix
    }
}

/// One option of a question as JevK5 reads it: its id and the text the model sees.
public struct JevK5Option: Sendable, Hashable {
    /// `true` or `false` for a noul, the criterion's name for a choice, the level's index for a
    /// score.
    public var id: String
    /// `"{id}: {description}"`.
    public var text: String

    /// Creates an option.
    public init(id: String, text: String) {
        self.id = id
        self.text = text
    }
}

extension JevK5Option {
    /// The `jevk5` package's `decision_options`: the options of a question in the caller's order,
    /// each written `"{id}: {description}"` as SemIf's JevBench mapping writes them.
    ///
    /// - A noul is `true` then `false`, each with its description from the criteria or, when
    ///   that is absent or false in Python's sense, `"The proposition is true."` (or `false`).
    /// - A choice is its criteria in order, each with its description or, when that is absent
    ///   or false, its own name.
    /// - A score is its levels in order, with their indices as ids.
    ///
    /// A description that is not a string is written as Python's `str()` of it, the f-string's
    /// replacement: `{'k': 'v'}`, `['a', 1]`, single-quoted reprs and `True`, `False`, `None`.
    public static func options(for question: Question) -> [JevK5Option] {
        let pairs: [(String, String)]
        switch question {
        case .noul(_, let criteria):
            pairs = [("true", criteria?.whenTrue), ("false", criteria?.whenFalse)].map {
                id, description in
                guard let description, description.isPythonTruthy else {
                    return (id, "The proposition is \(id).")
                }
                return (id, description.pythonStr)
            }
        case .choice(_, let criteria):
            pairs = criteria.map { entry in
                (entry.key, entry.value.isPythonTruthy ? entry.value.pythonStr : entry.key)
            }
        case .score(_, let levels):
            pairs = levels.enumerated().map { index, level in (String(index), level.pythonStr) }
        }
        return pairs.map { JevK5Option(id: $0.0, text: "\($0.0): \($0.1)") }
    }
}
