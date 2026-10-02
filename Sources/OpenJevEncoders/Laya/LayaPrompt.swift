// A port of laya 0.3.6 (NandhaKishorM/laya), `laya/common.py`: `QTYPES`, `serialize_state`,
// `render_criterion`, `render_options` and the texts `build_sequence` tokenizes; and of
// `laya/agent.py`, `Agent._to_internal`. The question is what upstream OpenJev (razorback16/openjev
// at dcd2094), `openjev/encoders.py`, `LayaEngine.read_batch` passes to laya. Apache-2.0. See
// THIRD_PARTY.md.

import OpenJevCore

/// One question as Laya reads it: laya's internal question and the texts its `build_sequence`
/// tokenizes.
///
/// Upstream's `LayaEngine.read_batch` hands laya each question as `{type, instructions, criteria}`,
/// with the instructions as ``/OpenJevCore/EncoderQuestion/instructions`` (upstream's `text_of`)
/// and the criteria as sent, and laya's `Agent._to_internal` turns that into `{t, ins, crit}`:
/// ``kind``, ``instructions`` and ``criteria``. Unlike Verdict's prompt, the criteria are not
/// `text_of` renderings: a description keeps its leading and trailing whitespace, and anything but
/// a string is written as compact JSON.
///
/// - A choice renders each option as its name when the description is `null` or the empty
///   string, else as `{name}: {description}`.
/// - A score renders each level as `level {index}: {description}`.
/// - A noul renders `false: {description}` then `true: {description}`, laya's order, the reverse
///   of the engine's `[P(true), P(false)]`. A missing, `null` or empty description reads
///   `no, the statement does not hold` and `yes, the statement holds`.
///
/// A description is the string as sent, or for anything else laya's `render_criterion`:
/// `json.dumps(value, ensure_ascii=False, separators=(", ", ": "))`, which is
/// ``/OpenJevCore/PythonJSONWriter/modelText(_:)``. Every "[MASK]" in the instructions, the options
/// and the state is replaced by a space before tokenizing, so that only laya's own markers are
/// masks.
public struct LayaPrompt: Sendable, Hashable {
    /// laya's `crit`: the criteria of the question as sent.
    public enum Criteria: Sendable, Hashable {
        /// A choice's options in order, each name with its description as sent.
        case choice(JSONObject)
        /// A score's levels in order, as sent.
        case score([JSONValue])
        /// A noul's descriptions of its outcomes, `nil` when absent.
        case noul(whenTrue: Described, whenFalse: Described)
    }

    /// The text laya's `build_sequence` replaces wherever the caller wrote it: the tokenizer's
    /// mask token, which marks each option.
    public static let maskToken = "[MASK]"

    /// What a noul's false outcome reads when it has no description.
    public static let defaultFalseDescription = "no, the statement does not hold"
    /// What a noul's true outcome reads when it has no description.
    public static let defaultTrueDescription = "yes, the statement holds"

    /// laya's `t`: the question type.
    public var kind: QuestionKind
    /// laya's `ins`: the instructions as text, upstream's `text_of` rendering.
    public var instructions: String
    /// laya's `crit`.
    public var criteria: Criteria

    /// Creates laya's question from a question the engine reads.
    public init(question: EncoderQuestion) {
        kind = question.kind
        instructions = question.instructions
        switch question.question {
        case .choice(_, let options):
            criteria = .choice(options)
        case .score(_, let levels):
            criteria = .score(levels)
        case .noul(_, let outcomes):
            criteria = .noul(
                whenTrue: outcomes?.whenTrue ?? nil, whenFalse: outcomes?.whenFalse ?? nil)
        }
    }

    /// Creates laya's question from its parts.
    public init(kind: QuestionKind, instructions: String, criteria: Criteria) {
        self.kind = kind
        self.instructions = instructions
        self.criteria = criteria
    }

    /// laya's `QTYPES` index of the question type, which the model reads as the question type
    /// plane: choice 0, score 1, noul 2.
    public var questionType: Int {
        Self.questionType(of: kind)
    }

    /// laya's `QTYPES`: choice 0, score 1, noul 2.
    public static func questionType(of kind: QuestionKind) -> Int {
        switch kind {
        case .choice: return 0
        case .score: return 1
        case .noul: return 2
        }
    }

    /// laya's `render_options`: the options in the order of their markers.
    public var options: [String] {
        switch criteria {
        case .choice(let options):
            return options.map { name, description in
                switch description {
                case .null, .string(""):
                    return name
                default:
                    return name + ": " + Self.renderCriterion(description)
                }
            }
        case .score(let levels):
            return levels.enumerated().map { index, level in
                "level \(index): " + Self.renderCriterion(level)
            }
        case .noul(let whenTrue, let whenFalse):
            return [
                "false: " + Self.renderOutcome(whenFalse, default: Self.defaultFalseDescription),
                "true: " + Self.renderOutcome(whenTrue, default: Self.defaultTrueDescription),
            ]
        }
    }

    /// The head `build_sequence` tokenizes after `[CLS]`: `{type} question: {instructions}`.
    public var head: String {
        "\(kind.rawValue) question: " + Self.replacingMasks(in: instructions)
    }

    /// The option texts `build_sequence` tokenizes after each `[MASK]` marker: a space, then the
    /// option.
    public var optionTexts: [String] {
        options.map { " " + Self.replacingMasks(in: $0) }
    }

    /// The state text `build_sequence` tokenizes after the options: laya's `serialize_state`,
    /// which is ``/OpenJevCore/StateText/render(_:)`` (a string as sent, anything else as
    /// `json.dumps(state, ensure_ascii=False)`), with every "[MASK]" replaced by a space.
    public static func stateText(_ state: JSONValue) -> String {
        replacingMasks(in: StateText.render(state))
    }

    /// laya's `render_criterion`: a string as it is, anything else as compact JSON with the
    /// separators `", "` and `": "` and non-ASCII text kept.
    ///
    /// - Precondition: The value holds no infinite or NaN float and no malformed integer text.
    ///   ``/OpenJevCore/JSONParser`` never produces either, and the schema builder has already
    ///   rendered every description with ``/OpenJevCore/TextOf``, which has the same precondition.
    public static func renderCriterion(_ value: JSONValue) -> String {
        if case .string(let text) = value {
            return text
        }
        do {
            return try PythonJSONWriter.modelText(value)
        } catch {
            preconditionFailure("a criterion cannot be written as JSON: \(error)")
        }
    }

    /// A noul outcome's description, or `fallback` when it is absent, `null` or empty: laya's
    /// `crit.get(...) not in (None, "")`.
    private static func renderOutcome(_ description: Described, default fallback: String)
        -> String
    {
        switch description {
        case .none, .some(.null), .some(.string("")):
            return fallback
        case .some(let value):
            return renderCriterion(value)
        }
    }

    /// `text.replace("[MASK]", " ")`: every occurrence, left to right, compared as Python
    /// compares strings, code point by code point. The mask token is ASCII, so comparing UTF-8
    /// bytes is the same thing, and a combining mark after the `]` does not hide a match the way
    /// Swift's character comparison would.
    public static func replacingMasks(in text: String) -> String {
        let mask = Array(maskToken.utf8)
        let bytes = Array(text.utf8)
        guard bytes.count >= mask.count else {
            return text
        }
        var output: [UInt8] = []
        var index = 0
        var replaced = false
        while index < bytes.count {
            if bytes[index] == mask[0], index + mask.count <= bytes.count,
                bytes[index..<(index + mask.count)].elementsEqual(mask)
            {
                output.append(UInt8(ascii: " "))
                index += mask.count
                replaced = true
            } else {
                output.append(bytes[index])
                index += 1
            }
        }
        return replaced ? String(decoding: output, as: UTF8.self) : text
    }
}
