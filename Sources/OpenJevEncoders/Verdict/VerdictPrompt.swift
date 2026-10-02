// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, function
// `verdict_prompt` and the constants `LABEL_MARKER`, `SEP_MARKER` and `ABSTAIN`, which follow
// core/formatting.py of Heman10x-NGU/Verdict-open-jev (v1.4 inference). Apache-2.0. See
// THIRD_PARTY.md.

import OpenJevCore

/// Verdict's model input for one question, as upstream's `verdict_prompt` writes it: the labels,
/// each after `<<LABEL>>`, then `<<SEP>>` and the body.
///
/// - A noul reads the labels `true: {instructions}` and `false: not {instructions}`, in that
///   order, and the body `Context:\n{context}\n\nEvaluate proposition: {instructions}`. Its
///   criteria are not read.
/// - A choice reads `It is {description}` for each option, or `It is {name}` when the
///   description is empty. A score reads `{description} (Value: {index})` for each level, with
///   the index written as Python writes `float(i)`: `0.0`, `1.0`, and so on. Both read the body
///   `Question: {instructions}\n\nContext:\n{context}`, or the context alone when the
///   instructions are empty.
/// - Every question ends with the label `insufficient evidence`, the abstention the model may
///   choose. ``VerdictCalibration`` drops its probability again.
///
/// The instructions and descriptions are ``/OpenJevCore/EncoderQuestion``'s, which are upstream's
/// `text_of` renderings, and the context is ``/OpenJevCore/StateText/render(_:)`` of the state.
public struct VerdictPrompt: Sendable, Hashable {
    /// The marker before each label, upstream's `LABEL_MARKER`.
    public static let labelMarker = "<<LABEL>>"
    /// The marker between the labels and the body, upstream's `SEP_MARKER`.
    public static let separatorMarker = "<<SEP>>"
    /// The last label of every question, upstream's `ABSTAIN`.
    public static let abstentionLabel = "insufficient evidence"

    /// The labels in order: one per option of the question, then ``abstentionLabel``.
    public var labels: [String]
    /// The text after ``separatorMarker``: the question and the context.
    public var body: String

    /// Creates the prompt of one question about a state rendered as `context`.
    public init(question: EncoderQuestion, context: String) {
        let instructions = question.instructions
        switch question.kind {
        case .noul:
            labels = ["true: \(instructions)", "false: not \(instructions)"]
            body = "Context:\n\(context)\n\nEvaluate proposition: \(instructions)"
        case .choice:
            labels = question.choices.map { choice in
                "It is \(choice.description.isEmpty ? choice.name : choice.description)"
            }
            body = Self.questionBody(instructions: instructions, context: context)
        case .score:
            labels = question.choices.enumerated().map { index, level in
                "\(level.description) (Value: \(Double(index).pythonRepr))"
            }
            body = Self.questionBody(instructions: instructions, context: context)
        }
        labels.append(Self.abstentionLabel)
    }

    /// The body of a choice or a score: the question and the context, or the context alone
    /// when the instructions are empty.
    private static func questionBody(instructions: String, context: String) -> String {
        instructions.isEmpty ? context : "Question: \(instructions)\n\nContext:\n\(context)"
    }

    /// The prompt the tokenizer reads.
    public var text: String {
        labels.map { Self.labelMarker + $0 }.joined() + Self.separatorMarker + body
    }

    /// Upstream's `k`: the options plus the abstention, which is how many of the model's logits
    /// the calibration reads.
    public var labelCount: Int { labels.count }
}
