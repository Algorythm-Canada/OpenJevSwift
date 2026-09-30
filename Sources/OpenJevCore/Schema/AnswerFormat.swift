// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, the
// `FORMATS` table. Apache-2.0. See THIRD_PARTY.md.

/// The shape of the answer the model writes: how questions are joined, what precedes each label,
/// and the sentence of the system text that asks for that shape.
///
/// `lines` is used for up to 10 read questions. `indexed` costs fewer tokens a question, so past
/// ten questions a schema is more likely to fit in one read.
public enum AnswerFormat: String, Sendable, Hashable, CaseIterable {
    /// One line per question: `"q1: A\nq2: 0"`.
    case lines
    /// One line for all questions, each id followed directly by its label: `"q1A q20"`.
    case indexed

    /// The text between two questions' answers: a newline or a space.
    public var join: String {
        switch self {
        case .lines: return "\n"
        case .indexed: return " "
        }
    }

    /// What precedes a label, with `{id}` standing for the question id: `"{id}: "` or `"{id}"`.
    public var leadTemplate: String {
        "{id}" + afterID
    }

    /// What follows the id in the lead: `": "` or nothing.
    public var afterID: String {
        switch self {
        case .lines: return ": "
        case .indexed: return ""
        }
    }

    /// The reply instruction that closes the system text.
    public var instruction: String {
        switch self {
        case .lines:
            return "Reply with one line per question, in this order, formatted as \"id: label\"."
        case .indexed:
            return
                "Reply on one line with each question's id immediately followed by its label, "
                + "separated by single spaces."
        }
    }

    /// The lead for one question, as Python's `lead.format(id=id)` gives it.
    public func lead(id: String) -> String {
        id + afterID
    }

    /// The format upstream picks for a schema with this many read questions: `lines` up to
    /// 10, `indexed` beyond.
    public static func forReadCount(_ count: Int) -> AnswerFormat {
        count <= 10 ? .lines : .indexed
    }
}
