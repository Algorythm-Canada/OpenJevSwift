// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, method
// `Engine.system_text`. Apache-2.0. See THIRD_PARTY.md.

/// The system prompt of a read: the questions, their allowed labels and the reply instruction.
public enum SystemText {
    /// The opening paragraph, before the first question.
    public static let opening =
        "Answer a fixed set of questions about the state the user provides. "
        + "Each question lists its allowed answers; reply with exactly one label per question.\n"

    /// The text shown for a question whose rendered instructions are empty.
    public static let defaultInstructions = "Answer about the state."

    /// The sentence appended when the questions are split across reads.
    public static let chunkedSentence =
        " A reply may cover only some of the questions; answer every line that is present."

    /// The system text for a group of read questions, byte for byte as upstream writes it.
    ///
    /// After ``opening``, each question gets `"\nQuestion {id}: {instructions}\n"` and one line
    /// per label:
    ///
    /// - noul: `"  {label}: {description}\n"`, or `"  {label}\n"` when undescribed;
    /// - score: `"  {label}: {description}\n"`, even when the description is empty;
    /// - choice: `"  {label}: {name} ({description})\n"`, or `"  {label}: {name}\n"` when
    ///   undescribed.
    ///
    /// Then a blank line and the format's instruction, and ``chunkedSentence`` when `chunked`
    /// is true. Labels and choices are paired in order; a question with more of one than the
    /// other lists the shorter count, as Python's `zip` does.
    public static func render(
        _ questions: [ReadQuestion], format: AnswerFormat, chunked: Bool
    ) -> String {
        var text = opening
        for question in questions {
            let instructions =
                question.instructions.isEmpty ? defaultInstructions : question.instructions
            text += "\nQuestion \(question.id): \(instructions)\n"
            for (choice, label) in zip(question.choices, question.labels) {
                let description = choice.description
                switch question.kind {
                case .noul:
                    text += description.isEmpty ? "  \(label)\n" : "  \(label): \(description)\n"
                case .score:
                    text += "  \(label): \(description)\n"
                case .choice:
                    text +=
                        description.isEmpty
                        ? "  \(label): \(choice.name)\n"
                        : "  \(label): \(choice.name) (\(description))\n"
                }
            }
        }
        text += "\n" + format.instruction
        if chunked {
            text += chunkedSentence
        }
        return text
    }
}
