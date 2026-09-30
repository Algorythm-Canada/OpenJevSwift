// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, method
// `Engine.answer_text`. Apache-2.0. See THIRD_PARTY.md.

/// The model turn's expected reply for a group of questions at chosen labels.
public enum AnswerText {
    /// Each question's lead and chosen label, joined by the format's join.
    ///
    /// All indices 0 give the template upstream resolves slots from: `"q1: A\nq2: 0\nq3: yes"`
    /// in the lines format, `"q1A q20 q3yes"` in the indexed format.
    ///
    /// - Precondition: `labelIndices` has one entry per question, and each is a valid index into
    ///   that question's labels.
    public static func render(
        _ questions: [ReadQuestion], labelIndices: [Int], format: AnswerFormat
    ) -> String {
        precondition(
            questions.count == labelIndices.count,
            "\(questions.count) questions but \(labelIndices.count) label indices")
        return zip(questions, labelIndices)
            .map { question, index in
                precondition(
                    question.labels.indices.contains(index),
                    "label index \(index) is out of range for question \(question.id)")
                return format.lead(id: question.id) + question.labels[index]
            }
            .joined(separator: format.join)
    }
}
