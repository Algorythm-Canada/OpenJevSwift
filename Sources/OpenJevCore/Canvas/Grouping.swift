// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, method
// `Engine.groups`. Apache-2.0. See THIRD_PARTY.md.

/// Splits a read's questions into the groups whose answer templates fit the canvas.
public enum ReadGrouping {
    /// The questions, in order, in the fewest groups whose template rows fit the canvas, as
    /// upstream's `Engine.groups` splits them.
    ///
    /// The questions are walked in order. A trial group is the current group plus the next
    /// question; its rows are `scaffold.count + enc(answerText at every first label).count + 1`.
    /// When the rows exceed the canvas and the current group is not empty, the current group is
    /// closed and the question starts a new one; otherwise the trial becomes the current group.
    /// So a single question that does not fit still gets a group, and ``TemplateResolver``
    /// refuses it with the canvas message.
    ///
    /// No questions give no groups. Upstream only calls `groups()` when there is something to
    /// read; its `[[]]` for an empty list is never observed.
    ///
    /// One read's label ids are never the binding limit: every question draws from the same
    /// label lists, so a whole schema's union is at most 255 choice letters, ten score digits and
    /// yes and no, well inside the 512 a read allows.
    ///
    /// - Throws: Whatever `tokenizer.encode` throws.
    public static func groups(
        _ questions: [ReadQuestion], format: AnswerFormat, geometry: CanvasGeometry,
        scaffold: [Int], tokenizer: any DecisionTokenizer
    ) throws -> [[ReadQuestion]] {
        var out: [[ReadQuestion]] = []
        var group: [ReadQuestion] = []
        for question in questions {
            let trial = group + [question]
            let rows = try rows(of: trial, format: format, scaffold: scaffold, tokenizer: tokenizer)
            if rows > geometry.canvas && !group.isEmpty {
                out.append(group)
                group = [question]
            } else {
                group = trial
            }
        }
        if !group.isEmpty {
            out.append(group)
        }
        return out
    }

    /// The rows a group's plain read needs: the scaffold, the answer text at every first label
    /// and the turn close. This is the number `groups()` compares with the canvas.
    ///
    /// - Throws: Whatever `tokenizer.encode` throws.
    public static func rows(
        of group: [ReadQuestion], format: AnswerFormat, scaffold: [Int],
        tokenizer: any DecisionTokenizer
    ) throws -> Int {
        let text = AnswerText.render(
            group, labelIndices: [Int](repeating: 0, count: group.count), format: format)
        return scaffold.count + (try tokenizer.encode(text, addSpecialTokens: false)).count + 1
    }
}
