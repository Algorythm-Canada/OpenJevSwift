// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, method
// `build_schema` of the encoder backends. Apache-2.0. See THIRD_PARTY.md.

/// One question an encoder backend reads (upstream's CLM and JevK5 backends).
///
/// It has no model-facing id or labels: those backends render each question themselves, so the
/// question as sent is kept alongside the rendered text.
public struct EncoderQuestion: Sendable {
    /// The caller's id for the question, its key in the request's `questions`.
    public var key: String
    /// The question type.
    public var kind: QuestionKind
    /// The question as sent, which holds the raw instructions and the original criteria.
    public var question: Question
    /// The instructions as ``TextOf/render(_:)`` gives them; empty when absent.
    public var instructions: String
    /// The answers in order: `yes` and `no`, each option's name, or each level's index, with
    /// their rendered descriptions.
    public var choices: [(name: String, description: String)]
    /// A score's levels as sent, for the answer's `legend`; `nil` for other types.
    public var legend: [JSONValue]?

    /// The instructions as sent, before rendering. An encoder renders an object itself rather
    /// than reading ``TextOf``'s JSON.
    public var rawInstructions: Described {
        question.instructions
    }

    /// Creates an encoder question.
    public init(
        key: String,
        kind: QuestionKind,
        question: Question,
        instructions: String,
        choices: [(name: String, description: String)],
        legend: [JSONValue]?
    ) {
        self.key = key
        self.kind = kind
        self.question = question
        self.instructions = instructions
        self.choices = choices
        self.legend = legend
    }
}

/// The questions an encoder backend reads, and the answers it needs no read for.
public struct EncoderQuestionSchema: Sendable {
    /// The questions to read, in request order.
    public var questions: [EncoderQuestion]
    /// The answers of single-option choices and single-level scores, keyed by the caller's id,
    /// in request order.
    public var forced: OrderedMap<Answer>

    /// Creates a schema.
    public init(questions: [EncoderQuestion], forced: OrderedMap<Answer>) {
        self.questions = questions
        self.forced = forced
    }
}

/// Turns a request's questions into an ``EncoderQuestionSchema``, as upstream's encoder
/// backends do.
///
/// The rules, limits, forced answers and error messages are those of
/// ``QuestionSchemaBuilder``, with the option limit set by ``maxChoices`` alone, since these
/// backends do not use single-token labels. There are no ids and no answer format.
public struct EncoderQuestionSchemaBuilder: Sendable {
    /// The most options one choice may have.
    public var maxChoices: Int

    /// Creates a builder.
    public init(maxChoices: Int) {
        self.maxChoices = maxChoices
    }

    /// The schema of a request's questions, in request order.
    ///
    /// - Throws: The ``SchemaError`` of the first question upstream refuses, located at
    ///   `["body", "questions", key, "criteria"]`.
    public func build(_ questions: OrderedMap<Question>) throws(SchemaError)
        -> EncoderQuestionSchema
    {
        // Upstream's encoder backends keep the engine's limit of 10 score levels.
        let rules = SchemaRules(maxChoices: maxChoices, maxScoreLevels: 10)
        var read: [EncoderQuestion] = []
        var forced = OrderedMap<Answer>()
        for (key, question) in questions {
            switch try rules.entry(key: key, question: question) {
            case .forced(let answer):
                forced.updateValue(answer, forKey: key)
            case .read(let kind, let choices, let legend):
                read.append(
                    EncoderQuestion(
                        key: key,
                        kind: kind,
                        question: question,
                        instructions: TextOf.render(question.instructions),
                        choices: choices,
                        legend: legend))
            }
        }
        return EncoderQuestionSchema(questions: read, forced: forced)
    }
}
