// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, method
// `Engine.build_schema`. Apache-2.0. See THIRD_PARTY.md.

/// The type of a question, as the schema and the prompt see it.
public enum QuestionKind: String, Sendable, Hashable, CaseIterable {
    /// A yes or no question.
    case noul
    /// A choice among named options.
    case choice
    /// A score over ordered levels.
    case score
}

/// One question the model reads, as upstream's `build_schema` describes it.
///
/// The model never sees the caller's key: it sees ``id`` (`q1` to `qN`), and answers map back by
/// position.
public struct ReadQuestion: Sendable {
    /// The caller's id for the question, its key in the request's `questions`.
    public var key: String
    /// The id the model sees: `q1` to `qN` over the read questions, in request order.
    public var id: String
    /// The question type.
    public var kind: QuestionKind
    /// The instructions as ``TextOf/render(_:)`` gives them; empty when absent.
    public var instructions: String
    /// The answers in label order: `yes` and `no`, each option's name, or each level's index,
    /// with their rendered descriptions.
    public var choices: [(name: String, description: String)]
    /// The label the model writes for each entry of ``choices``.
    public var labels: [String]
    /// A score's levels as sent, for the answer's `legend`; `nil` for other types.
    public var legend: [JSONValue]?

    /// Creates a read question.
    public init(
        key: String,
        id: String,
        kind: QuestionKind,
        instructions: String,
        choices: [(name: String, description: String)],
        labels: [String],
        legend: [JSONValue]?
    ) {
        self.key = key
        self.id = id
        self.kind = kind
        self.instructions = instructions
        self.choices = choices
        self.labels = labels
        self.legend = legend
    }
}

/// The questions of a request as the engine reads them, and the answers it needs no read for.
public struct QuestionSchema: Sendable {
    /// The questions to read, in request order, numbered `q1` to `qN`.
    public var questions: [ReadQuestion]
    /// The answers of single-option choices and single-level scores, keyed by the caller's id,
    /// in request order.
    public var forced: OrderedMap<Answer>
    /// The answer format: ``AnswerFormat/lines`` for up to 10 read questions,
    /// ``AnswerFormat/indexed`` beyond.
    public var format: AnswerFormat

    /// Creates a schema.
    public init(questions: [ReadQuestion], forced: OrderedMap<Answer>, format: AnswerFormat) {
        self.questions = questions
        self.forced = forced
        self.format = format
    }
}

/// Turns a request's questions into a ``QuestionSchema``, as upstream's `Engine.build_schema`
/// does.
///
/// - A noul reads `yes` and `no`, described by its criteria's `true` and `false`. Absent
///   criteria behave as empty ones.
/// - A choice needs at least one option. One option is answered without a read. More options
///   than there are choice labels, or than ``maxChoices``, is an error. Otherwise the options
///   are read in criteria order under the first labels.
/// - A score may have at most ``maxScoreLevels`` levels. One level is answered without a read.
///   Otherwise the levels are read under the labels `0` to `N-1`.
///
/// Forced answers have probability 1.0 and confidence 1.0; a forced score's `score` is 0.0.
/// The first failing question in request order throws.
///
/// Upstream also raises `unknown question type` for a type other than the three. That cannot
/// happen here, since ``Question`` has only the three cases: ``RequestValidator`` refuses such a
/// request with the generic 400 first, as upstream's API layer does.
public struct QuestionSchemaBuilder: Sendable {
    /// The single-token choice labels in order (`A`, `B`, `C`, ...), from the tokenizer.
    public var choiceLabels: [String]
    /// The most options one choice may have. Upstream's `MAX_CHOICES` is 255.
    public var maxChoices: Int
    /// The most levels one score may have. Upstream allows 10.
    public var maxScoreLevels: Int

    /// Creates a builder. The limit on options is the smaller of `maxChoices` and the number of
    /// labels.
    public init(choiceLabels: [String], maxChoices: Int = 255, maxScoreLevels: Int = 10) {
        self.choiceLabels = choiceLabels
        self.maxChoices = maxChoices
        self.maxScoreLevels = maxScoreLevels
    }

    /// The schema of a request's questions, in request order.
    ///
    /// - Throws: The ``SchemaError`` of the first question upstream refuses, located at
    ///   `["body", "questions", key, "criteria"]`.
    public func build(_ questions: OrderedMap<Question>) throws(SchemaError) -> QuestionSchema {
        let rules = SchemaRules(
            maxChoices: min(maxChoices, choiceLabels.count), maxScoreLevels: maxScoreLevels)
        var read: [ReadQuestion] = []
        var forced = OrderedMap<Answer>()
        for (key, question) in questions {
            switch try rules.entry(key: key, question: question) {
            case .forced(let answer):
                forced.updateValue(answer, forKey: key)
            case .read(let kind, let choices, let legend):
                read.append(
                    ReadQuestion(
                        key: key,
                        id: "q\(read.count + 1)",
                        kind: kind,
                        instructions: TextOf.render(question.instructions),
                        choices: choices,
                        labels: labels(kind, count: choices.count),
                        legend: legend))
            }
        }
        return QuestionSchema(
            questions: read, forced: forced, format: .forReadCount(read.count))
    }

    /// The labels of a read question with `count` answers.
    private func labels(_ kind: QuestionKind, count: Int) -> [String] {
        switch kind {
        case .noul: return ["yes", "no"]
        case .choice: return Array(choiceLabels.prefix(count))
        case .score: return (0..<count).map { String($0) }
        }
    }
}

/// What the schema rules decide for one question: a forced answer, or what to read.
enum SchemaEntry {
    /// The question has one possible answer and is not read.
    case forced(Answer)
    /// The question is read with these answers, and a score keeps its levels as the legend.
    case read(
        kind: QuestionKind, choices: [(name: String, description: String)], legend: [JSONValue]?)
}

/// The per-question part of upstream's two `build_schema` methods: the limits, the forced
/// answers and the rendered answers. ``QuestionSchemaBuilder`` and
/// ``EncoderQuestionSchemaBuilder`` both go through it.
struct SchemaRules {
    /// The most options one choice may have.
    var maxChoices: Int
    /// The most levels one score may have.
    var maxScoreLevels: Int

    /// The entry for one question.
    ///
    /// - Throws: A ``SchemaError`` for an empty choice, too many options or too many levels.
    func entry(key: String, question: Question) throws(SchemaError) -> SchemaEntry {
        let loc: [LocComponent] = ["body", "questions", .key(key), "criteria"]
        switch question {
        case .noul(_, let criteria):
            return .read(
                kind: .noul,
                choices: [
                    ("yes", TextOf.render(criteria?.whenTrue ?? nil)),
                    ("no", TextOf.render(criteria?.whenFalse ?? nil)),
                ],
                legend: nil)
        case .choice(_, let criteria):
            if criteria.isEmpty {
                throw SchemaError("Choice question must have at least one choice: \(key)", loc: loc)
            }
            if criteria.count == 1 {
                let only = criteria.keys[0]
                return .forced(
                    .choice(choice: only, probabilities: [only: 1.0], confidence: 1.0))
            }
            if criteria.count > maxChoices {
                throw SchemaError(
                    "Too many choices. Must have at most \(maxChoices) choices.", loc: loc)
            }
            return .read(
                kind: .choice,
                choices: criteria.map { (name: $0.key, description: TextOf.render($0.value)) },
                legend: nil)
        case .score(_, let criteria):
            if criteria.count > maxScoreLevels {
                throw SchemaError(
                    "Too many score levels. Must have at most \(maxScoreLevels) levels.", loc: loc)
            }
            if criteria.count == 1 {
                return .forced(
                    .score(score: 0.0, legend: criteria, probabilities: [1.0], confidence: 1.0))
            }
            return .read(
                kind: .score,
                choices: criteria.enumerated().map {
                    (name: String($0.offset), description: TextOf.render($0.element))
                },
                legend: criteria)
        }
    }
}
