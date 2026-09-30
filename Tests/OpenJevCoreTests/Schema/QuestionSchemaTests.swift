import OpenJevCore
import Testing

/// Compares ``QuestionSchemaBuilder`` with upstream's `Engine.build_schema` in
/// Fixtures/schemas/schemas.json, and checks the limits and text rules by hand.
@Suite("Question schema")
struct QuestionSchemaTests {
    private let encoder = WireEncoder()

    /// 255 distinct labels that do not depend on the fixtures.
    private static let labels = (0..<255).map { "L\($0)" }

    /// A choice question with `count` undescribed options named `o0`, `o1`, and so on.
    private static func choice(options count: Int) -> Question {
        .choice(
            instructions: nil,
            criteria: JSONObject(uniqueKeysWithValues: (0..<count).map { ("o\($0)", .null) }))
    }

    /// A score question with `count` levels named `level 0`, `level 1`, and so on.
    private static func score(levels count: Int) -> Question {
        .score(instructions: nil, criteria: (0..<count).map { .string("level \($0)") })
    }

    /// The error for question `q`'s criteria.
    private static func criteriaError(_ message: String) -> SchemaError {
        SchemaError(message, loc: ["body", "questions", "q", "criteria"])
    }

    private func build(_ questions: OrderedMap<Question>) throws(SchemaError) -> QuestionSchema {
        try QuestionSchemaBuilder(choiceLabels: Self.labels).build(questions)
    }

    private func buildError(_ questions: OrderedMap<Question>) -> SchemaError? {
        do {
            _ = try build(questions)
            return nil
        } catch {
            return error
        }
    }

    // MARK: Fixtures

    @Test(
        "Every recorded schema and schema error is reproduced",
        .enabled(
            if: UpstreamFixtures.exists("schemas/schemas.json", "labels.json")
                && FixtureTokenizer.exists,
            UpstreamFixtures.missingMessage))
    func recordedSchemas() throws {
        // The labels the engine would discover from the tokenizer, which must be the recorded
        // ones.
        let labels = try LabelDiscovery.choiceLabels(using: FixtureTokenizer.shared).labels
        #expect(labels == (try UpstreamFixtures.choiceLabels()))
        #expect(labels.count == 255)
        let cases = try UpstreamFixtures.cases("schemas/schemas.json")
        var failures: [String] = []
        var built = 0
        var refused = 0
        var unbuildable = 0
        for row in cases {
            let name = row["name"]?.stringValue ?? "?"
            do {
                switch try check(row, labels: labels) {
                case .built: built += 1
                case .refused: refused += 1
                case .notBuildable: unbuildable += 1
                case .failed(let problem): failures.append("\(name): \(problem)")
                }
            } catch {
                failures.append("\(name): \(error)")
            }
        }
        #expect(failures.isEmpty, "\(failures.count) failures: \(failures.prefix(10))")
        #expect(built == 29, "only \(built) schemas compared")
        #expect(refused == 9, "only \(refused) schema errors compared")
        #expect(unbuildable == 2, "\(unbuildable) unknown-type rows refused by the validator")
    }

    private enum Outcome {
        case built, refused, notBuildable
        case failed(String)
    }

    private func check(_ row: JSONValue, labels: [String]) throws -> Outcome {
        if row["api_reachable"]?.boolValue == false {
            // An unknown question type, which upstream's test hands to the engine directly.
            // Upstream's API refuses such a request before the engine, and so does
            // RequestValidator (#5), so there is no Question to build from.
            let body: JSONValue = [
                "state": "x", "model": "jev-latest", "questions": try #require(row["questions"]),
            ]
            do {
                _ = try RequestValidator().validate(body)
                return .failed("RequestValidator accepted an unknown question type")
            } catch {
                return error == .invalidRequest ? .notBuildable : .failed("\(error)")
            }
        }
        let body = try #require(row["request"])
        let request = try RequestValidator().validate(body)
        let builder = QuestionSchemaBuilder(choiceLabels: labels)
        if let expected = row["error"] {
            do {
                _ = try builder.build(request.questions)
                return .failed("built a schema; expected \(expected)")
            } catch {
                guard error.message == expected["message"]?.stringValue else {
                    return .failed("message \(error.message)")
                }
                guard JSONValue.array(error.loc.map(\.json)) == expected["loc"] else {
                    return .failed("loc \(error.loc)")
                }
                return .refused
            }
        }
        let schema = try builder.build(request.questions)
        let expected = try #require(row["schema"])
        let questions = JSONValue.array(schema.questions.map(Self.fixtureShape))
        let expectedQuestions = try #require(expected["questions"])
        if try encoder.bytes(json: questions) != encoder.bytes(json: expectedQuestions) {
            return .failed("questions \(try PythonJSONWriter.modelText(questions))")
        }
        let forced = JSONValue.object(
            JSONObject(uniqueKeysWithValues: schema.forced.map { ($0.key, $0.value.json) }))
        let expectedForced = try #require(expected["forced"])
        if try encoder.bytes(json: forced) != encoder.bytes(json: expectedForced) {
            return .failed("forced \(try PythonJSONWriter.modelText(forced))")
        }
        if expected["format"]?.stringValue != schema.format.rawValue {
            return .failed("format \(schema.format)")
        }
        return .built
    }

    /// A read question in the fixture's shape: `{key, id, type, instructions, choices, labels,
    /// legend}`, with `choices` as `[name, description]` pairs.
    private static func fixtureShape(_ question: ReadQuestion) -> JSONValue {
        [
            "key": .string(question.key),
            "id": .string(question.id),
            "type": .string(question.kind.rawValue),
            "instructions": .string(question.instructions),
            "choices": .array(
                question.choices.map { [.string($0.name), .string($0.description)] }),
            "labels": .array(question.labels.map { .string($0) }),
            "legend": question.legend.map { .array($0) } ?? .null,
        ]
    }

    // MARK: Choices

    @Test("A single-option choice is forced and not read")
    func singleOption() throws {
        let schema = try build(["only": Self.choice(options: 1)])
        #expect(schema.questions.isEmpty)
        #expect(schema.forced.keys == ["only"])
        #expect(
            schema.forced["only"]
                == .choice(choice: "o0", probabilities: ["o0": 1.0], confidence: 1.0))
        #expect(
            try encoder.string(#require(schema.forced["only"]))
                == #"{"type":"choice","choice":"o0","probabilities":{"o0":1.0},"confidence":1.0}"#)
    }

    @Test("255 options are read under the first 255 labels")
    func mostOptions() throws {
        let schema = try build(["q": Self.choice(options: 255)])
        let question = try #require(schema.questions.first)
        #expect(question.labels == Self.labels)
        #expect(question.choices.count == 255)
        #expect(question.choices.first?.name == "o0")
        #expect(question.choices.last?.name == "o254")
        #expect(question.choices.allSatisfy { $0.description.isEmpty })
    }

    @Test("256 options are refused")
    func tooManyOptions() {
        let error = buildError(["q": Self.choice(options: 256)])
        #expect(error?.message == "Too many choices. Must have at most 255 choices.")
        #expect(error?.loc == ["body", "questions", "q", "criteria"])
    }

    @Test("The option limit is the smaller of maxChoices and the label count")
    func optionLimit() throws {
        let fewLabels = QuestionSchemaBuilder(choiceLabels: ["A", "B", "C"])
        #expect(throws: Self.criteriaError("Too many choices. Must have at most 3 choices.")) {
            try fewLabels.build(["q": Self.choice(options: 4)])
        }
        let three = try fewLabels.build(["q": Self.choice(options: 3)])
        #expect(three.questions.first?.labels == ["A", "B", "C"])
        let lowLimit = QuestionSchemaBuilder(choiceLabels: Self.labels, maxChoices: 2)
        #expect(throws: Self.criteriaError("Too many choices. Must have at most 2 choices.")) {
            try lowLimit.build(["q": Self.choice(options: 3)])
        }
    }

    @Test("An empty choice is refused with its key")
    func emptyChoice() {
        let error = buildError(["it's": .choice(instructions: nil, criteria: [:])])
        #expect(error?.message == "Choice question must have at least one choice: it's")
        #expect(error?.loc == ["body", "questions", "it's", "criteria"])
    }

    // MARK: Scores

    @Test("A single-level score is forced with its level as the legend")
    func singleLevel() throws {
        let level: JSONValue = ["kind": "calm"]
        let schema = try build(["s": .score(instructions: nil, criteria: [level])])
        #expect(schema.questions.isEmpty)
        #expect(
            try encoder.string(#require(schema.forced["s"]))
                == #"{"type":"score","score":0.0,"legend":{"0":{"kind":"calm"}},"#
                + #""probabilities":{"0":1.0},"confidence":1.0}"#)
    }

    @Test("10 levels are read under 0 to 9 and keep their legend")
    func mostLevels() throws {
        let schema = try build(["s": Self.score(levels: 10)])
        let question = try #require(schema.questions.first)
        #expect(question.labels == (0..<10).map { String($0) })
        #expect(question.choices.map(\.name) == question.labels)
        #expect(question.choices.map(\.description) == (0..<10).map { "level \($0)" })
        #expect(question.legend == (0..<10).map { .string("level \($0)") })
    }

    @Test("11 levels are refused")
    func tooManyLevels() {
        let error = buildError(["s": Self.score(levels: 11)])
        #expect(error?.message == "Too many score levels. Must have at most 10 levels.")
        #expect(error?.loc == ["body", "questions", "s", "criteria"])
    }

    // MARK: Nouls

    @Test("A noul without criteria reads yes and no without descriptions")
    func noulWithoutCriteria() throws {
        let question = try #require(
            build(["n": .noul(instructions: nil, criteria: nil)]).questions.first)
        #expect(question.labels == ["yes", "no"])
        #expect(question.choices.map(\.name) == ["yes", "no"])
        #expect(question.choices.map(\.description) == ["", ""])
        #expect(question.legend == nil)
        #expect(question.instructions == "")
    }

    @Test("A noul with one side described leaves the other empty")
    func noulOneSide() throws {
        let onlyTrue = try #require(
            build(["n": .noul(instructions: nil, criteria: NoulCriteria(whenTrue: " Urgent "))])
                .questions.first)
        #expect(onlyTrue.choices.map(\.description) == ["Urgent", ""])
        let onlyFalse = try #require(
            build(["n": .noul(instructions: nil, criteria: NoulCriteria(whenFalse: ["a", 1]))])
                .questions.first)
        #expect(onlyFalse.choices.map(\.description) == ["", #"["a", 1]"#])
    }

    // MARK: Ids and format

    @Test("Ids skip forced questions and follow request order")
    func ids() throws {
        let schema = try build([
            "forced": Self.choice(options: 1),
            "first": .noul(instructions: "One", criteria: nil),
            "level": Self.score(levels: 1),
            "second": Self.choice(options: 2),
        ])
        #expect(schema.questions.map(\.key) == ["first", "second"])
        #expect(schema.questions.map(\.id) == ["q1", "q2"])
        #expect(schema.forced.keys == ["forced", "level"])
        #expect(schema.questions[0].instructions == "One")
    }

    @Test("The first failing question in request order throws")
    func firstErrorWins() {
        let error = buildError([
            "fine": Self.choice(options: 2),
            "levels": Self.score(levels: 11),
            "empty": .choice(instructions: nil, criteria: [:]),
        ])
        #expect(error?.loc == ["body", "questions", "levels", "criteria"])
    }

    @Test("Ten read questions use lines and eleven use indexed")
    func format() throws {
        func nouls(_ count: Int) -> OrderedMap<Question> {
            OrderedMap(
                uniqueKeysWithValues: (0..<count).map {
                    ("n\($0)", Question.noul(instructions: nil, criteria: nil))
                })
        }
        #expect(try build(nouls(10)).format == .lines)
        #expect(try build(nouls(11)).format == .indexed)
        var withForced = nouls(10)
        withForced.updateValue(Self.choice(options: 1), forKey: "forced")
        #expect(try build(withForced).format == .lines)
    }

    // MARK: Encoder variant

    @Test("The encoder builder keeps the raw question and applies the same rules")
    func encoderBuilder() throws {
        let instructions: JSONValue = ["goal": "  route  "]
        let questions: OrderedMap<Question> = [
            "forced": Self.choice(options: 1),
            "c": .choice(instructions: instructions, criteria: ["a": " first ", "b": .null]),
            "s": Self.score(levels: 2),
        ]
        let schema = try EncoderQuestionSchemaBuilder(maxChoices: 255).build(questions)
        let reference = try build(questions)
        #expect(schema.forced == reference.forced)
        #expect(schema.questions.map(\.key) == ["c", "s"])
        let choice = schema.questions[0]
        #expect(choice.kind == .choice)
        #expect(choice.rawInstructions == instructions)
        #expect(choice.instructions == #"{"goal": "  route  "}"#)
        #expect(choice.question == questions["c"])
        #expect(choice.choices.map(\.name) == ["a", "b"])
        #expect(choice.choices.map(\.description) == ["first", ""])
        #expect(schema.questions[1].legend == ["level 0", "level 1"])
        #expect(throws: Self.criteriaError("Too many choices. Must have at most 2 choices.")) {
            try EncoderQuestionSchemaBuilder(maxChoices: 2).build(["q": Self.choice(options: 3)])
        }
        #expect(throws: Self.criteriaError("Too many score levels. Must have at most 10 levels.")) {
            try EncoderQuestionSchemaBuilder(maxChoices: 255).build(["q": Self.score(levels: 11)])
        }
    }
}

/// Checks ``TextOf`` against Python's `text_of`.
@Suite("TextOf")
struct TextOfTests {
    @Test("Absent and null render as an empty string")
    func empty() {
        #expect(TextOf.render(nil) == "")
        #expect(TextOf.render(.null) == "")
    }

    @Test(
        "Strings are stripped as Python's str.strip() strips them",
        arguments: [
            ("\u{00A0}x y\u{00A0}", "x y"),
            ("\tx\n", "x"),
            ("\u{3000}x\u{3000}", "x"),
            ("\u{1C}\u{1F}\u{85}x\u{2028}\u{202F}", "x"),
            ("\u{200B}x\u{200B}", "\u{200B}x\u{200B}"),
            ("\u{FEFF}x", "\u{FEFF}x"),
            (" \u{2000}\u{200A} ", ""),
        ]
    )
    func stripping(input: String, expected: String) {
        #expect(Array(TextOf.render(.string(input)).utf8) == Array(expected.utf8))
    }

    @Test("Every scalar CPython's isspace() accepts, and no other, is whitespace")
    func whitespaceTable() {
        // CPython 3.14.7: [c for c in range(0x110000) if chr(c).isspace()]
        let python: Set<UInt32> = [
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680,
            0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009,
            0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
        ]
        let swift = Set(
            (UInt32(0)...0x10FFFF).lazy.compactMap(Unicode.Scalar.init)
                .filter(TextOf.isPythonWhitespace).map(\.value))
        #expect(swift == python)
    }

    @Test("A string is stripped scalar by scalar, not by grapheme")
    func combiningMark() {
        // " \u{301}x": the space and the accent form one grapheme, but Python strips the space.
        #expect(Array(TextOf.render(.string(" \u{301}x ")).utf8) == Array("\u{301}x".utf8))
    }

    @Test("Objects, arrays and other values render as json.dumps(ensure_ascii=False)")
    func structured() {
        #expect(
            TextOf.render(["b": "é", "a": [1, 2.5, nil, true]])
                == #"{"b": "é", "a": [1, 2.5, null, true]}"#)
        #expect(TextOf.render([]) == "[]")
        #expect(TextOf.render(.integer("3")) == "3")
        #expect(TextOf.render(.float(1.0)) == "1.0")
        #expect(TextOf.render(.bool(false)) == "false")
    }
}

/// Checks the strings of ``AnswerFormat`` against upstream's `FORMATS`.
@Suite("AnswerFormat")
struct AnswerFormatTests {
    @Test("The lines format")
    func lines() {
        #expect(AnswerFormat.lines.join == "\n")
        #expect(AnswerFormat.lines.leadTemplate == "{id}: ")
        #expect(AnswerFormat.lines.lead(id: "q7") == "q7: ")
        #expect(
            AnswerFormat.lines.instruction
                == "Reply with one line per question, in this order, formatted as \"id: label\".")
    }

    @Test("The indexed format")
    func indexed() {
        #expect(AnswerFormat.indexed.join == " ")
        #expect(AnswerFormat.indexed.leadTemplate == "{id}")
        #expect(AnswerFormat.indexed.lead(id: "q7") == "q7")
        #expect(
            AnswerFormat.indexed.instruction
                == "Reply on one line with each question's id immediately followed by its label, "
                + "separated by single spaces.")
    }
}
