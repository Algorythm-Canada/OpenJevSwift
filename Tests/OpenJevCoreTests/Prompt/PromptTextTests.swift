import OpenJevCore
import OpenJevTestSupport
import Testing

/// Compares ``SystemText``, ``AnswerText`` and ``StateText`` with upstream's `system_text`,
/// `answer_text` and state text in Fixtures/system-texts, Fixtures/templates and
/// Fixtures/tokenizer, and checks the line shapes by hand.
@Suite("Prompt text")
struct PromptTextTests {
    // MARK: Fixtures

    /// True when the text has exactly the UTF-8 bytes of the recorded string.
    private static func sameBytes(_ text: String, _ recorded: JSONValue?) throws -> Bool {
        try Array(text.utf8) == Array(#require(recorded?.stringValue).utf8)
    }

    @Test(
        "Every recorded system text is reproduced byte for byte",
        .enabled(
            if: UpstreamFixtures.exists("system-texts/system_texts.json", "labels.json"),
            UpstreamFixtures.missingMessage))
    func recordedSystemTexts() throws {
        let labels = try UpstreamFixtures.choiceLabels()
        var failures: [String] = []
        var compared = 0
        for row in try UpstreamFixtures.cases("system-texts/system_texts.json") {
            let name = row["name"]?.stringValue ?? "?"
            do {
                let request = try #require(row["request"])
                let schema = try UpstreamFixtures.schema(for: request, labels: labels)
                if row["format"]?.stringValue != schema.format.rawValue {
                    failures.append("\(name): format \(schema.format)")
                }
                let groups = try #require(row["groups"]?.arrayValue)
                for (index, group) in groups.enumerated() {
                    let questions = try UpstreamFixtures.questions(group["questions"], in: schema)
                    for chunked in [false, true] {
                        let key = chunked ? "chunked" : "unchunked"
                        let text = SystemText.render(
                            questions, format: schema.format, chunked: chunked)
                        if try !Self.sameBytes(text, group[key]) {
                            failures.append("\(name) group \(index) \(key): \(text)")
                        }
                        compared += 1
                    }
                }
                if let all = row["all_questions"] {
                    let text = SystemText.render(
                        schema.questions, format: schema.format, chunked: false)
                    if try !Self.sameBytes(text, all["unchunked"]) {
                        failures.append("\(name) all questions: \(text)")
                    }
                    compared += 1
                }
            } catch {
                failures.append("\(name): \(error)")
            }
        }
        #expect(failures.isEmpty, "\(failures.count) failures: \(failures.prefix(5))")
        #expect(compared == 58, "\(compared) texts compared")
    }

    @Test(
        "Every recorded answer text and alternative is reproduced byte for byte",
        .enabled(
            if: UpstreamFixtures.exists("templates/templates.json", "labels.json"),
            UpstreamFixtures.missingMessage))
    func recordedAnswerTexts() throws {
        let labels = try UpstreamFixtures.choiceLabels()
        var failures: [String] = []
        var compared = 0
        for row in try UpstreamFixtures.cases("templates/templates.json") {
            let name = row["name"]?.stringValue ?? "?"
            do {
                let request = try #require(row["request"])
                let schema = try UpstreamFixtures.schema(for: request, labels: labels)
                for group in try #require(row["groups"]?.arrayValue) {
                    let index = group["group"]?.intValue ?? -1
                    let questions = try UpstreamFixtures.questions(group["questions"], in: schema)
                    let expectedLabels = JSONValue.array(
                        questions.map { .array($0.labels.map { .string($0) }) })
                    if expectedLabels != group["labels"] {
                        failures.append("\(name) group \(index): labels")
                    }
                    let zeros = Array(repeating: 0, count: questions.count)
                    let text = AnswerText.render(
                        questions, labelIndices: zeros, format: schema.format)
                    if try !Self.sameBytes(text, group["answer_text"]) {
                        failures.append("\(name) group \(index): \(text)")
                    }
                    compared += 1
                    for alternative in try #require(group["alternatives"]?.arrayValue) {
                        var indices = zeros
                        let question = try #require(alternative["question"]?.intValue)
                        indices[question] = try #require(alternative["label"]?.intValue)
                        let text = AnswerText.render(
                            questions, labelIndices: indices, format: schema.format)
                        if try !Self.sameBytes(text, alternative["text"]) {
                            failures.append("\(name) group \(index) \(indices): \(text)")
                        }
                        compared += 1
                    }
                }
            } catch {
                failures.append("\(name): \(error)")
            }
        }
        #expect(failures.isEmpty, "\(failures.count) failures: \(failures.prefix(5))")
        #expect(compared == 972, "\(compared) answer texts compared")
    }

    @Test(
        "Every json_state text of the tokenizer corpus is a fixture state's StateText",
        .enabled(
            if: UpstreamFixtures.exists(
                "tokenizer/corpus.json", "schemas/schemas.json", "system-texts/system_texts.json",
                "templates/templates.json"),
            UpstreamFixtures.missingMessage))
    func recordedStateTexts() throws {
        var rendered: Set<[UInt8]> = []
        for file in [
            "schemas/schemas.json", "system-texts/system_texts.json", "templates/templates.json",
        ] {
            // The unknown-type rows of schemas.json have no request, so no state.
            for row in try UpstreamFixtures.cases(file) {
                if let state = row["request"]?["state"] {
                    rendered.insert(Array(StateText.render(state).utf8))
                }
            }
        }
        let corpus = try UpstreamFixtures.cases("tokenizer/corpus.json")
        let jsonStates = try corpus.filter {
            try #require($0["categories"]?.arrayValue).contains("json_state")
        }
        #expect(jsonStates.count == 3)
        for entry in jsonStates {
            let text = try #require(entry["text"]?.stringValue)
            #expect(rendered.contains(Array(text.utf8)), "no fixture state renders as \(text)")
        }
    }

    // MARK: System text

    private static let noul = ReadQuestion(
        key: "urgent", id: "q1", kind: .noul, instructions: "",
        choices: [("yes", "It is urgent"), ("no", "")], labels: ["yes", "no"], legend: nil)
    private static let choice = ReadQuestion(
        key: "team", id: "q2", kind: .choice, instructions: "Which team",
        choices: [("billing", "Payments"), ("sales", "")], labels: ["A", "B"], legend: nil)
    private static let score = ReadQuestion(
        key: "mood", id: "q3", kind: .score, instructions: "How angry",
        choices: [("0", "Calm"), ("1", "")], labels: ["0", "1"], legend: ["Calm", ""])

    @Test("Each question type writes its label lines, with and without descriptions")
    func lineShapes() {
        let text = SystemText.render(
            [Self.noul, Self.choice, Self.score], format: .lines, chunked: false)
        let expected =
            SystemText.opening
            + "\nQuestion q1: Answer about the state.\n  yes: It is urgent\n  no\n"
            + "\nQuestion q2: Which team\n  A: billing (Payments)\n  B: sales\n"
            + "\nQuestion q3: How angry\n  0: Calm\n  1: \n"
            + "\nReply with one line per question, in this order, formatted as \"id: label\"."
        #expect(Array(text.utf8) == Array(expected.utf8))
    }

    @Test("The chunked sentence appears only for chunked reads")
    func chunkedSentence() {
        for format in AnswerFormat.allCases {
            let plain = SystemText.render([Self.noul], format: format, chunked: false)
            let chunked = SystemText.render([Self.noul], format: format, chunked: true)
            #expect(plain.hasSuffix(format.instruction))
            #expect(!plain.contains("A reply may cover only some"))
            #expect(chunked == plain + SystemText.chunkedSentence)
        }
        #expect(
            SystemText.chunkedSentence
                == " A reply may cover only some of the questions; "
                + "answer every line that is present.")
    }

    @Test("Empty instructions read as the default text")
    func defaultInstructions() {
        #expect(SystemText.defaultInstructions == "Answer about the state.")
        let text = SystemText.render([Self.noul], format: .indexed, chunked: false)
        #expect(text.contains("\nQuestion q1: Answer about the state.\n"))
        #expect(
            text.hasPrefix(
                "Answer a fixed set of questions about the state the user provides. Each question "
                    + "lists its allowed answers; reply with exactly one label per question.\n"))
    }

    // MARK: Answer text

    @Test("Answer texts use the format's join and lead")
    func answerText() {
        let questions = [Self.choice, Self.score, Self.noul]
        #expect(
            AnswerText.render(questions, labelIndices: [0, 0, 0], format: .lines)
                == "q2: A\nq3: 0\nq1: yes")
        #expect(
            AnswerText.render(questions, labelIndices: [1, 1, 1], format: .indexed)
                == "q2B q31 q1no")
        #expect(AnswerText.render([], labelIndices: [], format: .lines) == "")
    }

    // MARK: State text

    @Test("A string state is sent as is, other states as json.dumps(ensure_ascii=False)")
    func stateText() {
        #expect(StateText.render("  Hi\n") == "  Hi\n")
        #expect(
            StateText.render(["name": "Zoë", "greeting": "你好 👋", "n": 1.0])
                == #"{"name": "Zoë", "greeting": "你好 👋", "n": 1.0}"#)
        #expect(
            StateText.render([1, "two", ["three": 3], nil, true])
                == #"[1, "two", {"three": 3}, null, true]"#)
    }
}
