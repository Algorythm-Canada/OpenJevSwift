import Foundation
import OpenJevCore
import OpenJevLetterReadout
import Testing

/// ``JevK5Prompt`` and ``JevK5Option`` against what upstream's `JevK5Engine.read_question` sent
/// for the fixture's requests (Fixtures/jevk5/reads.json): every option text and every pass's
/// prompt, byte for byte.
@Suite(
    "JevK5 prompt", .enabled(if: JevK5Fixtures.available, "Fixtures/jevk5/reads.json is missing"))
struct JevK5PromptTests {
    /// The read questions of a request, as the engine hands them to the backend.
    private static func questions(_ request: JevK5Fixtures.Request) throws -> [EncoderQuestion] {
        try EncoderQuestionSchemaBuilder(maxChoices: 255).build(request.request.questions).questions
    }

    @Test("decision_options gives the recorded options for every question")
    func options() throws {
        let reference = try JevK5Fixtures.reference()
        var compared = 0
        for request in reference.requests {
            let reads = reference.reads(of: request.name)
            let questions = try Self.questions(request)
            #expect(questions.map(\.key) == reads.map(\.key), "\(request.name): read questions")
            for (question, read) in zip(questions, reads) {
                #expect(
                    JevK5Option.options(for: question.question) == read.options,
                    "\(request.name).\(read.key)")
                compared += 1
            }
        }
        #expect(compared == reference.reads.count)
        #expect(compared >= 200)
    }

    @Test("Every pass's prompt is the jevk5 package's, byte for byte")
    func prompts() throws {
        let reference = try JevK5Fixtures.reference()
        var compared = 0
        var withText = 0
        for request in reference.requests {
            let reads = reference.reads(of: request.name)
            for (question, read) in zip(try Self.questions(request), reads) {
                for (index, pass) in read.passes.enumerated() {
                    let prompt = JevK5Prompt.text(
                        state: request.request.state, criterion: question.rawInstructions,
                        options: pass.texts)
                    let label = "\(request.name).\(read.key) pass \(index)"
                    if let recorded = pass.prompt {
                        #expect(prompt == recorded, "\(label)")
                        withText += 1
                    }
                    #expect(prompt.unicodeScalars.count == pass.characters, "\(label): length")
                    #expect(JevK5Fixtures.sha256(prompt) == pass.promptSHA256, "\(label): digest")
                    compared += 1
                }
            }
        }
        #expect(compared == reference.passes.count)
        #expect(withText > 250)
    }

    @Test("A question of at most 16 options is one pass over all of them")
    func onePassUpTo16() throws {
        let reference = try JevK5Fixtures.reference()
        for read in reference.reads where read.options.count <= 16 {
            #expect(read.passes.count == 1, "\(read.request).\(read.key)")
            #expect(read.passes.first?.texts == read.options.map(\.text))
        }
    }

    @Test("Python's str() of the values json.loads makes")
    func pythonStr() {
        let value: JSONValue = [
            "name": "it's", "say": "\"hi\"", "both": "'\"", "n": 1, "x": 1.0, "big": 1e16,
            "small": 0.1, "flag": true, "off": false, "none": .null, "list": [1, "a", [2.5]],
            "empty": [:], "é": "ü\n\t",
        ]
        #expect(
            JevK5Option.options(for: .choice(instructions: nil, criteria: ["k": value]))[0].text
                == "k: {'name': \"it's\", 'say': '\"hi\"', 'both': '\\'\"', 'n': 1, 'x': 1.0, "
                + "'big': 1e+16, 'small': 0.1, 'flag': True, 'off': False, 'none': None, "
                + "'list': [1, 'a', [2.5]], 'empty': {}, 'é': 'ü\\n\\t'}")
    }

    @Test("A false description falls back as `v or k` does")
    func falseDescriptions() {
        let choice = JevK5Option.options(
            for: .choice(
                instructions: nil,
                criteria: [
                    "a": .null, "b": "", "c": [:], "d": [], "e": 0, "f": 0.0, "g": false, "h": "0",
                    "i": 2,
                ]))
        #expect(
            choice.map(\.text) == [
                "a: a", "b: b", "c: c", "d: d", "e: e", "f: f", "g: g", "h: 0", "i: 2",
            ])
        let noul = JevK5Option.options(
            for: .noul(
                instructions: nil, criteria: NoulCriteria(whenTrue: "", whenFalse: ["no": 1])))
        #expect(
            noul == [
                JevK5Option(id: "true", text: "true: The proposition is true."),
                JevK5Option(id: "false", text: "false: {'no': 1}"),
            ])
        // A score's level is written as it is, even when false.
        let score = JevK5Option.options(
            for: .score(instructions: nil, criteria: ["", "low", ["x": .null]]))
        #expect(score.map(\.text) == ["0: ", "1: low", "2: {'x': None}"])
    }

    @Test("The criterion and the evidence go in as sent")
    func rawValues() {
        let prompt = JevK5Prompt.user(
            state: ["a": [1, 2.5]], criterion: ["check": "urgent"], options: ["true: yes"])
        #expect(
            prompt
                == #"{"evidence": {"a": [1, 2.5]}, "criterion": {"check": "urgent"}, "options": "#
                + #"[{"letter": "A", "description": "true: yes"}]}"#)
        // Absent instructions are null, and a string state is not stripped.
        #expect(
            JevK5Prompt.user(state: "  x  ", criterion: nil, options: [])
                == #"{"evidence": "  x  ", "criterion": null, "options": []}"#)
        #expect(
            JevK5Prompt.text(state: "s", criterion: "c", options: ["a"])
                == "<|im_start|>system\n" + JevK5Prompt.system + "<|im_end|>\n<|im_start|>user\n"
                + #"{"evidence": "s", "criterion": "c", "options": [{"letter": "A", "description": "a"}]}"#
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }
}
