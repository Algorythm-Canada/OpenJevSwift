import OpenJevCore
import OpenJevEncoders
import Testing

/// ``LayaPrompt`` against laya's `render_options`, `render_criterion` and `serialize_state` and the
/// texts its `build_sequence` tokenizes: every head, option and state of the reference corpus,
/// and the rules the corpus does not reach.
@Suite("Laya prompt")
struct LayaPromptTests {
    /// The read questions the schema builder makes of `questions`, in order.
    private func prompts(_ questions: OrderedMap<Question>) throws -> [LayaPrompt] {
        try EncoderQuestionSchemaBuilder(maxChoices: 255).build(questions).questions.map(
            LayaPrompt.init(question:))
    }

    @Test(
        "Every head and option of the reference corpus is the text laya tokenized",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func corpusTexts() throws {
        let reads = try LayaFixtures.readsByRequest()
        var compared = 0
        var mismatches: [String] = []
        for corpus in try LayaFixtures.corpus() {
            let recorded = try #require(reads[corpus.name], "no reads for \(corpus.name)")
            let questions = try LayaFixtures.questions(of: corpus)
            try #require(
                questions.map(\.key) == recorded.map(\.key), "\(corpus.name): question order")
            for (question, read) in zip(questions, recorded) {
                let prompt = LayaPrompt(question: question)
                if prompt.head != read.head || prompt.optionTexts != read.optionTexts
                    || prompt.questionType != read.qtype || prompt.options.count != read.options
                {
                    mismatches.append(read.name)
                }
                compared += 1
            }
        }
        #expect(compared == 200)
        #expect(mismatches.isEmpty, "\(mismatches.count) questions differ: \(mismatches.prefix(10))")
    }

    @Test(
        "Every state is the text laya tokenized, JSON states included",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func corpusStates() throws {
        let reference = try LayaFixtures.reference()
        var json = 0
        for corpus in try LayaFixtures.corpus() {
            let recorded = try #require(reference.stateTexts[corpus.name], "\(corpus.name)")
            #expect(LayaPrompt.stateText(corpus.request.state) == recorded, "\(corpus.name)")
            if corpus.request.state.stringValue == nil {
                json += 1
            }
        }
        #expect(reference.stateTexts.count == 26)
        #expect(json == 5)
    }

    @Test(
        "laya's question is what upstream hands laya: instructions as text, criteria as sent",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func internalQuestion() throws {
        let reads = try LayaFixtures.readsByRequest()
        for corpus in try LayaFixtures.corpus() {
            let recorded = try #require(reads[corpus.name])
            for (question, read) in zip(try LayaFixtures.questions(of: corpus), recorded) {
                let prompt = LayaPrompt(question: question)
                let sent = read.layaQuestion
                #expect(sent["type"]?.stringValue == prompt.kind.rawValue, "\(read.name)")
                #expect(sent["instructions"]?.stringValue == prompt.instructions, "\(read.name)")
                switch prompt.criteria {
                case .choice(let options):
                    #expect(sent["criteria"] == .object(options), "\(read.name)")
                case .score(let levels):
                    #expect(sent["criteria"] == .array(levels), "\(read.name)")
                case .noul(let whenTrue, let whenFalse):
                    // Upstream sends `criteria or {}`; laya reads only `true` and `false`.
                    let criteria = try #require(sent["criteria"]?.objectValue, "\(read.name)")
                    #expect(criteria["true"] == whenTrue, "\(read.name)")
                    #expect(criteria["false"] == whenFalse, "\(read.name)")
                }
            }
        }
    }

    @Test("A choice reads each name, with its description unless it is null or empty")
    func choiceOptions() throws {
        let prompt = try #require(
            try prompts([
                "c": .choice(
                    instructions: .string("  Which team?  "),
                    criteria: [
                        "none": .null, "empty": .string(""), "blank": .string("  "),
                        "text": .string(" kept as sent "),
                        "object": ["k": 1, "nested": ["z": [true, .null]]],
                        "array": [1, "a", 2.5], "floats": ["big": 1e16, "small": 1e-7],
                        "unicode": ["ключ": "🎉"], "zero": 0, "false": false,
                    ])
            ]).first)
        #expect(prompt.kind == .choice)
        #expect(prompt.questionType == 0)
        // The instructions are upstream's text_of: stripped.
        #expect(prompt.head == "choice question: Which team?")
        #expect(
            prompt.options == [
                "none", "empty", "blank:   ", "text:  kept as sent ",
                #"object: {"k": 1, "nested": {"z": [true, null]}}"#, #"array: [1, "a", 2.5]"#,
                #"floats: {"big": 1e+16, "small": 1e-07}"#, #"unicode: {"ключ": "🎉"}"#,
                "zero: 0", "false: false",
            ])
        #expect(prompt.optionTexts.first == " none")
    }

    @Test("A score reads every level by its index, a null one as null")
    func scoreLevels() throws {
        let prompt = try #require(
            try prompts([
                "s": .score(
                    instructions: ["scale": "1-3"],
                    criteria: [.string("  low  "), ["weight": 0.5], .null])
            ]).first)
        #expect(prompt.questionType == 1)
        #expect(prompt.head == #"score question: {"scale": "1-3"}"#)
        #expect(
            prompt.options == [#"level 0:   low  "#, #"level 1: {"weight": 0.5}"#, "level 2: null"])
    }

    @Test("A noul reads false then true, each with laya's phrase when it has no description")
    func noulOutcomes() throws {
        let read = try prompts([
            "plain": .noul(instructions: nil, criteria: nil),
            "nulls": .noul(
                instructions: .string("x"),
                criteria: NoulCriteria(whenTrue: .null, whenFalse: .string(""))),
            "described": .noul(
                instructions: .string("x"),
                criteria: NoulCriteria(whenTrue: ["means": "yes"], whenFalse: .string("  "))),
        ])
        #expect(read.map(\.questionType) == [2, 2, 2])
        #expect(read[0].head == "noul question: ")
        let defaults = [
            "false: " + LayaPrompt.defaultFalseDescription,
            "true: " + LayaPrompt.defaultTrueDescription,
        ]
        #expect(read[0].options == defaults)
        #expect(read[1].options == defaults)
        #expect(read[2].options == ["false:   ", #"true: {"means": "yes"}"#])
        #expect(defaults == ["false: no, the statement does not hold", "true: yes, the statement holds"])
    }

    @Test("Every [MASK] in the instructions, the options and the state becomes a space")
    func masksReplaced() throws {
        let prompt = try #require(
            try prompts([
                "c": .choice(
                    instructions: .string("Is [MASK] [MASK][MASK] here"),
                    criteria: ["[MASK]": .string("a [MASK] b"), "b": ["[MASK]": "[MASK]"]])
            ]).first)
        #expect(prompt.head == "choice question: Is      here")
        #expect(prompt.optionTexts == ["  : a   b", #" b: {" ": " "}"#])
        #expect(LayaPrompt.stateText(.string("[MASK]x[MASK]")) == " x ")
        #expect(LayaPrompt.stateText(["k": "[MASK]"]) == #"{"k": " "}"#)
        // Python compares code points: a combining mark after "]" does not hide the token, and
        // a partial token is left alone.
        #expect(LayaPrompt.replacingMasks(in: "[MASK]\u{301}") == " \u{301}")
        #expect(LayaPrompt.replacingMasks(in: "[[MASK]]") == "[ ]")
        #expect(LayaPrompt.replacingMasks(in: "[MASK") == "[MASK")
        #expect(LayaPrompt.replacingMasks(in: "") == "")
        #expect(LayaPrompt.replacingMasks(in: "naïve 🎉") == "naïve 🎉")
    }

    @Test("A state renders as serialize_state: a string as sent, anything else as JSON")
    func states() {
        #expect(LayaPrompt.stateText(.string("  kept  ")) == "  kept  ")
        #expect(LayaPrompt.stateText(["b": 1, "a": [1.5, "é"]]) == #"{"b": 1, "a": [1.5, "é"]}"#)
        #expect(LayaPrompt.stateText(["x", 1]) == #"["x", 1]"#)
        #expect(LayaPrompt.stateText(.null) == "null")
    }
}
