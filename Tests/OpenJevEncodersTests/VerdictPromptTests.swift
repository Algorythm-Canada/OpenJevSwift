import OpenJevCore
import OpenJevEncoders
import Testing

/// ``VerdictPrompt`` against upstream's `verdict_prompt`: its own unit test, and every prompt of
/// the reference corpus byte for byte.
@Suite("Verdict prompt")
struct VerdictPromptTests {
    /// The read questions the schema builder makes of `questions`, in order.
    private func questions(_ questions: OrderedMap<Question>) throws -> [EncoderQuestion] {
        try EncoderQuestionSchemaBuilder(maxChoices: 24).build(questions).questions
    }

    @Test("Matches upstream's test_verdict_prompt_matches_upstream_contract")
    func upstreamContract() throws {
        let read = try questions([
            "team": .choice(
                instructions: .string("Which team?"),
                criteria: ["outage": .string("service down"), "sales": .string("")]),
            "severity": .score(
                instructions: .string("How bad?"), criteria: [.string("low"), .string("high")]),
            "urgent": .noul(instructions: .string("It is urgent"), criteria: nil),
        ])
        let choice = VerdictPrompt(question: read[0], context: "ctx")
        #expect(
            choice.text
                == "<<LABEL>>It is service down<<LABEL>>It is sales<<LABEL>>insufficient evidence"
                + "<<SEP>>Question: Which team?\n\nContext:\nctx")
        #expect(choice.labelCount == 3)
        let score = VerdictPrompt(question: read[1], context: "ctx")
        #expect(
            score.text.hasPrefix(
                "<<LABEL>>low (Value: 0.0)<<LABEL>>high (Value: 1.0)<<LABEL>>"))
        let noul = VerdictPrompt(question: read[2], context: "ctx")
        #expect(
            noul.text
                == "<<LABEL>>true: It is urgent<<LABEL>>false: not It is urgent"
                + "<<LABEL>>insufficient evidence<<SEP>>Context:\nctx\n\n"
                + "Evaluate proposition: It is urgent")
        #expect(noul.labelCount == 3)
    }

    @Test("Empty instructions leave the context alone, and a noul still names them")
    func emptyInstructions() throws {
        let read = try questions([
            "c": .choice(instructions: nil, criteria: ["a": .null, "b": .string("  B  ")]),
            "s": .score(instructions: .string("   "), criteria: [.null, .string("high")]),
            "n": .noul(instructions: nil, criteria: nil),
        ])
        #expect(
            VerdictPrompt(question: read[0], context: "the state").text
                == "<<LABEL>>It is a<<LABEL>>It is B<<LABEL>>insufficient evidence<<SEP>>the state")
        #expect(
            VerdictPrompt(question: read[1], context: "the state").text
                == "<<LABEL>> (Value: 0.0)<<LABEL>>high (Value: 1.0)"
                + "<<LABEL>>insufficient evidence<<SEP>>the state")
        #expect(
            VerdictPrompt(question: read[2], context: "the state").text
                == "<<LABEL>>true: <<LABEL>>false: not <<LABEL>>insufficient evidence"
                + "<<SEP>>Context:\nthe state\n\nEvaluate proposition: ")
    }

    @Test("A noul's criteria are not read, and ten score levels count to 9.0")
    func noulCriteriaAndLevels() throws {
        let read = try questions([
            "plain": .noul(instructions: .string("x"), criteria: nil),
            "described": .noul(
                instructions: .string("x"),
                criteria: NoulCriteria(whenTrue: .string("yes it is"), whenFalse: .string("no"))),
            "levels": .score(
                instructions: .string("x"), criteria: (0..<10).map { .string("L\($0)") }),
        ])
        #expect(
            VerdictPrompt(question: read[0], context: "c")
                == VerdictPrompt(question: read[1], context: "c"))
        let levels = VerdictPrompt(question: read[2], context: "c")
        #expect(levels.labels.first == "L0 (Value: 0.0)")
        #expect(levels.labels.dropLast().last == "L9 (Value: 9.0)")
        #expect(levels.labels.last == VerdictPrompt.abstentionLabel)
        #expect(levels.labelCount == 11)
    }

    @Test(
        "Every prompt of the reference corpus matches upstream's byte for byte",
        .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage))
    func corpusParity() throws {
        let reads = try VerdictFixtures.readsByRequest()
        var compared = 0
        var mismatches: [String] = []
        for corpus in try VerdictFixtures.corpus() {
            let recorded = try #require(reads[corpus.name], "no reads for \(corpus.name)")
            let questions = try VerdictFixtures.questions(of: corpus)
            try #require(
                questions.map(\.key) == recorded.map(\.key), "\(corpus.name): question order")
            let context = StateText.render(corpus.request.state)
            for (question, read) in zip(questions, recorded) {
                let prompt = VerdictPrompt(question: question, context: context)
                if prompt.text != read.prompt || prompt.labelCount != read.k {
                    mismatches.append(read.name)
                }
                compared += 1
            }
        }
        #expect(compared == 200)
        #expect(mismatches.isEmpty, "\(mismatches.count) prompts differ: \(mismatches.prefix(10))")
    }
}
