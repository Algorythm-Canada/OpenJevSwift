import OpenJevCore
import Testing

@testable import OpenJevEncoders

/// ``LayaSequence`` against laya's `build_sequence`: the recorded sequences rebuilt from their
/// pieces, and each budget on sequences built to reach it.
@Suite("Laya sequence")
struct LayaSequenceTests {
    /// The ids laya's checkpoint uses.
    private let special = LayaSpecialTokens(
        classToken: 50_281, separator: 50_282, mask: 50_284, padding: 50_283)

    /// A tokenizer with one token per Unicode scalar, its value: lengths are easy to count.
    private struct ScalarTokenizer: LayaTokenizing {
        let specialTokens = LayaSpecialTokens(
            classToken: 50_281, separator: 50_282, mask: 50_284, padding: 50_283)
        func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    }

    /// `count` distinct ids that are not special.
    private func tokens(_ count: Int, from start: Int = 1000) -> [Int] {
        Array(start..<(start + count))
    }

    @Test(
        "The recorded pieces rebuild every recorded sequence and its markers",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func recordedSequences() throws {
        let reference = try LayaFixtures.reference()
        let tokenizer = LayaReplayTokenizer(reference)
        let byRequest = try LayaFixtures.readsByRequest()
        var mismatches: [String] = []
        var compared = 0
        for corpus in try LayaFixtures.corpus() {
            let reads = try #require(byRequest[corpus.name])
            let stateIDs = tokenizer.encode(LayaPrompt.stateText(corpus.request.state))
            for (question, read) in zip(try LayaFixtures.questions(of: corpus), reads) {
                let sequence = LayaSequence(
                    prompt: LayaPrompt(question: question), stateIDs: stateIDs,
                    tokenizer: tokenizer, maxLength: reference.maxLength,
                    headMaxLength: reference.headMaxLength)
                if sequence.ids != read.ids || sequence.markers != read.markers {
                    mismatches.append(read.name)
                }
                compared += 1
            }
        }
        #expect(compared == 200)
        #expect(mismatches.isEmpty, "\(mismatches.count) differ: \(mismatches.prefix(10))")
        // The corpus reaches the head's cut and the 48-token cap, and 25 states are cut.
        #expect(reference.reads.filter(\.truncated).count == 25)
        #expect(reference.reads.contains { $0.pieces.options.contains { $0.count == 48 } })
        #expect(reference.reads.contains { $0.pieces.head.count == 238 })
    }

    @Test("The sequence is [CLS] head [SEP], each option after its [MASK], [SEP], state, [SEP]")
    func layout() {
        let sequence = LayaSequence.build(
            head: [1, 2], options: [[3], [4, 5]], state: [6, 7, 8], special: special,
            maxLength: 1024, headMaxLength: 256)
        #expect(
            sequence.ids == [50_281, 1, 2, 50_282, 50_284, 3, 50_284, 4, 5, 50_282, 6, 7, 8, 50_282])
        #expect(sequence.markers == [4, 6])
    }

    @Test("An option keeps at most 48 tokens after its marker")
    func optionCap() {
        let sequence = LayaSequence.build(
            head: [1], options: [tokens(60), tokens(48, from: 2000)], state: [], special: special,
            maxLength: 1024, headMaxLength: 256)
        #expect(sequence.markers == [3, 52])
        #expect(Array(sequence.ids[4..<52]) == tokens(48))
        #expect(Array(sequence.ids[53..<101]) == tokens(48, from: 2000))
        #expect(sequence.ids.count == 1 + 1 + 1 + 49 + 49 + 1 + 1)
    }

    @Test("Options that leave fewer than 16 tokens are cut to an equal share, the head to 16")
    func optionShare() {
        // Six options of 45 tokens take 276 of the 256; each is cut to (256 - 16) / 6 = 40
        // tokens with its marker, which leaves the head 16.
        let options = (0..<6).map { tokens(45, from: 1000 + 100 * $0) }
        let sequence = LayaSequence.build(
            head: tokens(30, from: 9000), options: options, state: tokens(5, from: 8000),
            special: special, maxLength: 1024, headMaxLength: 256)
        #expect(Array(sequence.ids[1..<17]) == tokens(16, from: 9000))
        #expect(sequence.markers == (0..<6).map { 18 + 40 * $0 })
        for (index, marker) in sequence.markers.enumerated() {
            #expect(Array(sequence.ids[(marker + 1)..<(marker + 40)]) == Array(options[index].prefix(39)))
        }
        #expect(sequence.ids.count == 1 + 16 + 1 + 240 + 1 + 5 + 1)
    }

    @Test("A share never goes below 4 tokens, and the head never below 8")
    func floors() {
        // 100 options of 10 tokens: (256 - 16) / 100 = 2, so each keeps 4; they take 400 of
        // the 256, and the head keeps 8.
        let options = (0..<100).map { tokens(10, from: 1000 + 20 * $0) }
        let sequence = LayaSequence.build(
            head: tokens(20, from: 9000), options: options, state: [], special: special,
            maxLength: 1024, headMaxLength: 256)
        #expect(Array(sequence.ids[1..<9]) == tokens(8, from: 9000))
        #expect(sequence.markers.count == 100)
        #expect(zip(sequence.markers, sequence.markers.dropFirst()).allSatisfy { $1 - $0 == 4 })
        // Short options that fit leave the head its budget, however short the head is.
        let short = LayaSequence.build(
            head: tokens(3), options: [[1], [2]], state: [], special: special, maxLength: 1024,
            headMaxLength: 256)
        #expect(short.markers == [5, 7])
    }

    @Test("The state fills what is left of the maximum length, cut on the right")
    func stateRoom() {
        let sequence = LayaSequence.build(
            head: [1, 2], options: [[3]], state: tokens(2000), special: special, maxLength: 1024,
            headMaxLength: 256)
        #expect(sequence.ids.count == 1024)
        #expect(sequence.ids.last == special.separator)
        // [CLS] 1 2 [SEP] [MASK] 3 [SEP]: seven tokens, then the state, then [SEP].
        #expect(Array(sequence.ids[7..<1023]) == tokens(1016))
        let fits = LayaSequence.build(
            head: [1], options: [[2]], state: tokens(10), special: special, maxLength: 1024,
            headMaxLength: 256)
        #expect(fits.ids.count == 1 + 1 + 1 + 2 + 1 + 10 + 1)
    }

    @Test("A sequence is cut at the maximum length, and the markers past it are dropped")
    func overflow() {
        // 255 options keep 4 tokens each: 1,020 tokens with the head's 8 and three separators
        // pass 1,024, so the last markers are lost, which the backend refuses.
        let options = (0..<255).map { tokens(10, from: 1000 + 20 * $0) }
        let sequence = LayaSequence.build(
            head: tokens(20, from: 9000), options: options, state: tokens(50, from: 30_000),
            special: special, maxLength: 1024, headMaxLength: 256)
        #expect(sequence.ids.count == 1024)
        // Markers at 10 + 4k: the 255th, at 1,026, is past the end.
        #expect(sequence.markers.count == 254)
        #expect(sequence.markers.allSatisfy { $0 < 1024 })
        #expect(!sequence.ids.contains(30_000))  // the state got no room
        let error = LayaSequence.overflowError(model: "laya-1.0", headMaxLength: 256)
        #expect(
            error.message
                == "Too many choices for laya-1.0: a question's options must fit in 256 tokens.")
        #expect(error.loc == ["body"])
    }

    @Test("A prompt is tokenized into its head and its options, with their leading spaces")
    func fromPrompt() throws {
        let questions = try EncoderQuestionSchemaBuilder(maxChoices: 255).build([
            "c": .choice(instructions: .string("x"), criteria: ["a": .null, "bc": .string("d")])
        ]).questions
        let tokenizer = ScalarTokenizer()
        let prompt = LayaPrompt(question: questions[0])
        let sequence = LayaSequence(
            prompt: prompt, stateIDs: tokenizer.encode("S"), tokenizer: tokenizer,
            maxLength: 1024, headMaxLength: 256)
        let head = tokenizer.encode("choice question: x")
        #expect(
            sequence.ids
                == [50_281] + head + [50_282, 50_284] + tokenizer.encode(" a") + [50_284]
                + tokenizer.encode(" bc: d") + [50_282] + tokenizer.encode("S") + [50_282])
        #expect(sequence.markers == [head.count + 2, head.count + 5])
    }

    @Test("Python's floor division rounds toward minus infinity")
    func floorDivision() {
        #expect(LayaSequence.floorDivision(240, 6) == 40)
        #expect(LayaSequence.floorDivision(240, 7) == 34)
        #expect(LayaSequence.floorDivision(-6, 4) == -2)
        #expect(LayaSequence.floorDivision(-8, 4) == -2)
        #expect(LayaSequence.floorDivision(7, -2) == -4)
        #expect(LayaSequence.floorDivision(0, 5) == 0)
    }
}
