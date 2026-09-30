import Foundation
import OpenJevCore
import Testing

@Suite("Answer assembly")
struct AnswerAssemblyTests {
    private let encoder = WireEncoder()

    @Test(
        "Every recorded answer is rebuilt from its question and probabilities byte for byte",
        .enabled(if: WireFixtures.exists("answers.json"), WireFixtures.missingMessage))
    func recordedAnswers() throws {
        let rows = try #require(WireFixtures.load("answers.json")["answers"]?.arrayValue)
        #expect(rows.count >= 15)
        for row in rows {
            let name = row["name"]?.stringValue ?? "?"
            let question = try Question(json: #require(row["question"]))
            let probabilities = try #require(row["probabilities"]?.arrayValue).map {
                try #require($0.doubleValue)
            }
            let answer = Answer.make(for: question, probabilities: probabilities)
            #expect(try encoder.string(answer) == row["body_text"]?.stringValue, "\(name)")
        }
    }

    @Test(
        "The recorded full response is rebuilt, forced answers included",
        .enabled(if: WireFixtures.exists("answers.json"), WireFixtures.missingMessage))
    func recordedResponse() throws {
        let fixture = try #require(WireFixtures.load("answers.json")["response"])
        let questions = try #require(fixture["questions"]?.objectValue)
        let reads = try #require(fixture["reads"]?.objectValue)
        var answers = OrderedMap<Answer>()
        for (id, json) in questions {
            let question = try Question(json: json)
            // A question without a read is one upstream forces: its single option has
            // probability 1.
            var probabilities = [1.0]
            if let read = reads[id] {
                let values = try #require(read.arrayValue)
                probabilities = values.compactMap(\.doubleValue)
                #expect(probabilities.count == values.count, "\(id)")
            }
            let answer = Answer.make(for: question, probabilities: probabilities)
            answers.updateValue(answer, forKey: id)
        }
        let response = SystemOneResponse(
            model: "openjev-0.1", answers: answers, usage: Usage(inputTokens: 123, outputTokens: 0))
        #expect(try encoder.string(response) == fixture["body_text"]?.stringValue)
    }

    @Test("A tie picks the earliest option")
    func tie() {
        let question = Question.choice(
            instructions: nil, criteria: ["a": nil, "b": nil, "c": nil])
        let answer = Answer.make(for: question, probabilities: [0.25, 0.375, 0.375])
        guard case .choice(let choice, let probabilities, _) = answer else {
            Issue.record("not a choice answer")
            return
        }
        #expect(choice == "b")
        #expect(probabilities.keys == ["a", "b", "c"])
        #expect(probabilities.values == [0.25, 0.375, 0.375])
    }

    @Test("A score is the expected level and keeps its levels as sent")
    func score() {
        let levels: [JSONValue] = ["low", ["text": "high"]]
        let answer = Answer.make(
            for: .score(instructions: nil, criteria: levels), probabilities: [0.25, 0.75])
        #expect(
            answer
                == .score(
                    score: 0.75, legend: levels, probabilities: [0.25, 0.75],
                    confidence: Confidence.compute([0.25, 0.75])))
    }

    @Test("A noul answer is the probability of yes")
    func noul() {
        let answer = Answer.make(
            for: .noul(instructions: nil, criteria: nil), probabilities: [0.7, 0.3])
        #expect(answer == .noul(0.7))
    }

    @Test("Single-option questions read as certain give upstream's forced answers")
    func forced() throws {
        let choice = Answer.make(
            for: .choice(instructions: nil, criteria: ["only": nil]), probabilities: [1.0])
        let forcedChoice =
            #"{"type":"choice","choice":"only","probabilities":{"only":1.0},"confidence":1.0}"#
        #expect(try encoder.string(choice) == forcedChoice)
        let score = Answer.make(
            for: .score(instructions: nil, criteria: ["one"]), probabilities: [1.0])
        let forcedScore =
            #"{"type":"score","score":0.0,"legend":{"0":"one"},"#
            + #""probabilities":{"0":1.0},"confidence":1.0}"#
        #expect(try encoder.string(score) == forcedScore)
    }
}

@Suite("Confidence")
struct ConfidenceTests {
    @Test("Jev's documented example is about 0.596")
    func documentedExample() {
        let value = Confidence.compute([0.84, 0.159, 0.001])
        #expect(abs(value - 0.596) < 0.01)
        // The exact value CPython 3.14 gives for upstream's `confidence`.
        #expect(value == 0.5942682179296132)
    }

    @Test("Certain is 1, uniform is 0, one option is 1")
    func bounds() {
        #expect(Confidence.compute([1, 0, 0]) == 1.0)
        #expect(Confidence.compute([0.5, 0.5]) == 0.0)
        #expect(Confidence.compute(Array(repeating: 1.0 / 255.0, count: 255)) == 0.0)
        #expect(Confidence.compute([1.0]) == 1.0)
        #expect(Confidence.compute([]) == 1.0)
    }

    @Test("Confidence stays in [0, 1] for random distributions")
    func propertyRange() {
        var generator = SeededGenerator(state: 16)
        for _ in 0..<2000 {
            let count = Int.random(in: 2...255, using: &generator)
            var weights = (0..<count).map { _ in Double.random(in: 0..<1, using: &generator) }
            // Some vectors are sparse, so exact zeros are exercised too.
            if Bool.random(using: &generator) {
                for index in weights.indices {
                    if Double.random(in: 0..<1, using: &generator) < 0.8 {
                        weights[index] = 0
                    }
                }
                weights[0] += 1e-3
            }
            let total = weights.reduce(0, +)
            let value = Confidence.compute(weights.map { $0 / total })
            #expect(value >= 0 && value <= 1, "\(count) options gave \(value)")
        }
    }

    @Test(
        "Every recorded confidence matches",
        .enabled(if: DistributionFixtures.exists, DistributionFixtures.missingMessage))
    func recorded() throws {
        let cases = try DistributionFixtures.load().confidences
        for entry in cases {
            #expect(Confidence.compute(entry.probabilities) == entry.confidence, "\(entry.name)")
        }
    }
}

@Suite("Slot distribution")
struct SlotDistributionTests {
    @Test("A missing label gets the floor, the smallest top logprob less 5")
    func floor() {
        let top: [(tokenID: Int, logprob: Double)] = [(5, -0.1), (9, -2.5), (11, -4.0)]
        let result = SlotDistribution.compute(top: top, labelIDs: [9, 5, 77])
        // Label 77 is missing and gets -4.0 - 5.0 = -9.0.
        let logprobs = [-2.5, -0.1, -9.0]
        let exponentials = logprobs.map { exp($0 - -0.1) }
        let total = exponentials.reduce(0, +)
        for (got, want) in zip(result.probabilities, exponentials.map { $0 / total }) {
            #expect(abs(got - want) < 1e-15)
        }
        // The exact values CPython 3.14 gives for upstream's `slot_distribution`.
        let python = [0.08316229745681136, 0.9167126730858141, 0.00012502945737462924]
        #expect(result.probabilities == python)
        #expect(result.entropy == 0.3689587939182797)
    }

    @Test("The entropy is over the raw top-k set, not renormalised")
    func unnormalisedEntropy() {
        // The two entries hold 0.5 and 0.3 of the mass; 0.2 is outside the top-k.
        let result = SlotDistribution.compute(
            top: [(1, log(0.5)), (2, log(0.3))], labelIDs: [1, 2])
        let byHand = -(0.5 * log(0.5) + 0.3 * log(0.3))
        #expect(abs(result.entropy - byHand) < 1e-15)
        let renormalised = -(0.625 * log(0.625) + 0.375 * log(0.375))
        #expect(abs(result.entropy - renormalised) > 0.04)
        #expect(result.entropy == 0.7077654315777535)
        #expect(result.probabilities == [0.625, 0.37499999999999994])
    }

    @Test("The dictionary overload gives the same result")
    func dictionaryOverload() {
        let ordered = SlotDistribution.compute(top: [(5, -0.1), (9, -2.5)], labelIDs: [5, 9])
        let unordered = SlotDistribution.compute(top: [5: -0.1, 9: -2.5], labelIDs: [5, 9])
        #expect(ordered.probabilities == unordered.probabilities)
        #expect(abs(ordered.entropy - unordered.entropy) < 1e-15)
    }

    @Test("Probabilities sum to 1 and confidence stays in [0, 1] for random slots")
    func propertySum() {
        var generator = SeededGenerator(state: 6)
        for _ in 0..<2000 {
            let labelCount = Int.random(in: 1...255, using: &generator)
            let labelIDs = Array(1000..<(1000 + labelCount))
            var top: [(tokenID: Int, logprob: Double)] = []
            for tokenID in 0..<20 {
                top.append((tokenID, -Double.random(in: 0..<30, using: &generator)))
            }
            // A random share of the labels appears among the returned logprobs.
            for id in labelIDs where Bool.random(using: &generator) {
                top.append((id, -Double.random(in: 0..<40, using: &generator)))
            }
            let result = SlotDistribution.compute(top: top, labelIDs: labelIDs)
            let total = result.probabilities.reduce(0, +)
            #expect(abs(total - 1) < 1e-12, "sum \(total)")
            #expect(result.probabilities.allSatisfy { $0 >= 0 && $0 <= 1 })
            #expect(result.entropy >= 0)
            let confidence = Confidence.compute(result.probabilities)
            #expect(confidence >= 0 && confidence <= 1)
        }
    }

    @Test(
        "Every recorded slot distribution matches",
        .enabled(if: DistributionFixtures.exists, DistributionFixtures.missingMessage))
    func recorded() throws {
        let slots = try DistributionFixtures.load().slots
        for slot in slots {
            let result = SlotDistribution.compute(top: slot.top, labelIDs: slot.labelIDs)
            #expect(result.probabilities == slot.probabilities, "\(slot.name)")
            #expect(result.entropy == slot.entropy, "\(slot.name)")
        }
    }
}

@Suite("Read averaging")
struct ReadAveragingTests {
    @Test("Each label is averaged over the reads with Python's sum")
    func mean() {
        // CPython 3.14: sum([0.1] * 10) / 10 == 0.1, where a plain running total gives
        // 0.09999999999999999.
        #expect(ReadAveraging.mean(Array(repeating: [0.1, 0.9], count: 10))[0] == 0.1)
        let reads = [[0.3, 0.7], [0.3, 0.1], [0.3, 0.2]]
        #expect(ReadAveraging.mean(reads) == [0.3, 0.3333333333333333])
        #expect(ReadAveraging.mean([[0.25, 0.75]]) == [0.25, 0.75])
    }
}
