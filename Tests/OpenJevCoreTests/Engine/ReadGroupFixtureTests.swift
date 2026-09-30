import Foundation
import OpenJevCore
import Testing

/// Replays the `read_group` rows of Fixtures/distributions/distributions.json: upstream's own
/// `read_group` over synthetic log-probability maps, through ``DecisionEngine`` with a
/// ``StubBackend`` that answers each seed from the recorded maps.
@Suite(
    "Read group fixtures",
    .enabled(if: DistributionFixtures.exists, DistributionFixtures.missingMessage),
    .enabled(if: FixtureTokenizer.exists, FixtureTokenizer.missingMessage))
struct ReadGroupFixtureTests {
    /// The tolerance docs/09 allows on averaged probabilities.
    private static let tolerance = 1e-12

    @Test("Every read_group row gives the recorded means, billing and answers")
    func readGroupRows() async throws {
        let document = try JSONParser().parse(Data(contentsOf: DistributionFixtures.file))
        let rows = try #require(document["read_group"]?.arrayValue)
        #expect(rows.count >= 6)
        for row in rows {
            try await Self.check(row)
        }
    }

    private static func check(_ row: JSONValue) async throws {
        let name = row["name"]?.stringValue ?? "?"
        var request = try RequestValidator().validate(row["request"])
        let options = try #require(row["options"])
        request.steps = options["steps"]?.intValue
        request.samples = options["samples"]?.intValue
        request.think = options["think"]?.intValue
        request.sequential = options["sequential"]?.boolValue
        let seed = UInt64(try #require(row["seed"]?.intValue))

        var scripted: [UInt64: [[(tokenID: Int, logprob: Double)]]] = [:]
        let reads = try #require(row["reads"]?.arrayValue)
        for read in reads {
            let readSeed = UInt64(try #require(read["seed"]?.intValue))
            scripted[readSeed] = try #require(read["tops"]?.arrayValue).map {
                try PolicyFixtures.tops($0)
            }
        }
        let stub = StubBackend(scripted: scripted, scriptedPromptTokens: 100)
        let engine = try DecisionEngine(backend: stub)
        let decision = try await engine.decide(request, seed: seed)

        #expect(decision.inputTokens == row["billed"]?.intValue, "\(name): billed")
        #expect(decision.outputTokens == 0, "\(name): thought tokens")
        #expect(stub.reads.count == reads.count, "\(name): read count")
        let labelIDs = try #require(row["label_ids"]?.arrayValue).map {
            try PolicyFixtures.ints($0)
        }
        for read in stub.reads {
            #expect(read.slots.map(\.labelIDs) == labelIDs, "\(name): label ids")
            #expect(scripted[read.seed] != nil, "\(name): unrecorded seed \(read.seed)")
        }

        let means = try #require(row["means"]?.arrayValue).map { mean in
            try #require(mean.arrayValue).map { try #require($0.doubleValue) }
        }
        let recorded = try #require(row["answers"]?.objectValue)
        // The same doubles through the same operations in the same order: the answers match
        // byte for byte (D-020), and the tolerance below is docs/09's allowance.
        let answersJSON = JSONValue.object(
            JSONObject(uniqueKeysWithValues: decision.answers.map { ($0.key, $0.value.json) }))
        #expect(
            try String(decoding: WireEncoder().bytes(json: answersJSON), as: UTF8.self)
                == row["answers_text"]?.stringValue,
            "\(name): answers text")
        #expect(decision.answers.keys == request.questions.keys, "\(name): answer order")
        #expect(recorded.keys == decision.answers.keys, "\(name): recorded answer order")
        var readIndex = 0
        for (key, answer) in decision.answers {
            let expected = try Answer(json: #require(recorded[key]))
            let probabilities = Self.probabilities(of: answer)
            #expect(probabilities.count == means[readIndex].count, "\(name) \(key): label count")
            for (value, mean) in zip(probabilities, means[readIndex]) {
                #expect(abs(value - mean) <= tolerance, "\(name) \(key): mean \(value) vs \(mean)")
            }
            #expect(Self.close(answer, expected), "\(name) \(key): \(answer) vs \(expected)")
            readIndex += 1
        }
    }

    /// The label probabilities an answer carries, in label order.
    private static func probabilities(of answer: Answer) -> [Double] {
        switch answer {
        case .noul(let p): return [p, 1 - p]
        case .choice(_, let probabilities, _): return probabilities.values
        case .score(_, _, let probabilities, _): return probabilities
        }
    }

    /// Whether two answers agree to the tolerance in every number and exactly otherwise.
    private static func close(_ a: Answer, _ b: Answer) -> Bool {
        func near(_ x: Double, _ y: Double) -> Bool { abs(x - y) <= tolerance }
        switch (a, b) {
        case (.noul(let x), .noul(let y)):
            return near(x, y)
        case (.choice(let c1, let p1, let k1), .choice(let c2, let p2, let k2)):
            return c1 == c2 && p1.keys == p2.keys && near(k1, k2)
                && zip(p1.values, p2.values).allSatisfy { near($0, $1) }
        case (.score(let s1, let l1, let p1, let k1), .score(let s2, let l2, let p2, let k2)):
            return near(s1, s2) && l1 == l2 && p1.count == p2.count && near(k1, k2)
                && zip(p1, p2).allSatisfy { near($0, $1) }
        default:
            return false
        }
    }
}
