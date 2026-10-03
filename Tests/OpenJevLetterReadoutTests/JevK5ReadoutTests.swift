import Foundation
import OpenJevCore
import OpenJevLetterReadout
import Testing

/// ``JevK5Readout`` against the `jevk5` package's `spread` and upstream's letter softmax on
/// recorded logits (Fixtures/jevk5/reads.json): the distributions are the same doubles, so every
/// comparison is `==`.
@Suite(
    "JevK5 readout", .enabled(if: JevK5Fixtures.available, "Fixtures/jevk5/reads.json is missing"))
struct JevK5ReadoutTests {
    @Test("groups splits in order into near-equal runs, the longer first")
    func groups() {
        #expect(JevK5Readout.groups(20, into: 2) == [0..<10, 10..<20])
        #expect(JevK5Readout.groups(17, into: 2) == [0..<9, 9..<17])
        #expect(JevK5Readout.groups(33, into: 3) == [0..<11, 11..<22, 22..<33])
        let runs = JevK5Readout.groups(255, into: 16)
        #expect(runs.count == 16)
        #expect(runs.map(\.count) == [16] * 15 + [15])
        #expect(JevK5Readout.groups(3, into: 5).map(\.count) == [1, 1, 1, 0, 0])
    }

    @Test("spread over generated logits is the package's, bit for bit")
    func spreads() async throws {
        let reference = try JevK5Fixtures.reference()
        #expect(reference.spreads.count >= 16)
        for spread in reference.spreads {
            var next = 0
            let probabilities = try await JevK5Readout.spread(
                { texts in
                    defer { next += 1 }
                    guard next < spread.passes.count else {
                        throw FixtureFieldError(message: "\(spread.title): an extra pass")
                    }
                    let pass = spread.passes[next]
                    #expect(texts == pass.texts, "\(spread.title): pass \(next)'s texts")
                    return JevK5Readout.letterProbabilities(
                        logits: pass.logits, temperature: reference.temperature)
                }, texts: spread.texts, method: spread.method)
            #expect(next == spread.passes.count, "\(spread.title): passes")
            #expect(probabilities == spread.probabilities, "\(spread.title)")
        }
    }

    @Test("The corpus through the engine gives upstream's answers bit for bit, and its billing")
    func corpusThroughTheEngine() async throws {
        let reference = try JevK5Fixtures.reference()
        let backend = try JevK5Backend(
            model: ReplayLetterModel(reference), tokenizer: ReplayTokenizer(reference),
            temperature: reference.temperature)
        let engine = EncoderDecisionEngine(
            backend: backend, configuration: EncoderEngineConfiguration(warmUp: false))
        var answered = 0
        for request in reference.requests {
            let decision = try await engine.decide(request.request)
            #expect(decision.inputTokens == request.inputTokens, "\(request.name): input tokens")
            let reads = reference.reads(of: request.name)
            #expect(decision.inputTokens == reads.reduce(0) { $0 + $1.tokens })
            for read in reads {
                let answer = try #require(decision.answers[read.key], "\(request.name).\(read.key)")
                switch answer {
                case .noul(let yes):
                    #expect(yes == read.probabilities[0], "\(request.name).\(read.key)")
                case .choice(_, let probabilities, _):
                    #expect(
                        probabilities.values == read.probabilities, "\(request.name).\(read.key)")
                case .score(_, _, let probabilities, _):
                    #expect(probabilities == read.probabilities, "\(request.name).\(read.key)")
                }
                answered += 1
            }
        }
        #expect(answered == reference.reads.count)
        #expect(await backend.passCount == reference.passes.count)
    }
}

extension Array {
    /// `count` copies of each element, in order.
    fileprivate static func * (elements: Array, count: Int) -> Array {
        (0..<count).flatMap { _ in elements }
    }
}
