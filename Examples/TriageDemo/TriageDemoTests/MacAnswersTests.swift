import Foundation
import OpenJevCore
import OpenJevEncoders
import Testing
import TriageDemo

/// Compares this device's answers to the three samples with what `openjev decide --backend
/// verdict` answered on a Mac (mac-answers.json). The Mac reads on the GPU in batches of 16, an
/// iPhone on the Neural Engine one question at a time, so small differences are expected.
@MainActor
struct MacAnswersTests {
    /// The largest difference allowed in any probability.
    static let tolerance = 0.05

    /// Verdict's files are on this device once the app has launched with a connection; the test
    /// never downloads them.
    nonisolated static var verdictHeld: Bool {
        (try? EncoderPackageStore(environment: [:]).heldPackageDirectory(for: .verdict)) != nil
    }

    @Test(
        "Answers match the Mac's",
        .enabled(if: verdictHeld, "Verdict is not on this device: launch the app once, online"))
    func answersMatchTheMac() async throws {
        let fixture = try Self.fixture()
        let store = try EncoderPackageStore(environment: [:])
        let engine = EncoderDecisionEngine(backend: try await VerdictBackend.load(from: store))
        let model = TriageModel()

        var largest = (difference: 0.0, place: "")
        for sample in TriageQuestions.samples {
            let mac = try #require(fixture[sample.name]?.objectValue?["answers"]?.objectValue)
            let decision = try await engine.decide(model.request(for: sample.text))
            for (id, answer) in decision.answers {
                let expected = try Answer(json: try #require(mac[id]))
                for (index, (here, there)) in zip(
                    Self.probabilities(answer), Self.probabilities(expected)
                ).enumerated() {
                    let difference = abs(here - there)
                    if difference > largest.difference {
                        largest = (
                            difference,
                            "\(sample.name) \(id)[\(index)]: \(here) here, \(there) on the Mac"
                        )
                    }
                }
            }
        }
        let report = "Largest difference from the Mac: \(largest.difference) (\(largest.place))"
        print(report)
        // Kept in the result bundle, from a simulator or a device alike.
        Attachment.record(report, named: "largest-difference.txt")
        #expect(
            largest.difference <= Self.tolerance,
            "largest difference \(largest.difference) at \(largest.place)")
    }

    /// Every probability of an answer, a noul's as P(yes).
    static func probabilities(_ answer: Answer) -> [Double] {
        switch answer {
        case .noul(let yes): [yes]
        case .choice(_, let probabilities, _): probabilities.values
        case .score(_, _, let probabilities, _): probabilities
        }
    }

    static func fixture() throws -> JSONObject {
        let url = try #require(
            Bundle(for: BundleToken.self).url(forResource: "mac-answers", withExtension: "json"))
        return try #require(try JSONParser().parse(Data(contentsOf: url)).objectValue)
    }
}

private final class BundleToken {}
