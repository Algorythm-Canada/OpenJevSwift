import Foundation
import OpenJevEncoders
import Testing

/// ``VerdictCalibration`` against upstream's `read_batch` on the recorded float32 logits, and
/// the calibrator file format.
@Suite("Verdict calibration")
struct VerdictCalibrationTests {
    @Test(
        "The recorded logits give the recorded probabilities within 1e-6, per_k or global",
        .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage))
    func fixtureParity() throws {
        let reference = try VerdictFixtures.reference()
        let calibration = reference.calibrator
        var worst = 0.0
        var sources: [String: Int] = [:]
        for read in reference.reads {
            #expect(calibration.temperature(k: read.k) == read.temperature, "\(read.name)")
            let source = calibration.perK[read.k] == nil ? "global" : "per_k"
            #expect(source == read.temperatureSource, "\(read.name)")
            sources[source, default: 0] += 1
            let probabilities = calibration.probabilities(logits: read.logits, k: read.k)
            #expect(probabilities.count == read.options, "\(read.name)")
            for (a, b) in zip(probabilities, read.probabilities) {
                worst = max(worst, abs(a - b))
            }
        }
        #expect(worst < 1e-6, "largest difference \(worst)")
        // Both temperatures are exercised: 197 questions have a per_k entry and 3 do not.
        #expect(sources == ["per_k": 197, "global": 3])
    }

    @Test("A result that is not finite falls back to the uniform distribution over the options")
    func uniformFallback() {
        let calibration = VerdictCalibration(temperature: 2, perK: [:])
        let uniform3: [Double] = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        #expect(calibration.probabilities(logits: [1, .nan, 0, 2], k: 4) == uniform3)
        #expect(calibration.probabilities(logits: [.infinity, 0, 0, 2], k: 4) == uniform3)
        #expect(calibration.probabilities(logits: [0, 0, 0, .infinity], k: 4) == uniform3)
        #expect(
            calibration.probabilities(logits: [-.infinity, -.infinity, -.infinity], k: 3)
                == [0.5, 0.5])
        #expect(calibration.probabilities(logits: [0, 1, 200], k: 3) == [0.5, 0.5])
        // A logit of minus infinity for one option is a finite answer, not a fallback.
        #expect(calibration.probabilities(logits: [-.infinity, 0, 0], k: 3) == [0, 1])
        // A question without options, only the abstention, has no distribution.
        #expect(calibration.probabilities(logits: [1, 2, 3], k: 1) == [])
    }

    @Test("Reads only the first k logits, and drops and renormalises the abstention")
    func firstKAndAbstention() {
        let calibration = VerdictCalibration(temperature: 1, perK: [:])
        // The abstention (third) takes most of the mass; the options share the rest 1:e.
        let p = calibration.probabilities(logits: [0, 1, 5, 99, 99], k: 3)
        #expect(p.count == 2)
        #expect(abs(p[0] - 1 / (1 + exp(1.0))) < 1e-12)
        #expect(abs(p[1] - exp(1.0) / (1 + exp(1.0))) < 1e-12)
        #expect(abs(p.reduce(0, +) - 1) < 1e-12)
    }

    @Test("calibrator.json decodes as upstream reads it")
    func decoding() throws {
        let json = Data(#"{"temperature": 2.8039, "per_k": {"3": 5.0069, "25": 1.5144}}"#.utf8)
        let calibration = try JSONDecoder().decode(VerdictCalibration.self, from: json)
        #expect(
            calibration == VerdictCalibration(temperature: 2.8039, perK: [3: 5.0069, 25: 1.5144]))
        #expect(calibration.temperature(k: 3) == 5.0069)
        #expect(calibration.temperature(k: 4) == 2.8039)
        // Upstream reads per_k with cal.get("per_k", {}).
        let bare = try JSONDecoder().decode(
            VerdictCalibration.self, from: Data(#"{"temperature": 3}"#.utf8))
        #expect(bare == VerdictCalibration(temperature: 3, perK: [:]))
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                VerdictCalibration.self,
                from: Data(#"{"temperature": 3, "per_k": {"three": 1}}"#.utf8))
        }
        let roundTrip = try JSONDecoder().decode(
            VerdictCalibration.self, from: JSONEncoder().encode(calibration))
        #expect(roundTrip == calibration)
    }

    @Test(
        "The checkpoint's calibrator.json is the one the fixture recorded",
        .enabled(
            if: VerdictModelFiles.tokenizerDirectory != nil,
            VerdictModelFiles.missingTokenizerMessage),
        .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage))
    func checkpointCalibrator() throws {
        let folder = try #require(VerdictModelFiles.tokenizerDirectory)
        let calibration = try VerdictCalibration(
            contentsOf: folder.appendingPathComponent("calibrator.json"))
        #expect(calibration == (try VerdictFixtures.reference().calibrator))
    }
}
