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
        // Both options' float32 exponentials underflow to zero, as in upstream's PyTorch, which
        // returns the uniform answer here; double arithmetic would answer [0.3775, 0.6225].
        #expect(calibration.probabilities(logits: [0, 1, 400], k: 3) == [0.5, 0.5])
        // A logit of minus infinity for one option is a finite answer, not a fallback.
        #expect(calibration.probabilities(logits: [-.infinity, 0, 0], k: 3) == [0, 1])
        // A question without options, only the abstention, has no distribution.
        #expect(calibration.probabilities(logits: [1, 2, 3], k: 1) == [])
    }

    @Test("Subnormal float32 exponentials give upstream's answer, not the double one")
    func subnormalExponentials() {
        // Upstream's PyTorch (torch 2.13 on the CPU) answers [0.3802816867828369,
        // 0.6197183132171631] for these logits: e^-100 and e^-99.5 survive as float32
        // subnormals. Double arithmetic would answer [0.3775, 0.6225].
        let p = VerdictCalibration(temperature: 2, perK: [:]).probabilities(
            logits: [0, 1, 200], k: 3)
        #expect(p.count == 2)
        #expect(abs(p[0] - 0.380_281_686_782_836_9) < 1e-7, "\(p)")
        #expect(abs(p[1] - 0.619_718_313_217_163_1) < 1e-7, "\(p)")
    }

    @Test("Reads only the first k logits, and drops and renormalises the abstention")
    func firstKAndAbstention() {
        let calibration = VerdictCalibration(temperature: 1, perK: [:])
        // The abstention (third) takes most of the mass; the options share the rest 1:e.
        // The arithmetic is float32, as upstream's, so the comparison allows float32 rounding.
        let p = calibration.probabilities(logits: [0, 1, 5, 99, 99], k: 3)
        #expect(p.count == 2)
        #expect(abs(p[0] - 1 / (1 + exp(1.0))) < 1e-6)
        #expect(abs(p[1] - exp(1.0) / (1 + exp(1.0))) < 1e-6)
        #expect(abs(p.reduce(0, +) - 1) < 1e-6)
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
