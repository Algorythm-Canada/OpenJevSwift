import Foundation
import OpenJevEncoders
import Testing

/// The bounds the live parity test applies must catch a broken calibration: two planted bugs,
/// run on the recorded float32 logits, each break at least one of them, while the port meets
/// all three (docs/spikes/encoder-runtime.md, "Scope notes").
@Suite(
    "Planted calibration bugs",
    .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage))
struct PlantedBugTests {
    /// Every recorded question's distribution next to what `calibrate` makes of its logits.
    private func bounds(
        _ calibrate: (VerdictFixtures.Read, VerdictCalibration) -> [Double]
    ) throws -> ParityBounds {
        let reference = try VerdictFixtures.reference()
        return ParityBounds(
            reference.reads.map { read in
                (read.name, calibrate(read, reference.calibrator), read.probabilities)
            })
    }

    @Test("The port meets every bound")
    func portMeetsTheBounds() throws {
        let port = try bounds { read, calibration in
            calibration.probabilities(logits: read.logits, k: read.k)
        }
        #expect(port.violations.isEmpty, "\(port)")
        #expect(port.maxDifference < 1e-6)
    }

    @Test("The global temperature where per_k has one breaks a bound")
    func globalTemperature() throws {
        let planted = try bounds { read, calibration in
            VerdictCalibration.probabilities(
                logits: read.logits.prefix(read.k), temperature: calibration.temperature)
        }
        #expect(!planted.violations.isEmpty, "\(planted)")
    }

    @Test("Keeping the abstention's mass instead of renormalising breaks a bound")
    func abstentionKept() throws {
        let planted = try bounds { read, calibration in
            // The softmax over all k labels, read at the options: the abstention's probability
            // is dropped from the list but its mass is never given back to the options.
            let t = calibration.temperature(k: read.k)
            let scaled = read.logits.prefix(read.k).map { Double($0) / t }
            let top = scaled.max() ?? 0
            let exponentials = scaled.map { exp($0 - top) }
            let total = exponentials.reduce(0, +)
            return exponentials.dropLast().map { $0 / total }
        }
        #expect(!planted.violations.isEmpty, "\(planted)")
    }
}
