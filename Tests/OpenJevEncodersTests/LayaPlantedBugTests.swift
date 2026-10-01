import Foundation
import OpenJevCore
import OpenJevEncoders
import Testing

/// The bounds the live parity test applies must catch a broken read: the type's temperature
/// where a bucket has its own, run on the recorded float32 scores, breaks at least one of them,
/// while the port meets all three (docs/spikes/encoder-runtime.md, "Scope notes"). The
/// probabilities are compared before laya rounds them, as the live test compares them.
///
/// The other planted bug, a read one position off each marker, needs the model's scores next to
/// the markers, which laya.json does not record (it holds the scores at the markers only), so
/// ``LayaLiveTests`` plants it on the Core ML package's own output. Without a model, the backend
/// tests replay rows whose every other position is NaN, so a read off the markers fails there.
@Suite(
    "Laya planted bugs",
    .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
struct LayaPlantedBugTests {
    /// Every recorded question's unrounded distribution next to what `read` makes of it.
    private func bounds(_ read: (LayaFixtures.Read, LayaCalibration) -> [Double]) throws
        -> ParityBounds
    {
        let reference = try LayaFixtures.reference()
        let calibration = reference.calibration
        return ParityBounds(
            reference.reads.map { row in
                (row.name, read(row, calibration), row.probabilitiesUnrounded)
            })
    }

    @Test("The port meets every bound")
    func portMeetsTheBounds() throws {
        let port = try bounds { row, calibration in
            calibration.probabilities(logits: row.logits, kind: row.kind).map(Double.init)
        }
        #expect(port.violations.isEmpty, "\(port)")
        #expect(port.maxDifference < 1e-6)
    }

    @Test("The type's temperature where a bucket has its own breaks a bound")
    func typeTemperature() throws {
        let planted = try bounds { row, calibration in
            let typeOnly = LayaCalibration(
                temperatures: calibration.temperatures, temperaturesByOptions: [:],
                maxLength: calibration.maxLength, headMaxLength: calibration.headMaxLength)
            return typeOnly.probabilities(logits: row.logits, kind: row.kind).map(Double.init)
        }
        #expect(!planted.violations.isEmpty, "\(planted)")
    }
}
