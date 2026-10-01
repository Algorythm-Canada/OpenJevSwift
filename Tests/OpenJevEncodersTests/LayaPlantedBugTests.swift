import Foundation
import OpenJevCore
import OpenJevEncoders
import Testing

/// The bounds the live parity test applies must catch a broken read: two planted bugs, run on
/// the recorded float32 scores, each break at least one of them, while the port meets all three
/// (docs/spikes/encoder-runtime.md, "Scope notes"). The probabilities are compared before laya
/// rounds them, as the live test compares them.
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

    /// The scores a read at `offset` from each marker gets from a model's output row: the
    /// recorded scores at the markers, and a neutral 0 at every other position.
    private func shifted(_ row: LayaFixtures.Read, by offset: Int) -> [Float] {
        let scores = row.scoreRow(filler: 0)
        return row.markers.map { scores[$0 + offset] }
    }

    @Test("The port meets every bound")
    func portMeetsTheBounds() throws {
        let port = try bounds { row, calibration in
            calibration.probabilities(logits: shifted(row, by: 0), kind: row.kind).map(Double.init)
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

    @Test("Reading one position after each marker breaks a bound", arguments: [1, -1])
    func markersOffByOne(offset: Int) throws {
        let planted = try bounds { row, calibration in
            calibration.probabilities(logits: shifted(row, by: offset), kind: row.kind).map(
                Double.init)
        }
        #expect(!planted.violations.isEmpty, "\(planted)")
    }
}
