import EncoderHarness
import Foundation
import XCTest

/// Tokenization and calibration against Fixtures/encoders, without any model.
final class ParityTests: XCTestCase {
    /// swift-transformers tokenizes every Verdict prompt to upstream's ids, including the
    /// <<LABEL>> and <<SEP>> markers and the truncation at 512 tokens.
    func testVerdictTokenization() async throws {
        let locations = try requireTokenizers()
        let reference = try JSONDecoder().decode(
            VerdictReference.self, from: Data(contentsOf: locations.verdictFixture))
        let tokenizer = try await EncoderTokenizer.load(folder: locations.verdictTokenizer)
        let cls = try XCTUnwrap(tokenizer.id(of: "[CLS]"))
        let sep = try XCTUnwrap(tokenizer.id(of: "[SEP]"))
        XCTAssertEqual(tokenizer.id(of: "<<LABEL>>"), reference.classTokenIndex)
        var mismatches: [String] = []
        for read in reference.reads {
            let ids = VerdictInput.ids(
                prompt: read.prompt, tokenizer: tokenizer, cls: cls, sep: sep,
                maxLength: reference.maxLength)
            if ids != read.inputIds { mismatches.append("\(read.request)/\(read.key)") }
        }
        XCTAssertEqual(reference.reads.count, 200)
        XCTAssertEqual(
            reference.reads.filter(\.truncated).isEmpty, false, "the corpus has truncated prompts")
        XCTAssertEqual(mismatches, [], "\(mismatches.count) prompts tokenize differently")
    }

    /// The Swift build_sequence reproduces laya's token ids and [MASK] marker positions for every
    /// question, including the head budget, the 48-token option cap and the 1,024-token limit.
    func testLayaSequences() async throws {
        let locations = try requireTokenizers()
        let reference = try JSONDecoder().decode(
            LayaReference.self, from: Data(contentsOf: locations.layaFixture))
        let tokenizer = try await EncoderTokenizer.load(folder: locations.layaTokenizer)
        XCTAssertEqual(tokenizer.id(of: "[MASK]"), reference.specialTokens.mask)
        var mismatches: [String] = []
        for read in reference.reads {
            let state = try XCTUnwrap(reference.stateTexts[read.request])
            let (ids, markers) = LayaSequence.build(
                head: read.texts.head, options: read.texts.options, state: state,
                tokenizer: tokenizer,
                special: reference.specialTokens, maxLen: reference.maxLen,
                headMaxLen: reference.headMaxLen)
            if ids != read.ids || markers != read.markers {
                mismatches.append("\(read.request)/\(read.key)")
            }
        }
        XCTAssertEqual(reference.reads.count, 200)
        XCTAssertEqual(
            reference.reads.filter(\.truncated).isEmpty, false, "the corpus has truncated states")
        XCTAssertEqual(mismatches, [], "\(mismatches.count) sequences differ")
    }

    /// Upstream's Verdict calibration applied to the recorded float32 logits gives the recorded
    /// probabilities.
    func testVerdictCalibration() throws {
        let locations = try requireFixtures()
        let reference = try JSONDecoder().decode(
            VerdictReference.self, from: Data(contentsOf: locations.verdictFixture))
        var worst = 0.0
        for read in reference.reads {
            let t = VerdictCalibration.temperature(k: read.k, calibrator: reference.calibrator)
            XCTAssertEqual(t, read.temperature, "\(read.request)/\(read.key)")
            let p = VerdictCalibration.probabilities(
                logits: read.logits.map(Float.init)[...], temperature: t)
            XCTAssertEqual(p.count, read.options)
            for (a, b) in zip(p, read.probabilities) { worst = max(worst, abs(a - b)) }
        }
        XCTAssertLessThan(worst, 1e-6)
    }

    /// laya's temperature buckets, the [0.5, 5] clamp and the softmax applied to the recorded
    /// logits give the recorded probabilities before rounding.
    func testLayaCalibration() throws {
        let locations = try requireFixtures()
        let reference = try JSONDecoder().decode(
            LayaReference.self, from: Data(contentsOf: locations.layaFixture))
        XCTAssertEqual(
            reference.calibration.temperatureByOptions,
            reference.calibration.temperatureByOptionsRaw.mapValues { LayaCalibration.clamp($0) })
        var worst = 0.0
        for read in reference.reads {
            let t = LayaCalibration.temperature(
                qtype: read.qtype, k: read.markers.count, calibration: reference.calibration,
                qtypes: reference.qtypes)
            XCTAssertEqual(t, read.temperature, accuracy: 1e-12, "\(read.request)/\(read.key)")
            XCTAssertEqual(
                LayaCalibration.bucket(
                    qtype: read.qtype, k: read.markers.count, qtypes: reference.qtypes),
                read.bucket)
            let p = LayaCalibration.probabilities(
                logits: read.logits.map(Float.init), temperature: t)
            for (a, b) in zip(p, read.probabilitiesUnrounded) { worst = max(worst, abs(a - b)) }
        }
        XCTAssertLessThan(worst, 1e-6)
    }
}
