import Foundation
import OpenJevCore
import Testing

@testable import OpenJevEncoders

/// ``LayaCalibration`` against laya's `system_one` on the recorded float32 scores, the rounding
/// against laya's own answers, the renormalisation against what upstream published, and the
/// configuration file.
@Suite("Laya calibration")
struct LayaCalibrationTests {
    @Test(
        "The recorded scores give laya's buckets, temperatures and probabilities",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func fixtureParity() throws {
        let reference = try LayaFixtures.reference()
        let calibration = reference.calibration
        var worst = 0.0
        var exact = 0
        var sources: [String: Int] = [:]
        for read in reference.reads {
            let k = read.markers.count
            #expect(LayaCalibration.bucket(kind: read.kind, optionCount: k) == read.bucket)
            #expect(
                calibration.temperature(kind: read.kind, optionCount: k) == read.temperature,
                "\(read.name)")
            let source = calibration.temperaturesByOptions[read.bucket] == nil ? "type" : "bucket"
            sources[source, default: 0] += 1
            let probabilities = calibration.probabilities(logits: read.logits, kind: read.kind)
            #expect(probabilities.count == read.options, "\(read.name)")
            for (a, b) in zip(probabilities, read.probabilitiesUnrounded) {
                worst = max(worst, abs(Double(a) - b))
            }
            if probabilities.map(Double.init) == read.probabilitiesUnrounded {
                exact += 1
            }
        }
        #expect(worst < 1e-6, "largest difference \(worst)")
        // Laya's float32 arithmetic in numpy's order gives laya's probabilities bit for bit.
        if LayaFixtures.arithmeticIsLayas {
            #expect(exact == 200, "\(exact) of 200 bit for bit; largest difference \(worst)")
        }
        // Both temperatures are used: score:6-10 and score:2 have no bucket of their own.
        #expect(sources == ["bucket": 190, "type": 10])
    }

    @Test(
        "Rounding gives laya's answers exactly, and renormalising gives what upstream published",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func roundingAndRenormalisation() throws {
        let reference = try LayaFixtures.reference()
        var nouls = 0
        for read in reference.reads {
            let rounded = read.probabilitiesUnrounded.map(LayaCalibration.roundedToFourPlaces)
            let answer = try read.answerValues
            if read.kind == .noul {
                // laya answers P(true), its second marker.
                #expect([rounded[1]] == answer, "\(read.name)")
                nouls += 1
            } else {
                #expect(rounded == answer, "\(read.name): \(rounded) \(answer)")
            }
            let published = LayaCalibration.published(
                read.probabilitiesUnrounded.map { Float($0) }, kind: read.kind)
            #expect(published == read.probabilities, "\(read.name): \(published)")
        }
        #expect(nouls == 96)
    }

    @Test("Python's round to 4 places: the exact binary value, ties to even")
    func pythonRound() {
        let round = LayaCalibration.roundedToFourPlaces
        // Exact binary ties go to the even digit, as CPython's correctly rounded round does.
        #expect(round(0.031_25) == 0.0312)
        #expect(round(0.093_75) == 0.0938)
        #expect(round(0.156_25) == 0.1562)
        #expect(round(0.218_75) == 0.2188)
        // Decimal ties that are not binary ones go the way the binary value lies.
        #expect(round(0.000_05) == 0.0001)  // 5.0000000000000002e-05
        #expect(round(0.000_15) == 0.0001)  // 1.4999999999999999e-04
        #expect(round(2.675) == 2.675)
        #expect(round(0.387_581_050_395_965_6) == 0.3876)
        #expect(round(1) == 1)
        #expect(round(0.999_96) == 1)
        #expect(round(0) == 0)
        #expect(round(1e-300) == 0)
        #expect(round(.leastNonzeroMagnitude) == 0)
        #expect(round(-0.000_01).sign == .minus)
        #expect(round(-0.123_45) == -0.1235)  // -0.12345000000000000417
        #expect(round(.infinity) == .infinity)
        #expect(round(-.infinity) == -.infinity)
        #expect(round(.nan).isNaN)
        #expect(round(123_456_789.123_456_78) == 123_456_789.1235)
        #expect(round(1e300) == 1e300)
    }

    @Test("The rounding agrees with the C library's correctly rounded %.4f on random values")
    func roundingAgainstPrintf() {
        var generator = SystemRandomNumberGenerator()
        var values: [Double] = (0..<20_000).map { _ in Double.random(in: 0...1, using: &generator) }
        // Every tie with five decimals that a float can hold exactly: k / 32.
        values += (0...32).map { Double($0) / 32 }
        values += (0..<2000).map { _ in Double(Float.random(in: 0...1, using: &generator)) }
        for value in values {
            let printed = Double(String(format: "%.4f", value))
            #expect(LayaCalibration.roundedToFourPlaces(value) == printed, "\(value)")
        }
    }

    @Test("A noul publishes [P(true), 1 - P(true)], laya's second marker first")
    func noulOrder() {
        let calibration = LayaCalibration(temperatures: [1, 1, 1], temperaturesByOptions: [:])
        // Scores for false, then true.
        let p = calibration.distribution(logits: [0, 2], kind: .noul)
        let unrounded = calibration.probabilities(logits: [0, 2], kind: .noul)
        #expect(p == [LayaCalibration.roundedToFourPlaces(Double(unrounded[1])), 1 - p[0]])
        #expect(p[0] > 0.8)
        // A choice keeps its order, rounded and renormalised by Python's sum.
        let choice = LayaCalibration.published([0.333_33, 0.333_33, 0.333_34], kind: .choice)
        let rounded = [0.3333, 0.3333, 0.3333]
        #expect(choice == rounded.map { $0 / pythonSum(rounded) })
        #expect(LayaCalibration.published([], kind: .score) == [])
    }

    @Test("The softmax runs in float32 as numpy does, NaN and the infinities included")
    func float32Softmax() {
        let calibration = LayaCalibration(temperatures: [1, 1, 1], temperaturesByOptions: [:])
        let p = calibration.probabilities(logits: [1, 2, 3], kind: .choice)
        let e: [Float] = [exp(-2), exp(-1), 1]
        let total = e[0] + e[1] + e[2]
        #expect(p == e.map { $0 / total })
        #expect(calibration.probabilities(logits: [], kind: .choice) == [])
        let notANumber = calibration.probabilities(logits: [1, .nan], kind: .choice)
        #expect(notANumber.count == 2 && notANumber.allSatisfy { $0.isNaN })
        let infinite = calibration.probabilities(logits: [1, .infinity], kind: .choice)
        #expect(infinite.count == 2 && infinite.allSatisfy { $0.isNaN })
        #expect(calibration.probabilities(logits: [1, -.infinity], kind: .choice) == [1, 0])
        // numpy's pairwise sum: eight running sums from eight values on.
        let values = (0..<20).map { Float($0) * 0.1 + 1e-7 }
        var lanes = Array(values[0..<8])
        for index in 8..<16 {
            lanes[index - 8] += values[index]
        }
        var expected =
            ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        for index in 16..<20 {
            expected += values[index]
        }
        #expect(LayaCalibration.pairwiseSum(values) == expected)
    }

    @Test("Buckets name the type and the option count's range")
    func buckets() {
        let cases: [(QuestionKind, Int, String)] = [
            (.noul, 2, "noul:2"), (.choice, 1, "choice:2"), (.choice, 2, "choice:2"),
            (.choice, 3, "choice:3-5"), (.choice, 5, "choice:3-5"), (.choice, 6, "choice:6-10"),
            (.score, 10, "score:6-10"), (.choice, 11, "choice:11+"), (.choice, 255, "choice:11+"),
        ]
        for (kind, k, bucket) in cases {
            #expect(LayaCalibration.bucket(kind: kind, optionCount: k) == bucket)
        }
        let calibration = LayaCalibration(
            temperatures: [1.5, 2.5, 3.5], temperaturesByOptions: ["choice:11+": 0.1006])
        #expect(calibration.temperature(kind: .choice, optionCount: 12) == 0.5)
        #expect(calibration.temperature(kind: .choice, optionCount: 3) == 1.5)
        #expect(calibration.temperature(kind: .score, optionCount: 3) == 2.5)
        #expect(calibration.temperature(kind: .noul, optionCount: 2) == 3.5)
    }

    @Test("rl_agent_config.json decodes as laya reads it, every temperature clamped")
    func decoding() throws {
        let json = try JSONParser().parse(
            #"""
            {"max_len": 1024, "head_max_len": 256, "temperature": [1.0148, 9, -1],
             "temperature_by_options": {"noul:2": 1.9834, "choice:11+": 0.1006, "odd": "x",
                                        "flag": true}}
            """#)
        let calibration = try LayaCalibration(json: json)
        #expect(calibration.maxLength == 1024)
        #expect(calibration.headMaxLength == 256)
        #expect(calibration.temperatures == [1.0148, 5, 0.5])
        #expect(
            calibration.temperaturesByOptions == [
                "noul:2": 1.9834, "choice:11+": 0.5, "odd": 1, "flag": 1,
            ])
        // laya's defaults: temperature [1, 1, 1], no buckets, max_len 512, head_max_len 192.
        let bare = try LayaCalibration(json: [:])
        #expect(bare.temperatures == [1, 1, 1])
        #expect(bare.temperaturesByOptions.isEmpty)
        #expect(bare.maxLength == 512)
        #expect(bare.headMaxLength == 192)
        #expect(LayaCalibration.clamp(.nan) == 1)
        #expect(LayaCalibration.clamp(-.infinity) == 1)
        #expect(LayaCalibration.clamp(0.1006) == 0.5)
        #expect(LayaCalibration.clamp(7) == 5)
        #expect(LayaCalibration.clamp(2) == 2)
        let refused: [JSONValue] = [
            ["temperature": [1, 2]], ["temperature": "1"], ["temperature_by_options": [1]],
            ["max_len": 0], ["head_max_len": 1.5], ["max_len": -3], [1, 2],
        ]
        for config in refused {
            #expect(throws: EncoderLoadError.self, "\(config)") {
                try LayaCalibration(json: config)
            }
        }
    }

    @Test(
        "The checkpoint's rl_agent_config.json is the one the fixture recorded",
        .enabled(if: LayaModelFiles.tokenizer != nil, LayaModelFiles.missingTokenizerMessage),
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func checkpointConfiguration() throws {
        let file = try #require(LayaModelFiles.tokenizer).calibratorFile
        let calibration = try LayaCalibration(contentsOf: file)
        let reference = try LayaFixtures.reference()
        #expect(calibration == reference.calibration)
        #expect(calibration.temperatures == reference.temperature)
        #expect(calibration.temperaturesByOptions == reference.temperatureByOptions)
        #expect(reference.clamp == [0.5, 5])
        #expect(reference.temperatureByOptionsRaw["choice:11+"] == 0.100_582_808_256_149_29)
        #expect(calibration.temperaturesByOptions["choice:11+"] == 0.5)
        #expect(calibration.maxLength == 1024)
        #expect(calibration.headMaxLength == 256)
    }
}
