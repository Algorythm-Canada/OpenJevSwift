// A port of laya 0.3.6 (NandhaKishorM/laya): `laya/common.py`, `QTYPES`, `temp_bucket`,
// `TEMP_MIN`, `TEMP_MAX` and `clamp_temperature`; `laya/agent.py`, the configuration
// `Agent.__init__` reads from rl_agent_config.json and the calibration `Agent.system_one` applies
// to the scores at the markers. And of upstream OpenJev (razorback16/openjev at dcd2094),
// `openjev/encoders.py`, the distribution `LayaEngine.read_batch` publishes. Apache-2.0. See
// THIRD_PARTY.md.

import Foundation
import OpenJevCore

/// Laya's configuration, the checkpoint's rl_agent_config.json, and the calibration laya applies
/// with it.
///
/// A question with `k` options reads the model's scores at its `k` markers. They are divided by
/// the temperature of the question's bucket (``bucket(kind:optionCount:)``,
/// `temperature_by_options` in the file) or, when the bucket has none, by its type's
/// temperature (`temperature`, one per question type), and turned into probabilities with a
/// softmax. laya clamps every temperature to [0.5, 5] when it loads the file, because the
/// checkpoint's `choice:11+` bucket (0.1006) would sharpen the scores tenfold, and so does this.
///
/// laya rounds each probability to 4 decimal places (``roundedToFourPlaces(_:)``), and
/// upstream publishes a choice's or a score's rounded probabilities divided by their sum, and a
/// noul's rounded `P(true)` with `1 - P(true)` (``distribution(logits:kind:)``).
///
/// laya computes in float32 with numpy on the CPU, and so does this, in numpy's order of
/// operations: the division by the temperature, the softmax with its maximum subtracted, and
/// numpy's pairwise sum. On the recorded logits it reproduces laya's unrounded probabilities bit
/// for bit, so the rounded answers are laya's own.
public struct LayaCalibration: Sendable, Hashable {
    /// laya's `TEMP_MIN` and `TEMP_MAX`: the temperatures it applies.
    public static let temperatureRange: ClosedRange<Double> = 0.5...5.0
    /// The `max_len` laya uses when the file has none.
    public static let defaultMaxLength = 512
    /// The `head_max_len` laya uses when the file has none.
    public static let defaultHeadMaxLength = 192
    /// The question types laya knows, by their `QTYPES` index.
    public static let questionTypes = 3

    /// The temperature of each question type, by ``LayaPrompt/questionType(of:)``: choice,
    /// score and noul. Clamped.
    public var temperatures: [Double]
    /// The temperature of each bucket that has its own, such as `noul:2` or `choice:3-5`.
    /// Clamped.
    public var temperaturesByOptions: [String: Double]
    /// The longest sequence, `max_len`: 1,024 for the checkpoint.
    public var maxLength: Int
    /// The most tokens the head and the options share, `head_max_len`: 256 for the checkpoint.
    public var headMaxLength: Int

    /// Creates a calibration, clamping each temperature as laya does when it loads the file.
    ///
    /// - Precondition: There is a temperature for each of the three question types, and both
    ///   lengths are positive.
    public init(
        temperatures: [Double], temperaturesByOptions: [String: Double],
        maxLength: Int = defaultMaxLength, headMaxLength: Int = defaultHeadMaxLength
    ) {
        precondition(
            temperatures.count >= Self.questionTypes, "laya reads a temperature per question type")
        precondition(maxLength > 0 && headMaxLength > 0, "laya's lengths are positive")
        self.temperatures = temperatures.map(Self.clamp)
        self.temperaturesByOptions = temperaturesByOptions.mapValues(Self.clamp)
        self.maxLength = maxLength
        self.headMaxLength = headMaxLength
    }

    /// Reads an rl_agent_config.json file.
    ///
    /// - Throws: The file's read error, ``JSONParseError`` for a file that is not JSON, and
    ///   ``EncoderLoadError/invalidConfiguration(_:)`` for one laya could not run with.
    public init(contentsOf file: URL) throws {
        try self.init(json: JSONParser().parse(Data(contentsOf: file)))
    }

    /// Reads rl_agent_config.json as laya's `Agent.__init__` does: `temperature` (default
    /// `[1, 1, 1]`) and `temperature_by_options` (default none), each value through
    /// `clamp_temperature`; `max_len` (default 512) and `head_max_len` (default 192).
    ///
    /// `clamp_temperature` reads a value with Python's `float`: an integer, a float or a
    /// Boolean is that number. Anything else, and a number that is not finite, gives 1. A string
    /// also gives 1 here, where Python's `float` would parse one that holds a number.
    ///
    /// - Throws: ``EncoderLoadError/invalidConfiguration(_:)`` when `temperature` is not an array
    ///   of at least three values, `temperature_by_options` is not an object, or a length is not
    ///   a positive integer.
    public init(json: JSONValue) throws {
        func invalid(_ message: String) -> EncoderLoadError {
            .invalidConfiguration("rl_agent_config.json: \(message)")
        }
        guard case .object = json else {
            throw invalid("the file holds no object")
        }
        let temperatures: [Double]
        switch json["temperature"] {
        case .none:
            temperatures = [1, 1, 1]
        case .some(.array(let values)) where values.count >= Self.questionTypes:
            temperatures = values.map(Self.temperature(of:))
        default:
            throw invalid("temperature must be a list with one value per question type")
        }
        var byOptions: [String: Double] = [:]
        switch json["temperature_by_options"] {
        case .none:
            break
        case .some(.object(let buckets)):
            for (bucket, value) in buckets {
                byOptions[bucket] = Self.temperature(of: value)
            }
        default:
            throw invalid("temperature_by_options must be an object")
        }
        func length(_ key: String, default fallback: Int) throws -> Int {
            guard let value = json[key] else {
                return fallback
            }
            guard let length = value.intValue, length > 0 else {
                throw invalid("\(key) must be a positive integer, not \(value)")
            }
            return length
        }
        self.init(
            temperatures: temperatures, temperaturesByOptions: byOptions,
            maxLength: try length("max_len", default: Self.defaultMaxLength),
            headMaxLength: try length("head_max_len", default: Self.defaultHeadMaxLength))
    }

    /// A temperature as Python's `float` reads it, before clamping: `nan` for what it cannot
    /// read, which ``clamp(_:)`` turns into 1.
    private static func temperature(of value: JSONValue) -> Double {
        switch value {
        case .integer, .float:
            return value.doubleValue ?? .nan
        case .bool(let flag):
            return flag ? 1 : 0
        default:
            return .nan
        }
    }

    /// laya's `clamp_temperature`: the temperature confined to [0.5, 5], or 1 when it is not
    /// finite.
    public static func clamp(_ temperature: Double) -> Double {
        guard temperature.isFinite else {
            return 1
        }
        return min(temperatureRange.upperBound, max(temperatureRange.lowerBound, temperature))
    }

    /// laya's `temp_bucket`: the question type and the option count's range, `2`, `3-5`,
    /// `6-10` or `11+`, such as `noul:2` or `choice:11+`.
    public static func bucket(kind: QuestionKind, optionCount k: Int) -> String {
        let size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"
        return "\(kind.rawValue):\(size)"
    }

    /// The temperature laya applies to a question: its bucket's, else its type's.
    public func temperature(kind: QuestionKind, optionCount k: Int) -> Double {
        temperaturesByOptions[Self.bucket(kind: kind, optionCount: k)]
            ?? temperatures[LayaPrompt.questionType(of: kind)]
    }

    /// The probabilities laya computes from a question's scores at its markers, before it rounds
    /// them: in float32, the scores divided by the temperature, and their softmax.
    ///
    /// - Parameter logits: The scores at the markers, in laya's option order; one per option.
    /// - Returns: One probability per score, in the same order; empty for no scores.
    public func probabilities(logits: [Float], kind: QuestionKind) -> [Float] {
        guard !logits.isEmpty else {
            return []
        }
        let temperature = Float(temperature(kind: kind, optionCount: logits.count))
        let scaled = logits.map { $0 / temperature }
        // numpy's maximum propagates NaN, where Swift's max() depends on the order.
        let top = scaled.contains(where: \.isNaN) ? Float.nan : scaled.max() ?? 0
        let exponentials = scaled.map { exp($0 - top) }
        let total = Self.pairwiseSum(exponentials)
        return exponentials.map { $0 / total }
    }

    /// The distribution upstream publishes for a question, in the engine's option order
    /// (``EncoderQuestion/choices``), from its scores at the markers in laya's order.
    ///
    /// A choice's and a score's probabilities are rounded to 4 decimal places, as laya answers
    /// them, then divided by their sum, which Python's `sum` computes (``pythonSum(_:)``). A noul
    /// is `[P(true), 1 - P(true)]` with `P(true)` rounded: laya's markers are false then true, the
    /// engine's options true then false.
    ///
    /// - Returns: One probability per option; empty for a question without options.
    public func distribution(logits: [Float], kind: QuestionKind) -> [Double] {
        Self.published(probabilities(logits: logits, kind: kind), kind: kind)
    }

    /// The distribution upstream publishes from laya's unrounded probabilities, in laya's option
    /// order: rounded to 4 decimal places as laya answers them, then a choice's and a score's
    /// divided by their sum, and a noul's `P(true)` (laya's second option) with `1 - P(true)`.
    public static func published(_ probabilities: [Float], kind: QuestionKind) -> [Double] {
        let rounded = probabilities.map { roundedToFourPlaces(Double($0)) }
        if kind == .noul, rounded.count == 2 {
            return [rounded[1], 1 - rounded[1]]
        }
        let total = pythonSum(rounded)
        return rounded.map { $0 / total }
    }

    /// Python's `round(x, 4)` for a float: the value with 4 decimal places nearest to `x`'s
    /// exact binary value, ties to the even last digit, as the nearest double.
    ///
    /// CPython rounds the exact value to a decimal string and converts that back with correct
    /// rounding. This does the same with integers: `x` is `m * 2^e` exactly, so `x * 10^4` is
    /// `m * 625 * 2^(e + 4)`, which is rounded to an integer `n` half to even, and the result is
    /// the double nearest to `n / 10^4`. The infinities and NaN round to themselves, and the sign
    /// is kept, so a small negative value rounds to `-0.0`, as in Python.
    public static func roundedToFourPlaces(_ x: Double) -> Double {
        guard x.isFinite, x != 0 else {
            return x
        }
        let magnitude = x.magnitude
        // magnitude = significand * 2^exponent exactly, the significand an integer below 2^53.
        let significand: UInt64
        let exponent: Int
        if magnitude.isNormal {
            significand =
                magnitude.significandBitPattern | (1 << UInt64(Double.significandBitCount))
            exponent = Int(magnitude.exponent) - Double.significandBitCount
        } else {
            significand = magnitude.significandBitPattern
            exponent = Int(Double.leastNormalMagnitude.exponent) - Double.significandBitCount
        }
        // x * 10^4 = significand * 5^4 * 2^(exponent + 4), and significand * 5^4 < 2^63.
        let scaled = significand * 625
        let drop = -(exponent + 4)
        guard drop > 0 else {
            // x * 10^4 is an integer: x has at most 4 decimal places already.
            return x
        }
        let units: UInt64
        if drop >= 64 {
            units = 0
        } else {
            let kept = scaled >> UInt64(drop)
            let remainder = scaled - (kept << UInt64(drop))
            let half = UInt64(1) << UInt64(drop - 1)
            let roundsUp = remainder > half || (remainder == half && kept & 1 == 1)
            units = roundsUp ? kept + 1 : kept
        }
        let rounded: Double
        if units < (1 << 53) {
            // Both operands are exact, so the division rounds once, correctly.
            rounded = Double(units) / 10_000
        } else {
            rounded = Double("\(units)e-4") ?? x
        }
        return x < 0 ? -rounded : rounded
    }

    /// numpy's sum of a float32 array (`pairwise_sum_FLOAT`): in order below 8 values; with
    /// eight running sums up to 128; split in two halves, each a multiple of 8 long, above.
    static func pairwiseSum(_ values: [Float]) -> Float {
        0 + pairwiseSum(values[...])
    }

    private static func pairwiseSum(_ values: ArraySlice<Float>) -> Float {
        let n = values.count
        let base = values.startIndex
        if n < 8 {
            var total: Float = 0
            for value in values {
                total += value
            }
            return total
        }
        if n <= 128 {
            var sums = Array(values[base..<(base + 8)])
            var index = 8
            while index < n - n % 8 {
                for lane in 0..<8 {
                    sums[lane] += values[base + index + lane]
                }
                index += 8
            }
            var total =
                ((sums[0] + sums[1]) + (sums[2] + sums[3]))
                + ((sums[4] + sums[5]) + (sums[6] + sums[7]))
            while index < n {
                total += values[base + index]
                index += 1
            }
            return total
        }
        var half = n / 2
        half -= half % 8
        return pairwiseSum(values[base..<(base + half)])
            + pairwiseSum(values[(base + half)...])
    }
}
