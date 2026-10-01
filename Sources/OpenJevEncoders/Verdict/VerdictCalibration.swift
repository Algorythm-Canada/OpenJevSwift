// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, the
// calibrator that `VerdictEngine.load` reads and the calibration that `VerdictEngine.read_batch`
// applies to the model's logits. Apache-2.0. See THIRD_PARTY.md.

import Foundation

/// Verdict's calibrator, the checkpoint's calibrator.json, and the calibration upstream applies
/// with it.
///
/// A question with `k` labels (its options and the abstention) reads the model's first `k`
/// logits. They are divided by ``perK``'s temperature for `k`, or by the global ``temperature``
/// when `k` has none, turned into probabilities with a softmax, and the abstention's
/// probability is dropped and the rest renormalised. When that is not finite the answer is
/// uniform over the `k - 1` options.
///
/// The softmax, abstention drop and renormalisation use float32, as upstream does. Only the
/// returned probabilities are converted to `Double`.
public struct VerdictCalibration: Sendable, Hashable, Codable {
    /// The global temperature, calibrator.json's `temperature`.
    public var temperature: Double
    /// The temperature for each label count that has its own, calibrator.json's `per_k`.
    public var perK: [Int: Double]

    /// Creates a calibrator.
    public init(temperature: Double, perK: [Int: Double]) {
        self.temperature = temperature
        self.perK = perK
    }

    /// Reads a calibrator.json file.
    ///
    /// - Throws: The file's read error, or a `DecodingError` when it is not a calibrator.
    public init(contentsOf file: URL) throws {
        self = try JSONDecoder().decode(Self.self, from: Data(contentsOf: file))
    }

    private enum CodingKeys: String, CodingKey {
        case temperature
        case perK = "per_k"
    }

    /// Decodes calibrator.json as upstream reads it: `float(cal["temperature"])` and
    /// `{int(k): float(v) for k, v in cal.get("per_k", {}).items()}`.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        temperature = try container.decode(Double.self, forKey: .temperature)
        let byName = try container.decodeIfPresent([String: Double].self, forKey: .perK) ?? [:]
        var perK: [Int: Double] = [:]
        for (name, value) in byName {
            guard let k = Int(name) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .perK, in: container,
                    debugDescription: "per_k has the key \(name.debugDescription), not an integer")
            }
            perK[k] = value
        }
        self.perK = perK
    }

    /// Encodes the calibrator as calibrator.json writes it, with `per_k`'s keys as strings.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(temperature, forKey: .temperature)
        try container.encode(
            Dictionary(uniqueKeysWithValues: perK.map { (String($0.key), $0.value) }),
            forKey: .perK)
    }

    /// The temperature for a question with `k` labels: `per_k[k]`, or the global temperature.
    public func temperature(k: Int) -> Double {
        perK[k] ?? temperature
    }

    /// The calibrated distribution over a question's options from the model's logits.
    ///
    /// - Parameters:
    ///   - logits: The model's output row. Only the first `k` entries are read.
    ///   - k: The question's label count, its options plus the abstention.
    /// - Returns: `k - 1` probabilities in the question's option order; empty for a question
    ///   without options, which ``EncoderDecisionEngine`` then refuses as a broken contract.
    /// - Precondition: `logits` has at least `k` entries.
    public func probabilities(logits: [Float], k: Int) -> [Double] {
        Self.probabilities(logits: logits.prefix(k), temperature: temperature(k: k))
    }

    /// The first logits divided by the temperature, turned into probabilities with a softmax,
    /// with the last one (the abstention) dropped and the rest renormalised; uniform over the
    /// kept entries when the result's sum is not finite, as upstream's `math.isfinite(sum(p))`.
    /// Fewer than two logits leave no option, and the result is empty.
    public static func probabilities(logits: ArraySlice<Float>, temperature: Double) -> [Double] {
        guard logits.count >= 2 else {
            return []
        }
        let scaled = logits.map { $0 / Float(temperature) }
        let top = scaled.max() ?? 0
        let exponentials = scaled.map { exp($0 - top) }
        let total = exponentials.reduce(0, +)
        let kept = exponentials.dropLast().map { $0 / total }
        let keptTotal = kept.reduce(0, +)
        let probabilities = kept.map { $0 / keptTotal }
        guard probabilities.reduce(0, +).isFinite else {
            return Array(repeating: Double(1 / Float(kept.count)), count: kept.count)
        }
        return probabilities.map(Double.init)
    }
}
