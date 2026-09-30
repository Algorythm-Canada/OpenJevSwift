import Foundation

/// Verdict's calibration, as upstream's VerdictEngine.read_batch applies it to the model's logits.
public enum VerdictCalibration {
    /// calibrator.json's per_k[k], or its global temperature when k has no entry.
    public static func temperature(k: Int, calibrator: VerdictReference.Calibrator) -> Double {
        calibrator.perK[String(k)] ?? calibrator.temperature
    }

    /// The first k logits divided by the temperature, softmax, the abstention entry dropped and
    /// the rest renormalised; a uniform distribution when that is not finite. Upstream works in
    /// float32; this works in double, which differs by less than 1e-6.
    public static func probabilities(logits: ArraySlice<Float>, temperature: Double) -> [Double] {
        let z = logits.map { Double($0) / temperature }
        let top = z.max() ?? 0
        let e = z.map { exp($0 - top) }
        let total = e.reduce(0, +)
        let kept = e.dropLast().map { $0 / total }
        let sum = kept.reduce(0, +)
        let p = kept.map { $0 / sum }
        return p.allSatisfy(\.isFinite)
            ? p : Array(repeating: 1.0 / Double(kept.count), count: kept.count)
    }
}

/// Laya's calibration, as laya's Agent.system_one applies it to the scores at the markers.
public enum LayaCalibration {
    /// laya.common.temp_bucket.
    public static func bucket(qtype: Int, k: Int, qtypes: [String: Int]) -> String {
        let name = qtypes.first { $0.value == qtype }?.key ?? "choice"
        let size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"
        return "\(name):\(size)"
    }

    /// laya.common.clamp_temperature: confined to [0.5, 5], and 1 for anything not finite.
    public static func clamp(_ t: Double, low: Double = 0.5, high: Double = 5.0) -> Double {
        t.isFinite ? min(high, max(low, t)) : 1.0
    }

    /// temperature_by_options[bucket] if present, else temperature[qtype], both clamped.
    public static func temperature(
        qtype: Int, k: Int, calibration: LayaReference.Calibration, qtypes: [String: Int]
    )
        -> Double
    {
        let key = bucket(qtype: qtype, k: k, qtypes: qtypes)
        return clamp(calibration.temperatureByOptionsRaw[key] ?? calibration.temperatureRaw[qtype])
    }

    /// Softmax of the marker logits over the temperature, before laya rounds to 4 decimals.
    public static func probabilities(logits: [Float], temperature: Double) -> [Double] {
        let z = logits.map { Double($0) / temperature }
        let top = z.max() ?? 0
        let e = z.map { exp($0 - top) }
        let total = e.reduce(0, +)
        return e.map { $0 / total }
    }
}
