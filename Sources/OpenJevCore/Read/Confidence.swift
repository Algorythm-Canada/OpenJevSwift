// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, function
// `confidence`. Apache-2.0. See THIRD_PARTY.md.

import Foundation

/// How peaked an answer's distribution is.
public enum Confidence {
    /// `1 - H(p) / ln(K)`, clamped to `[0, 1]`: 1 is certain, 0 is uniform.
    ///
    /// `H(p)` is `-sum(x * ln x)` over the entries with `x > 0`, and `K` is `p.count`. The clamp
    /// follows Python's `max(0.0, min(1.0, value))`.
    ///
    /// With one entry or none, `ln(K)` is 0 and upstream's formula divides by zero. Upstream never
    /// reaches that case, because a question with a single option is answered without a read and
    /// its confidence is set to 1.0 directly. This function returns 1.0 for `K <= 1`, which is
    /// that same value.
    public static func compute(_ p: [Double]) -> Double {
        guard p.count > 1 else { return 1.0 }
        let entropy = -pythonSum(p.lazy.filter { $0 > 0 }.map { $0 * log($0) })
        let value = 1.0 - entropy / log(Double(p.count))
        // Python's min and max keep the first argument unless the second is strictly beyond it.
        let upper = value < 1.0 ? value : 1.0
        return upper > 0.0 ? upper : 0.0
    }
}
