// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, function
// `slot_distribution`. Apache-2.0. See THIRD_PARTY.md.

import Foundation

/// The label probabilities at one slot of a read, and the entropy that drives re-reads.
public enum SlotDistribution {
    /// The label distribution and the top-k entropy at one slot, from a backend's logprobs.
    ///
    /// Read-only logprobs are at temperature 1, so the labels' logprobs go into a softmax as
    /// they are:
    /// - a label missing from `top` gets the floor, the smallest logprob in `top` less 5;
    /// - the softmax runs over the labels only, after subtracting their largest logprob;
    /// - the entropy is `-sum(p * ln p)` over `p = exp(logprob)` for every entry of `top` with
    ///   `p > 0`. Those values are a top-k subset and are not renormalised, as upstream does; the
    ///   engine compares this entropy with `OPENJEV_AUTO_THRESHOLD`.
    ///
    /// `top` is in the order the backend returned it. The entropy is a compensated sum, as
    /// Python's, which can differ in its last bit between orders; use this overload when the
    /// order is known.
    ///
    /// - Parameters:
    ///   - top: The returned token ids and their logprobs. No id may appear twice.
    ///   - labelIDs: The label token ids, in option order.
    /// - Returns: One probability per label, in `labelIDs` order, and the entropy.
    /// - Precondition: `top` and `labelIDs` are not empty. A backend always returns at least one
    ///   token and a question always has at least one label, so an empty input is a programming
    ///   error, and this function stops rather than throws.
    public static func compute(
        top: [(tokenID: Int, logprob: Double)], labelIDs: [Int]
    ) -> (probabilities: [Double], entropy: Double) {
        guard let smallest = top.lazy.map(\.logprob).min() else {
            preconditionFailure("SlotDistribution.compute needs at least one top logprob")
        }
        let lookup = Dictionary(top.map { ($0.tokenID, $0.logprob) }) { _, _ in
            preconditionFailure("SlotDistribution.compute got a token id twice in top")
        }
        let floor = smallest - 5.0
        let labelLogprobs = labelIDs.map { lookup[$0] ?? floor }
        guard let largest = labelLogprobs.max() else {
            preconditionFailure("SlotDistribution.compute needs at least one label")
        }
        let exponentials = labelLogprobs.map { exp($0 - largest) }
        let total = pythonSum(exponentials)
        let probabilities = exponentials.map { $0 / total }
        let entropy = -pythonSum(
            top.lazy.map { exp($0.logprob) }.filter { $0 > 0 }.map { $0 * log($0) })
        return (probabilities, entropy)
    }

    /// The same as the ordered overload of `compute(top:labelIDs:)`, for a dictionary, whose
    /// iteration order is unspecified.
    ///
    /// The probabilities do not depend on the order. The entropy can differ in its last bit from
    /// Python's for the same entries in the backend's order.
    ///
    /// - Precondition: `top` and `labelIDs` are not empty.
    public static func compute(
        top: [Int: Double], labelIDs: [Int]
    ) -> (probabilities: [Double], entropy: Double) {
        compute(top: top.map { (tokenID: $0.key, logprob: $0.value) }, labelIDs: labelIDs)
    }
}
