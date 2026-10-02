import Foundation

/// The per-read report of the oracle parity tests (issue #31): when a slot's top label differs
/// from the oracle's, or its largest label probability difference exceeds ``threshold``, the
/// read's id, prompt key, width and steps and both distributions, so a failure shows what moved.
/// D-014's aggregate bounds stay the assertions; these figures are reported.
enum ReadDivergence {
    /// The largest |dp| of a slot above which the slot is reported.
    static let threshold = 0.05

    /// One slot's comparison.
    struct Slot {
        /// The slot's label token ids, in option order.
        var labelIDs: [Int]
        /// Our label probabilities, in option order.
        var ours: [Double]
        /// The oracle's.
        var oracle: [Double]
    }

    /// The index of the first largest value, as Python's `max(range(n), key=p.__getitem__)`.
    static func firstLargest(_ values: [Double]) -> Int {
        var best = 0
        for index in values.indices where values[index] > values[best] {
            best = index
        }
        return best
    }

    /// The report of one read, or nil when every slot keeps the oracle's top label and stays
    /// within ``threshold``.
    static func report(
        id: String, prompt: String, width: Int, steps: Int, slots: [Slot]
    ) -> String? {
        var lines: [String] = []
        for (index, slot) in slots.enumerated() {
            let largest = zip(slot.ours, slot.oracle).map { abs($0 - $1) }.max() ?? 0
            let topMoved = firstLargest(slot.ours) != firstLargest(slot.oracle)
            guard topMoved || largest > threshold else { continue }
            let reason = [
                topMoved ? "top label differs" : nil,
                largest > threshold ? String(format: "max |dp| %.4f", largest) : nil,
            ].compactMap { $0 }.joined(separator: ", ")
            lines.append(
                "  slot \(index) (\(reason)):\n"
                    + "    ours   \(render(slot.labelIDs, slot.ours))\n"
                    + "    oracle \(render(slot.labelIDs, slot.oracle))")
        }
        guard !lines.isEmpty else { return nil }
        return "\(id) (prompt \(prompt), width \(width), steps \(steps)):\n"
            + lines.joined(separator: "\n")
    }

    /// `[id: p, ...]` with four decimals.
    private static func render(_ ids: [Int], _ probabilities: [Double]) -> String {
        "["
            + zip(ids, probabilities).map { "\($0): \(String(format: "%.4f", $1))" }
            .joined(separator: ", ") + "]"
    }
}
