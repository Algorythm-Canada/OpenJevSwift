/// How far a set of distributions is from a reference, in the terms of the bounds spike #56 set
/// for the encoder backends (docs/spikes/encoder-runtime.md, "Scope notes"), in the style of
/// D-014: the largest and the mean absolute probability difference over every probability, and
/// the questions whose top answer changed although the reference's top two were at least 0.01
/// apart.
struct ParityBounds: CustomStringConvertible {
    static let maxDifferenceBound = 0.02
    static let meanDifferenceBound = 0.003
    static let topMarginBound = 0.01

    var maxDifference = 0.0
    var meanDifference = 0.0
    /// Questions whose top answer changed where the reference's margin was at least 0.01.
    var topChanges: [String] = []
    /// Questions whose top answer changed where the margin was smaller, which the bound allows.
    var closeTopChanges: [String] = []
    var questions = 0

    /// Compares each measured distribution with its reference.
    init(_ pairs: [(name: String, measured: [Double], reference: [Double])]) {
        var total = 0.0
        var count = 0
        for (name, measured, reference) in pairs {
            precondition(measured.count == reference.count, "\(name): lengths differ")
            for (a, b) in zip(measured, reference) {
                let difference = abs(a - b)
                maxDifference = max(maxDifference, difference.isNaN ? .infinity : difference)
                total += difference
                count += 1
            }
            let top = Self.top(reference)
            if Self.top(measured) != top {
                let sorted = reference.sorted(by: >)
                let margin = sorted.count > 1 ? sorted[0] - sorted[1] : 1
                if margin >= Self.topMarginBound {
                    topChanges.append(name)
                } else {
                    closeTopChanges.append(name)
                }
            }
            questions += 1
        }
        meanDifference = count > 0 ? total / Double(count) : 0
    }

    /// The index of the largest value; the first on a tie, as Python's `max` and upstream do.
    static func top(_ values: [Double]) -> Int {
        var best = 0
        for index in values.indices where values[index] > values[best] {
            best = index
        }
        return best
    }

    /// The bounds this comparison breaks, empty when it meets all three.
    var violations: [String] {
        var out: [String] = []
        if !(maxDifference <= Self.maxDifferenceBound) {
            out.append("largest difference \(maxDifference) > \(Self.maxDifferenceBound)")
        }
        if !(meanDifference <= Self.meanDifferenceBound) {
            out.append("mean difference \(meanDifference) > \(Self.meanDifferenceBound)")
        }
        if !topChanges.isEmpty {
            out.append("top answer changed with a margin of 0.01 or more: \(topChanges)")
        }
        return out
    }

    var description: String {
        "\(questions) questions: largest difference \(maxDifference), mean \(meanDifference), "
            + "top answers changed \(topChanges.count + closeTopChanges.count) "
            + "(with a margin under 0.01: \(closeTopChanges))"
    }
}
