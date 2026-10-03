// A port of the `jevk5` package by Alibi Serikbay (github.com/allebee/jevk5 at v0.2.2, 0571ef3),
// `jevk5/prompt.py`: `METHODS`, `TEMPERATURES`, `groups`, `spread`, `_combine`, `_knockout` and
// `_tree`; and of upstream OpenJev's letter softmax, the `read` inside
// `JevK5Engine.read_question` (razorback16/openjev at dcd2094, `openjev/encoders.py`).
// Apache-2.0. See THIRD_PARTY.md.

import Foundation
import OpenJevCore

/// How JevK5 turns the logits of one pass into a distribution, and how it reads a question with
/// more than 16 options in several passes.
///
/// The arithmetic is Python's, in Python's order: every sum is ``/OpenJevCore/pythonSum(_:)``
/// (CPython 3.12's compensated `sum`), `exp` and `pow` are the platform's `libm`, as CPython's
/// `math.exp` and `float.__pow__` are, and every sort keeps Python's stable order, so on the same
/// logits the distributions are the package's to the last bit.
public enum JevK5Readout {
    /// How more than 16 options are combined: `knockout`, the package's default and upstream's,
    /// or `tree`.
    public enum Method: String, Sendable, Hashable, CaseIterable {
        /// Each group of at most 16 options is read, then a final of 16 among the groups' best.
        case knockout
        /// One pass whose letters stand for whole groups, then one pass per group.
        case tree

        /// `TEMPERATURES`: the temperature that sharpens the combined distribution. 0.77 for the
        /// knockout, fitted on MASSIVE; the tree is unfitted (1.0).
        public var temperature: Double {
            switch self {
            case .knockout: return 0.77
            case .tree: return 1.0
            }
        }
    }

    /// One forward pass: the calibrated distribution over the letters of at most 16 option
    /// texts, in their order.
    public typealias Reader = (_ texts: [String]) async throws -> [Double]

    /// Upstream's letter softmax: `softmax((v - max) / temperature)` over the letters' logits,
    /// in the order the logits come.
    ///
    /// Upstream reads vLLM's logprobs, the logits minus the full vocabulary's normaliser, which
    /// cancels in the difference from the maximum (the package's llama.cpp client notes it), so
    /// the logits are read here directly.
    public static func letterProbabilities(logits: [Double], temperature: Double) -> [Double] {
        guard let top = logits.max() else { return [] }
        let weights = logits.map { exp(($0 - top) / temperature) }
        let total = pythonSum(weights)
        return weights.map { $0 / total }
    }

    /// `groups(n, count)`: `count` contiguous runs covering `0..<n`, their sizes differing by at
    /// most one, the longer ones first.
    public static func groups(_ count: Int, into groupCount: Int) -> [Range<Int>] {
        let base = count / groupCount
        let extra = count % groupCount
        var runs: [Range<Int>] = []
        var start = 0
        for group in 0..<groupCount {
            let stop = start + base + (group < extra ? 1 : 0)
            runs.append(start..<stop)
            start = stop
        }
        return runs
    }

    /// `spread(read, texts, method, temperature)`: a probability for every option, from a reader
    /// that answers at most 16 lettered options.
    ///
    /// Up to 16 options the result is `read(texts)` itself. Beyond, the options are split in
    /// their order into ceil(n / 16) groups of near-equal size and combined by `method`; the
    /// combined distribution is then sharpened by `temperature` (by default the method's): each
    /// value raised to `1 / temperature` and the whole renormalised. Both methods take
    /// ceil(n / 16) + 1 passes up to 256 options; the passes run one after another.
    public static func spread(
        _ read: Reader, texts: [String], method: Method = .knockout, temperature: Double? = nil
    ) async throws -> [Double] {
        if texts.count <= JevK5Prompt.letters.count {
            return try await read(texts)
        }
        var probabilities = try await combine(read, texts: texts, method: method)
        let temperature = temperature ?? method.temperature
        if temperature != 1.0 {
            let exponent = 1 / temperature
            probabilities = probabilities.map { pow($0, exponent) }
            let total = pythonSum(probabilities)
            probabilities = probabilities.map { $0 / total }
        }
        return probabilities
    }

    /// `_combine`: one pass up to 16 options, else the method's weights normalised.
    static func combine(_ read: Reader, texts: [String], method: Method) async throws -> [Double] {
        if texts.count <= JevK5Prompt.letters.count {
            return try await read(texts)
        }
        let weights: [Double]
        switch method {
        case .knockout:
            weights = try await knockout(read, texts: texts)
        case .tree:
            weights = try await tree(read, texts: texts)
        }
        let total = pythonSum(weights)
        return weights.map { $0 / total }
    }

    /// The number of groups of at most 16 that `count` options need, `-(-count // 16)`.
    static func groupCount(_ count: Int) -> Int {
        (count + JevK5Prompt.letters.count - 1) / JevK5Prompt.letters.count
    }

    /// `_knockout`: each group read, then a final of 16 among the groups' best.
    ///
    /// The final holds every group's top `16 // groups` options (at least one), its free places
    /// going to the most likely of the rest in any group. Finalists keep the final's
    /// distribution times the chance that the answer is a finalist; every other option gets its
    /// group's share of the final times its in-group probability. Ties go to the earlier option,
    /// as the package's stable sorts give them.
    static func knockout(_ read: Reader, texts: [String]) async throws -> [Double] {
        let letters = JevK5Prompt.letters.count
        let runs = groups(texts.count, into: groupCount(texts.count))
        var inner: [[Double]] = []
        for run in runs {
            inner.append(try await read(Array(texts[run])))
        }
        inner = inner.map { probabilities in
            let total = pythonSum(probabilities)
            return probabilities.map { $0 / total }
        }
        let keep = max(1, letters / runs.count)
        // `sorted(range(len(p)), key=lambda j: -p[j])`: by descending probability, ties in index
        // order.
        let ranked = inner.map { probabilities in
            probabilities.indices.sorted { a, b in
                let keyA = -probabilities[a]
                let keyB = -probabilities[b]
                return keyA < keyB || (keyA == keyB && a < b)
            }
        }
        var chosen = Set<Pair>()
        for (group, order) in ranked.enumerated() {
            for index in order.prefix(keep) {
                chosen.insert(Pair(group: group, index: index))
            }
        }
        // The rest in the order the generator walks them (group by group, each in its ranked
        // order), then stably sorted by descending probability.
        var rest: [Pair] = []
        for (group, order) in ranked.enumerated() {
            for index in order.dropFirst(keep) {
                rest.append(Pair(group: group, index: index))
            }
        }
        rest = rest.enumerated().sorted { a, b in
            let keyA = -inner[a.element.group][a.element.index]
            let keyB = -inner[b.element.group][b.element.index]
            return keyA < keyB || (keyA == keyB && a.offset < b.offset)
        }.map(\.element)
        for pair in rest.prefix(max(0, letters - chosen.count)) {
            chosen.insert(pair)
        }
        let tops = runs.indices.map { group in
            chosen.filter { $0.group == group }.map(\.index).sorted()
        }
        var finalists: [String] = []
        for (run, top) in zip(runs, tops) {
            for index in top {
                finalists.append(texts[run.lowerBound + index])
            }
        }
        let final = try await combine(read, texts: finalists, method: .knockout)
        // shares[g]: each finalist of group g with its probability in the final, in index order.
        var shares: [[(index: Int, value: Double)]] = []
        var at = 0
        for top in tops {
            shares.append(top.enumerated().map { offset, index in (index, final[at + offset]) })
            at += top.count
        }
        let inFinal = pythonSum(
            zip(inner, shares).map { probabilities, share in
                pythonSum(share.map(\.value)) * pythonSum(share.map { probabilities[$0.index] })
            })
        var weights: [Double] = []
        for (probabilities, share) in zip(inner, shares) {
            let mass = pythonSum(share.map(\.value))
            let finalByIndex = Dictionary(uniqueKeysWithValues: share.map { ($0.index, $0.value) })
            for (index, value) in probabilities.enumerated() {
                if let finalValue = finalByIndex[index] {
                    weights.append(finalValue * inFinal)
                } else {
                    weights.append(mass * value)
                }
            }
        }
        return weights
    }

    /// `_tree`: one pass whose letters stand for whole groups (each described as `"One of: "`
    /// and its members joined by `"; "`), then one pass per group:
    /// P(option) = P(its group) * P(option | its group).
    static func tree(_ read: Reader, texts: [String]) async throws -> [Double] {
        let runs = groups(
            texts.count, into: min(JevK5Prompt.letters.count, groupCount(texts.count)))
        let outer = try await read(
            runs.map { run in "One of: " + texts[run].joined(separator: "; ") })
        var weights: [Double] = []
        for (run, share) in zip(runs, outer) {
            let inner = try await combine(read, texts: Array(texts[run]), method: .tree)
            weights += inner.map { share * $0 }
        }
        return weights
    }

    /// An option in the knockout: its group and its index in the group.
    struct Pair: Hashable {
        var group: Int
        var index: Int
    }
}
