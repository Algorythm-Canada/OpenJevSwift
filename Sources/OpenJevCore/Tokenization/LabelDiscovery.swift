// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`,
// `Engine._single_token_labels` and the constant `MAX_CHOICES`. Apache-2.0. See THIRD_PARTY.md.

/// The choice labels a tokenizer allows and the single token id of each after `"q1: "`.
public struct LabelSet: Sendable, Hashable {
    /// The labels in order: `A`, `B`, `C`, and so on.
    public var labels: [String]
    /// The id each label becomes after the prefix, in the same order. The ids are distinct.
    public var labelIDs: [Int]

    /// Creates a label set.
    public init(labels: [String], labelIDs: [Int]) {
        self.labels = labels
        self.labelIDs = labelIDs
    }
}

/// The labels the engine reads answers under.
///
/// A slot on the canvas holds one token, so each answer label must be one token where the answer
/// template writes it. Noul and score labels are fixed; choice labels are discovered from the
/// tokenizer, as upstream's engine does at start-up.
public enum LabelDiscovery {
    /// The most choice labels discovery returns, upstream's `MAX_CHOICES` and Jev's limit on one
    /// choice's options.
    public static let maxChoices = 255

    /// The labels of a noul question: `yes`, then `no`.
    public static let noulLabels = ["yes", "no"]

    /// The labels of a score question's levels: `0` to `9`, the first `n` for `n` levels.
    public static let scoreLabels = (0..<10).map { String($0) }

    /// The prefix every candidate is tokenized after, the head of an answer line.
    public static let prefix = "q1: "

    /// The candidates in upstream's order: `A` to `Z`, `a` to `z`, then `AA` to `ZZ`.
    public static let candidates: [String] = {
        let upper = (UInt8(ascii: "A")...UInt8(ascii: "Z")).map { String(UnicodeScalar($0)) }
        let lower = (UInt8(ascii: "a")...UInt8(ascii: "z")).map { String(UnicodeScalar($0)) }
        let pairs = upper.flatMap { first in upper.map { first + $0 } }
        return upper + lower + pairs
    }()

    /// The choice labels that stay one token after ``prefix``, upstream's `_single_token_labels`.
    ///
    /// The base is `enc("q1: A")`. A candidate is kept when `enc("q1: " + candidate)` has as many
    /// tokens as the base, the same tokens except the last, and a last token no earlier label
    /// has. Discovery stops at ``maxChoices`` labels. `enc` is
    /// ``DecisionTokenizer/encode(_:addSpecialTokens:)`` without special tokens.
    ///
    /// - Throws: Whatever the tokenizer throws.
    public static func choiceLabels(using tokenizer: any DecisionTokenizer) throws -> LabelSet {
        let base = try tokenizer.encode(prefix + "A", addSpecialTokens: false)
        var labels: [String] = []
        var labelIDs: [Int] = []
        var seen: Set<Int> = []
        for candidate in candidates {
            let ids = try tokenizer.encode(prefix + candidate, addSpecialTokens: false)
            if ids.count == base.count, ids.dropLast() == base.dropLast(), let last = ids.last,
                !seen.contains(last)
            {
                seen.insert(last)
                labels.append(candidate)
                labelIDs.append(last)
            }
            if labels.count == maxChoices {
                break
            }
        }
        return LabelSet(labels: labels, labelIDs: labelIDs)
    }
}
