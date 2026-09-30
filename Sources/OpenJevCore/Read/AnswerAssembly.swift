// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, function
// `to_answer`. Apache-2.0. See THIRD_PARTY.md.

extension Answer {
    /// The answer to a question from its label probabilities, in Jev's shapes.
    ///
    /// - noul: the probability of yes, which is `p[0]`.
    /// - choice: the option at the first largest probability (Python's `max` keeps the first of
    ///   equal values), every option's probability keyed by its name in criteria order, and
    ///   ``Confidence/compute(_:)``.
    /// - score: the expected level `sum(i * p[i])`, the levels as sent, the probabilities in
    ///   level order, and ``Confidence/compute(_:)``.
    ///
    /// A single-option choice or a single-level score with `p == [1.0]` gives the same answer
    /// upstream forces for those questions without a read.
    ///
    /// - Precondition: `p.count` is the question's option count: 2 for noul, the number of
    ///   criteria otherwise.
    public static func make(for question: Question, probabilities p: [Double]) -> Answer {
        switch question {
        case .noul:
            precondition(p.count == 2, "a noul answer needs 2 probabilities, got \(p.count)")
            return .noul(p[0])
        case .choice(_, let criteria):
            precondition(
                p.count == criteria.count,
                "a choice answer needs \(criteria.count) probabilities, got \(p.count)")
            precondition(!p.isEmpty, "a choice answer needs at least one probability")
            var top = 0
            for index in p.indices where p[index] > p[top] {
                top = index
            }
            return .choice(
                choice: criteria.keys[top],
                probabilities: OrderedMap(uniqueKeysWithValues: zip(criteria.keys, p)),
                confidence: Confidence.compute(p))
        case .score(_, let criteria):
            precondition(
                p.count == criteria.count,
                "a score answer needs \(criteria.count) probabilities, got \(p.count)")
            let score = pythonSum(p.enumerated().lazy.map { Double($0.offset) * $0.element })
            return .score(
                score: score, legend: criteria, probabilities: p,
                confidence: Confidence.compute(p))
        }
    }
}
