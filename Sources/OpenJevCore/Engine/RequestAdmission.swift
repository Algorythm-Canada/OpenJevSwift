// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, the
// capability check and the queue bound of `EncoderEngine.decide`, and `openjev/engine.py`, the
// queue bound and the answer ordering of `Engine.decide`. Apache-2.0. See THIRD_PARTY.md.

/// The check upstream's encoder engines run before anything else: an option the backend cannot
/// honour, refused in upstream's field order.
///
/// ``DecisionEngine`` runs it against its backend's ``BackendCapabilities`` and
/// ``EncoderDecisionEngine`` against ``BackendCapabilities/readsOnly``, so the message, the
/// location and the order are decided once.
enum UnsupportedOptions {
    /// Refuses the first option of `images`, `steps`, `samples`, `think` and `sequential` that
    /// the request uses and `capabilities` lacks.
    ///
    /// `steps` and `samples` count as used above 1, `think` when it is not 0 and `sequential`
    /// when it is true, Python's truthiness of upstream's `unsupported` table.
    ///
    /// - Throws: ``SchemaError`` with `"{modelName} does not support {field}"` at
    ///   `["body", field]`.
    static func check(
        _ options: ReadOptions, hasImages: Bool, capabilities can: BackendCapabilities,
        modelName: String
    ) throws(SchemaError) {
        let unsupported: [(field: String, used: Bool)] = [
            ("images", hasImages && !can.images),
            ("steps", options.steps > 1 && !can.steps),
            ("samples", (options.samples ?? 0) > 1 && !can.samples),
            ("think", options.think != 0 && !can.think),
            ("sequential", options.sequential && !can.sequential),
        ]
        for entry in unsupported where entry.used {
            throw SchemaError(
                "\(modelName) does not support \(entry.field)", loc: ["body", .key(entry.field)])
        }
    }
}

/// Upstream's `waiting` counter and its bound, `if self.waiting >= self.s.max_queue: raise
/// Overloaded(...)`.
///
/// An engine actor holds one and calls ``admit(refusing:)`` when a request reaches the bound and
/// ``leave()`` when it finishes, on the actor, so the count is exact.
struct RequestQueue: Sendable {
    /// The most requests inside `decide` at once, `OPENJEV_MAX_QUEUE`.
    let limit: Int
    /// Requests inside `decide` right now.
    private(set) var waiting = 0

    /// Creates a queue bound.
    init(limit: Int) {
        self.limit = limit
    }

    /// Counts a request in, or refuses it when `limit` requests are already inside.
    ///
    /// - Throws: ``OverloadedError`` with `message`.
    mutating func admit(refusing message: String) throws(OverloadedError) {
        guard waiting < limit else {
            throw OverloadedError(message: message)
        }
        waiting += 1
    }

    /// Counts a request out.
    mutating func leave() {
        precondition(waiting > 0, "leave() without a matching admit()")
        waiting -= 1
    }
}

extension OrderedMap where Value == Answer {
    /// The answers reordered to `keys`, the request's question order, as upstream's `decide`
    /// returns `{k: answers[k] for k in questions}`.
    ///
    /// - Precondition: Every key has an answer.
    func ordered(as keys: [String]) -> OrderedMap<Answer> {
        var ordered = OrderedMap<Answer>()
        for key in keys {
            guard let answer = self[key] else {
                preconditionFailure("no answer for question \(key.pythonRepr)")
            }
            ordered.updateValue(answer, forKey: key)
        }
        return ordered
    }
}
