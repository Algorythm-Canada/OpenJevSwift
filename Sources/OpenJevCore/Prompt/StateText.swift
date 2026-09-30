// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, the state
// text of `Engine.decide`. Apache-2.0. See THIRD_PARTY.md.

/// The user turn's text: the state the questions are about.
public enum StateText {
    /// A string state as sent, anything else as `json.dumps(state, ensure_ascii=False)`.
    ///
    /// Unlike instructions and descriptions, a string state is not stripped.
    ///
    /// - Precondition: The state holds no infinite or NaN float and no malformed integer text.
    ///   ``JSONParser`` never produces either (decision D-016).
    public static func render(_ state: JSONValue) -> String {
        if case .string(let text) = state {
            return text
        }
        return TextOf.modelText(state)
    }
}
