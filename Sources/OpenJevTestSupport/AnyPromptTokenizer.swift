import OpenJevCore

/// ``FixtureTokenizer``, except that a chat prompt it has no recording of gets stand-in ids: one
/// per UTF-8 byte of the system and user texts.
///
/// The wire recordings use requests, such as one noul question about `x`, whose prompt the
/// tokenizer fixtures never rendered, and an SDK may send a request no fixture holds. The read
/// only needs to reach a stub backend, which never looks at the prompt's ids. Label discovery and
/// answer templates still come from the recordings, so the questions resolve as upstream resolves
/// them.
public struct AnyPromptTokenizer: DecisionTokenizer {
    /// The recorded tokenizations.
    public let recorded = FixtureTokenizer.shared

    /// Creates the tokenizer over the shared tables.
    public init() {}

    public func encode(_ text: String, addSpecialTokens: Bool) throws -> [Int] {
        try recorded.encode(text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool) throws -> String {
        try recorded.decode(ids, skipSpecialTokens: skipSpecialTokens)
    }

    public func chatPromptIDs(system: String, user: String, thinking: Bool) throws -> [Int] {
        if let ids = try? recorded.chatPromptIDs(system: system, user: user, thinking: thinking) {
            return ids
        }
        return (system + user).utf8.map(Int.init)
    }
}
