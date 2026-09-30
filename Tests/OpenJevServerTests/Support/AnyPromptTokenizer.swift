#if canImport(HummingbirdTesting)
    import OpenJevCore
    import OpenJevTestSupport

    /// ``FixtureTokenizer``, except that a chat prompt it has no recording of gets stand-in ids:
    /// one per UTF-8 byte of the system and user texts.
    ///
    /// The wire recordings use requests, such as one noul question about `x`, whose prompt the
    /// tokenizer fixtures never rendered. The tests that send them only need the read to reach
    /// a stub backend that fails or refuses, so the prompt's ids are never looked at. Label
    /// discovery and answer templates still come from the recordings, so the questions resolve
    /// as upstream resolves them.
    struct AnyPromptTokenizer: DecisionTokenizer {
        let recorded = FixtureTokenizer.shared

        func encode(_ text: String, addSpecialTokens: Bool) throws -> [Int] {
            try recorded.encode(text, addSpecialTokens: addSpecialTokens)
        }

        func decode(_ ids: [Int], skipSpecialTokens: Bool) throws -> String {
            try recorded.decode(ids, skipSpecialTokens: skipSpecialTokens)
        }

        func chatPromptIDs(system: String, user: String, thinking: Bool) throws -> [Int] {
            if let ids = try? recorded.chatPromptIDs(system: system, user: user, thinking: thinking)
            {
                return ids
            }
            return (system + user).utf8.map(Int.init)
        }
    }
#endif
