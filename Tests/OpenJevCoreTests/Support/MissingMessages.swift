import OpenJevTestSupport
import Testing

// The fixture loaders live in OpenJevTestSupport, which does not import Testing. These give the
// tests the `Comment` values that `.enabled(if:_:)` takes.

extension UpstreamFixtures {
    /// ``missingMessageText`` as a test comment.
    static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
}

extension WireFixtures {
    /// ``missingMessageText`` as a test comment.
    static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
}

extension PolicyFixtures {
    /// ``missingMessageText`` as a test comment.
    static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
}

extension FixtureTokenizer {
    /// ``missingMessageText`` as a test comment.
    static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
}
