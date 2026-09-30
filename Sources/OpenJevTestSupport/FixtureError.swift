import Foundation

/// A fixture file that is missing, malformed or lacks a value a loader needs.
///
/// The loaders throw it where a test would use `#require`; this module does not import Testing,
/// so that it stays a plain library every test target can depend on.
public struct FixtureError: Error, Sendable, Hashable, CustomStringConvertible {
    /// What was wrong.
    public var message: String

    /// Creates an error with a message.
    public init(_ message: String) {
        self.message = message
    }

    /// The message.
    public var description: String { message }
}

/// The value, or a ``FixtureError`` when it is `nil`. The fixture loaders' stand-in for
/// `#require`.
func unwrap<T>(_ value: T?, _ message: @autoclosure () -> String) throws -> T {
    guard let value else {
        throw FixtureError(message())
    }
    return value
}
