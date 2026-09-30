// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, class
// `SchemaError`. Apache-2.0. See THIRD_PARTY.md.

/// A request the model cannot answer as asked, surfaced as a 400 with a plain-string detail.
///
/// The image checks, the schema builder and the engine throw this error. The server turns it
/// into a response with `WireError.semantic400(_:)`: the body carries only the
/// message. The `loc` says where in the request the problem is; upstream logs it and never sends
/// it, and so does this project.
public struct SchemaError: Error, Sendable, Hashable, CustomStringConvertible {
    /// The reason, sent as the body's `detail`.
    public var message: String
    /// Where the problem is, starting with `body`. Logged, never sent.
    public var loc: [LocComponent]

    /// Creates an error. The location defaults to `["body"]`, as upstream's does.
    public init(_ message: String, loc: [LocComponent] = ["body"]) {
        self.message = message
        self.loc = loc
    }

    /// The location and the message, for logs and test failures.
    public var description: String {
        "\(loc.map(\.description).joined(separator: ".")): \(message)"
    }
}
