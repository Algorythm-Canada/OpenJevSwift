import Foundation
import OpenJevCore
import Testing

/// Loads the upstream wire recordings in Fixtures/wire.
///
/// Tools/fixtures/wire_tables.py writes them from upstream's FastAPI app. Tests that need a file
/// are disabled with a message when it is missing, rather than failing.
enum WireFixtures {
    /// Fixtures/wire, found relative to this source file.
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Wire
        .deletingLastPathComponent()  // OpenJevCoreTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root
        .appendingPathComponent("Fixtures/wire")

    /// The message shown when a file is missing.
    static let missingMessage: Comment =
        "Fixtures/wire is missing; run Tools/fixtures/.venv/bin/python Tools/fixtures/wire_tables.py"

    /// True when the named file exists.
    static func exists(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
    }

    /// The named file, parsed.
    static func load(_ name: String) throws -> JSONValue {
        try JSONParser().parse(Data(contentsOf: directory.appendingPathComponent(name)))
    }

    /// The request body bytes of a recorded case, with any placeholder expanded.
    static func bodyBytes(of request: JSONValue) throws -> [UInt8]? {
        if let encoded = request["body_base64"]?.stringValue {
            return try #require(Data(base64Encoded: encoded)).map { $0 }
        }
        guard var text = request["body_text"]?.stringValue else { return nil }
        if let expand = request["body_expand"] {
            let placeholder = try #require(expand["placeholder"]?.stringValue)
            let repeated = try #require(expand["repeat"]?.stringValue)
            let count = try #require(expand["count"]?.intValue)
            text = text.replacingOccurrences(
                of: placeholder, with: String(repeating: repeated, count: count))
        }
        return Array(text.utf8)
    }
}
