import Foundation
import OpenJevCore

/// Loads the upstream wire recordings in Fixtures/wire.
///
/// Tools/fixtures/wire_tables.py writes them from upstream's FastAPI app. Tests that need a file
/// are disabled with a message when it is missing, rather than failing.
public enum WireFixtures {
    /// Fixtures/wire.
    public static let directory = UpstreamFixtures.directory.appendingPathComponent("wire")

    /// The message shown when a file is missing.
    public static let missingMessageText =
        "Fixtures/wire is missing; run Tools/fixtures/.venv/bin/python Tools/fixtures/wire_tables.py"

    /// True when the named file exists.
    public static func exists(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
    }

    /// The named file, parsed.
    public static func load(_ name: String) throws -> JSONValue {
        try JSONParser().parse(Data(contentsOf: directory.appendingPathComponent(name)))
    }

    /// The recorded case named `name` in `cases.json`.
    public static func recordedCase(named name: String) throws -> JSONValue {
        let cases = try unwrap(load("cases.json")["cases"]?.arrayValue, "cases.json: cases")
        return try unwrap(cases.first { $0["name"]?.stringValue == name }, "no case \(name)")
    }

    /// The `GET /v1/models` listing recorded for `backend` in `models.json`.
    public static func listing(forBackend backend: String) throws -> JSONValue {
        let listings = try unwrap(
            load("models.json")["listings"]?.arrayValue, "models.json: listings")
        return try unwrap(
            listings.first { $0["backend"]?.stringValue == backend },
            "no listing for the backend \(backend)")
    }

    /// The request body bytes of a recorded case, with any placeholder expanded.
    public static func bodyBytes(of request: JSONValue) throws -> [UInt8]? {
        if let encoded = request["body_base64"]?.stringValue {
            return try unwrap(Data(base64Encoded: encoded), "body_base64 is not base64").map { $0 }
        }
        guard var text = request["body_text"]?.stringValue else { return nil }
        if let expand = request["body_expand"] {
            let placeholder = try unwrap(
                expand["placeholder"]?.stringValue, "body_expand: placeholder")
            let repeated = try unwrap(expand["repeat"]?.stringValue, "body_expand: repeat")
            let count = try unwrap(expand["count"]?.intValue, "body_expand: count")
            text = text.replacingOccurrences(
                of: placeholder, with: String(repeating: repeated, count: count))
        }
        return Array(text.utf8)
    }
}
