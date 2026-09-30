import Foundation
import OpenJevCore
import Testing

/// Loads the CPython reference tables in Fixtures/python-json.
///
/// Tools/fixtures/python_json_tables.py writes them. Tests that need a table are disabled with
/// a message when it is missing, rather than failing.
enum PythonFixtures {
    /// Fixtures/python-json, found relative to this source file.
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // JSON
        .deletingLastPathComponent()  // OpenJevCoreTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root
        .appendingPathComponent("Fixtures/python-json")

    /// The message shown when a table is missing.
    static let missingMessage: Comment =
        "Fixtures/python-json is missing; run python3 Tools/fixtures/python_json_tables.py"

    /// True when the named table exists.
    static func exists(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
    }

    /// The rows of the named table.
    static func rows(_ name: String) throws -> [JSONValue] {
        let data = try Data(contentsOf: directory.appendingPathComponent(name))
        let table = try JSONParser().parse(data)
        return table["rows"]?.arrayValue ?? []
    }
}
