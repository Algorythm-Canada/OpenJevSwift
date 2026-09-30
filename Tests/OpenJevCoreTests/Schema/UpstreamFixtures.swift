import Foundation
import OpenJevCore
import Testing

/// Loads the engine tables in Fixtures/ that Tools/fixtures/upstream_tables.py writes from
/// upstream's engine and the pinned tokenizer: schemas, system texts, templates, the tokenizer
/// corpus and the choice labels.
///
/// Tests that need a file are disabled with a message when it is missing, rather than failing.
enum UpstreamFixtures {
    /// Fixtures/, found relative to this source file.
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Schema
        .deletingLastPathComponent()  // OpenJevCoreTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root
        .appendingPathComponent("Fixtures")

    /// The message shown when a file is missing.
    static let missingMessage: Comment =
        "An engine fixture is missing; run make upstream, make fixtures-venv and make fixtures"

    /// True when every named file exists. Paths are relative to Fixtures/.
    static func exists(_ paths: String...) -> Bool {
        paths.allSatisfy {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    /// The named file, parsed. The path is relative to Fixtures/.
    static func load(_ path: String) throws -> JSONValue {
        try JSONParser().parse(Data(contentsOf: directory.appendingPathComponent(path)))
    }

    /// The `cases` array of the named file.
    static func cases(_ path: String) throws -> [JSONValue] {
        try #require(load(path)["cases"]?.arrayValue)
    }

    /// The 255 single-token choice labels of labels.json, in order.
    static func choiceLabels() throws -> [String] {
        let labels = try #require(load("labels.json")["labels"]?.arrayValue)
        return try labels.map { try #require($0.stringValue) }
    }

    /// The schema upstream's engine builds for a recorded request body, built the same way: the
    /// body through ``RequestValidator``, then ``QuestionSchemaBuilder`` with the recorded labels.
    static func schema(for request: JSONValue, labels: [String]) throws -> QuestionSchema {
        let decoded = try RequestValidator().validate(request)
        return try QuestionSchemaBuilder(choiceLabels: labels).build(decoded.questions)
    }

    /// The read questions whose ids are listed, in the listed order.
    static func questions(_ ids: JSONValue?, in schema: QuestionSchema) throws -> [ReadQuestion] {
        let list = try #require(ids?.arrayValue)
        return try list.map { id in
            try #require(schema.questions.first { $0.id == id.stringValue }, "no question \(id)")
        }
    }
}
