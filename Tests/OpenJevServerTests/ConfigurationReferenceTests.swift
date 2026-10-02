import Foundation
import OpenJevCore
import Testing

@testable import OpenJevServer

/// The configuration reference, `Sources/OpenJevServer/Documentation.docc/Configuration.md`,
/// against the code and against `docs/deployment.md`, so the documented table cannot drift from
/// what the server reads.
///
/// The table must list exactly the variables ``ServerSettings/init(environment:)`` reads, found in
/// its source, and `OPENJEV_ENCODER_MODELS`, which the encoder store reads, documented as unset
/// since the store takes a missing or empty value for no folder. Each documented default must be
/// the code's: setting a variable to it, or to the empty string when the table says unset, must
/// give the same settings as leaving it out, and for the three numbers an empty value leaves at
/// their default, the setting itself must hold the documented value or nothing. And every variable
/// the settings table of `docs/deployment.md` lists must be in the reference with the same default.
@Suite("Configuration reference")
struct ConfigurationReferenceTests {
    /// The repository root, found relative to this source file.
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // OpenJevServerTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root

    /// The reference, relative to the root.
    static let referencePath = "Sources/OpenJevServer/Documentation.docc/Configuration.md"
    /// The deployment guide, relative to the root.
    static let deploymentPath = "docs/deployment.md"
    /// The source of ``ServerSettings``, whose environment reads the table lists.
    static let settingsSourcePath = "Sources/OpenJevServer/ServerSettings.swift"
    /// The source of the encoder store, which reads ``storeVariable``.
    static let storeSourcePath = "Sources/OpenJevEncoders/Store/EncoderPackageStore.swift"
    /// The one variable the server reads outside ``ServerSettings``: the encoder store's local
    /// packages folder, read on macOS only.
    static let storeVariable = "OPENJEV_ENCODER_MODELS"

    /// The variables read with `env.optionalInteger` and `env.optionalDouble`, as upstream's
    /// `_env_num` reads, for which an empty value means the default as a missing one does, so
    /// setting one to the empty string cannot tell an unset default from any other. Each maps to
    /// its setting, nil when the settings hold none.
    static let numberSettings: [String: @Sendable (ServerSettings) -> Double?] = [
        "OPENJEV_MLX_CACHE_LIMIT_GB": { $0.mlxCacheLimitGB },
        "OPENJEV_MLX_PROMPT_CACHE": { Double($0.mlxPromptCache) },
        "OPENJEV_ENCODER_FUNCTIONS": { $0.encoderFunctions.map(Double.init) },
    ]

    /// One row of a settings table: the variable and its documented default, nil for unset.
    struct Row: Hashable, CustomStringConvertible {
        var name: String
        var defaultValue: String?

        var description: String { "\(name) (\(defaultValue.map { "`\($0)`" } ?? "unset"))" }
    }

    /// A file of the repository as text.
    static func text(_ path: String) throws -> String {
        try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
    }

    /// The rows of the markdown tables in `path` whose first cell is one `OPENJEV_` variable in
    /// code voice, with the second cell's default: a value in code voice, or `unset`.
    static func rows(of path: String) throws -> [Row] {
        var rows: [Row] = []
        for line in try text(path).split(separator: "\n") where line.hasPrefix("| `OPENJEV_") {
            let cells = line.split(separator: "|", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            // "", the variable, the default, and the rest.
            let name = try #require(codeVoice(cells[1]), "\(path): \(line)")
            let cell = cells[2]
            if cell == "unset" {
                rows.append(Row(name: name, defaultValue: nil))
            } else {
                let value: String = try #require(
                    codeVoice(cell), "\(path): the default of \(name) is neither code nor unset")
                rows.append(Row(name: name, defaultValue: value))
            }
        }
        return rows
    }

    /// The text of a cell that is one span of code voice, such as `` `8080` ``.
    static func codeVoice(_ cell: String) -> String? {
        guard cell.count >= 2, cell.first == "`", cell.last == "`" else { return nil }
        let inner = cell.dropFirst().dropLast()
        return inner.contains("`") ? nil : String(inner)
    }

    /// The variables `ServerSettings(environment:)` reads: the name in every
    /// `env.<reader>("OPENJEV_...")` call of its source whose reader's name starts with `prefix`.
    static func serverSettingsReads(prefix: String = "") throws -> Set<String> {
        let source = try text(settingsSourcePath)
        let read = try Regex(#"env\.\#(prefix)[a-zA-Z]+\(\s*"(OPENJEV_[A-Z0-9_]+)""#)
        return Set(
            source.matches(of: read).compactMap { $0.output[1].substring.map(String.init) })
    }

    @Test("The table lists each variable the server reads, once")
    func namesAreTheReads() throws {
        let rows = try Self.rows(of: Self.referencePath)
        let names = rows.map(\.name)
        #expect(Set(names).count == names.count, "a variable is listed twice: \(names)")
        let reads = try Self.serverSettingsReads()
        // ServerSettings reads the 31 variables of upstream's Settings and this port's.
        #expect(reads.count >= 31, "the source scan found \(reads.count) reads")
        let expected = reads.union([Self.storeVariable])
        let unread = Set(names).subtracting(expected).sorted()
        let unlisted = expected.subtracting(names).sorted()
        #expect(unread.isEmpty, "listed but never read: \(unread)")
        #expect(unlisted.isEmpty, "read but not listed: \(unlisted)")
    }

    /// The store's source, since `OpenJevEncoders` exists only on Apple platforms and this test runs
    /// on Linux too. `EncoderPackageStoreTests` checks there what the read does: a folder for a
    /// value, none for a missing or empty one.
    @Test("The encoder store reads the variable the table lists for it")
    func storeReadsItsVariable() throws {
        let source = try Self.text(Self.storeSourcePath)
        #expect(source.contains("localModelsVariable = \"\(Self.storeVariable)\""))
        #expect(
            source.contains("environment[Self.localModelsVariable]"),
            "EncoderPackageStore(environment:) no longer reads \(Self.storeVariable)")
    }

    @Test("Each documented default is the one ServerSettings applies")
    func defaultsAreTheCode() throws {
        let defaults = try ServerSettings(environment: [:])
        let rows = try Self.rows(of: Self.referencePath)
        #expect(!rows.isEmpty)
        // The store's variable has no ServerSettings to compare with: a missing or empty value
        // means no folder (EncoderPackageStoreTests), so the table must say unset.
        let store = try #require(rows.first { $0.name == Self.storeVariable })
        #expect(store.defaultValue == nil, "\(store) but the store's default is no folder")
        for row in rows where row.name != Self.storeVariable {
            let value = row.defaultValue ?? ""
            let settings = try ServerSettings(environment: [row.name: value])
            #expect(settings == defaults, "\(row.name)=\(value.pythonRepr) is not the default")
            // An empty value cannot show these defaults, so read the setting itself.
            if let setting = Self.numberSettings[row.name] {
                let held = setting(defaults)
                #expect(
                    held == row.defaultValue.flatMap(Double.init),
                    "\(row) but the settings hold \(held.map { "\($0)" } ?? "nothing")")
            }
        }
        let numberReads = try Self.serverSettingsReads(prefix: "optional")
        #expect(
            Set(Self.numberSettings.keys) == numberReads,
            "numberSettings must list the optional reads: \(numberReads.sorted())")
    }

    @Test("docs/deployment.md's settings table agrees with the reference")
    func deploymentAgrees() throws {
        var reference: [String: Row] = [:]
        for row in try Self.rows(of: Self.referencePath) {
            reference[row.name] = row
        }
        let deployment = try Self.rows(of: Self.deploymentPath)
        #expect(deployment.count >= 10, "deployment.md lists \(deployment.count) variables")
        for row in deployment {
            #expect(reference[row.name] == row, "deployment.md has \(row)")
        }
    }
}
