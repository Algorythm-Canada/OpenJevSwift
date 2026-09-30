import Foundation
import OpenJevCore
import Testing

/// Checks that every file under Fixtures/ records the pins it was generated from.
///
/// The scripts in Tools/fixtures start every file with a `generator` object. A file regenerated
/// after a pin moved (another upstream commit or tokenizer revision) fails here, so a stale or
/// mixed set of fixtures is caught before any test compares against it. When a pin moves, update
/// the constants below together with the Makefile, THIRD_PARTY.md and the scripts.
@Suite("Fixture pins")
struct FixturePinTests {
    /// The upstream OpenJev commit, as `UPSTREAM_OPENJEV_COMMIT` in the Makefile.
    static let upstreamCommit = "dcd2094"
    /// The tokenizer the fixtures were computed with, as THIRD_PARTY.md pins it.
    static let tokenizerRepository = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
    static let tokenizerRevision = "a7a81407613811e8ba63af92ac0d852b809e191f"
    /// The encoder checkpoints Fixtures/encoders was computed with, as Tools/encoders/common.py
    /// pins them (checked below). Those files use each checkpoint's own tokenizer.
    static let verdictRevision = "8af2496eb63c7fa66d7d234e1f62629380030eb4"
    static let layaRevision = "1a793eb568e6718f15941d08f85432581df534e3"

    /// The repository root, found relative to this source file.
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Fixtures
        .deletingLastPathComponent()  // OpenJevCoreTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root

    static let fixtures = root.appendingPathComponent("Fixtures")

    /// Every JSON file under Fixtures/, relative to it, sorted.
    static func files() -> [String] {
        let enumerator = FileManager.default.enumerator(atPath: fixtures.path)
        let paths = enumerator?.compactMap { $0 as? String } ?? []
        return paths.filter { $0.hasSuffix(".json") }.sorted()
    }

    @Test("Every fixture file records the pins it was generated from")
    func pins() throws {
        let files = Self.files()
        #expect(files.count >= 22, "found only \(files.count) fixture files")
        var problems: [String] = []
        for file in files {
            let value = try JSONParser().parse(
                Data(contentsOf: Self.fixtures.appendingPathComponent(file)))
            guard let generator = value["generator"] else {
                problems.append("\(file): no generator object")
                continue
            }
            problems += Self.problems(in: generator, of: file)
        }
        #expect(problems.isEmpty, "\(problems.count) problems: \(problems.prefix(10))")
    }

    @Test("The expected upstream commit is the Makefile's")
    func makefileAgrees() throws {
        let makefile = try String(
            contentsOf: Self.root.appendingPathComponent("Makefile"), encoding: .utf8)
        #expect(makefile.contains("UPSTREAM_OPENJEV_COMMIT := \(Self.upstreamCommit)\n"))
        let thirdParty = try String(
            contentsOf: Self.root.appendingPathComponent("THIRD_PARTY.md"), encoding: .utf8)
        #expect(thirdParty.contains("`\(Self.upstreamCommit)`"))
        #expect(thirdParty.contains("`\(Self.tokenizerRevision.prefix(8))`"))
    }

    @Test("The expected encoder pins are the ones Tools/encoders/common.py generates with")
    func encoderScriptsAgree() throws {
        let common = try String(
            contentsOf: Self.root.appendingPathComponent("Tools/encoders/common.py"),
            encoding: .utf8)
        #expect(common.contains("\nUPSTREAM_COMMIT = \"\(Self.upstreamCommit)\"\n"))
        #expect(common.contains("\nVERDICT_REVISION = \"\(Self.verdictRevision)\"\n"))
        #expect(common.contains("\nLAYA_REVISION = \"\(Self.layaRevision)\"\n"))
    }

    /// What is wrong with one file's generator object, if anything.
    ///
    /// python-json/ holds CPython reference tables, which involve neither upstream nor the
    /// tokenizer. wire/ was recorded from upstream with a stand-in tokenizer. encoders/ holds the
    /// Verdict and Laya reference outputs that Tools/encoders records through upstream's code
    /// with each model's own tokenizer, so it pins the checkpoints instead of the tokenizer.
    /// model/ holds the DiffusionGemma checkpoint's own JSON files, which involve no upstream
    /// code, so it pins the checkpoint (the tokenizer's repository and revision) and not upstream.
    /// Every other file comes from upstream's code with the real tokenizer and records both pins
    /// and the version of the script that wrote it.
    static func problems(in generator: JSONValue, of file: String) -> [String] {
        var out: [String] = []
        func expect(_ key: String, _ value: String) {
            let found = generator[key]?.stringValue
            if found != value {
                out.append("\(file): generator.\(key) is \(found ?? "missing"), expected \(value)")
            }
        }
        let encoders = file.hasPrefix("encoders/")
        let scripts = encoders ? "Tools/encoders" : "Tools/fixtures"
        if generator["script"]?.stringValue?.hasPrefix(scripts + "/") != true {
            out.append("\(file): generator.script does not name a script in \(scripts)")
        }
        if generator["python"]?.stringValue == nil {
            out.append("\(file): generator.python is missing")
        }
        if file.hasPrefix("python-json/") {
            return out
        }
        if file.hasPrefix("model/") {
            expect("model_repo", tokenizerRepository)
            expect("model_revision", tokenizerRevision)
            if generator["version"]?.intValue == nil {
                out.append("\(file): generator.version is missing")
            }
            return out
        }
        expect("upstream", "razorback16/openjev")
        expect("upstream_commit", upstreamCommit)
        if file.hasPrefix("wire/") {
            return out
        }
        if encoders {
            expect("verdict_revision", verdictRevision)
            expect("laya_revision", layaRevision)
        } else {
            expect("tokenizer_repo", tokenizerRepository)
            expect("tokenizer_revision", tokenizerRevision)
        }
        if generator["version"]?.intValue == nil {
            out.append("\(file): generator.version is missing")
        }
        return out
    }
}
