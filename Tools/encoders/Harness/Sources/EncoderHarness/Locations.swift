import Foundation

/// Where the harness finds the reference fixtures, the two tokenizers and the Core ML packages.
///
/// An iPhone run reads everything from the `Staged` folder that Tools/encoders/stage_harness.sh
/// fills and Tools/encoders/HarnessApp.swiftpm copies into its bundle:
///
///     Staged/fixtures/verdict.json, laya.json
///     Staged/tokenizers/verdict/tokenizer.json, tokenizer_config.json
///     Staged/tokenizers/laya/tokenizer.json, tokenizer_config.json
///     Staged/models/<variant>.mlpackage
///
/// On macOS the harness can instead read the repository's Fixtures/encoders, the tokenizers in the
/// Hugging Face cache at the revisions the fixtures record, and the packages that
/// Tools/encoders/convert_*.py wrote to ~/Library/Caches/OpenJevSwift/encoders (or to
/// OPENJEV_ENCODER_MODELS).
public struct HarnessLocations: Sendable {
    public let fixtures: URL
    public let verdictTokenizer: URL
    public let layaTokenizer: URL
    public let models: URL

    public var verdictFixture: URL { fixtures.appendingPathComponent("verdict.json") }
    public var layaFixture: URL { fixtures.appendingPathComponent("laya.json") }

    public init(fixtures: URL, verdictTokenizer: URL, layaTokenizer: URL, models: URL) {
        self.fixtures = fixtures
        self.verdictTokenizer = verdictTokenizer
        self.layaTokenizer = layaTokenizer
        self.models = models
    }

    /// The staged folder's layout, if it holds the fixtures.
    public static func staged(in folder: URL) -> HarnessLocations? {
        let locations = HarnessLocations(
            fixtures: folder.appendingPathComponent("fixtures"),
            verdictTokenizer: folder.appendingPathComponent("tokenizers/verdict"),
            layaTokenizer: folder.appendingPathComponent("tokenizers/laya"),
            models: folder.appendingPathComponent("models"))
        return locations.hasFixtures ? locations : nil
    }

    /// The repository's fixtures with the Hugging Face cache and the converted-model cache.
    public static func repository(root: URL, environment: [String: String]) -> HarnessLocations? {
        let fixtures = root.appendingPathComponent("Fixtures/encoders")
        guard
            let generator = try? FixtureFile.generator(
                of: fixtures.appendingPathComponent("verdict.json"))
        else { return nil }
        // The iOS Simulator runs on the Mac and can read its files; its own home is the app's.
        let home = URL(fileURLWithPath: environment["SIMULATOR_HOST_HOME"] ?? NSHomeDirectory())
        let hub =
            environment["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
            ?? environment["HF_HOME"].map { URL(fileURLWithPath: $0).appendingPathComponent("hub") }
            ?? home.appendingPathComponent(".cache/huggingface/hub")
        let models =
            environment["OPENJEV_ENCODER_MODELS"].map { URL(fileURLWithPath: $0) }
            ?? home.appendingPathComponent("Library/Caches/OpenJevSwift/encoders")
        return HarnessLocations(
            fixtures: fixtures,
            verdictTokenizer: hub.appendingPathComponent(
                "models--heman10x--rlcd-modernbert-151m/snapshots/\(generator.verdictRevision)"),
            layaTokenizer: hub.appendingPathComponent(
                "models--convaiinnovations--laya-typed-decisions/snapshots/\(generator.layaRevision)/tokenizer"
            ),
            models: models)
    }

    public var hasFixtures: Bool {
        FileManager.default.fileExists(atPath: verdictFixture.path)
            && FileManager.default.fileExists(atPath: layaFixture.path)
    }

    public var hasTokenizers: Bool {
        [verdictTokenizer, layaTokenizer].allSatisfy {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("tokenizer.json").path)
        }
    }

    public func package(named name: String) -> URL {
        models.appendingPathComponent("\(name).mlpackage")
    }

    public func hasPackage(named name: String) -> Bool {
        FileManager.default.fileExists(atPath: package(named: name).path)
    }
}
