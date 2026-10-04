import Foundation
import OpenJevCore
import OpenJevDiffusionGemma
import OpenJevLetterReadout
import OpenJevServer
import Testing

/// Where JevK5's files come from: `OPENJEV_JEVK5_MODEL`'s two forms, the published conversions at
/// their pinned commits, a folder's required files, and the conversions' digests against
/// `Tools/jevk5/convert.py`, which wrote them.
@Suite("JevK5 checkpoint")
struct JevK5CheckpointTests {
    @Test("OPENJEV_JEVK5_MODEL names a folder or a Hub repository, as OPENJEV_MLX_MODEL does")
    func settingForms() {
        let home = NSHomeDirectory()
        #expect(
            JevK5ModelFiles.source(setting: "~/models/jevk5")
                == .directory(URL(fileURLWithPath: home + "/models/jevk5", isDirectory: true)))
        #expect(
            JevK5ModelFiles.source(setting: "/opt/jevk5")
                == .directory(URL(fileURLWithPath: "/opt/jevk5", isDirectory: true)))
        #expect(
            JevK5ModelFiles.source(setting: "org/repo@abc123")
                == .hub(repository: "org/repo", revision: "abc123"))
        #expect(
            JevK5ModelFiles.source(setting: "org/repo")
                == .hub(repository: "org/repo", revision: nil))
        // A conversion's repository takes the conversion's pinned commit.
        #expect(
            JevK5ModelFiles.source(setting: JevK5Checkpoint.defaultSetting)
                == JevK5Checkpoint.eightBit.hubSource)
        #expect(
            JevK5ModelFiles.source(setting: "Algorythm-Canada/jevk5-0.2-mlx-4bit")
                == JevK5Checkpoint.fourBit.hubSource)
        #expect(
            JevK5ModelFiles.source(setting: "Algorythm-Canada/jevk5-0.2-mlx-8bit@main")
                == .hub(repository: "Algorythm-Canada/jevk5-0.2-mlx-8bit", revision: "main"))
    }

    @Test("The server's default is the 8-bit conversion's repository, the platform's on macOS")
    func serverDefault() throws {
        #expect(try ServerSettings().jevk5Model == JevK5Checkpoint.defaultSetting)
        #expect(JevK5Checkpoint.defaultSetting == "Algorythm-Canada/jevk5-0.2-mlx-8bit")
        #expect(JevK5Checkpoint.platformDefault == .eightBit)
        #expect(
            try ServerSettings(environment: ["OPENJEV_JEVK5_MODEL": "/tmp/x"]).jevk5Model
                == "/tmp/x")
    }

    @Test("Both conversions are published: each repository loads at the commit the Hub gave it")
    func publishedDefaults() {
        #expect(
            JevK5Checkpoint.eightBit.hubSource
                == .hub(
                    repository: "Algorythm-Canada/jevk5-0.2-mlx-8bit",
                    revision: "d19a6f09b42fdd3b4ff7fc85b1ebbda2a79bfa4a"))
        #expect(
            JevK5Checkpoint.fourBit.hubSource
                == .hub(
                    repository: "Algorythm-Canada/jevk5-0.2-mlx-4bit",
                    revision: "e3807fbf27a8b8f7ad277e331935bbc4368513db"))
        // Commits, which the resolver uses as they are, not branches it would ask the Hub about.
        for checkpoint in JevK5Checkpoint.all {
            #expect(checkpoint.revision.count == 40, "\(checkpoint.repository)")
            #expect(
                checkpoint.revision.allSatisfy { $0.isHexDigit && !$0.isUppercase },
                "\(checkpoint.repository)")
        }
    }

    @Test("The default resolves at its pinned commit: with the Hub unreachable, its snapshot loads")
    func defaultResolvesAtItsCommit() async throws {
        let cache = HubCacheLocation(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("jevk5-cache-\(UUID().uuidString)", isDirectory: true))
        defer { try? FileManager.default.removeItem(at: cache.directory) }
        let checkpoint = JevK5Checkpoint.eightBit
        let snapshot = cache.snapshotDirectory(checkpoint.repository, commit: checkpoint.revision)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        for name in JevK5ModelFiles.requiredFiles where name != "model.safetensors.index.json" {
            try Data("{}".utf8).write(to: snapshot.appendingPathComponent(name))
        }
        try Data(#"{"weight_map": {"a": "model.safetensors"}}"#.utf8)
            .write(to: snapshot.appendingPathComponent("model.safetensors.index.json"))
        try Data().write(to: snapshot.appendingPathComponent("model.safetensors"))
        // Nothing listens on port 9, so only a cached snapshot of the commit asked for can load.
        let resolver = ModelResolver(endpoint: URL(string: "http://127.0.0.1:9")!, maxAttempts: 1)
        for source in [
            JevK5ModelFiles.source(setting: JevK5Checkpoint.defaultSetting),
            .hub(repository: checkpoint.repository, revision: nil),
        ] {
            #expect(
                try await JevK5ModelFiles.resolve(source, cache: cache, resolver: resolver)
                    == snapshot, "\(source)")
        }
        // A revision named in the setting is kept: `main` has no snapshot here and needs the Hub.
        await #expect(throws: ModelResolverError.self) {
            _ = try await JevK5ModelFiles.resolve(
                JevK5ModelFiles.source(setting: JevK5Checkpoint.defaultSetting + "@main"),
                cache: cache, resolver: resolver)
        }
    }

    @Test("A folder must hold the checkpoint's files and every shard its index names")
    func folderFiles() async throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("jevk5-files-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(JevK5ModelFiles.missingFiles(in: folder) == JevK5ModelFiles.requiredFiles)
        for name in JevK5ModelFiles.requiredFiles where name != "model.safetensors.index.json" {
            try Data("{}".utf8).write(to: folder.appendingPathComponent(name))
        }
        try Data(#"{"weight_map": {"a": "model.safetensors"}}"#.utf8)
            .write(to: folder.appendingPathComponent("model.safetensors.index.json"))
        #expect(JevK5ModelFiles.missingFiles(in: folder) == ["model.safetensors"])
        await #expect(throws: JevK5LoadError.self) {
            _ = try await JevK5ModelFiles.resolve(.directory(folder))
        }
        try Data().write(to: folder.appendingPathComponent("model.safetensors"))
        #expect(JevK5ModelFiles.missingFiles(in: folder).isEmpty)
        #expect(try await JevK5ModelFiles.resolve(.directory(folder)) == folder)
    }

    @Test("The conversions' digests are the ones Tools/jevk5/convert.py pins")
    func digestsMatchTheScript() throws {
        let script = try String(
            contentsOf: JevK5Fixtures.root.appendingPathComponent("Tools/jevk5/convert.py"),
            encoding: .utf8)
        #expect(script.contains("SOURCE_REVISION = \"\(JevK5Checkpoint.fourBit.sourceRevision)\""))
        for checkpoint in JevK5Checkpoint.all {
            // The script's OUTPUTS block for these bits, up to the next one.
            let start = try #require(script.range(of: "    \(checkpoint.bits): {\n"))
            let block = script[start.upperBound...].prefix { $0 != "}" }
            #expect(checkpoint.files.count == 11, "\(checkpoint.repository)")
            for file in checkpoint.files {
                #expect(
                    block.contains("\"\(file.name)\": (\(file.bytes), \"\(file.sha256)\"),"),
                    "\(checkpoint.repository): \(file.name)")
            }
            #expect(
                script.contains("\(checkpoint.bits): \"\(checkpoint.repository)\""),
                "\(checkpoint.repository) is the script's PUBLISHED_AS")
        }
    }
}
