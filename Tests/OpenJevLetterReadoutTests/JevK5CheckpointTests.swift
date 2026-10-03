import Foundation
import OpenJevCore
import OpenJevDiffusionGemma
import OpenJevLetterReadout
import OpenJevServer
import Testing

/// Where JevK5's files come from: `OPENJEV_JEVK5_MODEL`'s two forms, the unpublished default, a
/// folder's required files, and the conversions' digests against `Tools/jevk5/convert.py`, which
/// wrote them.
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
        // A conversion's repository takes the conversion's pinned revision, none until published.
        #expect(
            JevK5ModelFiles.source(setting: JevK5Checkpoint.defaultSetting)
                == JevK5Checkpoint.fourBit.hubSource)
        #expect(
            JevK5ModelFiles.source(setting: "Algorythm-Canada/jevk5-0.2-mlx-8bit@main")
                == .hub(repository: "Algorythm-Canada/jevk5-0.2-mlx-8bit", revision: "main"))
    }

    @Test("The server's default is the 4-bit conversion's repository")
    func serverDefault() throws {
        #expect(try ServerSettings().jevk5Model == JevK5Checkpoint.defaultSetting)
        #expect(JevK5Checkpoint.defaultSetting == "Algorythm-Canada/jevk5-0.2-mlx-4bit")
        #expect(
            try ServerSettings(environment: ["OPENJEV_JEVK5_MODEL": "/tmp/x"]).jevk5Model
                == "/tmp/x")
    }

    @Test("An unpublished conversion is refused before any request reaches the Hub")
    func unpublishedDefault() async throws {
        #expect(!JevK5Checkpoint.fourBit.isPublished)
        // Nothing listens on port 9: a request would fail with a transport error instead.
        let resolver = ModelResolver(endpoint: URL(string: "http://127.0.0.1:9")!, maxAttempts: 1)
        do {
            _ = try await JevK5ModelFiles.resolve(
                JevK5ModelFiles.source(setting: JevK5Checkpoint.defaultSetting),
                cache: HubCacheLocation(directory: FileManager.default.temporaryDirectory),
                resolver: resolver)
            Issue.record("the unpublished default resolved")
        } catch let error as JevK5LoadError {
            #expect(error == .notPublished(repository: "Algorythm-Canada/jevk5-0.2-mlx-4bit"))
            #expect(error.description.contains("Tools/jevk5/convert.py"))
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
