import Foundation
import OpenJevCore
import OpenJevEncoders
import Testing

/// ``EncoderPackageStore``: the SHA-256 check, downloads from file URLs, the local models
/// folder and the Hugging Face fallback, and the manifest the library embeds for Verdict.
@Suite("Encoder package store")
struct EncoderPackageStoreTests {
    /// A folder of its own under the temporary folder, removed by the caller.
    private func temporaryFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("OpenJevEncodersTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    /// Writes `text` to `folder/path`, creating the folders, and returns its URL.
    @discardableResult
    private func write(_ text: String, to path: String, in folder: URL) throws -> URL {
        let file = folder.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: file)
        return file
    }

    /// A manifest entry for a file, with its real size and digest.
    private func entry(_ path: String, for file: URL) throws -> EncoderPackageManifest.File {
        EncoderPackageManifest.File(
            path: path, url: file,
            bytes: try Data(contentsOf: file).count,
            sha256: try EncoderPackageStore.sha256(of: file))
    }

    /// A small package, tokenizer and calibrator under `folder/source`, and a manifest that
    /// publishes them at their file URLs.
    private func sourceManifest(in folder: URL) throws -> EncoderPackageManifest {
        let source = folder.appendingPathComponent("source", isDirectory: true)
        let packagePaths = [
            "Manifest.json", "Data/com.apple.CoreML/model.mlmodel",
            "Data/com.apple.CoreML/weights/weight.bin",
        ]
        let packageFiles = try packagePaths.map { path in
            try entry(
                path,
                for: write(
                    "contents of \(path)", to: path.replacingOccurrences(of: "/", with: "--"),
                    in: source))
        }
        let tokenizerFiles = try ["tokenizer.json", "tokenizer_config.json"].map { name in
            try entry(name, for: write("{\"name\": \"\(name)\"}", to: name, in: source))
        }
        let calibrator = try entry(
            "calibrator.json",
            for: write(
                #"{"temperature": 2.8039, "per_k": {"3": 5.0069}}"#, to: "calibrator.json",
                in: source))
        return EncoderPackageManifest(
            model: "verdict-1.4", package: "test-package",
            minimumOS: .init(iOS: 18, macOS: 15),
            checkpoint: .init(repository: "owner/checkpoint", revision: "abc123"),
            packageFiles: packageFiles, tokenizerFiles: tokenizerFiles, calibrator: calibrator)
    }

    @Test("SHA-256 digests are those of the standard's test vectors")
    func digests() throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(
            try EncoderPackageStore.sha256(of: write("abc", to: "abc", in: folder))
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        #expect(
            try EncoderPackageStore.sha256(of: write("", to: "empty", in: folder))
                == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }

    @Test("Verification accepts the manifest's digest and refuses a corrupted file")
    func verification() throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = try write("the weights", to: "weight.bin", in: folder)
        let expected = try entry("weights/weight.bin", for: file)
        try EncoderPackageStore.verify(file, against: expected, named: "weights/weight.bin")

        try write("the weighTs", to: "weight.bin", in: folder)
        let corrupted = #expect(throws: EncoderPackageError.self) {
            try EncoderPackageStore.verify(file, against: expected, named: "weights/weight.bin")
        }
        guard case .digestMismatch(let name, _, let wanted, let actual) = corrupted else {
            Issue.record("expected a digest mismatch, got \(String(describing: corrupted))")
            return
        }
        #expect(name == "weights/weight.bin")
        #expect(wanted == expected.sha256)
        #expect(actual == (try EncoderPackageStore.sha256(of: file)))
        #expect(corrupted?.description.contains("the file was not used") == true)

        try write("the weights, longer", to: "weight.bin", in: folder)
        let truncated = #expect(throws: EncoderPackageError.self) {
            try EncoderPackageStore.verify(file, against: expected, named: "weights/weight.bin")
        }
        guard case .sizeMismatch = truncated else {
            Issue.record("expected a size mismatch, got \(String(describing: truncated))")
            return
        }
    }

    @Test("Downloads each file once, checks it, and keeps it out of backups")
    func downloads() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let manifest = try sourceManifest(in: folder)
        let storeFolder = folder.appendingPathComponent("store", isDirectory: true)
        let store = EncoderPackageStore(directory: storeFolder)
        let locations = try await store.locations(for: manifest)

        let root = storeFolder.appendingPathComponent("test-package", isDirectory: true)
        #expect(locations.packageDirectory == root.appendingPathComponent("test-package.mlpackage"))
        #expect(locations.tokenizerDirectory == root.appendingPathComponent("tokenizer"))
        #expect(
            locations.calibratorFile == root.appendingPathComponent("tokenizer/calibrator.json"))
        for file in manifest.packageFiles {
            let local = locations.packageDirectory.appendingPathComponent(file.path)
            #expect(try String(contentsOf: local, encoding: .utf8) == "contents of \(file.path)")
        }
        #expect(
            try VerdictCalibration(contentsOf: locations.calibratorFile)
                == VerdictCalibration(temperature: 2.8039, perK: [3: 5.0069]))
        let excluded = try root.resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(excluded.isExcludedFromBackup == true)

        // A second call finds everything in place: nothing is downloaded, so the sources can go.
        try FileManager.default.removeItem(at: folder.appendingPathComponent("source"))
        #expect(try await store.locations(for: manifest) == locations)
    }

    @Test("A download that does not match the manifest is refused and not kept")
    func corruptDownload() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        var manifest = try sourceManifest(in: folder)
        manifest.packageFiles[2].sha256 = String(repeating: "0", count: 64)
        let store = EncoderPackageStore(
            directory: folder.appendingPathComponent("store", isDirectory: true))
        let error = await #expect(throws: EncoderPackageError.self) {
            try await store.locations(for: manifest)
        }
        guard case .digestMismatch(let name, let url, _, _) = error else {
            Issue.record("expected a digest mismatch, got \(String(describing: error))")
            return
        }
        #expect(name == "test-package.mlpackage/Data/com.apple.CoreML/weights/weight.bin")
        #expect(url == manifest.packageFiles[2].url)
        let kept = folder.appendingPathComponent(
            "store/test-package/test-package.mlpackage/Data/com.apple.CoreML/weights/weight.bin")
        #expect(!FileManager.default.fileExists(atPath: kept.path))
    }

    @Test("A package for a newer OS, or a path outside the folder, is refused before any download")
    func refusedManifests() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let storeFolder = folder.appendingPathComponent("store", isDirectory: true)
        let store = EncoderPackageStore(directory: storeFolder)
        var future = try sourceManifest(in: folder)
        future.minimumOS = .init(iOS: 99, macOS: 99)
        let tooOld = await #expect(throws: EncoderPackageError.self) {
            try await store.locations(for: future)
        }
        guard case .unsupportedOperatingSystem = tooOld else {
            Issue.record("expected an unsupported OS, got \(String(describing: tooOld))")
            return
        }
        for path in ["../escape.bin", "/etc/escape", "Data//model.mlmodel", "./Manifest.json", ""] {
            var escaping = try sourceManifest(in: folder)
            escaping.packageFiles[0].path = path
            let error = await #expect(throws: EncoderPackageError.self, "\(path)") {
                try await store.locations(for: escaping)
            }
            #expect(error == .invalidPath(path))
        }
        var badPackage = try sourceManifest(in: folder)
        badPackage.package = "../elsewhere"
        await #expect(throws: EncoderPackageError.invalidPath("../elsewhere")) {
            try await store.locations(for: badPackage)
        }
        #expect(!FileManager.default.fileExists(atPath: storeFolder.path))
    }

    @Test("A newer manifest's changed file is downloaded again")
    func changedManifest() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        var manifest = try sourceManifest(in: folder)
        let store = EncoderPackageStore(
            directory: folder.appendingPathComponent("store", isDirectory: true))
        let locations = try await store.locations(for: manifest)
        // Same size, other bytes: only the recorded digest tells the files apart.
        let source = try write(
            "contents of Manifest.jsoN", to: "Manifest.json",
            in: folder.appendingPathComponent("source"))
        manifest.packageFiles[0] = try entry("Manifest.json", for: source)
        _ = try await store.locations(for: manifest)
        #expect(
            try String(
                contentsOf: locations.packageDirectory.appendingPathComponent("Manifest.json"),
                encoding: .utf8) == "contents of Manifest.jsoN")
    }

    @Test("The local models folder is used instead of downloading")
    func localModels() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let manifest = EncoderPackageManifest.verdict
        let local = folder.appendingPathComponent("models", isDirectory: true)
        try write("{}", to: "verdict-m18-fp16.mlpackage/Manifest.json", in: local)
        for name in ["tokenizer.json", "tokenizer_config.json", "calibrator.json"] {
            try write("{}", to: "verdict-m18-fp16/tokenizer/\(name)", in: local)
        }
        let storeFolder = folder.appendingPathComponent("store", isDirectory: true)
        let store = EncoderPackageStore(directory: storeFolder, localModelsDirectory: local)
        let locations = try await store.locations(for: manifest)
        #expect(
            locations.packageDirectory == local.appendingPathComponent("verdict-m18-fp16.mlpackage")
        )
        #expect(
            locations.tokenizerDirectory
                == local.appendingPathComponent("verdict-m18-fp16/tokenizer"))
        #expect(
            locations.calibratorFile
                == local.appendingPathComponent("verdict-m18-fp16/tokenizer/calibrator.json"))
        #expect(!FileManager.default.fileExists(atPath: storeFolder.path))
    }

    @Test("Without a tokenizer folder, the local models read the Hugging Face cache snapshot")
    func huggingFaceFallback() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let manifest = EncoderPackageManifest.verdict
        let local = folder.appendingPathComponent("models", isDirectory: true)
        let hub = folder.appendingPathComponent("hub", isDirectory: true)
        try write("{}", to: "verdict-m18-fp16.mlpackage/Manifest.json", in: local)
        let snapshot = manifest.checkpoint.snapshot(in: hub)
        #expect(
            snapshot.path.hasSuffix(
                "hub/models--heman10x--rlcd-modernbert-151m/snapshots/"
                    + "8af2496eb63c7fa66d7d234e1f62629380030eb4"))
        let store = EncoderPackageStore(
            directory: folder.appendingPathComponent("store"), localModelsDirectory: local,
            huggingFaceHubDirectory: hub)

        let missing = await #expect(throws: EncoderPackageError.self) {
            try await store.locations(for: manifest)
        }
        #expect(
            missing?.description.contains("tokenizer.json, tokenizer_config.json, calibrator.json")
                == true)

        for name in ["tokenizer.json", "tokenizer_config.json", "calibrator.json"] {
            try write("{}", to: name, in: snapshot)
        }
        let locations = try await store.locations(for: manifest)
        #expect(locations.tokenizerDirectory == snapshot)

        let empty = EncoderPackageStore(
            directory: folder.appendingPathComponent("store"),
            localModelsDirectory: folder.appendingPathComponent("elsewhere"))
        let noPackage = await #expect(throws: EncoderPackageError.self) {
            try await empty.locations(for: manifest)
        }
        #expect(noPackage?.description.contains("OPENJEV_ENCODER_MODELS") == true)
    }

    @Test("The environment names the local models folder and the Hugging Face cache")
    func environment() throws {
        let store = try EncoderPackageStore(environment: ["OPENJEV_ENCODER_MODELS": "/models"])
        #expect(store.localModelsDirectory?.path == "/models")
        #expect(store.directory.path.hasSuffix("OpenJevSwift/encoders"))
        #expect(
            try EncoderPackageStore(environment: ["OPENJEV_ENCODER_MODELS": ""])
                .localModelsDirectory == nil)
        #expect(try EncoderPackageStore(environment: [:]).localModelsDirectory == nil)
        let hub = EncoderPackageStore.huggingFaceHubDirectory
        #expect(hub(["HF_HUB_CACHE": "/a/hub", "HF_HOME": "/b"]).path == "/a/hub")
        #expect(hub(["HF_HOME": "/b", "XDG_CACHE_HOME": "/c"]).path == "/b/hub")
        #expect(hub(["XDG_CACHE_HOME": "/c"]).path == "/c/huggingface/hub")
        #expect(hub([:]).path == NSHomeDirectory() + "/.cache/huggingface/hub")
    }

    @Test(
        "The embedded manifest publishes Verdict's package, tokenizer and calibrator",
        .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage))
    func verdictManifest() throws {
        let manifest = EncoderPackageManifest.verdict
        #expect(manifest.model == KnownEncoderModels.verdict.name)
        #expect(manifest.package == EncoderPackageSpec.verdict.name)
        #expect(manifest.minimumOS == .init(iOS: 18, macOS: 15))
        #expect(manifest.checkpoint.repository == "heman10x/rlcd-modernbert-151m")
        #expect(manifest.checkpoint.revision == (try VerdictFixtures.reference().verdictRevision))
        #expect(
            manifest.packageFiles.map(\.path) == [
                "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin",
                "Manifest.json",
            ])
        let release =
            "https://github.com/Algorythm-Canada/openjev-models/releases/download/"
            + "verdict-m18-fp16-v1/"
        #expect(
            manifest.packageFiles.map(\.url.absoluteString) == [
                release + "Data--com.apple.CoreML--model.mlmodel",
                release + "Data--com.apple.CoreML--weights--weight.bin",
                release + "Manifest.json",
            ])
        let checkpoint =
            "https://huggingface.co/heman10x/rlcd-modernbert-151m/resolve/"
            + manifest.checkpoint.revision + "/"
        #expect(manifest.tokenizerFiles.map(\.path) == ["tokenizer.json", "tokenizer_config.json"])
        #expect(
            (manifest.tokenizerFiles + [manifest.calibrator]).map(\.url.absoluteString)
                == ["tokenizer.json", "tokenizer_config.json", "calibrator.json"].map {
                    checkpoint + $0
                })
        let files = manifest.packageFiles + manifest.tokenizerFiles + [manifest.calibrator]
        #expect(files.allSatisfy { $0.bytes > 0 })
        #expect(
            files.allSatisfy { file in
                file.sha256.count == 64
                    && file.sha256.allSatisfy { "0123456789abcdef".contains($0) }
            })
    }

    @Test(
        "The embedded manifest's tokenizer and calibrator are the checkpoint's, digests included",
        .enabled(
            if: VerdictModelFiles.tokenizerDirectory != nil,
            VerdictModelFiles.missingTokenizerMessage))
    func manifestMatchesCheckpointFiles() throws {
        // These files are the checkpoint's own at a pinned revision, the same on every machine.
        let manifest = EncoderPackageManifest.verdict
        let tokenizer = try #require(VerdictModelFiles.tokenizerDirectory)
        for file in manifest.tokenizerFiles + [manifest.calibrator] {
            try EncoderPackageStore.verify(
                tokenizer.appendingPathComponent(file.path), against: file, named: file.path)
        }
    }

    @Test(
        "The embedded manifest lists the files of the converted package",
        .enabled(
            if: VerdictModelFiles.packageDirectory != nil,
            VerdictModelFiles.missingPackageMessage))
    func manifestListsPackageFiles() throws {
        // Only the paths: every conversion writes new identifiers into Manifest.json, so a
        // package converted on another machine has other digests. Tools/encoders/manifest.py
        // --check compares the digests with the package the release publishes.
        let package = try #require(VerdictModelFiles.packageDirectory)
        let paths = FileManager.default.enumerator(atPath: package.path)?
            .compactMap { $0 as? String }
            .filter { path in
                var isFolder: ObjCBool = false
                return FileManager.default.fileExists(
                    atPath: package.appendingPathComponent(path).path, isDirectory: &isFolder)
                    && !isFolder.boolValue && !path.hasSuffix(".DS_Store")
            }
        #expect(paths?.sorted() == EncoderPackageManifest.verdict.packageFiles.map(\.path))
    }
}
