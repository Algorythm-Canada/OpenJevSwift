import Foundation
import OpenJevCore
import OpenJevEncoders
import Testing

/// ``EncoderPackageStore`` for Laya: the five embedded manifests, the checkpoint's tokenizer under
/// `tokenizer/`, the tokenizer fetched without a package, the packages a device holds, and which
/// of the iPhone's four packages a sequence uses.
@Suite("Laya package store")
struct LayaPackageStoreTests {
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
            path: path, url: file, bytes: try Data(contentsOf: file).count,
            sha256: try EncoderPackageStore.sha256(of: file))
    }

    /// A small stand-in for one of Laya's packages, with Laya's tokenizer layout, published at
    /// file URLs under `folder/source/{package}`.
    private func manifest(
        _ package: String, in folder: URL, downloads: Bool = true
    ) throws -> EncoderPackageManifest {
        let source = folder.appendingPathComponent("source/\(package)", isDirectory: true)
        let packageFiles = try ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel"].map {
            try entry($0, for: write("\(package) \($0)", to: $0, in: source))
        }
        let tokenizerFiles = try ["tokenizer.json", "tokenizer_config.json"].map {
            try entry($0, for: write("{\"name\": \"\($0)\"}", to: "tokenizer/\($0)", in: source))
        }
        let configuration = try entry(
            "rl_agent_config.json",
            for: write(
                #"{"max_len": 1024, "head_max_len": 256}"#, to: "rl_agent_config.json", in: source))
        return EncoderPackageManifest(
            model: "laya-1.0", package: package, minimumOS: .init(iOS: 18, macOS: 15),
            checkpoint: .init(repository: "owner/laya", revision: "abc123"),
            packageDownloadsEnabled: downloads, packageFiles: packageFiles,
            tokenizerFiles: tokenizerFiles, calibrator: configuration,
            checkpointTokenizerFolder: "tokenizer")
    }

    @Test("The embedded manifests publish Laya's five packages, tokenizer and configuration")
    func embeddedManifests() throws {
        let checkpoint =
            "https://huggingface.co/convaiinnovations/laya-typed-decisions/resolve/"
            + "1a793eb568e6718f15941d08f85432581df534e3/"
        let release = "https://github.com/Algorythm-Canada/openjev-models/releases/download/"
        let byLength = EncoderPackageManifest.layaByLength
        #expect(byLength.keys.sorted() == EncoderPackageSpec.layaSequenceLengths)
        let all = [EncoderPackageManifest.laya] + byLength.keys.sorted().map { byLength[$0]! }
        #expect(
            all.map(\.package) == [
                "laya-m18-fp16", "laya-f18-b1s128-fp16", "laya-f18-b1s256-fp16",
                "laya-f18-b1s512-fp16", "laya-f18-b1s1024-fp16",
            ])
        #expect(EncoderPackageManifest.laya.package == EncoderPackageSpec.layaMultifunction.name)
        for (length, manifest) in byLength {
            #expect(manifest.package == EncoderPackageSpec.laya(sequenceLength: length).name)
        }
        for manifest in all {
            #expect(manifest.model == KnownEncoderModels.laya.name)
            #expect(manifest.minimumOS == .init(iOS: 18, macOS: 15))
            #expect(manifest.packageDownloadsEnabled)
            #expect(manifest.checkpoint.repository == "convaiinnovations/laya-typed-decisions")
            #expect(manifest.checkpointTokenizerFolder == "tokenizer")
            #expect(
                manifest.packageFiles.map(\.path) == [
                    "Data/com.apple.CoreML/model.mlmodel",
                    "Data/com.apple.CoreML/weights/weight.bin", "Manifest.json",
                ])
            let tag = release + manifest.package + "-v1/"
            #expect(
                manifest.packageFiles.map(\.url.absoluteString) == [
                    tag + "Data--com.apple.CoreML--model.mlmodel",
                    tag + "Data--com.apple.CoreML--weights--weight.bin", tag + "Manifest.json",
                ])
            #expect(
                manifest.tokenizerFiles.map(\.path) == ["tokenizer.json", "tokenizer_config.json"])
            #expect(
                manifest.tokenizerFiles.map(\.url.absoluteString) == [
                    checkpoint + "tokenizer/tokenizer.json",
                    checkpoint + "tokenizer/tokenizer_config.json",
                ])
            #expect(manifest.calibrator.path == "rl_agent_config.json")
            #expect(manifest.calibrator.url.absoluteString == checkpoint + "rl_agent_config.json")
            // Every package shares the checkpoint's files.
            #expect(manifest.tokenizerFiles == EncoderPackageManifest.laya.tokenizerFiles)
            #expect(manifest.calibrator == EncoderPackageManifest.laya.calibrator)
            let files = manifest.packageFiles + manifest.tokenizerFiles + [manifest.calibrator]
            #expect(files.allSatisfy { $0.bytes > 0 })
            #expect(
                files.allSatisfy { file in
                    file.sha256.count == 64
                        && file.sha256.allSatisfy { "0123456789abcdef".contains($0) }
                })
        }
    }

    @Test("A manifest encoded before the checkpoint's tokenizer folder existed still decodes")
    func manifestDecoding() throws {
        let laya = EncoderPackageManifest.laya
        let decoded = try JSONDecoder().decode(
            EncoderPackageManifest.self, from: JSONEncoder().encode(laya))
        #expect(decoded == laya)
        // Remove the key, as a manifest encoded by an earlier release lacks it.
        var object = try #require(
            try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(EncoderPackageManifest.verdict))
                as? [String: Any])
        #expect(object.removeValue(forKey: "checkpointTokenizerFolder") != nil)
        let earlier = try JSONDecoder().decode(
            EncoderPackageManifest.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(earlier == EncoderPackageManifest.verdict)
        #expect(earlier.checkpointTokenizerFolder.isEmpty)
    }

    @Test(
        "The checkpoint is the fixture's, and its files are the ones the manifests check",
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    func checkpointRevision() throws {
        let reference = try LayaFixtures.reference()
        #expect(EncoderPackageManifest.laya.checkpoint.repository == reference.layaRepository)
        #expect(EncoderPackageManifest.laya.checkpoint.revision == reference.layaRevision)
    }

    @Test(
        "The embedded tokenizer and configuration digests are the checkpoint's",
        .enabled(if: LayaModelFiles.tokenizer != nil, LayaModelFiles.missingTokenizerMessage))
    func manifestMatchesCheckpointFiles() throws {
        let manifest = EncoderPackageManifest.laya
        let found = try #require(LayaModelFiles.tokenizer)
        for file in manifest.tokenizerFiles {
            try EncoderPackageStore.verify(
                found.tokenizerDirectory.appendingPathComponent(file.path), against: file,
                named: file.path)
        }
        try EncoderPackageStore.verify(
            found.calibratorFile, against: manifest.calibrator, named: manifest.calibrator.path)
    }

    @Test(
        "Each embedded manifest lists the files of its converted package",
        .enabled(
            if: LayaModelFiles.multifunctionPackage != nil
                && LayaModelFiles.packagesByLength.count == 4,
            LayaModelFiles.missingPackageMessage))
    func manifestsListPackageFiles() throws {
        // Only the paths: every conversion writes new identifiers into Manifest.json, so a
        // package converted on another machine has other digests. manifest.py --check compares
        // the digests with the packages the releases publish.
        let manifests = [EncoderPackageManifest.laya] + EncoderPackageManifest.layaByLength.values
        for manifest in manifests {
            let package = try #require(EncoderModelFiles.package(manifest.package))
            let paths = FileManager.default.enumerator(atPath: package.path)?
                .compactMap { $0 as? String }
                .filter { path in
                    var isFolder: ObjCBool = false
                    return FileManager.default.fileExists(
                        atPath: package.appendingPathComponent(path).path, isDirectory: &isFolder)
                        && !isFolder.boolValue && !path.hasSuffix(".DS_Store")
                }
            #expect(paths?.sorted() == manifest.packageFiles.map(\.path), "\(manifest.package)")
        }
    }

    @Test("Without a tokenizer folder, the local models read the snapshot's tokenizer/ folder")
    func huggingFaceFallback() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let manifest = EncoderPackageManifest.laya
        let local = folder.appendingPathComponent("models", isDirectory: true)
        let hub = folder.appendingPathComponent("hub", isDirectory: true)
        try write("{}", to: "laya-m18-fp16.mlpackage/Manifest.json", in: local)
        let snapshot = manifest.checkpoint.snapshot(in: hub)
        let store = EncoderPackageStore(
            directory: folder.appendingPathComponent("store"), localModelsDirectory: local,
            huggingFaceHubDirectory: hub)

        // The tokenizer at the snapshot's root is not Laya's layout.
        for name in ["tokenizer.json", "tokenizer_config.json", "rl_agent_config.json"] {
            try write("{}", to: name, in: snapshot)
        }
        let missing = await #expect(throws: EncoderPackageError.self) {
            try await store.locations(for: manifest)
        }
        #expect(
            missing?.description.contains(
                "tokenizer.json, tokenizer_config.json, rl_agent_config.json") == true)
        // The error names every folder looked in, the snapshot's root included, where the
        // configuration file is sought apart from the tokenizer.
        guard case .missingLocalTokenizer(_, let searched) = missing else {
            Issue.record("expected a missing tokenizer, got \(String(describing: missing))")
            return
        }
        #expect(
            searched.map(\.standardizedFileURL.path)
                == [
                    local.appendingPathComponent("laya-m18-fp16/tokenizer"),
                    snapshot.appendingPathComponent("tokenizer"), snapshot,
                ].map(\.standardizedFileURL.path))

        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try write("{}", to: "tokenizer/\(name)", in: snapshot)
        }
        let locations = try await store.locations(for: manifest)
        #expect(
            locations.packageDirectory == local.appendingPathComponent("laya-m18-fp16.mlpackage"))
        #expect(locations.tokenizerDirectory == snapshot.appendingPathComponent("tokenizer"))
        #expect(locations.calibratorFile == snapshot.appendingPathComponent("rl_agent_config.json"))
        #expect(manifest.checkpointFiles(in: hub) == locations.tokenizer)

        // The tokenizer alone needs no package: the iPhone's set has none at first.
        let tokenizer = try await store.tokenizerLocations(
            for: try #require(EncoderPackageManifest.layaByLength[128]))
        #expect(tokenizer == locations.tokenizer)
        // A tokenizer folder beside the local package wins over the cache.
        for name in ["tokenizer.json", "tokenizer_config.json", "rl_agent_config.json"] {
            try write("{}", to: "laya-m18-fp16/tokenizer/\(name)", in: local)
        }
        let own = try await store.locations(for: manifest)
        #expect(own.tokenizerDirectory == local.appendingPathComponent("laya-m18-fp16/tokenizer"))
        #expect(
            own.calibratorFile
                == local.appendingPathComponent("laya-m18-fp16/tokenizer/rl_agent_config.json"))
    }

    @Test("The tokenizer is fetched without the package, also before the package is published")
    func tokenizerWithoutPackage() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        var manifest = try manifest("laya-f18-b1s128-fp16", in: folder, downloads: false)
        let storeFolder = folder.appendingPathComponent("store", isDirectory: true)
        let store = EncoderPackageStore(directory: storeFolder)
        let root = storeFolder.appendingPathComponent("laya-f18-b1s128-fp16", isDirectory: true)

        let tokenizer = try await store.tokenizerLocations(for: manifest)
        #expect(tokenizer.tokenizerDirectory == root.appendingPathComponent("tokenizer"))
        #expect(
            tokenizer.calibratorFile
                == root.appendingPathComponent("tokenizer/rl_agent_config.json"))
        #expect(
            try LayaCalibration(contentsOf: tokenizer.calibratorFile).headMaxLength == 256)
        #expect(
            !FileManager.default.fileExists(
                atPath: root.appendingPathComponent("laya-f18-b1s128-fp16.mlpackage").path))
        #expect(try store.heldPackageDirectory(for: manifest) == nil)
        await #expect(
            throws: EncoderPackageError.packageDownloadsUnavailable("laya-f18-b1s128-fp16")
        ) {
            try await store.locations(for: manifest)
        }

        // Once published, the package comes down; the tokenizer is not fetched again.
        manifest.packageDownloadsEnabled = true
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try FileManager.default.removeItem(
                at: folder.appendingPathComponent("source/laya-f18-b1s128-fp16/tokenizer/\(name)"))
        }
        let locations = try await store.locations(for: manifest)
        #expect(locations.tokenizer == tokenizer)
        #expect(
            try store.heldPackageDirectory(for: manifest)
                == root.appendingPathComponent("laya-f18-b1s128-fp16.mlpackage"))
    }

    @Test("A package is held when every file was checked and keeps its size, and nothing else")
    func heldPackages() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let manifest = try manifest("laya-f18-b1s256-fp16", in: folder)
        let store = EncoderPackageStore(
            directory: folder.appendingPathComponent("store", isDirectory: true))
        #expect(try store.heldPackageDirectory(for: manifest) == nil)
        let locations = try await store.locations(for: manifest)
        #expect(try store.heldPackageDirectory(for: manifest) == locations.packageDirectory)
        // A file whose size changed is not held, and nothing is downloaded to find out.
        try write("changed", to: "Manifest.json", in: locations.packageDirectory)
        #expect(try store.heldPackageDirectory(for: manifest) == nil)
        // A manifest whose digest changed is not held.
        _ = try await store.locations(for: manifest)
        var newer = manifest
        newer.packageFiles[0].sha256 = String(repeating: "a", count: 64)
        #expect(try store.heldPackageDirectory(for: newer) == nil)
        #expect(try store.heldPackageDirectory(for: manifest) == locations.packageDirectory)

        // The local models hold a package when its folder exists.
        let local = folder.appendingPathComponent("models", isDirectory: true)
        let localStore = EncoderPackageStore(
            directory: folder.appendingPathComponent("unused"), localModelsDirectory: local)
        #expect(try localStore.heldPackageDirectory(for: manifest) == nil)
        try write("{}", to: "laya-f18-b1s256-fp16.mlpackage/Manifest.json", in: local)
        #expect(
            try localStore.heldPackageDirectory(for: manifest)
                == local.appendingPathComponent("laya-f18-b1s256-fp16.mlpackage"))
    }

    @available(macOS 15, iOS 18, *)
    @Test("A sequence uses the smallest package the device holds; past 1,024 tokens none")
    func packageChoice() async throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let specs = EncoderPackageSpec.layaSequenceLengths.map {
            EncoderPackageSpec.laya(sequenceLength: $0)
        }
        let names = specs.map(\.name)
        #expect(names == EncoderPackageSpec.layaSequenceLengths.map { "laya-f18-b1s\($0)-fp16" })
        var manifests: [String: EncoderPackageManifest] = [:]
        for name in names {
            manifests[name] = try manifest(name, in: folder)
        }
        let byName = manifests
        let store = EncoderPackageStore(
            directory: folder.appendingPathComponent("store", isDirectory: true))
        // The device holds the 128- and 512-token packages.
        for name in ["laya-f18-b1s128-fp16", "laya-f18-b1s512-fp16"] {
            _ = try await store.locations(for: try #require(byName[name]))
        }

        // The runner asks the store which packages the device holds. It is handed each held
        // package at a folder that does not exist, so compiling the package it picks throws
        // missingFile naming that package, before Core ML is involved.
        let unbuilt = folder.appendingPathComponent("unbuilt", isDirectory: true)
        func unbuiltFolder(of name: String) -> URL {
            unbuilt.appendingPathComponent(name + ".mlpackage", isDirectory: true)
        }
        let source = CoreMLPackagesByLength.Source(
            held: { spec in
                try store.heldPackageDirectory(for: byName[spec.name]!).map { _ in
                    unbuilt.appendingPathComponent(spec.name + ".mlpackage", isDirectory: true)
                }
            },
            fetch: { spec in
                try await store.locations(for: byName[spec.name]!).packageDirectory
            })
        let model = CoreMLPackagesByLength(
            specs: specs.reversed(), computeUnits: .cpuAndNeuralEngine, source: source)
        #expect(model.specs == specs)
        #expect(model.spec(holding: 1024) == specs[3])
        #expect(model.spec(holding: 1025) == nil)

        /// The error a row of `length` tokens gets: which package the runner picked.
        func outcome(_ length: Int) async -> EncoderLoadError? {
            let row: [[Int32]] = [
                [Int32](repeating: 7, count: length), [Int32](repeating: 1, count: length),
                [Int32](repeating: 0, count: length),
            ]
            do {
                _ = try await model.run([row])
                return nil
            } catch {
                return error as? EncoderLoadError
            }
        }
        let held = ["laya-f18-b1s128-fp16", "laya-f18-b1s512-fp16"]
        let cases: [(length: Int, package: String)] = [
            (1, "laya-f18-b1s128-fp16"), (128, "laya-f18-b1s128-fp16"),
            (129, "laya-f18-b1s512-fp16"), (512, "laya-f18-b1s512-fp16"),
        ]
        for (length, package) in cases {
            #expect(
                await outcome(length) == .missingFile(unbuiltFolder(of: package)),
                "\(length) tokens")
        }
        // A row longer than every package the device holds names the one to fetch.
        let tooLong = await outcome(600)
        #expect(tooLong == .noPackage(length: 600, package: "laya-f18-b1s1024-fp16", held: held))
        #expect(tooLong?.description.contains("prefetch(lengths:)") == true)
        // Past 1,024 tokens no package takes the sequence, and nothing is fetched.
        #expect(await outcome(1025) == .noPackage(length: 1025, package: nil, held: held))
        let beyond = await #expect(throws: EncoderLoadError.self) {
            try await model.prefetch(lengths: [1025])
        }
        #expect(beyond == .noPackage(length: 1025, package: nil, held: held))

        // A package fetched since is used by the next read that it takes.
        _ = try await store.locations(for: try #require(byName["laya-f18-b1s256-fp16"]))
        #expect(await outcome(129) == .missingFile(unbuiltFolder(of: "laya-f18-b1s256-fp16")))
        #expect(await outcome(300) == .missingFile(unbuiltFolder(of: "laya-f18-b1s512-fp16")))
        // Nothing compiled, so nothing was created to load.
        #expect(await model.loadedLengths.isEmpty)
    }
}
