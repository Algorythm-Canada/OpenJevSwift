import Foundation
import OpenJevDiffusionGemma
import Testing

/// ``ModelResolver`` against ``TestHubServer``, a local server with the Hub API's JSON shapes and
/// three small files, one of them LFS, each test in a temporary cache.
@Suite("Model resolution and download", .serialized)
struct ModelResolverTests {
    private let repository = TestHubServer.repository
    private let commit = TestHubServer.commit

    /// A fresh temporary cache directory.
    private func temporaryCache() throws -> HubCacheLocation {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-hub-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return HubCacheLocation(directory: directory)
    }

    private func withServer<T>(
        _ body: (TestHubServer, ModelResolver, HubCacheLocation) async throws -> T
    ) async throws -> T {
        let server = try TestHubServer()
        try await server.start()
        defer { server.stop() }
        let cache = try temporaryCache()
        defer { try? FileManager.default.removeItem(at: cache.directory) }
        let resolver = ModelResolver(endpoint: server.endpoint, maxAttempts: 3)
        return try await body(server, resolver, cache)
    }

    @Test("A cold download creates blobs, relative snapshot symlinks and refs/main")
    func coldDownload() async throws {
        try await withServer { server, resolver, cache in
            let reports = ProgressLog()
            let resolution = try await resolver.resolution(
                of: .hub(repository: repository, revision: nil), cache: cache,
                progress: { reports.append($0) })
            let root = cache.repositoryDirectory(repository)
            #expect(root.lastPathComponent == "models--test-org--tiny-model")
            let snapshot = cache.snapshotDirectory(repository, commit: commit)
            #expect(resolution.directory == snapshot)
            #expect(resolution.commit == commit)
            #expect(resolution.downloadedFiles == 3)
            let total = TestHubServer.files.reduce(0) { $0 + $1.content.count }
            #expect(resolution.downloadedBytes == total)

            let ref = try String(
                contentsOf: root.appendingPathComponent("refs/main"), encoding: .utf8)
            #expect(ref == commit)
            let manager = FileManager.default
            for file in TestHubServer.files {
                let link = snapshot.appendingPathComponent(file.path)
                #expect(
                    try manager.destinationOfSymbolicLink(atPath: link.path)
                        == "../../blobs/\(file.blobID)")
                let blob = root.appendingPathComponent("blobs/\(file.blobID)")
                #expect(try Data(contentsOf: blob) == file.content)
                #expect(try Data(contentsOf: link) == file.content)
            }
            #expect(file(named: "model.safetensors").blobID.count == 64)
            #expect(file(named: "config.json").blobID.count == 40)
            let blobs = try manager.contentsOfDirectory(
                atPath: root.appendingPathComponent("blobs").path)
            #expect(!blobs.contains { $0.hasSuffix(".incomplete") })

            let last = try #require(reports.values.last)
            #expect(last.filesDone == 3 && last.fileCount == 3)
            #expect(last.bytesDone == total && last.totalBytes == total)
            #expect(last.downloadedBytes == total)
            #expect(server.downloads.count == 3)
        }
    }

    @Test("A second resolve downloads nothing")
    func secondResolve() async throws {
        try await withServer { server, resolver, cache in
            _ = try await resolver.resolve(
                .hub(repository: repository, revision: nil), cache: cache)
            let before = server.downloads.count
            let again = try await resolver.resolution(
                of: .hub(repository: repository, revision: commit), cache: cache)
            #expect(again.downloadedFiles == 0 && again.downloadedBytes == 0)
            #expect(server.downloads.count == before)
            // A snapshot link that is gone is relinked to the blob already there, not downloaded.
            let link = cache.snapshotDirectory(repository, commit: commit)
                .appendingPathComponent("config.json")
            try FileManager.default.removeItem(at: link)
            let relinked = try await resolver.resolution(
                of: .hub(repository: repository, revision: commit), cache: cache)
            #expect(relinked.downloadedFiles == 0)
            #expect(server.downloads.count == before)
            #expect(FileManager.default.fileExists(atPath: link.path))
        }
    }

    @Test("A download closed partway resumes with a Range request and the right digest")
    func interruptedDownload() async throws {
        try await withServer { server, resolver, cache in
            server.interruptNextLFSDownload(after: 100_000)
            let directory = try await resolver.resolve(
                .hub(repository: repository, revision: commit), cache: cache)
            let shard = file(named: "model.safetensors")
            #expect(
                try TokenizerFiles.sha256Hex(
                    of: directory.appendingPathComponent("model.safetensors")) == shard.sha256)
            let attempts = server.downloads.filter { $0.path.hasSuffix("/model.safetensors") }
            #expect(attempts.count == 2)
            #expect(attempts.first?.headers["range"] == nil)
            // The second request resumes from the bytes that reached the disk, at most the
            // 100,000 the server wrote before it closed (the socket may drop some of them).
            let range = try #require(attempts.last?.headers["range"])
            let offset = try #require(Int(range.dropFirst(6).dropLast()))
            #expect(range.hasPrefix("bytes=") && range.hasSuffix("-"))
            #expect(offset > 0 && offset <= 100_000)
        }
    }

    @Test("A corrupted LFS file is refused naming the file and both digests, and is not kept")
    func corruptedFile() async throws {
        try await withServer { server, resolver, cache in
            server.corruptLFSFile(true)
            let shard = file(named: "model.safetensors")
            do {
                _ = try await resolver.resolve(
                    .hub(repository: repository, revision: commit), cache: cache)
                Issue.record("the corrupted file was accepted")
            } catch let error as ModelResolverError {
                guard
                    case .digestMismatch(
                        let repo, let revision, let name, let algorithm, let expected, let actual) =
                        error
                else {
                    Issue.record("unexpected error \(error)")
                    return
                }
                #expect(repo == repository && revision == commit)
                #expect(name == "model.safetensors" && algorithm == "SHA-256")
                #expect(expected == shard.sha256 && actual != shard.sha256)
                #expect(error.description.contains(expected) && error.description.contains(actual))
            }
            let blobs = cache.repositoryDirectory(repository).appendingPathComponent("blobs")
            let names = try FileManager.default.contentsOfDirectory(atPath: blobs.path)
            #expect(!names.contains { $0.hasPrefix(shard.sha256) })
            // Once the server is fixed, the same cache completes.
            server.corruptLFSFile(false)
            _ = try await resolver.resolve(
                .hub(repository: repository, revision: commit), cache: cache)
        }
    }

    @Test("A revision the Hub does not have is named in the error")
    func missingRevision() async throws {
        try await withServer { _, resolver, cache in
            let error = await #expect(throws: ModelResolverError.self) {
                try await resolver.resolve(
                    .hub(repository: repository, revision: "v9"), cache: cache)
            }
            #expect(error == .revisionNotFound(repository: repository, revision: "v9"))
            #expect(error?.description == "test-org/tiny-model has no revision v9")
        }
    }

    @Test("A 401 says the repository is gated and names HF_TOKEN")
    func unauthorized() async throws {
        try await withServer { server, resolver, cache in
            server.respondToEverything(with: 401)
            let error = await #expect(throws: ModelResolverError.self) {
                try await resolver.resolve(
                    .hub(repository: repository, revision: nil), cache: cache)
            }
            let text = try #require(error?.description)
            #expect(text.contains("gated"))
            #expect(text.contains("HF_TOKEN"))
            #expect(text.contains(repository))
            #expect(text.contains("main"))
        }
    }

    @Test("An empty token is not sent; a token is sent as a Bearer header")
    func tokens() async throws {
        try await withServer { server, resolver, cache in
            _ = try await resolver.resolve(
                .hub(repository: repository, revision: nil), cache: cache, token: "")
            #expect(!server.requests.isEmpty)
            #expect(server.requests.allSatisfy { $0.headers["authorization"] == nil })
            let other = try temporaryCache()
            defer { try? FileManager.default.removeItem(at: other.directory) }
            let count = server.requests.count
            _ = try await resolver.resolve(
                .hub(repository: repository, revision: nil), cache: other, token: "hf_abc")
            #expect(
                server.requests.dropFirst(count).allSatisfy {
                    $0.headers["authorization"] == "Bearer hf_abc"
                })
        }
        #expect(HubCacheLocation.token(environment: ["HF_TOKEN": ""]) == nil)
        #expect(HubCacheLocation.token(environment: [:]) == nil)
        #expect(HubCacheLocation.token(environment: ["HF_TOKEN": "hf_abc"]) == "hf_abc")
    }

    @Test("A redirect to another host drops the token; a whitespace token is not sent")
    func redirectDropsTheToken() async throws {
        try await withServer { server, resolver, cache in
            server.redirectLFSFileToAnotherHost(true)
            let directory = try await resolver.resolve(
                .hub(repository: repository, revision: commit), cache: cache, token: "hf_abc")
            let shard = file(named: "model.safetensors")
            #expect(
                try TokenizerFiles.sha256Hex(
                    of: directory.appendingPathComponent("model.safetensors")) == shard.sha256)
            let cdn = server.requests.filter { $0.path.hasPrefix("/cdn/") }
            #expect(cdn.count == 1)
            #expect(cdn.allSatisfy { $0.headers["authorization"] == nil })
            let hub = server.requests.filter { !$0.path.hasPrefix("/cdn/") }
            #expect(hub.allSatisfy { $0.headers["authorization"] == "Bearer hf_abc" })

            let other = try temporaryCache()
            defer { try? FileManager.default.removeItem(at: other.directory) }
            let count = server.requests.count
            _ = try await resolver.resolve(
                .hub(repository: repository, revision: commit), cache: other, token: "  ")
            #expect(
                server.requests.dropFirst(count).allSatisfy { $0.headers["authorization"] == nil })
        }
    }

    @Test("A commit already in the cache resolves when the Hub cannot be reached")
    func offlineSnapshot() async throws {
        let cache = try temporaryCache()
        defer { try? FileManager.default.removeItem(at: cache.directory) }
        let server = try TestHubServer()
        try await server.start()
        let endpoint = server.endpoint
        _ = try await ModelResolver(endpoint: endpoint).resolve(
            .hub(repository: repository, revision: nil), cache: cache)
        server.stop()
        try await Task.sleep(for: .milliseconds(100))
        let offline = ModelResolver(endpoint: endpoint, maxAttempts: 1)
        let directory = try await offline.resolve(
            .hub(repository: repository, revision: commit), cache: cache)
        #expect(directory == cache.snapshotDirectory(repository, commit: commit))
        // refs/main is how a branch resolves offline too.
        let byBranch = try await offline.resolve(
            .hub(repository: repository, revision: "main"), cache: cache)
        #expect(byBranch == directory)
    }

    @Test("A local directory resolves without network; one without config.json fails clearly")
    func localDirectory() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-local-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for file in TestHubServer.files {
            try file.content.write(to: directory.appendingPathComponent(file.path))
        }
        // Port 9 (discard) is closed: any request would fail.
        let resolver = ModelResolver(endpoint: URL(string: "http://127.0.0.1:9")!)
        #expect(try await resolver.resolve(.directory(directory)) == directory)

        try FileManager.default.removeItem(at: directory.appendingPathComponent("config.json"))
        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("model.safetensors"))
        let error = await #expect(throws: ModelResolverError.self) {
            try await resolver.resolve(.directory(directory))
        }
        #expect(error == .incompleteDirectory(directory, missing: ["config.json"]))
        #expect(error?.description.contains("lacks config.json") == true)

        // With config.json back, the absent shard the index names is reported.
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        let shard = await #expect(throws: ModelResolverError.self) {
            try await resolver.resolve(.directory(directory))
        }
        #expect(shard == .incompleteDirectory(directory, missing: ["model.safetensors"]))
    }

    @Test("Repository ids that are not org/name, or that would leave the cache, are refused")
    func unsafeNames() async throws {
        let resolver = ModelResolver(endpoint: URL(string: "http://127.0.0.1:9")!)
        for bad in ["../evil", "org/../x", "single", "org/name/extra", "org/"] {
            let error = await #expect(throws: ModelResolverError.self) {
                try await resolver.resolve(.hub(repository: bad, revision: nil))
            }
            #expect(error == .invalidRepository(bad))
        }
    }

    @Test("The cache follows HF_HUB_CACHE, then HF_HOME, then XDG_CACHE_HOME, then ~/.cache")
    func cacheLocation() {
        #expect(
            HubCacheLocation(environment: ["HF_HUB_CACHE": "/a", "HF_HOME": "/b"]).directory.path
                == "/a")
        #expect(HubCacheLocation(environment: ["HF_HOME": "/b"]).directory.path == "/b/hub")
        #expect(
            HubCacheLocation(environment: ["XDG_CACHE_HOME": "/c", "HF_HUB_CACHE": ""]).directory
                .path == "/c/huggingface/hub")
        #expect(
            HubCacheLocation(environment: [:]).directory.path
                == NSHomeDirectory() + "/.cache/huggingface/hub")
    }

    @Test("A setting names a path, a pinned preset, or a repository at a revision")
    func sourceSetting() {
        #expect(ModelSource(setting: "mlx-community/diffusiongemma-26B-A4B-it-4bit") == .fourBit)
        #expect(ModelSource(setting: "mlx-community/diffusiongemma-26B-A4B-it-8bit") == .eightBit)
        #expect(ModelSource(setting: "mlx-community/diffusiongemma-26B-A4B-it-bf16") == .bf16)
        #expect(
            ModelSource(setting: "mlx-community/diffusiongemma-26B-A4B-it-4bit@main")
                == .hub(
                    repository: "mlx-community/diffusiongemma-26B-A4B-it-4bit", revision: "main"))
        #expect(ModelSource(setting: "org/other") == .hub(repository: "org/other", revision: nil))
        #expect(ModelSource(setting: "org/other@") == .hub(repository: "org/other", revision: nil))
        #expect(
            ModelSource(setting: "/models/dg")
                == .directory(URL(fileURLWithPath: "/models/dg", isDirectory: true)))
        for case .hub(_, let revision?) in ModelSource.presets {
            #expect(revision.count == 40 && revision.allSatisfy(\.isHexDigit))
        }
    }

    private func file(named path: String) -> TestHubServer.File {
        TestHubServer.files.first { $0.path == path }!
    }
}

/// The progress reports of one resolve.
final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var reports: [ModelResolver.Progress] = []
    var values: [ModelResolver.Progress] { lock.withLock { reports } }
    func append(_ report: ModelResolver.Progress) { lock.withLock { reports.append(report) } }
}
