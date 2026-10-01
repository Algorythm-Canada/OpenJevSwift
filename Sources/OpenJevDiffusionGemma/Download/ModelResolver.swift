// Model resolution and download (issue #30): a local directory as is, or a Hub repository
// resolved to a commit and downloaded into the Hugging Face cache in huggingface_hub's layout,
// with resume and digest checks, so upstream OpenJev and mlx-vlm read the same files.

import CryptoKit
import Foundation

/// Why a model source could not be resolved. Every case names the repository, the revision and,
/// where there is one, the file.
public enum ModelResolverError: Error, Sendable, Hashable, CustomStringConvertible {
    /// A local directory lacks files a checkpoint needs.
    case incompleteDirectory(URL, missing: [String])
    /// The repository id is not `{org}/{name}` of letters, digits, `-`, `_` and `.`.
    case invalidRepository(String)
    /// The Hub answered 401 or 403: the repository is gated or private, or the token cannot
    /// read it.
    case unauthorized(
        repository: String, revision: String, file: String?, status: Int,
        tokenSent: Bool)
    /// The Hub has no such revision (404 on the revision or the tree).
    case revisionNotFound(repository: String, revision: String)
    /// The Hub lists a file it then does not serve (404 on the download).
    case fileNotFound(repository: String, revision: String, file: String)
    /// Another HTTP status.
    case httpStatus(repository: String, revision: String, file: String?, status: Int)
    /// The Hub's JSON was not what its API returns.
    case invalidResponse(repository: String, revision: String, reason: String)
    /// A tree path that would leave the snapshot directory, or an id that is not a digest.
    case unsafePath(repository: String, revision: String, path: String)
    /// A downloaded file is not the size the tree lists.
    case sizeMismatch(
        repository: String, revision: String, file: String, expected: Int,
        actual: Int)
    /// A downloaded file's digest is not the one the tree lists. The download is not kept.
    case digestMismatch(
        repository: String, revision: String, file: String, algorithm: String,
        expected: String, actual: String)
    /// The network failed and retrying did not help.
    case transport(repository: String, revision: String, file: String?, reason: String)

    public var description: String {
        switch self {
        case .incompleteDirectory(let url, let missing):
            return "\(url.path) is not a DiffusionGemma checkpoint: it lacks "
                + missing.joined(separator: ", ")
        case .invalidRepository(let repository):
            return "\(repository) is not a Hugging Face repository id of the form org/name"
        case .unauthorized(let repository, let revision, let file, let status, let tokenSent):
            let what = file.map { "\($0) of " } ?? ""
            let token =
                tokenSent
                ? "the token in \(HubCacheLocation.tokenVariable) cannot read it"
                : "set \(HubCacheLocation.tokenVariable) to a token that can read it"
            return "HTTP \(status) for \(what)\(repository) at \(revision): the repository is "
                + "gated or private; \(token)"
        case .revisionNotFound(let repository, let revision):
            return "\(repository) has no revision \(revision)"
        case .fileNotFound(let repository, let revision, let file):
            return "\(repository) at \(revision) lists \(file) but does not serve it (HTTP 404)"
        case .httpStatus(let repository, let revision, let file, let status):
            let what = file.map { "\($0) of " } ?? ""
            return "HTTP \(status) for \(what)\(repository) at \(revision)"
        case .invalidResponse(let repository, let revision, let reason):
            return "the Hub's answer for \(repository) at \(revision) is not usable: \(reason)"
        case .unsafePath(let repository, let revision, let path):
            return "\(repository) at \(revision) lists \(path), which is not a safe cache path"
        case .sizeMismatch(let repository, let revision, let file, let expected, let actual):
            return "\(file) of \(repository) at \(revision) is \(actual) bytes; the Hub lists "
                + "\(expected)"
        case .digestMismatch(
            let repository, let revision, let file, let algorithm, let expected, let actual):
            return "\(file) of \(repository) at \(revision) has \(algorithm) \(actual); the Hub "
                + "lists \(expected)"
        case .transport(let repository, let revision, let file, let reason):
            let what = file.map { "\($0) of " } ?? ""
            return "downloading \(what)\(repository) at \(revision) failed: \(reason)"
        }
    }
}

/// Resolves a ``ModelSource`` to a local directory, downloading what the Hugging Face cache
/// lacks.
///
/// The downloader is this module's own rather than swift-transformers' `HubApi`, whose snapshots
/// go to `downloadBase/models/{org}/{repo}/` and so cannot share the checkpoint upstream and
/// mlx-vlm already keep in huggingface_hub's layout (``HubCacheLocation``).
public struct ModelResolver: Sendable {
    /// The files a checkpoint directory must hold before the loader reads it.
    public static let requiredFiles = ["config.json", "model.safetensors.index.json"]

    /// Where the download stands: files and bytes done out of the tree's totals.
    public struct Progress: Sendable, Hashable {
        /// The repository, or the directory's path for a local source.
        public var repository: String
        /// The commit being resolved; empty before it is known.
        public var commit: String
        /// The files present and checked.
        public var filesDone: Int
        /// The files the tree lists (or that were asked for).
        public var fileCount: Int
        /// The bytes present, the current file's partial bytes included.
        public var bytesDone: Int
        /// The bytes the tree lists.
        public var totalBytes: Int
        /// The bytes this resolve downloaded so far.
        public var downloadedBytes: Int
        /// The file being downloaded, nil between files.
        public var currentFile: String?
    }

    /// What a resolve found and did.
    public struct Resolution: Sendable, Hashable {
        /// The directory to load: the snapshot, or the local directory.
        public var directory: URL
        /// The commit, nil for a local directory.
        public var commit: String?
        /// The files downloaded.
        public var downloadedFiles: Int
        /// The bytes downloaded, resumed bytes excluded.
        public var downloadedBytes: Int
    }

    /// The Hub, `https://huggingface.co` unless a test serves its own.
    public var endpoint: URL
    /// How many times a file's download is attempted before the transport error is thrown.
    public var maxAttempts: Int
    private let configuration: URLSessionConfiguration
    private let session: URLSession

    /// A resolver for `endpoint`.
    public init(
        endpoint: URL = URL(string: "https://huggingface.co")!, maxAttempts: Int = 5,
        configuration: URLSessionConfiguration = .ephemeral
    ) {
        self.endpoint = endpoint
        self.maxAttempts = max(1, maxAttempts)
        self.configuration = configuration
        session = URLSession(configuration: configuration)
    }

    /// The directory `source` names, downloading into `cache` what it lacks.
    ///
    /// See ``resolution(of:cache:token:files:progress:)``.
    public func resolve(
        _ source: ModelSource, cache: HubCacheLocation = .standard, token: String? = nil,
        progress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        try await resolution(of: source, cache: cache, token: token, progress: progress).directory
    }

    /// Resolves `source`.
    ///
    /// A directory is used as is once ``requiredFiles`` are found in it. A Hub source is
    /// resolved to a commit: a 40-hex revision is one already; a branch or tag (nil is `main`) is
    /// asked of the Hub API and `refs/<name>` is written in the cache. Then every file of the
    /// tree (or of `files` alone) is looked for in the snapshot and downloaded when missing:
    /// into `blobs/<id>.incomplete` with HTTP Range resume, checked (SHA-256 for LFS files; the
    /// size and the git blob SHA-1 for the others), renamed to `blobs/<id>` and linked from the
    /// snapshot with a relative symlink. When the Hub cannot be reached and the revision is a
    /// commit, a snapshot that holds ``requiredFiles`` and every shard its index names is used
    /// without it.
    ///
    /// - Parameters:
    ///   - token: the access token, sent as `Authorization: Bearer`; nil or empty sends none.
    ///   - files: the tree paths to fetch, nil for all of them. When it is set the snapshot is
    ///     not required to be a complete checkpoint.
    /// - Throws: ``ModelResolverError``, or the file system's errors.
    public func resolution(
        of source: ModelSource, cache: HubCacheLocation = .standard, token: String? = nil,
        files: Set<String>? = nil, progress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> Resolution {
        switch source {
        case .directory(let directory):
            let missing = Self.missingFiles(in: directory)
            guard missing.isEmpty else {
                throw ModelResolverError.incompleteDirectory(directory, missing: missing)
            }
            return Resolution(
                directory: directory, commit: nil, downloadedFiles: 0, downloadedBytes: 0)
        case .hub(let repository, let revision):
            let token = token.flatMap {
                let trimmed = $0.trimmingCharacters(in: .whitespaces)
                return trimmed.isEmpty ? nil : trimmed
            }
            return try await HubSnapshot(
                resolver: self, cache: cache, repository: repository,
                revision: revision ?? "main", token: token, progress: progress
            ).resolve(files: files)
        }
    }

    /// The ``requiredFiles`` `directory` lacks, then the shards its index names that are absent.
    public static func missingFiles(in directory: URL) -> [String] {
        let manager = FileManager.default
        var missing = requiredFiles.filter {
            !manager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        if missing.isEmpty,
            let data = try? Data(
                contentsOf: directory.appendingPathComponent("model.safetensors.index.json")),
            let index = try? JSONDecoder().decode(ShardIndex.self, from: data)
        {
            missing += Set(index.weightMap.values).sorted().compactMap { shard in
                guard isContainedPath(shard) else {
                    return "\(shard) (the index names a path outside the directory)"
                }
                return manager.fileExists(atPath: directory.appendingPathComponent(shard).path)
                    ? nil : shard
            }
        }
        return missing
    }

    /// True when `path` is relative and has no empty, `.` or `..` component, so appending it to
    /// a directory stays inside that directory.
    static func isContainedPath(_ path: String) -> Bool {
        !path.isEmpty && !path.hasPrefix("/")
            && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
                !$0.isEmpty && $0 != "." && $0 != ".."
            }
    }

    private struct ShardIndex: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    // MARK: Requests

    /// A GET of `url` with the token and `Accept-Encoding: identity`, so byte ranges are of the
    /// file itself.
    func request(_ url: URL, token: String?, range: Int? = nil) -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let range, range > 0 {
            request.setValue("bytes=\(range)-", forHTTPHeaderField: "Range")
        }
        return request
    }

    /// The body and response of a small request.
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        return (data, http)
    }

    /// Streams `request` into `handle`, which is positioned at `offset`, reporting the bytes in
    /// the file after each chunk.
    func transfer(
        _ request: URLRequest, into handle: FileHandle, offset: Int,
        onBytes: @escaping @Sendable (Int) -> Void
    ) async -> FileTransfer.Outcome {
        let transfer = FileTransfer(handle: handle, offset: offset, onBytes: onBytes)
        let session = URLSession(
            configuration: configuration, delegate: transfer, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return await transfer.run(request, in: session)
    }
}

/// One file download streamed to disk by a URLSession data delegate, so a 5 GB shard is never
/// held in memory and its partial bytes survive an interruption.
///
/// `@unchecked Sendable`: URLSession calls the delegate on its serial delegate queue, and the
/// lock guards the state ``run(_:in:)`` reads when the task completes.
final class FileTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    /// How a transfer ended.
    struct Outcome: Sendable {
        /// The HTTP status, 0 when no response arrived.
        var status: Int
        /// The bytes in the file afterwards.
        var bytes: Int
        /// The bytes this transfer wrote.
        var written: Int
        /// True when the server answered a range request with something other than the range
        /// asked for, so the file must start again.
        var restart: Bool
        /// The transport or file error, if any.
        var error: String?
    }

    private let lock = NSLock()
    private let handle: FileHandle
    private let offset: Int
    private let onBytes: @Sendable (Int) -> Void
    private var status = 0
    private var bytes: Int
    private var written = 0
    private var restart = false
    private var failure: String?
    private var continuation: CheckedContinuation<Outcome, Never>?

    init(handle: FileHandle, offset: Int, onBytes: @escaping @Sendable (Int) -> Void) {
        self.handle = handle
        self.offset = offset
        self.onBytes = onBytes
        bytes = offset
    }

    /// Runs the request to its end. Cancelling the calling task cancels the data task, which
    /// then completes with an error and resumes the continuation once, as any completion does.
    func run(_ request: URLRequest, in session: URLSession) async -> Outcome {
        let task = session.dataTask(with: request)
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.withLock { self.continuation = continuation }
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        let http = response as? HTTPURLResponse
        lock.lock()
        let disposition = accept(
            status: http?.statusCode ?? 0, range: http?.value(forHTTPHeaderField: "Content-Range"))
        lock.unlock()
        completionHandler(disposition)
    }

    /// Follows a redirect, without the token when it leaves the host: the Hub sends LFS files to
    /// its CDN, which needs no token, as huggingface_hub does.
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        var request = request
        if request.url?.host != task.originalRequest?.url?.host {
            request.setValue(nil, forHTTPHeaderField: "Authorization")
        }
        completionHandler(request)
    }

    /// Whether to take the response's body, and what it means for the file. Called under the
    /// lock.
    private func accept(status code: Int, range: String?) -> URLSession.ResponseDisposition {
        status = code
        switch code {
        case 200 where offset > 0:
            // The server ignored the range: the file starts again.
            do {
                try handle.truncate(atOffset: 0)
                bytes = 0
                return .allow
            } catch {
                failure = "\(error)"
                return .cancel
            }
        case 200:
            return .allow
        case 206 where Self.rangeStart(range) == offset:
            return .allow
        case 206:
            restart = true
            return .cancel
        default:
            return .cancel
        }
    }

    /// Appends a chunk to the file and returns the bytes in it, or nil after a write error.
    /// Called under the lock.
    private func append(_ data: Data, cancelling task: URLSessionDataTask) -> Int? {
        do {
            try handle.write(contentsOf: data)
            bytes += data.count
            written += data.count
            return bytes
        } catch {
            failure = "\(error)"
            task.cancel()
            return nil
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        let total = append(data, cancelling: dataTask)
        lock.unlock()
        if let total {
            onBytes(total)
        }
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
    ) {
        lock.lock()
        let waiting = self.continuation
        self.continuation = nil
        var reason = self.failure
        if reason == nil, let error, (200...299).contains(status), !restart {
            reason = error.localizedDescription
        }
        if reason == nil, status == 0, let error {
            reason = error.localizedDescription
        }
        let outcome = Outcome(
            status: status, bytes: bytes, written: written, restart: restart, error: reason)
        lock.unlock()
        waiting?.resume(returning: outcome)
    }

    /// The first byte of `bytes start-end/total`.
    static func rangeStart(_ header: String?) -> Int? {
        guard let header, header.hasPrefix("bytes ") else { return nil }
        let range = header.dropFirst(6)
        guard let dash = range.firstIndex(of: "-") else { return nil }
        return Int(range[..<dash])
    }
}

/// One Hub resolve: the commit, the tree, then each file.
private struct HubSnapshot {
    let resolver: ModelResolver
    let cache: HubCacheLocation
    let repository: String
    let revision: String
    let token: String?
    let progress: (@Sendable (ModelResolver.Progress) -> Void)?

    /// One file the tree lists.
    struct TreeFile: Sendable {
        let path: String
        let size: Int
        /// The blob id: the LFS SHA-256, else the git blob SHA-1.
        let id: String
        let isLFS: Bool
    }

    private var repositoryDirectory: URL { cache.repositoryDirectory(repository) }
    private var blobs: URL {
        repositoryDirectory.appendingPathComponent("blobs", isDirectory: true)
    }

    func resolve(files wanted: Set<String>?) async throws -> ModelResolver.Resolution {
        try checkRepository()
        let isCommit = Self.isCommitHash(revision)
        let commit: String
        let tree: [TreeFile]
        do {
            commit = isCommit ? revision.lowercased() : try await resolveCommit()
            tree = try await fetchTree(commit: commit)
        } catch let error as URLError {
            // A cancelled load is cancelled, not offline.
            if Task.isCancelled || error.code == .cancelled {
                throw CancellationError()
            }
            // Offline: a commit whose snapshot is complete needs no network.
            if let commit = offlineCommit(), wanted == nil {
                let snapshot = cache.snapshotDirectory(repository, commit: commit)
                if ModelResolver.missingFiles(in: snapshot).isEmpty {
                    return ModelResolver.Resolution(
                        directory: snapshot, commit: commit, downloadedFiles: 0,
                        downloadedBytes: 0)
                }
            }
            throw ModelResolverError.transport(
                repository: repository, revision: revision, file: nil,
                reason: error.localizedDescription)
        }
        if !isCommit {
            try writeReference(commit)
        }

        let selected = wanted.map { names in tree.filter { names.contains($0.path) } } ?? tree
        if let wanted {
            let listed = Set(tree.map(\.path))
            if let absent = wanted.subtracting(listed).sorted().first {
                throw ModelResolverError.fileNotFound(
                    repository: repository, revision: commit, file: absent)
            }
        }
        let snapshot = cache.snapshotDirectory(repository, commit: commit)
        let manager = FileManager.default
        try manager.createDirectory(at: blobs, withIntermediateDirectories: true)
        try manager.createDirectory(at: snapshot, withIntermediateDirectories: true)

        let totalBytes = selected.reduce(0) { $0 + $1.size }
        var state = ModelResolver.Progress(
            repository: repository, commit: commit, filesDone: 0, fileCount: selected.count,
            bytesDone: 0, totalBytes: totalBytes, downloadedBytes: 0, currentFile: nil)
        progress?(state)
        var downloadedFiles = 0
        for file in selected {
            let link = snapshot.appendingPathComponent(file.path)
            if !isPresent(file, at: link) {
                state.currentFile = file.path
                progress?(state)
                if let added = try await download(file, commit: commit, progress: state) {
                    state.downloadedBytes += added
                    downloadedFiles += 1
                }
                try self.link(file, at: link, snapshot: snapshot)
            }
            state.filesDone += 1
            state.bytesDone += file.size
            state.currentFile = nil
            progress?(state)
        }
        if wanted == nil {
            let missing = ModelResolver.missingFiles(in: snapshot)
            guard missing.isEmpty else {
                throw ModelResolverError.incompleteDirectory(snapshot, missing: missing)
            }
        }
        return ModelResolver.Resolution(
            directory: snapshot, commit: commit, downloadedFiles: downloadedFiles,
            downloadedBytes: state.downloadedBytes)
    }

    // MARK: Hub API

    private func checkRepository() throws {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        guard parts.count == 2,
            parts.allSatisfy({
                !$0.isEmpty && $0 != "." && $0 != ".."
                    && $0.unicodeScalars.allSatisfy { allowed.contains($0) && $0.isASCII }
            })
        else { throw ModelResolverError.invalidRepository(repository) }
    }

    static func isCommitHash(_ revision: String) -> Bool {
        revision.count == 40 && revision.allSatisfy(\.isHexDigit)
    }

    private static func encoded(_ component: String) -> String {
        component.addingPercentEncoding(
            withAllowedCharacters: CharacterSet.alphanumerics.union(
                CharacterSet(charactersIn: "-_.~"))) ?? component
    }

    private func apiURL(_ suffix: String) -> URL {
        resolver.endpoint.appendingPathComponent("api/models/\(repository)/\(suffix)")
    }

    /// Throws the error a refused metadata request stands for.
    private func check(_ response: HTTPURLResponse, file: String?) throws {
        switch response.statusCode {
        case 200...299:
            return
        case 401, 403:
            throw ModelResolverError.unauthorized(
                repository: repository, revision: revision, file: file,
                status: response.statusCode, tokenSent: token != nil)
        case 404:
            throw ModelResolverError.revisionNotFound(repository: repository, revision: revision)
        default:
            throw ModelResolverError.httpStatus(
                repository: repository, revision: revision, file: file,
                status: response.statusCode)
        }
    }

    /// `GET /api/models/{repo}/revision/{revision}`'s `sha`.
    private func resolveCommit() async throws -> String {
        struct Info: Decodable { let sha: String }
        let url = URL(
            string: apiURL("revision").absoluteString + "/" + Self.encoded(revision))!
        let (data, response) = try await resolver.data(for: resolver.request(url, token: token))
        try check(response, file: nil)
        guard let info = try? JSONDecoder().decode(Info.self, from: data),
            Self.isCommitHash(info.sha)
        else {
            throw ModelResolverError.invalidResponse(
                repository: repository, revision: revision, reason: "no commit sha")
        }
        return info.sha.lowercased()
    }

    /// `GET /api/models/{repo}/tree/{commit}?recursive=true`, following `Link: rel="next"`.
    private func fetchTree(commit: String) async throws -> [TreeFile] {
        struct Entry: Decodable {
            struct LFS: Decodable {
                let oid: String
                let size: Int
            }
            let type: String
            let oid: String?
            let size: Int?
            let path: String
            let lfs: LFS?
        }
        var next: URL? = URL(
            string: apiURL("tree").absoluteString + "/\(commit)?recursive=true")
        var files: [TreeFile] = []
        while let url = next {
            let (data, response) = try await resolver.data(
                for: resolver.request(url, token: token))
            try check(response, file: nil)
            let entries: [Entry]
            do {
                entries = try JSONDecoder().decode([Entry].self, from: data)
            } catch {
                throw ModelResolverError.invalidResponse(
                    repository: repository, revision: commit, reason: "the tree: \(error)")
            }
            for entry in entries where entry.type == "file" {
                try files.append(
                    treeFile(
                        entry.path, oid: entry.oid, size: entry.size,
                        lfs: entry.lfs.map { ($0.oid, $0.size) }, commit: commit))
            }
            next = Self.nextLink(response.value(forHTTPHeaderField: "Link"), relativeTo: url)
            // The token goes only to the Hub: a next page on another origin is refused.
            if let page = next, !Self.sameOrigin(page, resolver.endpoint) {
                throw ModelResolverError.invalidResponse(
                    repository: repository, revision: commit,
                    reason: "the tree's next page is on another host, \(page.host ?? "none")")
            }
        }
        guard !files.isEmpty else {
            throw ModelResolverError.invalidResponse(
                repository: repository, revision: commit, reason: "the tree lists no file")
        }
        return files.sorted { $0.path < $1.path }
    }

    private func treeFile(
        _ path: String, oid: String?, size: Int?, lfs: (oid: String, size: Int)?, commit: String
    ) throws -> TreeFile {
        guard ModelResolver.isContainedPath(path) else {
            throw ModelResolverError.unsafePath(
                repository: repository, revision: commit, path: path)
        }
        let id: String
        let bytes: Int
        if let lfs {
            id = lfs.oid.lowercased()
            bytes = lfs.size
            guard id.count == 64 else {
                throw ModelResolverError.unsafePath(
                    repository: repository, revision: commit, path: path)
            }
        } else {
            guard let oid, let size else {
                throw ModelResolverError.invalidResponse(
                    repository: repository, revision: commit,
                    reason: "\(path) has no oid or size")
            }
            id = oid.lowercased()
            bytes = size
            guard id.count == 40 else {
                throw ModelResolverError.unsafePath(
                    repository: repository, revision: commit, path: path)
            }
        }
        guard id.allSatisfy(\.isHexDigit), bytes >= 0 else {
            throw ModelResolverError.unsafePath(
                repository: repository, revision: commit, path: path)
        }
        return TreeFile(path: path, size: bytes, id: id, isLFS: lfs != nil)
    }

    /// True when `a` and `b` have the same scheme, host and port.
    static func sameOrigin(_ a: URL, _ b: URL) -> Bool {
        func port(_ url: URL) -> Int? {
            url.port ?? (url.scheme == "https" ? 443 : url.scheme == "http" ? 80 : nil)
        }
        return a.scheme?.lowercased() == b.scheme?.lowercased()
            && a.host?.lowercased() == b.host?.lowercased() && port(a) == port(b)
    }

    /// The `rel="next"` URL of a `Link` header, resolved against `base`.
    static func nextLink(_ header: String?, relativeTo base: URL? = nil) -> URL? {
        guard let header else { return nil }
        for part in header.split(separator: ",") where part.contains("rel=\"next\"") {
            if let open = part.firstIndex(of: "<"), let close = part.firstIndex(of: ">"),
                open < close
            {
                return URL(
                    string: String(part[part.index(after: open)..<close]), relativeTo: base)?
                    .absoluteURL
            }
        }
        return nil
    }

    /// The commit the revision names without asking the Hub: itself, or `refs/<revision>`.
    private func offlineCommit() -> String? {
        if Self.isCommitHash(revision) {
            return revision.lowercased()
        }
        let ref = repositoryDirectory.appendingPathComponent("refs/\(revision)")
        guard let text = try? String(contentsOf: ref, encoding: .utf8) else { return nil }
        let commit = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return Self.isCommitHash(commit) ? commit : nil
    }

    /// Writes `refs/<revision>` as huggingface_hub does: the commit, no newline.
    private func writeReference(_ commit: String) throws {
        let components = revision.split(separator: "/", omittingEmptySubsequences: false)
        guard components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw ModelResolverError.unsafePath(
                repository: repository, revision: revision, path: "refs/\(revision)")
        }
        let ref = repositoryDirectory.appendingPathComponent("refs/\(revision)")
        try FileManager.default.createDirectory(
            at: ref.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(commit.utf8).write(to: ref, options: .atomic)
    }

    // MARK: Files

    /// True when the snapshot path exists and, through its symlink, has the tree's size.
    private func isPresent(_ file: TreeFile, at link: URL) -> Bool {
        let path = link.resolvingSymlinksInPath().path
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
            let size = (attributes[.size] as? NSNumber)?.intValue
        else { return false }
        return size == file.size
    }

    /// Downloads a file into its blob. Returns the bytes downloaded, or nil when the blob was
    /// already there (another snapshot shares it) and only the link is missing.
    private func download(
        _ file: TreeFile, commit: String, progress state: ModelResolver.Progress
    ) async throws -> Int? {
        let manager = FileManager.default
        let blob = blobs.appendingPathComponent(file.id)
        if let attributes = try? manager.attributesOfItem(atPath: blob.path),
            (attributes[.size] as? NSNumber)?.intValue == file.size
        {
            return nil
        }
        let partial = blobs.appendingPathComponent(file.id + ".incomplete")
        let url = resolver.endpoint.appendingPathComponent(
            "\(repository)/resolve/\(commit)/\(file.path)")
        var downloaded = 0
        var lastError = ""
        var attempt = 0
        while true {
            attempt += 1
            if !manager.fileExists(atPath: partial.path) {
                manager.createFile(atPath: partial.path, contents: nil)
            }
            var offset =
                (try? manager.attributesOfItem(atPath: partial.path)[.size] as? NSNumber)?
                .intValue ?? 0
            if offset > file.size {
                try Data().write(to: partial)
                offset = 0
            }
            if offset == file.size {
                break
            }
            if attempt > resolver.maxAttempts {
                throw ModelResolverError.transport(
                    repository: repository, revision: commit, file: file.path, reason: lastError)
            }
            let handle = try FileHandle(forWritingTo: partial)
            try handle.seek(toOffset: UInt64(offset))
            let progress = self.progress
            let base = state
            let reportEvery = 4 << 20
            let reported = ReportedBytes()
            let resumedAt = offset
            let outcome = await resolver.transfer(
                resolver.request(url, token: token, range: offset), into: handle, offset: offset
            ) { bytes in
                guard let progress, reported.shouldReport(bytes, every: reportEvery) else {
                    return
                }
                var current = base
                current.bytesDone += bytes
                current.downloadedBytes += bytes - resumedAt
                progress(current)
            }
            try? handle.close()
            // A cancelled load stops here, keeping the partial file for the next run.
            try Task.checkCancellation()
            downloaded += outcome.written
            switch outcome.status {
            case 401, 403:
                throw ModelResolverError.unauthorized(
                    repository: repository, revision: commit, file: file.path,
                    status: outcome.status, tokenSent: token != nil)
            case 404:
                throw ModelResolverError.fileNotFound(
                    repository: repository, revision: commit, file: file.path)
            case 416:
                // The range cannot be served: start the file again.
                try Data().write(to: partial)
                lastError = "HTTP 416 for a resumed range"
                continue
            case 200, 206, 0:
                break
            default:
                if outcome.status >= 500 {
                    lastError = "HTTP \(outcome.status)"
                    continue
                }
                throw ModelResolverError.httpStatus(
                    repository: repository, revision: commit, file: file.path,
                    status: outcome.status)
            }
            if outcome.restart {
                try Data().write(to: partial)
                lastError = "the server sent another range than the one asked for"
                continue
            }
            if let error = outcome.error {
                lastError = error
                continue
            }
            if outcome.bytes != file.size {
                lastError = "the response ended at \(outcome.bytes) of \(file.size) bytes"
                continue
            }
        }
        do {
            try verify(partial, file, commit: commit)
        } catch {
            try? manager.removeItem(at: partial)
            throw error
        }
        if manager.fileExists(atPath: blob.path) {
            try manager.removeItem(at: blob)
        }
        try manager.moveItem(at: partial, to: blob)
        return downloaded
    }

    /// Checks a downloaded file: its size, then its SHA-256 for an LFS file or the SHA-1 of
    /// `blob <size>\0` and the content for a git file.
    private func verify(_ url: URL, _ file: TreeFile, commit: String) throws {
        let size =
            (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?
            .intValue ?? -1
        guard size == file.size else {
            throw ModelResolverError.sizeMismatch(
                repository: repository, revision: commit, file: file.path, expected: file.size,
                actual: size)
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let actual: String
        let algorithm: String
        if file.isLFS {
            var hasher = SHA256()
            while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
                hasher.update(data: chunk)
            }
            actual = Self.hex(hasher.finalize())
            algorithm = "SHA-256"
        } else {
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(file.size)\0".utf8))
            while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
                hasher.update(data: chunk)
            }
            actual = Self.hex(hasher.finalize())
            algorithm = "git blob SHA-1"
        }
        guard actual == file.id else {
            throw ModelResolverError.digestMismatch(
                repository: repository, revision: commit, file: file.path,
                algorithm: algorithm, expected: file.id, actual: actual)
        }
    }

    private static func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Links `snapshots/<commit>/<path>` to `../../blobs/<id>` (one more `..` per directory of
    /// the path), replacing whatever was there.
    private func link(_ file: TreeFile, at link: URL, snapshot: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(
            at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
        if (try? manager.destinationOfSymbolicLink(atPath: link.path)) != nil
            || manager.fileExists(atPath: link.path)
        {
            try manager.removeItem(at: link)
        }
        let depth = file.path.split(separator: "/").count - 1
        let target =
            String(repeating: "../", count: depth + 2) + "blobs/" + file.id
        try manager.createSymbolicLink(atPath: link.path, withDestinationPath: target)
    }
}

/// Throttles progress reports to one per `every` bytes. Called on the transfer's delegate
/// queue only, so the lock is uncontended; it is there for `Sendable`.
private final class ReportedBytes: @unchecked Sendable {
    private let lock = NSLock()
    private var last = 0

    func shouldReport(_ bytes: Int, every: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard bytes - last >= every else { return false }
        last = bytes
        return true
    }
}
