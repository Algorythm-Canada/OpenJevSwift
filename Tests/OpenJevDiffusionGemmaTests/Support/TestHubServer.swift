import CryptoKit
import Foundation
import Network

/// A local HTTP server that answers as the Hugging Face Hub does for one fake repository: the
/// revision and tree JSON of its API and the files under `resolve/`, with HTTP Range, so the
/// resolver's tests run without the network.
///
/// `@unchecked Sendable`: the listener and connections run on one serial queue, and the lock
/// guards what the tests read and set.
final class TestHubServer: @unchecked Sendable {
    /// One file of the fake repository.
    struct File {
        var path: String
        var content: Data
        var isLFS: Bool

        var sha256: String {
            SHA256.hash(data: content).map { String(format: "%02x", $0) }.joined()
        }
        var gitBlobSHA1: String {
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(content.count)\0".utf8))
            hasher.update(data: content)
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }
        /// The blob id huggingface_hub names the file by.
        var blobID: String { isLFS ? sha256 : gitBlobSHA1 }
    }

    /// A request as the server saw it.
    struct Request: Sendable {
        var path: String
        var headers: [String: String]
    }

    static let repository = "test-org/tiny-model"
    static let commit = "0123456789abcdef0123456789abcdef01234567"

    /// config.json, the shard index and one LFS shard of 300,000 bytes.
    static let files: [File] = {
        var shard = Data(count: 300_000)
        for index in shard.indices {
            shard[index] = UInt8(truncatingIfNeeded: index &* 31 &+ 7)
        }
        return [
            File(
                path: "config.json", content: Data(#"{"model_type": "diffusion_gemma"}"#.utf8),
                isLFS: false),
            File(
                path: "model.safetensors.index.json",
                content: Data(#"{"weight_map": {"w": "model.safetensors"}}"#.utf8),
                isLFS: false),
            File(path: "model.safetensors", content: shard, isLFS: true),
        ]
    }()

    private let queue = DispatchQueue(label: "TestHubServer")
    private let listener: NWListener
    private let lock = NSLock()
    private var seen: [Request] = []
    private var status: Int?
    private var corrupt = false
    private var interruptAfter: Int?
    private var redirect = false

    /// The server's base URL, once started.
    private(set) var endpoint = URL(string: "http://127.0.0.1:0")!

    init() throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
    }

    /// Starts listening and sets ``endpoint``.
    func start() async throws {
        let port: UInt16 = try await withCheckedThrowingContinuation { continuation in
            let resumed = Resumed()
            listener.stateUpdateHandler = { [listener] state in
                switch state {
                case .ready:
                    if resumed.first() {
                        continuation.resume(returning: listener.port?.rawValue ?? 0)
                    }
                case .failed(let error):
                    if resumed.first() { continuation.resume(throwing: error) }
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                self?.accept(connection)
            }
            listener.start(queue: queue)
        }
        endpoint = URL(string: "http://127.0.0.1:\(port)")!
    }

    func stop() {
        listener.cancel()
    }

    /// Every request so far.
    var requests: [Request] { lock.withLock { seen } }
    /// The file downloads so far.
    var downloads: [Request] { requests.filter { $0.path.contains("/resolve/") } }

    /// Answers every request with `status` (401, say) until reset with nil.
    func respondToEverything(with status: Int?) { lock.withLock { self.status = status } }
    /// Serves the LFS file with one byte changed.
    func corruptLFSFile(_ on: Bool) { lock.withLock { corrupt = on } }
    /// Redirects downloads of the LFS file to `localhost`, another host than `127.0.0.1`, as the
    /// Hub redirects LFS files to its CDN.
    func redirectLFSFileToAnotherHost(_ on: Bool) { lock.withLock { redirect = on } }
    /// Closes the next download of the LFS file after `bytes` bytes of its body.
    func interruptNextLFSDownload(after bytes: Int) { lock.withLock { interruptAfter = bytes } }

    // MARK: Serving

    private final class Resumed: @unchecked Sendable {
        private let lock = NSLock()
        private var done = false
        func first() -> Bool {
            lock.withLock {
                defer { done = true }
                return !done
            }
        }
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(connection, buffer: Data())
    }

    private func receive(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) {
            [weak self] data, _, complete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buffer[..<end.lowerBound], as: UTF8.self)
                self.respond(to: head, on: connection)
            } else if complete || error != nil {
                connection.cancel()
            } else {
                self.receive(connection, buffer: buffer)
            }
        }
    }

    private func respond(to head: String, on connection: NWConnection) {
        let lines = head.components(separatedBy: "\r\n")
        let path = lines.first?.split(separator: " ").dropFirst().first.map(String.init) ?? "/"
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...]
                .trimmingCharacters(in: .whitespaces)
        }
        let (forced, corrupt, interrupt, redirect) = lock.withLock {
            seen.append(Request(path: path, headers: headers))
            return (status, self.corrupt, interruptAfter, self.redirect)
        }
        if let forced {
            send(connection, status: forced, body: Data(#"{"error": "refused"}"#.utf8))
            return
        }
        let repository = Self.repository
        if path.hasPrefix("/cdn/"),
            let file = Self.files.first(where: { path == "/cdn/\($0.path)" })
        {
            serve(file.content, range: headers["range"], on: connection, cutAfter: nil)
            return
        }
        if redirect, let lfs = Self.files.first(where: \.isLFS),
            path == "/\(repository)/resolve/\(Self.commit)/\(lfs.path)"
        {
            let location = "http://localhost:\(endpoint.port ?? 0)/cdn/\(lfs.path)"
            send(connection, status: 302, body: Data(), extraHeaders: ["Location: \(location)"])
            return
        }
        if path == "/api/models/\(repository)/revision/main" {
            send(connection, status: 200, body: Data(#"{"sha": "\#(Self.commit)"}"#.utf8))
        } else if path.hasPrefix("/api/models/\(repository)/revision/")
            || (path.hasPrefix("/api/models/\(repository)/tree/")
                && !path.hasPrefix("/api/models/\(repository)/tree/\(Self.commit)"))
        {
            send(connection, status: 404, body: Data(#"{"error": "Revision not found"}"#.utf8))
        } else if path == "/api/models/\(repository)/tree/\(Self.commit)?recursive=true" {
            send(connection, status: 200, body: Self.treeJSON())
        } else if path.hasPrefix("/\(repository)/resolve/\(Self.commit)/"),
            let file = Self.files.first(where: {
                path == "/\(repository)/resolve/\(Self.commit)/\($0.path)"
            })
        {
            var content = file.content
            if file.isLFS && corrupt {
                content[content.count / 2] ^= 0xFF
            }
            var cut: Int?
            if file.isLFS, let interrupt {
                cut = interrupt
                lock.withLock { interruptAfter = nil }
            }
            serve(content, range: headers["range"], on: connection, cutAfter: cut)
        } else {
            send(connection, status: 404, body: Data("Entry not found".utf8))
        }
    }

    private static func treeJSON() -> Data {
        let entries: [[String: Any]] =
            files.map { file in
                var entry: [String: Any] = [
                    "type": "file", "oid": file.gitBlobSHA1, "size": file.content.count,
                    "path": file.path,
                ]
                if file.isLFS {
                    // The Hub's tree gives an LFS file the pointer's oid and size at the top level.
                    entry["oid"] = String(repeating: "f", count: 40)
                    entry["size"] = 134
                    entry["lfs"] = [
                        "oid": file.sha256, "size": file.content.count, "pointerSize": 134,
                    ]
                }
                return entry
            } + [
                [
                    "type": "directory", "oid": String(repeating: "e", count: 40), "size": 0,
                    "path": "docs",
                ]
            ]
        return try! JSONSerialization.data(withJSONObject: entries)
    }

    private func serve(_ content: Data, range: String?, on connection: NWConnection, cutAfter: Int?)
    {
        var start = 0
        if let range, range.hasPrefix("bytes="), range.hasSuffix("-"),
            let from = Int(range.dropFirst(6).dropLast())
        {
            start = from
        }
        guard start < content.count else {
            send(connection, status: 416, body: Data())
            return
        }
        let body = content[start...]
        var extra: [String] = []
        if start > 0 {
            extra.append("Content-Range: bytes \(start)-\(content.count - 1)/\(content.count)")
        }
        send(
            connection, status: start > 0 ? 206 : 200, body: Data(body), extraHeaders: extra,
            cutAfter: cutAfter)
    }

    private func send(
        _ connection: NWConnection, status: Int, body: Data, extraHeaders: [String] = [],
        cutAfter: Int? = nil
    ) {
        let reason =
            [
                200: "OK", 206: "Partial Content", 401: "Unauthorized", 403: "Forbidden",
                404: "Not Found", 416: "Range Not Satisfiable",
            ][status] ?? "Status"
        var head = "HTTP/1.1 \(status) \(reason)\r\nContent-Length: \(body.count)\r\n"
        head += "Connection: close\r\n"
        for header in extraHeaders { head += header + "\r\n" }
        head += "\r\n"
        var bytes = Data(head.utf8)
        bytes.append(cutAfter.map { body.prefix($0) } ?? body)
        connection.send(
            content: bytes,
            completion: .contentProcessed { _ in
                connection.cancel()
            })
    }
}
