import Foundation
import Observation
import OpenJevCore
import OpenJevEncoders

/// Loads Verdict once: downloads it through the library's own store on first launch, with
/// progress, then compiles, loads and warms it up. Later launches find the checked files in
/// Application Support and need no network.
@MainActor
@Observable
final class ModelLoader {
    enum Phase: Equatable {
        case idle
        /// Bytes on the device so far, of the manifest's total.
        case downloading(received: Int64, total: Int64)
        case loading
        case ready
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var engine: EncoderDecisionEngine?

    /// The manifest the store downloads, embedded in the library with each file's SHA-256.
    private let manifest = EncoderPackageManifest.verdict

    func load() async {
        // Once only: a second call while a load runs, such as a double tap on Try again, would
        // download into the same folder twice.
        switch phase {
        case .idle, .failed: break
        case .downloading, .loading, .ready: return
        }
        // A session of our own, so the app can watch the store's downloads for progress.
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 60
        let session = URLSession(configuration: configuration)
        // The watcher stops however the load ends, so a retry never sees a dead session's bytes.
        var progress: Task<Void, Never>?
        defer {
            progress?.cancel()
            session.finishTasksAndInvalidate()
        }
        do {
            phase = .loading
            let store = try EncoderPackageStore(environment: [:], session: session)
            let held = try store.heldPackageDirectory(for: manifest) != nil
            if !held {
                let total = Self.totalBytes(of: manifest)
                let onDisk = Self.bytesOnDisk(of: manifest, in: store.directory)
                phase = .downloading(received: onDisk, total: total)
                progress = Task { await watch(session, onDisk: onDisk, total: total) }
            }
            let locations = try await store.locations(for: manifest)
            progress?.cancel()
            phase = .loading
            let backend = try await VerdictBackend.load(
                configuration: VerdictBackend.Configuration(locations: locations))
            let engine = EncoderDecisionEngine(backend: backend)
            try await engine.warmUp()
            self.engine = engine
            phase = .ready
        } catch {
            phase = .failed(Self.message(for: error))
        }
    }

    /// Polls the session's tasks and adds the bytes each has received to what was already on
    /// disk. The store downloads one file at a time, so a finished task's last count stays in.
    private func watch(_ session: URLSession, onDisk: Int64, total: Int64) async {
        var received: [Int: Int64] = [:]
        while !Task.isCancelled {
            for task in await session.allTasks {
                received[task.taskIdentifier] = task.countOfBytesReceived
            }
            let bytes = min(onDisk + received.values.reduce(0, +), total)
            if case .downloading = phase {
                phase = .downloading(received: bytes, total: total)
            }
            try? await Task.sleep(for: .milliseconds(200))
        }
    }

    /// Every file the store fetches for the manifest: the package, the tokenizer and the
    /// calibrator.
    private static func totalBytes(of manifest: EncoderPackageManifest) -> Int64 {
        let files = manifest.packageFiles + manifest.tokenizerFiles + [manifest.calibrator]
        return files.reduce(0) { $0 + Int64($1.bytes) }
    }

    /// The bytes of files an earlier, interrupted launch already put in place, at the layout
    /// `EncoderPackageStore` documents: `{package}/{package}.mlpackage/` and `{package}/tokenizer/`.
    private static func bytesOnDisk(of manifest: EncoderPackageManifest, in directory: URL)
        -> Int64
    {
        let root = directory.appendingPathComponent(manifest.package, isDirectory: true)
        let entries =
            manifest.packageFiles.map { (manifest.package + ".mlpackage/" + $0.path, $0) }
            + (manifest.tokenizerFiles + [manifest.calibrator]).map { ("tokenizer/" + $0.path, $0) }
        return entries.reduce(0) { sum, entry in
            let path = root.appendingPathComponent(entry.0).path
            let size =
                (try? FileManager.default.attributesOfItem(atPath: path)[.size])
                as? NSNumber
            return size?.intValue == entry.1.bytes ? sum + Int64(entry.1.bytes) : sum
        }
    }

    /// A message a person can act on.
    private static func message(for error: any Error) -> String {
        if let error = error as? URLError {
            switch error.code {
            case .notConnectedToInternet, .networkConnectionLost, .timedOut,
                .cannotFindHost, .cannotConnectToHost, .dataNotAllowed:
                return "The download stopped: \(error.localizedDescription) "
                    + "The first launch needs a connection to fetch Verdict (about 310 MB); "
                    + "after that the app works offline."
            case .secureConnectionFailed, .serverCertificateUntrusted,
                .serverCertificateHasUnknownRoot, .serverCertificateNotYetValid,
                .serverCertificateHasBadDate:
                // A network that inspects TLS (a corporate proxy) presents its own certificate,
                // which this device may not trust. The files come from github.com and
                // huggingface.co.
                return "The download failed: \(error.localizedDescription) "
                    + "If this network inspects encrypted traffic, this device must trust its "
                    + "certificate for github.com and huggingface.co, or try another network."
            default:
                return "The download failed: \(error.localizedDescription)"
            }
        }
        if let error = error as? EncoderPackageError {
            return "Verdict could not be installed: \(error.description)"
        }
        return "Verdict could not be loaded: \(String(describing: error))"
    }
}
