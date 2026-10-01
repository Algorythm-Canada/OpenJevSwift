import Foundation
import Testing

/// The `openjev` binary SwiftPM built next to the test bundle, run as a child process.
///
/// `swift build --build-tests` and `swift test` build every product, the executable included,
/// into the directory that holds the test bundle (`.build/debug` with the native build system,
/// `.build/out/Products/Debug` with Swift Build).
enum BuiltBinary {
    /// The binary, or `nil` when it is not where the products are.
    static let url: URL? = candidates.first {
        FileManager.default.isExecutableFile(atPath: $0.path)
    }

    /// Where the binary is looked for: beside the test bundle, which is a folder ending in
    /// `.xctest` on macOS and an executable file of that name on Linux, and beside the running
    /// test executable.
    static var candidates: [URL] {
        var directories: [URL] = []
        for bundle in [Bundle(for: Locator.self).bundleURL, Bundle.main.bundleURL] {
            directories.append(
                bundle.pathExtension == "xctest" ? bundle.deletingLastPathComponent() : bundle)
        }
        if let executable = CommandLine.arguments.first {
            directories.append(URL(fileURLWithPath: executable).deletingLastPathComponent())
        }
        return directories.map { $0.appendingPathComponent("openjev") }
    }

    /// The comment of a missing binary.
    static var missingMessage: String {
        "no built openjev binary at \(candidates.map(\.path)); run swift build --build-tests"
    }

    /// A class in the test bundle, for `Bundle(for:)`.
    private final class Locator {}

    /// What a finished child process did.
    struct Outcome: Sendable {
        var status: Int32
        var standardOutput: Data
        var standardError: Data

        var output: String { String(decoding: standardOutput, as: UTF8.self) }
        var errors: String { String(decoding: standardError, as: UTF8.self) }
    }

    /// A child process with its output going to files, for tests that read it as it runs.
    final class Child: @unchecked Sendable {
        let process = Process()
        let outputFile: URL
        let errorFile: URL

        /// Starts the binary with `arguments`, `environment` and `input` on standard input.
        init(arguments: [String], environment: [String: String], input: Data = Data()) throws {
            guard let url = BuiltBinary.url else {
                throw BinaryMissing(description: BuiltBinary.missingMessage)
            }
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("openjev-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            outputFile = directory.appendingPathComponent("stdout")
            errorFile = directory.appendingPathComponent("stderr")
            let inputFile = directory.appendingPathComponent("stdin")
            for file in [outputFile, errorFile] {
                _ = FileManager.default.createFile(atPath: file.path, contents: nil)
            }
            try input.write(to: inputFile)
            process.executableURL = url
            process.arguments = arguments
            process.environment = environment
            process.standardInput = try FileHandle(forReadingFrom: inputFile)
            process.standardOutput = try FileHandle(forWritingTo: outputFile)
            process.standardError = try FileHandle(forWritingTo: errorFile)
            try process.run()
        }

        /// Standard error so far.
        var errorsSoFar: String {
            (try? String(contentsOf: errorFile, encoding: .utf8)) ?? ""
        }

        /// Waits up to `limit` for the process to exit and returns what it did, or kills it and
        /// returns `nil`.
        func wait(upTo limit: Duration) async -> Outcome? {
            let clock = ContinuousClock()
            let deadline = clock.now + limit
            while process.isRunning && clock.now < deadline {
                try? await Task.sleep(for: .milliseconds(10))
            }
            guard !process.isRunning else {
                kill(process.processIdentifier, SIGKILL)
                return nil
            }
            return Outcome(
                status: process.terminationStatus,
                standardOutput: (try? Data(contentsOf: outputFile)) ?? Data(),
                standardError: (try? Data(contentsOf: errorFile)) ?? Data())
        }

        /// Sends SIGTERM, as launchd does to stop a job.
        func terminate() {
            process.terminate()
        }

        /// Sends SIGTERM unless the process has exited, for a test's cleanup on every path.
        func terminateIfRunning() {
            if process.isRunning {
                process.terminate()
            }
        }
    }

    /// Runs the binary to its end.
    static func run(
        _ arguments: [String], environment: [String: String] = [:], input: Data = Data(),
        upTo limit: Duration = .seconds(60)
    ) async throws -> Outcome {
        let child = try Child(arguments: arguments, environment: environment, input: input)
        guard let outcome = await child.wait(upTo: limit) else {
            throw BinaryMissing(description: "openjev \(arguments) did not exit within \(limit)")
        }
        return outcome
    }

    /// The binary is missing, or did not behave.
    struct BinaryMissing: Error, CustomStringConvertible {
        var description: String
    }

    /// The environment a child gets: the test process's, without any `OPENJEV_` variable, so
    /// the developer's own settings do not leak into the checks, plus `extra`.
    static func environment(_ extra: [String: String] = [:]) -> [String: String] {
        var environment = ProcessInfo.processInfo.environment.filter {
            !$0.key.hasPrefix("OPENJEV_")
        }
        environment.merge(extra) { $1 }
        return environment
    }
}
