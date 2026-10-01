import Foundation
import Logging
import OpenJevCore
import OpenJevServer
import OpenJevTestSupport
import Testing

@testable import openjev

/// Runs `openjev` commands in the test process with a context of the test's own: an environment,
/// standard input, captured output and log lines, and backends that include stubs.
enum CommandHarness {
    /// What a command did.
    struct Outcome: Sendable {
        /// The exit status.
        var status: Int32
        /// Everything written to standard output.
        var standardOutput: [UInt8]
        /// Everything written to standard error.
        var standardError: [UInt8]
        /// Every log line, `{level} {message}`.
        var logLines: [String]

        /// Standard output as text.
        var output: String { String(decoding: standardOutput, as: UTF8.self) }
        /// Standard error as text.
        var errors: String { String(decoding: standardError, as: UTF8.self) }
    }

    /// Runs `arguments` and returns what the command did.
    static func run(
        _ arguments: [String], environment: [String: String] = [:], input: [UInt8] = [],
        backends: BackendRegistry = .standard,
        onServing: @escaping @Sendable (Int, @escaping @Sendable () async -> Void) async -> Void =
            { _, _ in }
    ) async -> Outcome {
        let capture = Capture()
        let context = CommandContext(
            environment: environment, backends: backends,
            readStandardInput: { input },
            writeStandardOutput: { capture.output($0) },
            writeStandardError: { capture.error($0) },
            makeLogHandler: { _ in CapturingLogHandler(capture: capture) },
            shutdownSignals: [], onServing: onServing)
        let status = await OpenJevCommand.execute(arguments, in: context)
        return capture.outcome(status: status)
    }

    /// The standard backends and three of the test's: `stub` (DiffusionGemma's engine over
    /// `diffusion`), `stub-encoder` (the encoder engine over `encoder`, whose listing only the
    /// loaded backend knows) and `broken`, whose load fails.
    static func backends(
        diffusion: StubBackend = StubBackend(),
        encoder: StubQuestionReadBackend = StubQuestionReadBackend(),
        loads: LoadCounter = LoadCounter()
    ) -> BackendRegistry {
        BackendRegistry.standard
            .adding(
                BackendRegistry.Backend(
                    name: "stub", modelName: "openjev-0.1", kind: .diffusion,
                    servedModels: .diffusionGemma,
                    availability: .available { _, _ in
                        loads.increment()
                        return DecisionBackendProvider { _ in diffusion }
                    })
            )
            .adding(
                BackendRegistry.Backend(
                    name: "stub-encoder", modelName: encoder.modelInfo.name, kind: .encoder,
                    servedModels: nil,
                    availability: .available { _, willWarmUp in
                        loads.increment()
                        return QuestionReadBackendProvider(
                            load: { _ in encoder }, willWarmUp: willWarmUp)
                    })
            )
            .adding(
                BackendRegistry.Backend(
                    name: "broken", modelName: "broken-1", kind: .encoder, servedModels: nil,
                    availability: .available { _, _ in
                        QuestionReadBackendProvider { _ in throw LoadFailure() }
                    }))
    }

    /// The error the `broken` backend's load throws.
    struct LoadFailure: Error, CustomStringConvertible {
        var description: String { "the weights are missing" }
    }
}

/// Counts provider creations, which is how a test sees that a backend was loaded.
final class LoadCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    /// The loads so far.
    var count: Int {
        lock.withLock { value }
    }

    func increment() {
        lock.withLock { value += 1 }
    }
}

/// What a command wrote, collected from any task.
final class Capture: @unchecked Sendable {
    private let lock = NSLock()
    private var standardOutput: [UInt8] = []
    private var standardError: [UInt8] = []
    private var lines: [String] = []

    func output(_ bytes: [UInt8]) {
        lock.withLock { standardOutput += bytes }
    }

    func error(_ bytes: [UInt8]) {
        lock.withLock { standardError += bytes }
    }

    func log(_ line: String) {
        lock.withLock { lines.append(line) }
    }

    /// The log lines so far.
    var logLines: [String] {
        lock.withLock { lines }
    }

    func outcome(status: Int32) -> CommandHarness.Outcome {
        lock.withLock {
            CommandHarness.Outcome(
                status: status, standardOutput: standardOutput, standardError: standardError,
                logLines: lines)
        }
    }
}

/// A log handler that writes `{level} {message}` lines into a ``Capture``.
struct CapturingLogHandler: LogHandler {
    let capture: Capture
    var metadata: Logger.Metadata = [:]
    var logLevel: Logger.Level = .info

    subscript(metadataKey key: String) -> Logger.Metadata.Value? {
        get { metadata[key] }
        set { metadata[key] = newValue }
    }

    func log(event: LogEvent) {
        capture.log("\(event.level) \(event.message)")
    }
}

/// A value set once, which any number of tasks wait for.
final class Handoff<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Value?
    private var waiters: [CheckedContinuation<Value, Never>] = []

    /// Sets the value and wakes the waiters; later calls do nothing.
    func resolve(_ value: Value) {
        let woken = lock.withLock { () -> [CheckedContinuation<Value, Never>] in
            guard result == nil else { return [] }
            result = value
            defer { waiters.removeAll() }
            return waiters
        }
        for waiter in woken {
            waiter.resume(returning: value)
        }
    }

    /// The value, once it is set.
    var value: Value {
        get async {
            await withCheckedContinuation { continuation in
                let ready = lock.withLock { () -> Value? in
                    if let result {
                        return result
                    }
                    waiters.append(continuation)
                    return nil
                }
                if let ready {
                    continuation.resume(returning: ready)
                }
            }
        }
    }
}

extension WireFixtures {
    /// ``missingMessageText`` as a test comment.
    static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
}

extension PolicyFixtures {
    /// ``missingMessageText`` as a test comment.
    static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
}
