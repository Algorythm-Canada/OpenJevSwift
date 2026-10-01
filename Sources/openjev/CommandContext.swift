import Foundation
import Logging
import UnixSignals

/// What a command reads and writes outside its arguments: the environment, the standard streams,
/// the backends it may load, its loggers and how `serve` is stopped.
///
/// The commands read ``current``, which is the process's own unless a test runs a command with
/// another, so a test sees exactly what a user would without touching the process.
struct CommandContext: Sendable {
    /// The environment the `OPENJEV_*` variables are read from.
    var environment: [String: String]
    /// The backends `OPENJEV_BACKEND` may name.
    var backends: BackendRegistry
    /// Reads standard input to its end.
    var readStandardInput: @Sendable () throws -> [UInt8]
    /// Writes to standard output.
    var writeStandardOutput: @Sendable ([UInt8]) -> Void
    /// Writes to standard error.
    var writeStandardError: @Sendable ([UInt8]) -> Void
    /// The log handler of every logger a command makes, standard error's by default.
    var makeLogHandler: @Sendable (_ label: String) -> any LogHandler
    /// The signals that start `serve`'s graceful shutdown.
    var shutdownSignals: [UnixSignal]
    /// Called once `serve` listens, with the port and a function that starts its graceful
    /// shutdown as a signal would.
    var onServing: ServingHook

    /// What ``onServing`` is: the port, and the function that starts the graceful shutdown.
    typealias ServingHook = @Sendable (_ port: Int, _ shutDown: @escaping ShutDown) async -> Void
    /// Starts `serve`'s graceful shutdown.
    typealias ShutDown = @Sendable () async -> Void

    /// The context commands run in.
    @TaskLocal static var current = CommandContext.process

    /// The process's environment and streams, the standard backends, SIGINT and SIGTERM.
    static var process: CommandContext {
        CommandContext(
            environment: ProcessInfo.processInfo.environment,
            backends: .standard,
            readStandardInput: {
                try FileHandle.standardInput.readToEnd().map { Array($0) } ?? []
            },
            writeStandardOutput: { bytes in
                try? FileHandle.standardOutput.write(contentsOf: Data(bytes))
            },
            writeStandardError: { bytes in
                try? FileHandle.standardError.write(contentsOf: Data(bytes))
            },
            makeLogHandler: { label in StreamLogHandler.standardError(label: label) },
            shutdownSignals: [.sigterm, .sigint],
            onServing: { _, _ in })
    }

    /// A logger named `openjev` at `level`.
    func logger(level: Logger.Level) -> Logger {
        var logger = Logger(label: "openjev", factory: makeLogHandler)
        logger.logLevel = level
        return logger
    }
}
