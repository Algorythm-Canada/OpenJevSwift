import OpenJevServer

/// The exit statuses of `openjev`, for launchd, scripts and docs/deployment.md.
enum ExitStatus: Int32, Sendable, CaseIterable {
    /// The command did what it was asked; for `serve`, a clean shutdown.
    case success = 0
    /// Anything else: an unreadable request file, a backend that failed during a decision, an
    /// address already in use, or a shutdown that cut requests short.
    case failure = 1
    /// Invalid settings: a variable or a flag upstream's checks refuse, an unknown backend, or a
    /// command line the parser refuses. The message names the variable.
    case invalidSettings = 2
    /// The backend cannot run: this build does not have it yet, or it failed to load.
    case backendUnavailable = 3
    /// `decide` only: the request was refused, which the server answers with a 4xx or a 529.
    case refused = 4
}

/// What stops a command: an exit status and what it writes to standard error first.
struct CommandFailure: Error, Sendable, Equatable {
    /// The exit status.
    var status: ExitStatus
    /// The bytes written to standard error, as they are.
    var standardError: [UInt8]

    /// A failure that prints `openjev: {message}` and a newline.
    init(_ status: ExitStatus, message: String) {
        self.status = status
        self.standardError = Array("openjev: \(message)\n".utf8)
    }

    /// A failure that prints `body` exactly, as `decide` prints a wire error body.
    init(_ status: ExitStatus, body: [UInt8]) {
        self.status = status
        self.standardError = body
    }

    /// The settings the environment gives, or the failure for invalid settings, which repeats
    /// upstream's message.
    static func settings(_ environment: [String: String]) throws(CommandFailure)
        -> ServerSettings
    {
        do {
            return try ServerSettings(environment: environment)
        } catch {
            throw CommandFailure(.invalidSettings, message: error.message)
        }
    }
}
