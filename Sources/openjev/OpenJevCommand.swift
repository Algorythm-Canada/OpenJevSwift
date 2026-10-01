import ArgumentParser
import Foundation
import OpenJevCore

/// The root command of the `openjev` command line tool: `serve`, `decide` and `models`.
///
/// Without a subcommand it prints its help. The `--version` flag prints the package version.
@main
struct OpenJevCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "openjev",
        abstract: "A native Swift implementation of the OpenJev decision server.",
        discussion: """
            Exit statuses: 0 success, and for serve a clean shutdown; 1 any other failure; 2 \
            invalid settings or command line, with the message naming the variable; 3 a backend \
            this build lacks or that failed to load; 4 a request decide was refused.
            """,
        version: openJevCoreVersion,
        subcommands: [ServeCommand.self, DecideCommand.self, ModelsCommand.self])

    /// Runs the command line and exits with its status.
    static func main() async {
        let status = await execute(Array(CommandLine.arguments.dropFirst()), in: .process)
        Foundation.exit(status)
    }

    /// Parses `arguments`, runs the command in `context` and returns the exit status, writing
    /// what the command and the parser print to the context's streams.
    ///
    /// A command line the parser refuses is invalid settings, status 2, as it is for Python's
    /// argparse, rather than the parser's 64.
    static func execute(_ arguments: [String], in context: CommandContext) async -> Int32 {
        await CommandContext.$current.withValue(context) {
            do {
                var command = try parseAsRoot(arguments)
                if var command = command as? any AsyncParsableCommand {
                    try await command.run()
                } else {
                    try command.run()
                }
                return ExitStatus.success.rawValue
            } catch let failure as CommandFailure {
                context.writeStandardError(failure.standardError)
                return failure.status.rawValue
            } catch {
                let code = exitCode(for: error)
                let message = fullMessage(for: error)
                if code == .success {
                    if !message.isEmpty {
                        context.writeStandardOutput(Array((message + "\n").utf8))
                    }
                    return ExitStatus.success.rawValue
                }
                if !message.isEmpty {
                    context.writeStandardError(Array((message + "\n").utf8))
                }
                return code == .validationFailure
                    ? ExitStatus.invalidSettings.rawValue : code.rawValue
            }
        }
    }
}
