import ArgumentParser
import Foundation
import OpenJevCore
import OpenJevServer

/// `openjev decide`: one decision without a server, printed as the wire JSON.
struct DecideCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decide",
        abstract: "Answer one POST /v1/systemone request without a server.",
        discussion: """
            The request body is read from the file --request names, or from standard input \
            without it or with -, and treated as JSON. The response body is printed to standard \
            output exactly as the server sends it, without a trailing newline: the same \
            answers, through the same engine, for scripts and for regression files. A refusal \
            prints the server's error body to standard error and exits 4; a backend that fails \
            during the decision prints its 503 body and exits 1. The settings are read from the \
            OPENJEV_* variables, as serve reads them; the warm-up read is skipped.
            """)

    @OptionGroup var backend: BackendOption

    @Option(
        name: .customLong("request"),
        help: ArgumentHelp(
            "A file holding the request body; - or none reads standard input.", valueName: "file"))
    var requestFile: String?

    func run() async throws {
        let context = CommandContext.current
        let environment = ModelsCommand.withoutWarmUp(backend.applied(to: context.environment))
        let settings = try CommandFailure.settings(environment)
        let selected = try context.backends.backend(named: settings.backend)
        // Refuse a backend this build lacks before reading a body from a terminal.
        _ = try selected.provider(environment: environment)
        let body = try readBody(context)
        let service = try await selected.load(settings: settings, environment: environment)
        let outcome: Result<[UInt8], CommandFailure>
        do {
            outcome = .success(
                try await SystemOneHandler(settings: settings, service: service)
                    .respond(toJSONBody: body))
        } catch let error as WireError {
            outcome = .failure(Self.failure(for: error))
        } catch {
            // A 200 the server could not write either: Starlette's plain-text 500.
            outcome = .failure(CommandFailure(.failure, body: Array("Internal Server Error".utf8)))
        }
        await (service as? any ModelReleasing)?.close()
        context.writeStandardOutput(try outcome.get())
    }

    /// The request body: the file's bytes, or standard input's.
    private func readBody(_ context: CommandContext) throws(CommandFailure) -> [UInt8] {
        if let requestFile, requestFile != "-" {
            do {
                return Array(try Data(contentsOf: URL(fileURLWithPath: requestFile)))
            } catch {
                throw CommandFailure(
                    .failure, message: "cannot read \(requestFile): \(error.localizedDescription)")
            }
        }
        do {
            return try context.readStandardInput()
        } catch {
            throw CommandFailure(.failure, message: "cannot read standard input: \(error)")
        }
    }

    /// The error body the server sends for `error`, on standard error: exit 4 for a refusal,
    /// which is a 4xx or the 529 of a full queue, and 1 for a failure, the 503 of a backend that
    /// failed.
    static func failure(for error: WireError) -> CommandFailure {
        let refused = error.status < 500 || error.status == 529
        let bytes = (try? WireEncoder().bytes(error)) ?? Array("Internal Server Error".utf8)
        return CommandFailure(refused ? .refused : .failure, body: bytes)
    }
}
