import ArgumentParser
import OpenJevCore
import OpenJevServer

/// `openjev models`: the `GET /v1/models` body of the selected backend.
struct ModelsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "Print the GET /v1/models body the server would send for the backend.",
        discussion: """
            The body is printed exactly as the server sends it, without a trailing newline. A \
            backend's listing needs no model, so nothing is loaded, and the listing of mlx and \
            laya is printed although this build cannot serve them yet. The settings are read \
            from the OPENJEV_* variables, as serve reads them, and checked the same way.
            """)

    @OptionGroup var backend: BackendOption

    func run() async throws {
        let context = CommandContext.current
        // A listing that needs the model needs no warm-up read.
        let environment = Self.withoutWarmUp(backend.applied(to: context.environment))
        let settings = try CommandFailure.settings(environment)
        let selected = try context.backends.backend(named: settings.backend)
        let served: ServedModels
        if let listed = selected.servedModels {
            served = listed
        } else {
            let service = try await selected.load(settings: settings, environment: environment)
            served = service.servedModels
            await (service as? any ModelReleasing)?.close()
        }
        let body = ModelsResponse(models: served.listing)
        // A listing holds strings only, so writing it cannot fail.
        guard let bytes = try? WireEncoder().bytes(body) else {
            throw CommandFailure(.failure, message: "the model listing could not be written")
        }
        context.writeStandardOutput(bytes)
    }

    /// The environment with the warm-up off: a command that reads once gains nothing from it.
    static func withoutWarmUp(_ environment: [String: String]) -> [String: String] {
        var environment = environment
        environment["OPENJEV_WARMUP"] = "0"
        return environment
    }
}
