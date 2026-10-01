import ArgumentParser

/// `--backend`, which every subcommand takes.
struct BackendOption: ParsableArguments {
    @Option(
        help: ArgumentHelp(
            "The backend: mlx, laya or verdict. Overrides OPENJEV_BACKEND.", valueName: "name"))
    var backend: String?

    /// `environment` with the flag over its variable.
    func applied(to environment: [String: String]) -> [String: String] {
        var environment = environment
        if let backend {
            environment["OPENJEV_BACKEND"] = backend
        }
        return environment
    }
}
