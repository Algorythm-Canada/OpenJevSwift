import ArgumentParser
import OpenJevCore

/// The root command of the `openjev` command line tool.
///
/// Without a subcommand it prints its help. The `--version` flag prints the package version.
@main
struct OpenJevCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "openjev",
        abstract: "A native Swift implementation of the OpenJev decision server.",
        version: openJevCoreVersion
    )
}
