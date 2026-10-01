import ArgumentParser
import OpenJevServer
import Testing

@testable import openjev

/// The command line of every subcommand and flag, and the precedence of a flag over its
/// variable (issue #40).
@Suite("Argument parsing")
struct ArgumentParsingTests {
    /// The parsed command, as the root parses it.
    private func parse<Command: ParsableCommand>(
        _ arguments: [String], as type: Command.Type
    ) throws -> Command {
        try #require(try OpenJevCommand.parseAsRoot(arguments) as? Command)
    }

    @Test("serve takes every flag")
    func serveFlags() throws {
        let serve = try parse(
            [
                "serve", "--backend", "verdict", "--host", "0.0.0.0", "--port", "9000",
                "--log-level", "debug", "--no-warmup", "--shutdown-timeout", "2.5",
            ], as: ServeCommand.self)
        #expect(serve.backend.backend == "verdict")
        #expect(serve.host == "0.0.0.0")
        #expect(serve.port == "9000")
        #expect(serve.logLevel == "debug")
        #expect(serve.noWarmup)
        #expect(serve.shutdownTimeout == 2.5)
        #expect(
            serve.applied(to: [:]) == [
                "OPENJEV_BACKEND": "verdict", "OPENJEV_HOST": "0.0.0.0", "OPENJEV_PORT": "9000",
                "OPENJEV_LOG_LEVEL": "debug", "OPENJEV_WARMUP": "0",
            ])
    }

    @Test("serve without flags leaves every variable as it is")
    func serveDefaults() throws {
        let serve = try parse(["serve"], as: ServeCommand.self)
        #expect(serve.backend.backend == nil)
        #expect(serve.host == nil && serve.port == nil && serve.logLevel == nil)
        #expect(!serve.noWarmup)
        #expect(serve.shutdownTimeout == 30)
        let environment = [
            "OPENJEV_BACKEND": "laya", "OPENJEV_PORT": "81", "OPENJEV_WARMUP": "1", "HOME": "/h",
        ]
        #expect(serve.applied(to: environment) == environment)
    }

    @Test("A flag wins over its variable, and a variable without a flag still applies")
    func precedence() throws {
        let environment = [
            "OPENJEV_BACKEND": "mlx", "OPENJEV_HOST": "10.0.0.1", "OPENJEV_PORT": "1234",
            "OPENJEV_LOG_LEVEL": "error", "OPENJEV_WARMUP": "1", "OPENJEV_MAX_QUEUE": "7",
        ]
        let serve = try parse(
            [
                "serve", "--backend", "verdict", "--host", "127.0.0.2", "--port", "9",
                "--log-level", "debug", "--no-warmup",
            ], as: ServeCommand.self)
        let settings = try ServerSettings(environment: serve.applied(to: environment))
        #expect(settings.backend == "verdict")
        #expect(settings.host == "127.0.0.2")
        #expect(settings.port == 9)
        #expect(settings.logLevel == .debug)
        #expect(!settings.warmup)
        #expect(settings.maxQueue == 7)

        let plain = try ServerSettings(
            environment: try parse(["serve"], as: ServeCommand.self).applied(to: environment))
        #expect(plain.backend == "mlx" && plain.host == "10.0.0.1" && plain.port == 1234)
        #expect(plain.logLevel == .error && plain.warmup)

        let decide = try parse(["decide", "--backend", "laya"], as: DecideCommand.self)
        #expect(decide.backend.applied(to: environment)["OPENJEV_BACKEND"] == "laya")
        let models = try parse(["models", "--backend", "verdict"], as: ModelsCommand.self)
        #expect(models.backend.applied(to: environment)["OPENJEV_BACKEND"] == "verdict")
        #expect(
            try parse(["models"], as: ModelsCommand.self).backend.applied(to: environment)
                == environment)
    }

    @Test("decide takes --backend and --request")
    func decideFlags() throws {
        let decide = try parse(
            ["decide", "--backend", "verdict", "--request", "body.json"], as: DecideCommand.self)
        #expect(decide.backend.backend == "verdict")
        #expect(decide.requestFile == "body.json")
        let stdin = try parse(["decide"], as: DecideCommand.self)
        #expect(stdin.backend.backend == nil && stdin.requestFile == nil)
        #expect(try parse(["decide", "--request", "-"], as: DecideCommand.self).requestFile == "-")
    }

    @Test("models takes --backend")
    func modelsFlags() throws {
        #expect(
            try parse(["models", "--backend", "laya"], as: ModelsCommand.self).backend.backend
                == "laya")
        #expect(try parse(["models"], as: ModelsCommand.self).backend.backend == nil)
    }

    @Test("A command line the parser refuses exits 2 with its message and usage")
    func refusedCommandLines() async {
        let cases: [([String], String)] = [
            (["serve", "--shutdown-timeout=-1"], "--shutdown-timeout must be 0 or more seconds"),
            (["serve", "--shutdown-timeout", "-1"], "Missing value for '--shutdown-timeout"),
            (["serve", "--shutdown-timeout", "soon"], "--shutdown-timeout"),
            (["serve", "--prot", "9"], "Unknown option '--prot'"),
            (["decide", "--request"], "Missing value for '--request <file>'"),
            (["models", "extra"], "Unexpected argument 'extra'"),
            (["nope"], "Unexpected argument 'nope'"),
            (["serve", "--no-warmup=yes"], "--no-warmup"),
        ]
        for (arguments, message) in cases {
            let outcome = await CommandHarness.run(arguments)
            #expect(outcome.status == 2, "\(arguments)")
            #expect(outcome.errors.contains(message), "\(arguments): \(outcome.errors)")
            #expect(outcome.errors.contains("Usage: openjev"), "\(arguments): \(outcome.errors)")
            #expect(outcome.standardOutput.isEmpty, "\(arguments)")
        }
    }

    @Test("Every flag's help names the variable it overrides")
    func helpNamesTheVariables() async {
        let serve = await CommandHarness.run(["serve", "--help"])
        #expect(serve.status == 0)
        for (flag, variable) in [
            ("--backend", "OPENJEV_BACKEND"), ("--host", "OPENJEV_HOST"),
            ("--port", "OPENJEV_PORT"), ("--log-level", "OPENJEV_LOG_LEVEL"),
            ("--no-warmup", "OPENJEV_WARMUP=0"),
        ] {
            #expect(serve.output.contains(flag), "\(flag)")
            #expect(serve.output.contains(variable), "\(variable)")
        }
        #expect(serve.output.contains("--shutdown-timeout"))
        for subcommand in ["decide", "models"] {
            let help = await CommandHarness.run([subcommand, "--help"])
            #expect(help.status == 0)
            #expect(help.output.contains("--backend") && help.output.contains("OPENJEV_BACKEND"))
        }
        #expect(await CommandHarness.run(["decide", "--help"]).output.contains("--request"))
    }

    @Test("The root prints its help, its subcommands and the version")
    func root() async {
        let help = await CommandHarness.run([])
        #expect(help.status == 0)
        for subcommand in ["serve", "decide", "models"] {
            #expect(help.output.contains(subcommand))
        }
        #expect(help.output.contains("Exit statuses"))
        let version = await CommandHarness.run(["--version"])
        #expect(version.status == 0)
        #expect(version.output == "0.1.0-dev\n")
    }
}
