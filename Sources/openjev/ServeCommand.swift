// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/__main__.py`:
// `uvicorn.run(create_app(), host=OPENJEV_HOST, port=OPENJEV_PORT, log_level=OPENJEV_LOG_LEVEL)`,
// with the backend choice of `create_app` and the graceful shutdown uvicorn gives SIGINT and
// SIGTERM. Apache-2.0. See THIRD_PARTY.md.

import ArgumentParser
import Foundation
import Logging
import OpenJevCore
import OpenJevServer
import ServiceLifecycle

/// `openjev serve`: load the model and serve the OpenJev API until SIGINT or SIGTERM.
struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Load the model and serve the OpenJev API until SIGINT or SIGTERM.",
        discussion: """
            The settings are the OPENJEV_* variables upstream reads, with upstream's defaults \
            and checks; each flag overrides its variable. The phases, then one line per request \
            (method, path, status, milliseconds, request id), go to standard error. On SIGINT or \
            SIGTERM the server stops accepting, lets the requests in flight finish within \
            --shutdown-timeout, releases the model and exits 0; requests still in flight then \
            are cancelled and the exit status is 1. docs/deployment.md has the variables, a \
            launchd job and the exit statuses.
            """)

    @OptionGroup var backend: BackendOption

    @Option(
        help: ArgumentHelp("The address to bind. Overrides OPENJEV_HOST.", valueName: "address"))
    var host: String?

    @Option(
        help: ArgumentHelp(
            "The port to bind; 0 picks a free one. Overrides OPENJEV_PORT.", valueName: "port"))
    var port: String?

    @Option(
        name: .customLong("log-level"),
        help: ArgumentHelp(
            "trace, debug, info, notice, warning, error or critical. Overrides OPENJEV_LOG_LEVEL.",
            valueName: "level"))
    var logLevel: String?

    @Flag(
        name: .customLong("no-warmup"),
        help: "Skip the warm-up read before serving. Sets OPENJEV_WARMUP=0.")
    var noWarmup = false

    @Option(
        name: .customLong("shutdown-timeout"),
        help: ArgumentHelp(
            "Seconds the requests in flight get to finish after SIGINT or SIGTERM, 0 to 86400.",
            valueName: "seconds"))
    var shutdownTimeout: Double = 30

    /// The longest `--shutdown-timeout`, a day.
    static let longestShutdownTimeout: Double = 86_400

    func validate() throws {
        guard shutdownTimeout >= 0, shutdownTimeout <= Self.longestShutdownTimeout else {
            throw ValidationError(
                "--shutdown-timeout must be between 0 and "
                    + "\(SettingsSummary.number(Self.longestShutdownTimeout)) seconds")
        }
    }

    /// `environment` with each flag that was given over its variable.
    func applied(to environment: [String: String]) -> [String: String] {
        var environment = backend.applied(to: environment)
        let flags = [
            ("OPENJEV_HOST", host), ("OPENJEV_PORT", port), ("OPENJEV_LOG_LEVEL", logLevel),
        ]
        for (name, value) in flags {
            if let value {
                environment[name] = value
            }
        }
        if noWarmup {
            environment["OPENJEV_WARMUP"] = "0"
        }
        return environment
    }

    func run() async throws {
        let context = CommandContext.current
        let environment = applied(to: context.environment)
        let settings = try CommandFailure.settings(environment)
        let selected = try context.backends.backend(named: settings.backend)
        _ = try selected.provider(environment: environment)
        let logger = context.logger(level: Logger.Level(settings.logLevel))
        logger.info(
            "\(SettingsSummary.line(settings, environment: environment, kind: selected.kind))")
        for (name, url) in settings.modelRoutesWithoutCredentials {
            logger.info("forwarding \(name) to \(url)")
        }
        logger.info("loading \(selected.modelName) (OPENJEV_BACKEND=\(selected.name))")
        let service = try await selected.load(
            settings: settings, environment: environment,
            willWarmUp: { logger.info("warming up") })
        if !settings.warmup {
            logger.info("warm-up skipped (OPENJEV_WARMUP=0)")
        }

        let running = RunningGroup()
        let server = DecisionServer(
            settings: settings, service: service, logger: logger,
            onServerRunning: { port in
                logger.info("serving on \(settings.host):\(port)")
                await context.onServing(port) { await running.triggerGracefulShutdown() }
            })
        var configuration = ServiceGroupConfiguration(
            services: [server], gracefulShutdownSignals: context.shutdownSignals, logger: logger)
        configuration.maximumGracefulShutdownDuration = .seconds(shutdownTimeout)
        let group = ServiceGroup(configuration: configuration)
        running.set(group)
        do {
            try await group.run()
        } catch is ShutdownInterrupted {
            throw CommandFailure(
                .failure,
                message: "the requests in flight did not finish within the shutdown timeout "
                    + "(\(SettingsSummary.number(shutdownTimeout)) s) and were cancelled")
        } catch {
            throw CommandFailure(.failure, message: "the server stopped: \(error)")
        }
        logger.info("stopped")
    }
}

/// The service group of a running `serve`, for the shutdown function handed to
/// ``CommandContext/onServing``, which exists before the group does.
final class RunningGroup: @unchecked Sendable {
    // Guarded by `lock`.
    private let lock = NSLock()
    private var group: ServiceGroup?

    /// Keeps the group.
    func set(_ group: ServiceGroup) {
        lock.withLock { self.group = group }
    }

    /// Starts the group's graceful shutdown, as SIGTERM does.
    func triggerGracefulShutdown() async {
        let group = lock.withLock { self.group }
        await group?.triggerGracefulShutdown()
    }
}

/// The settings line `serve` writes first: what the server will do, never the API key or the
/// origin secret, and the model routes by name. A line per route follows it, with the route's URL
/// without its credentials.
enum SettingsSummary {
    /// `settings: host=... port=...`, the settings that apply to the backend's kind.
    static func line(
        _ settings: ServerSettings, environment: [String: String],
        kind: BackendRegistry.Backend.Kind
    ) -> String {
        var parts = [
            "host=\(settings.host)", "port=\(settings.port)", "backend=\(settings.backend)",
            "log_level=\(settings.logLevel.rawValue)", "warmup=\(settings.warmup ? "on" : "off")",
            "max_queue=\(settings.maxQueue)", "max_questions=\(settings.maxQuestions)",
            "max_body_bytes=\(settings.maxBodyBytes)",
        ]
        switch kind {
        case .diffusion:
            parts += [
                "max_inflight=\(settings.maxInflight)", "canvas=\(settings.canvas)",
                "mlx_model=\(settings.mlxModel)",
                "mlx_cache_limit_gb=\(settings.mlxCacheLimitGB.map(number) ?? "unset")",
                "mlx_prompt_cache=\(settings.mlxPromptCache)",
                "mlx_max_prompt=\(settings.mlxMaxPrompt)",
                "auto_threshold=\(number(settings.autoThreshold))", "auto_max=\(settings.autoMax)",
            ]
        case .encoder:
            parts.append("encoder_batch=\(settings.encoderBatch)")
            parts.append("encoder_functions=\(settings.encoderFunctions.map(String.init) ?? "all")")
            let local = environment["OPENJEV_ENCODER_MODELS"].flatMap { $0.isEmpty ? nil : $0 }
            parts.append("encoder_models=\(local ?? "downloads")")
        }
        parts += [
            "api_key=\(settings.apiKey.isEmpty ? "unset" : "set")",
            "origin_secret=\(settings.originSecret.isEmpty ? "unset" : "set")",
            "model_routes="
                + (settings.modelRoutes.isEmpty
                    ? "none" : settings.modelRoutes.keys.joined(separator: ",")),
        ]
        return "settings: " + parts.joined(separator: " ")
    }

    /// A number without a trailing `.0`: `30`, `2.5`.
    static func number(_ value: Double) -> String {
        value.rounded() == value && abs(value) < 1e15 ? String(Int(value)) : String(value)
    }
}

extension Logger.Level {
    /// The swift-log level of an `OPENJEV_LOG_LEVEL` name.
    init(_ level: ServerSettings.LogLevel) {
        switch level {
        case .trace: self = .trace
        case .debug: self = .debug
        case .info: self = .info
        case .notice: self = .notice
        case .warning: self = .warning
        case .error: self = .error
        case .critical: self = .critical
        }
    }
}
