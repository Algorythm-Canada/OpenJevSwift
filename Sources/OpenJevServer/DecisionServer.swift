// A port of upstream OpenJev (razorback16/openjev at dcd2094), the serving half of
// `openjev/__main__.py` (`uvicorn.run`) and the end of `lifespan` in `openjev/api.py`, which awaits
// the engine's `close()` once uvicorn has stopped. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Hummingbird
    import Logging
    import OpenJevCore
    import ServiceLifecycle

    /// The OpenJev server over a loaded service, run as a swift-service-lifecycle `Service`: it
    /// serves until a graceful shutdown, lets the requests in flight finish, then releases the
    /// service's model (``ModelReleasing``).
    ///
    /// On a graceful shutdown the listening socket closes at once, so a new connection is
    /// refused; an idle connection is closed, and one with a request in flight is closed once its
    /// answer has been written. When the service group that runs the server cancels it, which a
    /// group does when the shutdown takes longer than its `maximumGracefulShutdownDuration`, the
    /// requests still in flight are cancelled and ``run()`` throws ``ShutdownInterrupted``
    /// after releasing the model; cancelled with no request in flight, it stops as cleanly as
    /// after a graceful shutdown. The model is released however the server ends.
    ///
    /// ```swift
    /// var configuration = ServiceGroupConfiguration(
    ///     services: [DecisionServer(settings: settings, service: service, logger: logger)],
    ///     gracefulShutdownSignals: [.sigterm, .sigint], logger: logger)
    /// configuration.maximumGracefulShutdownDuration = .seconds(30)
    /// try await ServiceGroup(configuration: configuration).run()
    /// ```
    public struct DecisionServer: Service, CustomStringConvertible {
        /// The settings: the host, the port and everything the routes read.
        public let settings: ServerSettings
        /// The loaded service.
        public let service: any SystemOneService
        /// The logger of the server and of every request.
        public let logger: Logger
        private let onServerRunning: @Sendable (_ port: Int) async -> Void

        /// Creates the server. `onServerRunning` gets the port the server listens on once it
        /// does, which is the one bound when the settings' port is 0.
        public init(
            settings: ServerSettings, service: any SystemOneService, logger: Logger,
            onServerRunning: @escaping @Sendable (_ port: Int) async -> Void = { _ in }
        ) {
            self.settings = settings
            self.service = service
            self.logger = logger
            self.onServerRunning = onServerRunning
        }

        /// The name the service group logs.
        public var description: String { "OpenJev" }

        /// Serves until a graceful shutdown or a cancellation, then releases the model.
        ///
        /// - Throws: ``ShutdownInterrupted`` when the server was cancelled while it was answering
        ///   requests, which the cancellation cut short; and the error that stopped the server
        ///   otherwise, such as an address already in use.
        public func run() async throws {
            let requests = RequestsInFlight()
            let application = OpenJevApplication.application(
                settings: settings, service: service, logger: logger,
                connections: ConnectionRegistry(), inFlight: requests,
                onServerRunning: onServerRunning)
            let logger = logger
            do {
                try await withTaskCancellationHandler {
                    try await withGracefulShutdownHandler {
                        try await application.run()
                    } onGracefulShutdown: {
                        logger.info(
                            "shutting down: no new connections, finishing the requests in flight")
                    }
                } onCancel: {
                    requests.serverCancelled()
                }
            } catch {
                await release()
                if requests.cutShort {
                    throw ShutdownInterrupted()
                }
                // Cancelled with nothing in flight: nothing was cut short.
                if Task.isCancelled {
                    return
                }
                throw error
            }
            await release()
            if requests.cutShort {
                throw ShutdownInterrupted()
            }
        }

        /// Releases the service's model, when the service adopts ``ModelReleasing``.
        private func release() async {
            guard let releasing = service as? any ModelReleasing else { return }
            await releasing.close()
            logger.info("released \(service.servedModels.version)")
        }
    }

    /// The server was cancelled while it was answering requests, which the cancellation cut short:
    /// a service group cancels its services when a graceful shutdown takes longer than its
    /// `maximumGracefulShutdownDuration`.
    public struct ShutdownInterrupted: Error, Sendable, Hashable, CustomStringConvertible {
        /// Creates the error.
        public init() {}

        /// What happened.
        public var description: String {
            "the server was cancelled while requests were in flight"
        }
    }
#endif
