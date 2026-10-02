// A port of upstream OpenJev (razorback16/openjev at dcd2094), `create_app` in `openjev/api.py`:
// the `/health`, `/v1/models` and `/v1/systemone` routes, their middleware and error answers, the
// model routes, and the host and port `openjev/__main__.py` binds. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import HTTPTypes
    import Hummingbird
    import HummingbirdCore
    import Logging
    import NIOCore
    import OpenJevCore

    /// Builds the OpenJev HTTP application.
    public enum OpenJevApplication {
        /// The routes over a loaded service, behind upstream's `request_id_and_auth` in its
        /// order: the request log, the headers middleware, then authentication and the body cap
        /// for `/v1/`. Requests are not watched for clients that go away;
        /// ``application(settings:service:logger:onServerRunning:)`` builds a server that watches
        /// them.
        public static func router(
            settings: ServerSettings, service: any SystemOneService
        ) -> Router<OpenJevRequestContext> {
            router(settings: settings, service: service, connections: nil)
        }

        /// The routes, cancelling the decision of a client that goes away when its connection is
        /// in `connections`.
        static func router(
            settings: ServerSettings, service: any SystemOneService,
            connections: ConnectionRegistry?
        ) -> Router<OpenJevRequestContext> {
            let router = Router(context: OpenJevRequestContext.self)
            router.add(middleware: RequestLogMiddleware())
            router.add(middleware: ResponseHeadersMiddleware())
            router.add(
                middleware: AuthenticationMiddleware(
                    originSecret: settings.originSecret, apiKey: settings.apiKey))
            router.add(middleware: BodyCapMiddleware(limit: settings.maxBodyBytes))
            let routes = Routes(
                settings: settings, handler: SystemOneHandler(settings: settings, service: service),
                router: ModelRouter(settings: settings), connections: connections)
            router.get("/health") { _, _ in try routes.health() }
            router.get("/v1/models") { _, _ in try routes.models() }
            router.post("/v1/systemone") { request, context in
                try await routes.systemOne(request, context: context)
            }
            return router
        }

        /// The application over a loaded service, bound to the settings' host and port, as
        /// upstream's `lifespan` loads the engine before uvicorn binds. Every connection is
        /// watched, so a client that goes away cancels its decision. `onServerRunning` gets the
        /// port the server listens on, which is the one bound when the settings' port is 0.
        ///
        /// ``DecisionServer`` runs it with a graceful shutdown and releases the model after.
        public static func application(
            settings: ServerSettings, service: any SystemOneService, logger: Logger,
            onServerRunning: @escaping @Sendable (_ port: Int) async -> Void = { _ in }
        ) -> Application<RouterResponder<OpenJevRequestContext>> {
            application(
                settings: settings, service: service, logger: logger,
                connections: ConnectionRegistry(), inFlight: RequestsInFlight(),
                onServerRunning: onServerRunning)
        }

        /// ``application(settings:service:logger:onServerRunning:)`` with the registry its
        /// connections join and the record of the requests a cancellation cuts short.
        static func application(
            settings: ServerSettings, service: any SystemOneService, logger: Logger,
            connections: ConnectionRegistry, inFlight: RequestsInFlight,
            onServerRunning: @escaping @Sendable (_ port: Int) async -> Void
        ) -> Application<RouterResponder<OpenJevRequestContext>> {
            Application(
                router: router(settings: settings, service: service, connections: connections),
                server: server(connections: connections, inFlight: inFlight),
                configuration: ApplicationConfiguration(
                    address: .hostname(settings.host, port: settings.port)),
                onServerRunning: { channel in
                    await onServerRunning(channel.localAddress?.port ?? settings.port)
                },
                logger: logger)
        }

        /// The HTTP/1 server: every connection is watched for its client going away and joins
        /// `connections`, and a request that a cancellation reaches before the connection has
        /// taken its whole answer is noted in `inFlight`.
        static func server(
            connections: ConnectionRegistry, inFlight: RequestsInFlight
        ) -> HTTPServerBuilder {
            HTTPServerBuilder { responder in
                HTTP1Channel(
                    responder: inFlight.noting(responder),
                    configuration: HTTP1Channel.Configuration(
                        additionalChannelHandlers: [
                            ClientDisconnectHandler(registry: connections)
                        ]))
            }
        }
    }

    /// The route handlers.
    struct Routes: Sendable {
        let settings: ServerSettings
        let handler: SystemOneHandler
        /// The model routes, `OPENJEV_MODEL_ROUTES`.
        let router: ModelRouter
        /// The server's connections, or `nil` when requests are not watched.
        let connections: ConnectionRegistry?

        /// `GET /health`: `{"status":"ok"}`.
        func health() throws -> Response {
            try WireResponses.ok(Health())
        }

        /// `GET /v1/models`: the service's listing, then each routed model this server does not
        /// serve, upstream's `models_list`. The routed servers are not asked.
        func models() throws -> Response {
            let served = handler.service.servedModels
            return try WireResponses.ok(
                ModelsResponse(models: served.listing(routedNames: router.routes.keys)))
        }

        /// `POST /v1/systemone`: the body as FastAPI reads it and its shape, then either the
        /// routed server's answer for a model routed elsewhere (``ModelRouter``), or
        /// ``SystemOneHandler``'s, while the client is there.
        ///
        /// - Throws: A ``WireError`` for every refusal, which the headers middleware renders: the
        ///   body's 400 and 422s (``RequestBodyReader``), the shape's, the 503 naming a
        ///   ``ForwardingFailure`` for a routed server that did not answer, and every error of
        ///   ``SystemOneHandler``; ``ClientDisconnected`` when the client went away before the
        ///   answer was ready.
        func systemOne(
            _ request: Request, context: OpenJevRequestContext
        ) async throws -> Response {
            let log = RefusalLog(context: context)
            let reader = RequestBodyReader(maxBodyBytes: settings.maxBodyBytes)
            let bytes = try await reader.bytes(of: request)
            let body: JSONValue?
            do {
                body = try reader.value(of: bytes, headers: request.headers)
            } catch {
                throw log.validation(error)
            }
            let wireRequest = try handler.validated(body, log: log)
            let watch = connections?.watch(for: context.channel)
            // upstream: a model routed to another container and not served here is passed through
            if let url = router.url(
                for: wireRequest.model, servedModels: handler.service.servedModels)
            {
                return try await forward(
                    bytes, request: request, to: url, model: wireRequest.model, log: log,
                    watch: watch)
            }
            let answer = try await handler.respond(to: wireRequest, log: log, watch: watch)
            return WireResponses.json(status: .ok, bytes: answer)
        }

        /// The routed server's answer for a request, while the client is there: a client that
        /// goes away cancels the exchange.
        ///
        /// - Throws: The 503 naming the ``ForwardingFailure``, or the type of another error, for
        ///   an exchange that failed, logged at error level with the routed model's name, as a
        ///   backend failure is (D-031); ``ClientDisconnected`` when the client went away first.
        private func forward(
            _ bytes: ByteBuffer, request: Request, to url: String, model: String,
            log: RefusalLog, watch: ConnectionWatch?
        ) async throws -> Response {
            let router = router
            let headers = request.headers
            do {
                return try await ClientConnection.cancellingOnDisconnect(watch) {
                    try await router.forward(bytes, headers: headers, to: url)
                }
            } catch let error as ClientDisconnected {
                throw error
            } catch {
                // upstream: type(e).__name__ of httpx's error; a request the server cancelled
                // while stopping is a CancellationError, as a decision is.
                let name =
                    (error as? ForwardingFailure)?.name ?? String(describing: type(of: error))
                let failure = WireError.backendUnavailable503(name)
                log.failure(failure, forwarding: model)
                throw failure
            }
        }
    }

    /// The `GET /health` body.
    private struct Health: WireEncodable {
        var json: JSONValue { ["status": "ok"] }
    }
#endif
