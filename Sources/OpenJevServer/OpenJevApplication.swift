// A port of upstream OpenJev (razorback16/openjev at dcd2094), `create_app` in `openjev/api.py`:
// the `/health`, `/v1/models` and `/v1/systemone` routes, their middleware and error answers, and
// the host and port `openjev/__main__.py` binds. Model routes (#38) come with their own issue.
// Apache-2.0. See THIRD_PARTY.md.

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
                connections: connections)
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
        /// connections join and the count of its requests in flight.
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
        /// `connections`, and every request counts in `inFlight` until its response is written.
        static func server(
            connections: ConnectionRegistry, inFlight: RequestsInFlight
        ) -> HTTPServerBuilder {
            HTTPServerBuilder { responder in
                HTTP1Channel(
                    responder: inFlight.counting(responder),
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
        /// The server's connections, or `nil` when requests are not watched.
        let connections: ConnectionRegistry?

        /// `GET /health`: `{"status":"ok"}`.
        func health() throws -> Response {
            try WireResponses.ok(Health())
        }

        /// `GET /v1/models`: the service's listing.
        func models() throws -> Response {
            try WireResponses.ok(ModelsResponse(models: handler.service.servedModels.listing))
        }

        /// `POST /v1/systemone`: the body as FastAPI reads it, then ``SystemOneHandler``, while
        /// the client is there.
        ///
        /// - Throws: A ``WireError`` for every refusal, which the headers middleware renders: the
        ///   body's 400 and 422s (``RequestBodyReader``) and every error of
        ///   ``SystemOneHandler/respond(to:log:watch:)``; ``ClientDisconnected`` when the client
        ///   went away before the answer was ready.
        func systemOne(
            _ request: Request, context: OpenJevRequestContext
        ) async throws -> Response {
            let log = RefusalLog(context: context)
            let body: JSONValue?
            do {
                body = try await RequestBodyReader(maxBodyBytes: settings.maxBodyBytes)
                    .value(of: request)
            } catch let error as WireError {
                throw log.validation(error)
            }
            let bytes = try await handler.respond(
                to: body, log: log, watch: connections?.watch(for: context.channel))
            return WireResponses.json(status: .ok, bytes: bytes)
        }
    }

    /// The `GET /health` body.
    private struct Health: WireEncodable {
        var json: JSONValue { ["status": "ok"] }
    }
#endif
