// A port of upstream OpenJev (razorback16/openjev at dcd2094), `create_app` in `openjev/api.py`:
// the `/health`, `/v1/models` and `/v1/systemone` routes, and the host and port
// `openjev/__main__.py` binds. Authentication (#36), the full error contract (#35), capacity and
// shutdown (#37) and model routes (#38) come with their own issues. Apache-2.0. See
// THIRD_PARTY.md.

#if canImport(Hummingbird)
    import HTTPTypes
    import Hummingbird
    import OpenJevCore

    /// Builds the OpenJev HTTP application.
    public enum OpenJevApplication {
        /// Loads the service through the provider, as upstream's `lifespan` loads the engine
        /// before serving, and returns the application bound to the settings' host and port.
        public static func make(
            settings: ServerSettings, provider: some BackendProvider
        ) async throws -> Application<RouterResponder<OpenJevRequestContext>> {
            let service = try await provider.makeService(settings: settings)
            return Application(
                router: router(settings: settings, service: service),
                configuration: ApplicationConfiguration(
                    address: .hostname(settings.host, port: settings.port)))
        }

        /// The routes over a loaded service, with the headers middleware in front of them.
        public static func router(
            settings: ServerSettings, service: any SystemOneService
        ) -> Router<OpenJevRequestContext> {
            let router = Router(context: OpenJevRequestContext.self)
            router.add(middleware: ResponseHeadersMiddleware())
            let routes = Routes(settings: settings, service: service)
            router.get("/health") { _, _ in try routes.health() }
            router.get("/v1/models") { _, _ in try routes.models() }
            router.post("/v1/systemone") { request, _ in try await routes.systemOne(request) }
            return router
        }
    }

    /// The route handlers.
    struct Routes: Sendable {
        let settings: ServerSettings
        let service: any SystemOneService

        /// `GET /health`: `{"status":"ok"}`.
        func health() throws -> Response {
            try WireResponses.ok(Health())
        }

        /// `GET /v1/models`: the service's listing.
        func models() throws -> Response {
            try WireResponses.ok(ModelsResponse(models: service.servedModels.listing))
        }

        /// `POST /v1/systemone`, in upstream's order: the body, its shape, the model name, the
        /// questions cap, then the engine. The engine's model time goes to `server-timing`.
        ///
        /// - Throws: A ``WireError`` for every refusal, which the headers middleware renders; the
        ///   backend's own errors, which it renders as a 500.
        func systemOne(_ request: Request) async throws -> Response {
            let body = try await parsedBody(request)
            let wireRequest = try RequestValidator().validate(body)
            let served = service.servedModels
            guard served.accepts(wireRequest.model) else {
                throw WireError.unknownModel(wireRequest.model)
            }
            // A request's questions fan out into reads; the cap bounds one body's work.
            if wireRequest.questions.count > settings.maxQuestions {
                throw WireError.semantic400(
                    "at most \(settings.maxQuestions) questions per request")
            }
            let decision: Decision
            do {
                decision = try await service.decide(wireRequest)
            } catch let error as SchemaError {
                throw WireError.semantic400(error)
            } catch let error as OverloadedError {
                throw WireError.overloaded529(error.message)
            }
            ModelTimeRecorder.record(decision.modelTime)
            return try WireResponses.ok(
                SystemOneResponse(
                    model: served.version, answers: decision.answers,
                    usage: Usage(
                        inputTokens: decision.inputTokens, outputTokens: decision.outputTokens)))
        }

        /// The body, read up to `OPENJEV_MAX_BODY_BYTES` and parsed, or `nil` when it is empty.
        ///
        /// - Throws: The 413 past the cap; FastAPI's `json_invalid` 422 for a body that is not
        ///   JSON, whose `ctx.error` is the parser's description rather than Python's `json`
        ///   message (issue #35); and whatever reading the body throws otherwise.
        private func parsedBody(_ request: Request) async throws -> JSONValue? {
            let bytes: ByteBuffer
            do {
                bytes = try await request.body.collect(upTo: settings.maxBodyBytes)
            } catch let error as any HTTPResponseError where error.status == .contentTooLarge {
                throw WireError.bodyTooLarge413(limit: settings.maxBodyBytes)
            }
            if bytes.readableBytes == 0 {
                return nil
            }
            let parser = JSONParser(
                options: JSONParser.Options(maximumBytes: settings.maxBodyBytes))
            do {
                return try parser.parse(bytes.readableBytesView)
            } catch {
                throw WireError.validation422([
                    ValidationErrorItem(
                        type: "json_invalid", loc: ["body", .index(error.offset)],
                        msg: "JSON decode error", input: [:],
                        ctx: ["error": .string(error.description)])
                ])
            }
        }
    }

    /// The `GET /health` body.
    private struct Health: WireEncodable {
        var json: JSONValue { ["status": "ok"] }
    }
#endif
