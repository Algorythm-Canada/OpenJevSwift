// A port of upstream OpenJev (razorback16/openjev at dcd2094), `create_app` in `openjev/api.py`:
// the `/health`, `/v1/models` and `/v1/systemone` routes, their middleware and error answers, and
// the host and port `openjev/__main__.py` binds. Capacity and shutdown (#37) and model routes
// (#38) come with their own issues. Apache-2.0. See THIRD_PARTY.md.

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

        /// The routes over a loaded service, behind upstream's `request_id_and_auth` in its
        /// order: the headers middleware, then authentication and the body cap for `/v1/`.
        public static func router(
            settings: ServerSettings, service: any SystemOneService
        ) -> Router<OpenJevRequestContext> {
            let router = Router(context: OpenJevRequestContext.self)
            router.add(middleware: ResponseHeadersMiddleware())
            router.add(
                middleware: AuthenticationMiddleware(
                    originSecret: settings.originSecret, apiKey: settings.apiKey))
            router.add(middleware: BodyCapMiddleware(limit: settings.maxBodyBytes))
            let routes = Routes(settings: settings, service: service)
            router.get("/health") { _, _ in try routes.health() }
            router.get("/v1/models") { _, _ in try routes.models() }
            router.post("/v1/systemone") { request, context in
                try await routes.systemOne(request, context: context)
            }
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
        /// questions cap, then the engine. The engine's model time goes to `server-timing`, and
        /// each refusal upstream logs is logged as ``RefusalLog`` describes.
        ///
        /// - Throws: A ``WireError`` for every refusal, which the headers middleware renders:
        ///   the body's 400 and 422s (``RequestBodyReader``), the shape's 422 or 400, the unknown
        ///   model, the questions cap, and for the engine's errors a ``SchemaError`` as the
        ///   plain-detail 400, an ``OverloadedError`` as the 529, a ``BackendRefusal`` as the 400
        ///   `the model rejected this request` and anything else as the 503 naming its type.
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
            let wireRequest: SystemOneRequest
            do {
                wireRequest = try RequestValidator().validate(body)
            } catch let error where error == .invalidRequest {
                log.invalidRequest(error, problems: RequestValidator().problems(body))
                throw error
            } catch {
                throw log.validation(error)
            }
            let served = service.servedModels
            guard served.accepts(wireRequest.model) else {
                throw WireError.unknownModel(wireRequest.model)
            }
            // A request's questions fan out into reads; the cap bounds one body's work.
            if wireRequest.questions.count > settings.maxQuestions {
                let refusal = WireError.semantic400(
                    "at most \(settings.maxQuestions) questions per request")
                log.semantic(refusal, at: ["body", "questions"])
                throw refusal
            }
            let decision = try await decide(wireRequest, log: log)
            ModelTimeRecorder.record(decision.modelTime)
            return try WireResponses.ok(
                SystemOneResponse(
                    model: served.version, answers: decision.answers,
                    usage: Usage(
                        inputTokens: decision.inputTokens, outputTokens: decision.outputTokens)))
        }

        /// The most characters of a backend's refusal the 400 repeats, as upstream's
        /// `Upstream(str(msg)[:500])` keeps.
        static let refusalCharacters = 500

        /// The service's decision, or the answer for its error.
        private func decide(_ request: SystemOneRequest, log: RefusalLog) async throws -> Decision {
            do {
                return try await service.decide(request)
            } catch let error as SchemaError {
                let refusal = WireError.semantic400(error)
                log.semantic(refusal, at: error.loc)
                throw refusal
            } catch let error as OverloadedError {
                throw WireError.overloaded529(error.message)
            } catch let error as BackendRefusal {
                let reason = error.reason.unicodeScalars.prefix(Self.refusalCharacters)
                let refusal = WireError.modelRejected400(
                    String(String.UnicodeScalarView(reason)))
                log.semantic(refusal, at: ["body"])
                throw refusal
            } catch {
                // Upstream names the httpx error its vLLM backend raised: type(e).__name__.
                let failure = WireError.backendUnavailable503(String(describing: type(of: error)))
                log.failure(failure)
                throw failure
            }
        }
    }

    /// The `GET /health` body.
    private struct Health: WireEncodable {
        var json: JSONValue { ["status": "ok"] }
    }
#endif
