// A port of upstream OpenJev (razorback16/openjev at dcd2094), the `systemone` route of
// `openjev/api.py` once FastAPI has read the body: the shape, the model name, the questions cap,
// the engine and the mapping of its errors. The route forwards a routed model's request between
// the shape and the model name (`ModelRouter`). Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Logging
    import OpenJevCore

    /// `POST /v1/systemone` from the read body to the bytes of the answer, in upstream's order: the
    /// body's shape, the model name, the questions cap, then the service.
    ///
    /// The route and `openjev decide` both answer through it, so the command prints the bytes the
    /// server sends. Each refusal upstream logs is logged as `RefusalLog` describes. Only the
    /// route forwards a routed model's request (`ModelRouter`); `openjev decide` answers with the
    /// loaded model alone, so a model it does not serve is the unknown-model 400 there.
    public struct SystemOneHandler: Sendable {
        /// The settings: the questions cap and the body cap.
        public let settings: ServerSettings
        /// The loaded service.
        public let service: any SystemOneService

        /// Creates a handler over a loaded service.
        public init(settings: ServerSettings, service: any SystemOneService) {
            self.settings = settings
            self.service = service
        }

        /// The body of the 200 the server answers when `body` is sent as JSON, which is how
        /// `openjev decide` sends its request: the cap the body middleware applies, the reading
        /// `json.loads` gives a JSON body, then the shape, the model name, the questions cap and
        /// the service. Nothing is logged.
        ///
        /// - Throws: The ``/OpenJevCore/WireError`` the server answers with: the 413 of a body over
        ///   `OPENJEV_MAX_BODY_BYTES`, the 400 and 422s of a body that is not JSON, the shape's
        ///   refusals and every error of the service; and ``/OpenJevCore/JSONWriteError`` for an
        ///   answer the server could not write either, its plain-text 500.
        public func respond(toJSONBody body: [UInt8]) async throws -> [UInt8] {
            let log = RefusalLog(logger: Self.silent, requestID: "")
            guard body.count <= settings.maxBodyBytes else {
                throw WireError.bodyTooLarge413(limit: settings.maxBodyBytes)
            }
            let value: JSONValue?
            do {
                value =
                    body.isEmpty
                    ? nil : try RequestBodyReader(maxBodyBytes: settings.maxBodyBytes).parse(body)
            } catch {
                throw log.validation(error)
            }
            return try await respond(to: value, log: log, watch: nil)
        }

        /// The body of the 200 for a read body (`nil` when there was none), deciding while the
        /// client of `watch` is there: a client that goes away cancels the decision.
        ///
        /// - Throws: A ``WireError`` for every refusal, which the headers middleware renders: the
        ///   shape's 422 or 400 (``validated(_:log:)``), then every error of the overload that
        ///   takes the validated request.
        func respond(
            to body: JSONValue?, log: RefusalLog, watch: ConnectionWatch?
        ) async throws -> [UInt8] {
            try await respond(to: validated(body, log: log), log: log, watch: watch)
        }

        /// The request in a read body (`nil` when there was none), as pydantic validates
        /// `SystemOneRequest` before the route runs. The route forwards a request for a routed
        /// model once its body has passed this, and before the model name is checked.
        ///
        /// - Throws: The shape's 422, or the 400 `Invalid request.` for an unknown question type,
        ///   each logged as upstream logs it.
        func validated(_ body: JSONValue?, log: RefusalLog) throws(WireError) -> SystemOneRequest {
            do {
                return try RequestValidator().validate(body)
            } catch let error where error == .invalidRequest {
                log.invalidRequest(error, problems: RequestValidator().problems(body))
                throw error
            } catch {
                throw log.validation(error)
            }
        }

        /// The body of the 200 for a validated request, in upstream's order after the shape: the
        /// model name, the questions cap, then the service, deciding while the client of `watch`
        /// is there.
        ///
        /// - Throws: A ``WireError`` for every refusal, which the headers middleware renders: the
        ///   unknown model, the questions cap, and for the service's errors a ``SchemaError`` as
        ///   the plain-detail 400, an ``OverloadedError`` as the 529, a ``BackendRefusal`` as the
        ///   400 `the model rejected this request` and anything else as the 503 naming its type;
        ///   ``ClientDisconnected`` when the client went away first; and ``JSONWriteError`` for an
        ///   answer that cannot be written.
        func respond(
            to wireRequest: SystemOneRequest, log: RefusalLog, watch: ConnectionWatch?
        ) async throws -> [UInt8] {
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
            let decision: Decision
            do {
                decision = try await ClientConnection.cancellingOnDisconnect(watch) {
                    try await service.decide(wireRequest)
                }
            } catch let error as ClientDisconnected {
                throw error
            } catch {
                throw answer(for: error, log: log)
            }
            return try WireEncoder().bytes(
                SystemOneResponse(
                    model: served.version, answers: decision.answers,
                    usage: Usage(
                        inputTokens: decision.inputTokens, outputTokens: decision.outputTokens)))
        }

        /// The most characters of a backend's refusal the 400 repeats, as upstream's
        /// `Upstream(str(msg)[:500])` keeps.
        static let refusalCharacters = 500

        /// A logger that writes nothing, for `openjev decide`, which prints the error body itself.
        static let silent = Logger(label: "openjev.silent") { _ in SwiftLogNoOpLogHandler() }

        /// The answer for an error of the service.
        private func answer(for error: any Error, log: RefusalLog) -> WireError {
            switch error {
            case let error as SchemaError:
                let refusal = WireError.semantic400(error)
                log.semantic(refusal, at: error.loc)
                return refusal
            case let error as OverloadedError:
                return WireError.overloaded529(error.message)
            case let error as BackendRefusal:
                let reason = error.reason.unicodeScalars.prefix(Self.refusalCharacters)
                let refusal = WireError.modelRejected400(
                    String(String.UnicodeScalarView(reason)))
                log.semantic(refusal, at: ["body"])
                return refusal
            default:
                // Upstream names the httpx error its vLLM backend raised: type(e).__name__. A
                // decision the server cancelled while stopping is a CancellationError here.
                let failure = WireError.backendUnavailable503(String(describing: type(of: error)))
                log.failure(failure)
                return failure
            }
        }
    }
#endif
