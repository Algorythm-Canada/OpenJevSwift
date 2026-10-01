// A port of upstream OpenJev (razorback16/openjev at dcd2094), `log_invalid` in `openjev/api.py`
// and its calls from `invalid_body` and `semantic_error`. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Hummingbird
    import Logging
    import OpenJevCore

    /// Upstream's `log_invalid`: one warning for each refused request, naming where it was wrong
    /// and why, never its body, state or instructions, so that common client mistakes are
    /// visible without keeping anyone's data.
    ///
    /// The line is `{status} {request_id} {problems}`, the problems joined by `; `: `loc: type`
    /// for a validation error and `loc: reason` for a plain-detail 400, the `loc` components
    /// joined by `.`. Only the refusals upstream logs are logged: the 422s, the invalid-request
    /// 400 and the plain-detail 400s. Authentication, the body cap, a body FastAPI cannot read,
    /// an unknown model and a full queue are not. A backend failure, which upstream does not log
    /// either, is logged at error level with the message of its 503, since the request log
    /// (``RequestLogMiddleware``) shows only its status, and so is a routed server that did not
    /// answer a forwarded request.
    struct RefusalLog: Sendable {
        /// Where the lines go.
        let logger: Logger
        /// The request's id, `req_` and 32 hex characters, or empty outside a request.
        let requestID: String

        /// The log of one request: its logger and its id.
        init(context: OpenJevRequestContext) {
            self.init(logger: context.logger, requestID: context.requestID)
        }

        /// A log writing to `logger` for the request `requestID`.
        init(logger: Logger, requestID: String) {
            self.logger = logger
            self.requestID = requestID
        }

        /// Logs a 422's problems, and returns the error to throw. Any other error is returned
        /// without a log line.
        func validation(_ error: WireError) -> WireError {
            if case .validation(let items) = error.body {
                warning(
                    status: error.status,
                    items.map { ValidationProblem(loc: $0.loc, type: $0.type).description })
            }
            return error
        }

        /// Logs the invalid-request 400 with every problem of its body, the unknown question
        /// types included.
        func invalidRequest(_ error: WireError, problems: [ValidationProblem]) {
            warning(status: error.status, problems.map(\.description))
        }

        /// Logs a plain-detail 400 with the location of its problem, as `semantic_error` does.
        func semantic(_ error: WireError, at loc: [LocComponent]) {
            guard case .plain(let body) = error.body else { return }
            warning(status: error.status, [SchemaError(body.detail, loc: loc).description])
        }

        /// Logs a backend failure at error level, with the message of its answer.
        func failure(_ error: WireError) {
            guard case .typed(let body) = error.body else { return }
            let text = Self.line(status: error.status, requestID: requestID, [body.message])
            logger.error("\(text)")
        }

        /// Logs a forwarded request that got no answer at error level, with the message of its
        /// answer and the routed model's name, never the route's URL: upstream does not log it,
        /// and only the 503's status would show otherwise.
        func failure(_ error: WireError, forwarding model: String) {
            guard case .typed(let body) = error.body else { return }
            let text = Self.line(
                status: error.status, requestID: requestID,
                ["\(body.message) (forwarding \(model))"])
            logger.error("\(text)")
        }

        private func warning(status: Int, _ problems: [String]) {
            let text = Self.line(status: status, requestID: requestID, problems)
            logger.warning("\(text)")
        }

        /// The text of a line: the status, the request id (`-` when there is none) and the
        /// problems, or `invalid request` when there are none.
        static func line(status: Int, requestID: String, _ problems: [String]) -> String {
            let id = requestID.isEmpty ? "-" : requestID
            let text = problems.isEmpty ? "invalid request" : problems.joined(separator: "; ")
            return "\(status) \(id) \(text)"
        }
    }
#endif
