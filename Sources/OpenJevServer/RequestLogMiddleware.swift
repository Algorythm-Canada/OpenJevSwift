// The server's request log (issue #40), in the place of the access log uvicorn writes for upstream
// OpenJev (razorback16/openjev at dcd2094): one line per request, without the client's address or
// the query string uvicorn includes. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import Logging

    /// Logs one line per request at info level, once its response is ready:
    /// `{method} {path} {status} {milliseconds}ms {request id}`, for example
    /// `POST /v1/systemone 200 41.8ms req_0123456789abcdef0123456789abcdef`.
    ///
    /// The path is the request's path without its query string. A body, a header value and a
    /// query string are never written. The request id is the one ``ResponseHeadersMiddleware``
    /// gave the response, and the time runs from when the request reached the router to when
    /// its response was ready. A client that went away before its answer shows 499. It runs
    /// first, outside every other middleware, so every request gets its line, refusals included.
    struct RequestLogMiddleware: RouterMiddleware {
        typealias Context = OpenJevRequestContext

        func handle(
            _ request: Request, context: Context,
            next: (Request, Context) async throws -> Response
        ) async throws -> Response {
            let clock = ContinuousClock()
            let started = clock.now
            do {
                let response = try await next(request, context)
                log(
                    request, status: Int(response.status.code), response: response,
                    elapsed: clock.now - started, context: context)
                return response
            } catch {
                // The headers middleware answers every error itself; this is a safety net.
                log(
                    request, status: 500, response: nil, elapsed: clock.now - started,
                    context: context)
                throw error
            }
        }

        private func log(
            _ request: Request, status: Int, response: Response?, elapsed: Duration,
            context: Context
        ) {
            var logger = context.logger
            // Hummingbird's own request id would only confuse the line, which carries upstream's.
            logger[metadataKey: "hb.request.id"] = nil
            let id = response?.headers[HeaderName.requestID]
            let text = Self.line(
                method: request.method.rawValue, path: request.uri.path, status: status,
                elapsed: elapsed, requestID: id)
            logger.info("\(text)")
        }

        /// The text of a line. The milliseconds have one decimal, as `server-timing` writes them,
        /// and a missing request id is `-`.
        static func line(
            method: String, path: String, status: Int, elapsed: Duration, requestID: String?
        ) -> String {
            let milliseconds = ServerTiming.format(ServerTiming.milliseconds(elapsed))
            return "\(method) \(path) \(status) \(milliseconds)ms \(requestID ?? "-")"
        }
    }
#endif
