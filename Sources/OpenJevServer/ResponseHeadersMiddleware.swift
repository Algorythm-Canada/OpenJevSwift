// A port of upstream OpenJev (razorback16/openjev at dcd2094), the `request_id_and_auth`
// middleware of `openjev/api.py`, whose authentication and body cap are
// `AuthenticationMiddleware` and `BodyCapMiddleware`. The `model_ns` context variable of
// `openjev/engine.py` is the core's `ModelTimeRecorder`. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import NIOCore
    import OpenJevCore

    /// The request context of the OpenJev routes: Hummingbird's core storage, the request id
    /// the headers middleware assigns and the connection the request came on.
    public struct OpenJevRequestContext: RequestContext {
        /// Hummingbird's per-request storage.
        public var coreContext: CoreRequestContextStorage
        /// `req_` and 32 lowercase hex characters, upstream's `request.state.request_id`. Empty
        /// until ``ResponseHeadersMiddleware`` runs.
        public var requestID: String
        /// The connection's channel, through which the route learns that a client has gone away.
        public let channel: any Channel

        /// Creates a context for a request.
        public init(source: ApplicationRequestContextSource) {
            self.coreContext = CoreRequestContextStorage(source: source)
            self.requestID = ""
            self.channel = source.channel
        }
    }

    /// Adds upstream's `server-timing`, `x-typesafe-request-id` and `x-request-id` headers to
    /// every response, error responses included, and turns a thrown error into its response.
    ///
    /// The `model` time is what the engine adds to the request's ``ModelTimeRecorder`` as each
    /// backend call ends, refused and failed requests included; the `server` time is the rest of
    /// the wall time, never below zero; `total` is the wall time. Each is in milliseconds with one
    /// decimal, as upstream's `{:.1f}` writes them. The time of a request forwarded to another
    /// server (`OPENJEV_MODEL_ROUTES`) will count as model time when issue #38 forwards it, as
    /// upstream's `forward` adds it.
    struct ResponseHeadersMiddleware: RouterMiddleware {
        typealias Context = OpenJevRequestContext

        func handle(
            _ request: Request, context: Context,
            next: (Request, Context) async throws -> Response
        ) async throws -> Response {
            var context = context
            let requestID = RequestIdentifier.make()
            context.requestID = requestID
            let recorder = ModelTimeRecorder()
            let clock = ContinuousClock()
            let started = clock.now
            var response: Response
            do {
                response = try await ModelTimeRecorder.$current.withValue(recorder) {
                    try await next(request, context)
                }
            } catch {
                response = WireResponses.response(for: error, context: context)
            }
            let total = clock.now - started
            response.headers[HeaderName.serverTiming] = ServerTiming.header(
                model: recorder.total, total: total)
            response.headers[HeaderName.typesafeRequestID] = requestID
            response.headers[HeaderName.requestID] = requestID
            return response
        }
    }

    /// The header names the server sets beyond Hummingbird's.
    enum HeaderName {
        static let serverTiming = named("server-timing")
        static let typesafeRequestID = named("x-typesafe-request-id")
        static let requestID = named("x-request-id")

        /// A header name from a lowercase token.
        static func named(_ token: String) -> HTTPField.Name {
            guard let name = HTTPField.Name(token) else {
                preconditionFailure("\(token) is not a valid header name")
            }
            return name
        }
    }

    /// Upstream's request id, `"req_" + secrets.token_hex(16)`.
    enum RequestIdentifier {
        /// A fresh id: `req_` and 32 lowercase hex characters from the system's random source.
        static func make() -> String {
            var generator = SystemRandomNumberGenerator()
            return "req_" + hex(generator.next()) + hex(generator.next())
        }

        /// 16 lowercase hex digits, zero-padded.
        private static func hex(_ value: UInt64) -> String {
            let digits = String(value, radix: 16)
            return String(repeating: "0", count: 16 - digits.count) + digits
        }
    }

    /// The `server-timing` value.
    enum ServerTiming {
        /// `model;dur=A, server;dur=B, total;dur=C`, with `B = max(0, C - A)`.
        static func header(model: Duration, total: Duration) -> String {
            let modelMS = milliseconds(model)
            let totalMS = milliseconds(total)
            // Not max(0, x): Swift's max returns -0.0 for (0.0, -0.0), which prints as -0.0.
            let difference = totalMS - modelMS
            let serverMS = difference > 0 ? difference : 0
            return "model;dur=\(format(modelMS)), server;dur=\(format(serverMS)), "
                + "total;dur=\(format(totalMS))"
        }

        /// The duration in milliseconds.
        static func milliseconds(_ duration: Duration) -> Double {
            let (seconds, attoseconds) = duration.components
            return Double(seconds) * 1e3 + Double(attoseconds) / 1e15
        }

        /// One decimal, rounded from the binary value as Python's `{:.1f}` rounds it.
        static func format(_ value: Double) -> String {
            String(format: "%.1f", value)
        }
    }
#endif
