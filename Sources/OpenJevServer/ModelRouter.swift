// A port of upstream OpenJev (razorback16/openjev at dcd2094), `forward` and the routed-model
// branch of the `systemone` route in `openjev/api.py` (OPENJEV_MODEL_ROUTES), with the
// `httpx.AsyncClient(timeout=httpx.Timeout(settings.forward_timeout, connect=5.0))` its `lifespan`
// creates. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import AsyncHTTPClient
    import Foundation
    import HTTPTypes
    import Hummingbird
    import NIOCore
    import NIOHTTP1
    import NIOPosix
    import OpenJevCore

    /// Upstream's model routes, `OPENJEV_MODEL_ROUTES`: a request for a model another OpenJev
    /// server serves is passed to that server, and its answer comes back unchanged, so one
    /// origin serves every model.
    ///
    /// A request is forwarded when its body has passed validation, its `model` has a route and
    /// this server does not accept that name, upstream's `req.model in settings.model_routes and
    /// req.model not in model_names`. It goes to `{url}/v1/systemone` as the bytes the client
    /// sent, with only the client's `authorization`, `x-origin-secret` and `content-type`
    /// headers, the first field of each, as Starlette reads them. A URL with credentials sends
    /// them as HTTP Basic authorization in place of the client's, as httpx does.
    ///
    /// The routed server's status and body come back unchanged, with only its `content-type`
    /// and `retry-after` (several fields joined with `, `, as httpx joins them). This server adds
    /// its own request ids and `server-timing`, whose `model` counts the whole exchange, network
    /// included, as upstream's `forward` adds it to `model_ns`.
    ///
    /// Each forwarded request has its own client, made and shut down inside the request, on
    /// swift-nio's shared event loops; nothing outlives the request (decision D-039). The client
    /// waits 5 seconds for a connection, `OPENJEV_FORWARD_TIMEOUT` for each write and read, never
    /// follows a redirect, speaks HTTP/1.1, sends no `accept-encoding` and decodes no body, so
    /// the bytes that come back are the bytes the routed server sent.
    struct ModelRouter: Sendable {
        /// The routes: a model name, then the URL of the server that serves it, without a
        /// trailing slash.
        let routes: OrderedMap<String>
        /// Seconds the client waits for each write and read, `OPENJEV_FORWARD_TIMEOUT`.
        let forwardTimeout: Double

        /// The router of the settings' `OPENJEV_MODEL_ROUTES` and `OPENJEV_FORWARD_TIMEOUT`.
        init(settings: ServerSettings) {
            self.routes = settings.modelRoutes
            self.forwardTimeout = settings.forwardTimeout
        }

        /// The headers a forwarded request carries, upstream's `FORWARD_HEADERS`.
        static let forwardedHeaders: [HTTPField.Name] = [
            .authorization, HeaderName.named("x-origin-secret"), .contentType,
        ]

        /// The headers of the routed server's answer that come back.
        static let keptHeaders: [HTTPField.Name] = [.contentType, HeaderName.named("retry-after")]

        /// How long the client waits for a connection, upstream's `connect=5.0`.
        static let connectTimeout = TimeAmount.seconds(5)

        /// The URL of the server `model` is routed to, or `nil` when this server accepts the name
        /// or no route names it: a request for a model this server serves is answered here, even
        /// when a route names it too.
        func url(for model: String, servedModels: ServedModels) -> String? {
            guard !servedModels.accepts(model) else { return nil }
            return routes[model]
        }

        /// Forwards a request to `{url}/v1/systemone` and returns the routed server's answer, as
        /// upstream's `forward` does. The time from sending the request to reading the whole
        /// answer is added to the request's ``ModelTimeRecorder``, also when the exchange fails.
        ///
        /// - Throws: A ``ForwardingFailure`` when the routed server could not be reached or did not
        ///   answer, which the route answers with the 503 naming it; `CancellationError` when the
        ///   request was cancelled; and any other error of the client as it is.
        func forward(
            _ body: ByteBuffer, headers: HTTPFields, to url: String
        ) async throws -> Response {
            let target = url + "/v1/systemone"
            try Self.check(target)
            var request = HTTPClientRequest(url: target)
            request.method = .POST
            for name in Self.forwardedHeaders {
                if let value = headers[values: name].first {
                    request.headers.add(name: name.canonicalName, value: value)
                }
            }
            if let credentials = Self.basicAuthorization(target) {
                request.headers.replaceOrAdd(name: "authorization", value: credentials)
            }
            request.body = .bytes(body)

            let client = HTTPClient(
                eventLoopGroup: MultiThreadedEventLoopGroup.singleton,
                configuration: configuration)
            let clock = ContinuousClock()
            let started = clock.now
            let outcome: Result<Response, any Error>
            do {
                let response = try await client.execute(request, deadline: .distantFuture)
                let content = try await response.body.collect(upTo: .max)
                outcome = .success(Self.answer(response, body: content))
            } catch {
                outcome = .failure(error)
            }
            // upstream: the other container's time, network included, is the model's time here
            ModelTimeRecorder.record(clock.now - started)
            // The shutdown is not interrupted by cancellation, so the client never outlives the
            // request; nothing is in flight on it any more.
            try? await client.shutdown()
            switch outcome {
            case .success(let response):
                return response
            case .failure(let error):
                throw Self.failure(for: error)
            }
        }

        /// The client's configuration: upstream's timeouts, no redirects, HTTP/1.1, a connection
        /// that is not retried, and no body decoding.
        var configuration: HTTPClient.Configuration {
            let readTimeout = Self.timeAmount(seconds: forwardTimeout)
            var timeout = HTTPClient.Configuration.Timeout(
                connect: Self.connectTimeout, read: readTimeout)
            timeout.write = readTimeout
            var configuration = HTTPClient.Configuration(
                redirectConfiguration: .disallow, timeout: timeout, decompression: .disabled)
            configuration.httpVersion = .http1Only
            // A refused connection is the 503 at once, as httpx's ConnectError is, rather than
            // retried until the connect timeout.
            configuration.connectionPool.retryConnectionEstablishment = false
            return configuration
        }

        /// The answer of the routed server: its status and body, and its `content-type` and
        /// `retry-after` alone.
        static func answer(_ response: HTTPClientResponse, body: ByteBuffer) -> Response {
            var fields = HTTPFields()
            for name in keptHeaders {
                let values = response.headers[name.canonicalName]
                if !values.isEmpty {
                    fields[name] = values.joined(separator: ", ")
                }
            }
            return Response(
                status: HTTPResponse.Status(
                    code: Int(response.status.code),
                    reasonPhrase: response.status.reasonPhrase),
                headers: fields, body: ResponseBody(byteBuffer: body))
        }

        /// Refuses a URL httpx would refuse before sending: one Foundation cannot parse is
        /// `InvalidURL`, and one without an `http` or `https` scheme and a host is
        /// `UnsupportedProtocol`.
        static func check(_ target: String) throws(ForwardingFailure) {
            guard let parsed = URL(string: target) else {
                throw .invalidURL
            }
            guard let scheme = parsed.scheme?.lowercased(), ["http", "https"].contains(scheme),
                let host = parsed.host, !host.isEmpty
            else {
                throw .unsupportedProtocol
            }
        }

        /// `Basic` and the base64 of `user:password`, percent-decoded, for a URL that holds
        /// credentials, as httpx authenticates such a URL; `nil` for one without.
        static func basicAuthorization(_ target: String) -> String? {
            guard let components = URLComponents(string: target),
                components.user != nil || components.password != nil
            else {
                return nil
            }
            let pair = (components.user ?? "") + ":" + (components.password ?? "")
            return "Basic " + Data(pair.utf8).base64EncodedString()
        }

        /// `seconds` as a timeout, or `nil`, no timeout, for one too long to express: Python's
        /// `float` lets `OPENJEV_FORWARD_TIMEOUT` be `inf`.
        static func timeAmount(seconds: Double) -> TimeAmount? {
            // 9e9 seconds is 285 years, which still fits in TimeAmount's Int64 of nanoseconds.
            guard seconds < 9e9 else { return nil }
            return .nanoseconds(Int64(seconds * 1e9))
        }

        /// The error the route answers for a failed exchange: cancellation as
        /// `CancellationError`, a transport failure as the ``ForwardingFailure`` httpx would
        /// raise for it, and anything else as it is.
        static func failure(for error: any Error) -> any Error {
            if error is CancellationError || (error as? HTTPClientError) == .cancelled {
                return CancellationError()
            }
            return ForwardingFailure(error) ?? error
        }
    }

    /// Why a forwarded request got no answer, named after the httpx exception upstream's 503
    /// names in its place (`type(e).__name__`): the route answers it with
    /// `inference backend unavailable: {name}` and `retry-after: 2`.
    enum ForwardingFailure: String, Error, Sendable, CaseIterable {
        /// The connection was refused, or the name did not resolve.
        case connectError = "ConnectError"
        /// No connection within 5 seconds.
        case connectTimeout = "ConnectTimeout"
        /// The routed server sent nothing for `OPENJEV_FORWARD_TIMEOUT` seconds.
        case readTimeout = "ReadTimeout"
        /// The request could not be written for `OPENJEV_FORWARD_TIMEOUT` seconds.
        case writeTimeout = "WriteTimeout"
        /// Reading the answer failed, such as a connection reset.
        case readError = "ReadError"
        /// Writing the request failed.
        case writeError = "WriteError"
        /// The routed server closed the connection without an answer, or sent one that is not
        /// HTTP.
        case remoteProtocolError = "RemoteProtocolError"
        /// The route's URL has no `http` or `https` scheme, or no host.
        case unsupportedProtocol = "UnsupportedProtocol"
        /// The route's URL cannot be parsed.
        case invalidURL = "InvalidURL"

        /// The name the 503 gives.
        var name: String { rawValue }

        /// The failure an error of the HTTP client or of swift-nio is, or `nil` for another
        /// error.
        init?(_ error: any Error) {
            switch error {
            case let failure as ForwardingFailure:
                self = failure
            case let error as HTTPClientError:
                guard let failure = Self.failure(error) else { return nil }
                self = failure
            case is NIOConnectionError:
                self = .connectError
            case let error as ChannelError:
                switch error {
                case .connectTimeout:
                    self = .connectTimeout
                case .eof, .ioOnClosedChannel, .alreadyClosed, .outputClosed, .inputClosed:
                    self = .remoteProtocolError
                default:
                    return nil
                }
            case let error as IOError:
                self = Self.failure(error)
            case is HTTPParserError:
                self = .remoteProtocolError
            default:
                // A TLS handshake that fails is httpx's ConnectError; the TLS errors are
                // swift-nio-ssl's types, which this target does not import.
                guard String(reflecting: type(of: error)).hasPrefix("NIOSSL.") else { return nil }
                self = .connectError
            }
        }

        /// The failure an `HTTPClientError` is, or `nil` for one that is not a transport
        /// failure.
        private static func failure(_ error: HTTPClientError) -> ForwardingFailure? {
            switch error {
            case .connectTimeout, .tlsHandshakeTimeout, .httpProxyHandshakeTimeout,
                .socksHandshakeTimeout:
                return .connectTimeout
            case .readTimeout, .deadlineExceeded:
                return .readTimeout
            case .writeTimeout:
                return .writeTimeout
            case .remoteConnectionClosed, .uncleanShutdown, .bodyLengthMismatch:
                return .remoteProtocolError
            case .invalidURL:
                return .invalidURL
            case .emptyScheme, .emptyHost:
                return .unsupportedProtocol
            default:
                // unsupportedScheme(_:) carries the scheme, so it cannot be matched as a value.
                return error.shortDescription == "Unsupported scheme" ? .unsupportedProtocol : nil
            }
        }

        /// The failure a socket error is: a connection that could not be made is httpx's
        /// ConnectError, and any other failure is a read's or a write's, by the system call.
        private static func failure(_ error: IOError) -> ForwardingFailure {
            let connecting: Set<Int32> = [
                ECONNREFUSED, ENETUNREACH, EHOSTUNREACH, ETIMEDOUT, EADDRNOTAVAIL, ENETDOWN,
            ]
            if connecting.contains(error.errnoCode) {
                return .connectError
            }
            return String(describing: error).contains("write") ? .writeError : .readError
        }
    }
#endif
