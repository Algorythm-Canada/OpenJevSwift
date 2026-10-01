#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import Logging
    import NIOCore
    import OpenJevCore
    import OpenJevTestSupport
    import Testing

    /// A stand-in for the OpenJev server a model is routed to: a Hummingbird application on an
    /// ephemeral port of 127.0.0.1 whose `POST /v1/systemone` does what its ``Behaviour`` says
    /// and records every request it receives, as upstream's tests stand in for it with httpx's
    /// `MockTransport`.
    final class RoutedTarget: Sendable {
        /// What the target does with a request.
        enum Behaviour: Sendable {
            /// Answers with the status, the header fields in order and the body, after `delay`.
            case answer(status: Int, headers: [Header], body: [UInt8], delay: Duration = .zero)
            /// Reads the request and never answers, until the target stops.
            case silent
        }

        /// One header field of an answer.
        struct Header: Sendable {
            var name: String
            var value: String
        }

        /// A request as the target received it.
        struct Received: Sendable {
            /// The path, with any query.
            var path: String
            /// The `host` header, which swift-http-types keeps apart from the other fields.
            var authority: String?
            /// Every header field but `host`, in the order received.
            var headers: HTTPFields
            /// The body's bytes.
            var body: [UInt8]

            /// The header names, lowercased, in the order received.
            var headerNames: [String] {
                headers.map { $0.name.canonicalName }
            }

            /// The value of the first field named `name`.
            func header(_ name: String) -> String? {
                HTTPField.Name(name).flatMap { headers[values: $0].first }
            }
        }

        /// The port the target listens on.
        let port: Int
        private let recorder: Recorder

        private init(port: Int, recorder: Recorder) {
            self.port = port
            self.recorder = recorder
        }

        /// `http://127.0.0.1:{port}`, the URL a route names.
        var url: String { "http://127.0.0.1:\(port)" }

        /// The requests received so far, in order.
        var received: [Received] { recorder.requests }

        /// The answer upstream's recording stubbed for a row whose `stub` holds a `route`, as
        /// httpx builds it: `json` rendered compactly with `application/json`, `text` with
        /// `text/plain; charset=utf-8`, then the stub's own headers; `raise: ReadTimeout` is
        /// ``Behaviour/silent``, which the client's read timeout ends.
        ///
        /// - Throws: ``FixtureError`` for a stub this map does not know, so that a new kind of
        ///   recording cannot pass by being ignored.
        static func behaviour(stubbing route: JSONValue) throws -> Behaviour {
            if let raised = route["raise"]?.stringValue {
                guard raised == "ReadTimeout" else {
                    throw FixtureError("a route stub raising \(raised) has no stand-in here")
                }
                return .silent
            }
            let status = try #require(route["status"]?.intValue, "route stub: status")
            var headers: [Header] = []
            let body: [UInt8]
            if let json = route["json"] {
                body = try WireEncoder().bytes(json: json)
                headers.append(Header(name: "content-type", value: "application/json"))
            } else if let text = route["text"]?.stringValue {
                body = Array(text.utf8)
                headers.append(Header(name: "content-type", value: "text/plain; charset=utf-8"))
            } else {
                throw FixtureError("a route stub without json or text has no stand-in here")
            }
            for (name, value) in route["headers"]?.objectValue ?? [:] {
                headers.append(
                    Header(name: name, value: try #require(value.stringValue, "header \(name)")))
            }
            return .answer(status: status, headers: headers, body: body)
        }

        /// Runs `body` with a target that behaves as `behaviour` says, and stops the target
        /// afterwards, cancelling a request it still holds.
        static func run<Value: Sendable>(
            _ behaviour: Behaviour, _ body: @Sendable (RoutedTarget) async throws -> Value
        ) async throws -> Value {
            let recorder = Recorder()
            // The answer's fields, built here so that a bad name fails the test, not the handler.
            var fields = HTTPFields()
            if case .answer(_, let headers, _, _) = behaviour {
                for header in headers {
                    let name = try #require(HTTPField.Name(header.name), "\(header.name)")
                    fields.append(HTTPField(name: name, value: header.value))
                }
            }
            let answerFields = fields
            let router = Router()
            router.post("/v1/systemone") { request, _ -> Response in
                let bytes = try await request.body.collect(upTo: .max)
                recorder.append(
                    Received(
                        path: request.uri.description, authority: request.head.authority,
                        headers: request.headers, body: Array(bytes.readableBytesView)))
                switch behaviour {
                case .silent:
                    try await Task.sleep(for: .seconds(3600))
                    return Response(status: .noContent)
                case .answer(let status, _, let body, let delay):
                    if delay > .zero {
                        try await Task.sleep(for: delay)
                    }
                    return Response(
                        status: HTTPResponse.Status(code: status), headers: answerFields,
                        body: ResponseBody(byteBuffer: ByteBuffer(bytes: body)))
                }
            }
            let (ports, portContinuation) = AsyncStream.makeStream(of: Int.self)
            var logger = Logger(label: "RoutedTarget")
            logger.logLevel = .critical
            let application = Application(
                router: router,
                configuration: ApplicationConfiguration(address: .hostname("127.0.0.1", port: 0)),
                onServerRunning: { channel in
                    portContinuation.yield(channel.localAddress?.port ?? 0)
                },
                logger: logger)
            return try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    defer { portContinuation.finish() }
                    try await application.run()
                }
                defer { group.cancelAll() }
                var iterator = ports.makeAsyncIterator()
                guard let port = await iterator.next(), port > 0 else {
                    throw HarnessError("the routed target did not listen")
                }
                return try await body(RoutedTarget(port: port, recorder: recorder))
            }
        }

        /// The requests a target received, shared with its route handler.
        private final class Recorder: @unchecked Sendable {
            // Guarded by `lock`.
            private let lock = NSLock()
            private var list: [Received] = []

            var requests: [Received] {
                lock.withLock { list }
            }

            func append(_ request: Received) {
                lock.withLock { list.append(request) }
            }
        }
    }
#endif
