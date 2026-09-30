#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import Logging
    import NIOEmbedded
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// Builds the OpenJev application over stub backends and sends it requests through
    /// Hummingbird's in-process test client, or straight to its responder.
    enum ServerHarness {
        /// A DiffusionGemma engine over ``StubBackend`` and the fixture tokenizer, configured
        /// from the settings, as ``DecisionBackendProvider`` builds it.
        static func diffusionService(
            _ settings: ServerSettings, backend: StubBackend = StubBackend()
        ) async throws -> any SystemOneService {
            try await DecisionBackendProvider { _ in backend }.makeService(settings: settings)
        }

        /// An encoder engine over ``StubQuestionReadBackend``, without the warm-up read unless
        /// the settings ask for it.
        static func encoderService(
            _ backend: StubQuestionReadBackend = StubQuestionReadBackend(),
            settings: ServerSettings? = nil
        ) async throws -> any SystemOneService {
            try await QuestionReadBackendProvider { _ in backend }
                .makeService(settings: settings ?? ServerSettings(warmup: false))
        }

        /// Runs `body` with a test client of the application over `service`; `nil` settings are
        /// upstream's defaults. Request loggers derive from `logger` when one is given.
        static func withClient<Value: Sendable>(
            settings: ServerSettings? = nil,
            service: any SystemOneService,
            logger: Logger? = nil,
            _ body: @Sendable (any TestClientProtocol) async throws -> Value
        ) async throws -> Value {
            let app = Application(
                router: OpenJevApplication.router(
                    settings: try settings ?? ServerSettings(), service: service),
                logger: logger)
            return try await app.test(.router, body)
        }

        /// A request with string headers and a body. The test client sets `Content-Length`
        /// whenever there is a body.
        static func send(
            _ client: any TestClientProtocol, _ method: HTTPRequest.Method, _ path: String,
            headers: [String: String] = [:], body: [UInt8]? = nil
        ) async throws -> TestResponse {
            try await client.execute(
                uri: path, method: method, headers: try fields(headers),
                body: body.map { ByteBuffer(bytes: $0) })
        }

        /// A `POST /v1/systemone` of a wire request, rendered as upstream's fixtures render it.
        static func post(
            _ client: any TestClientProtocol, _ request: JSONValue
        ) async throws -> TestResponse {
            try await send(
                client, .post, "/v1/systemone", headers: ["content-type": "application/json"],
                body: try WireEncoder().bytes(json: request))
        }

        /// Sends a request built by hand straight to the application's responder, as the router
        /// test client does, but with exactly the headers given and the body as a stream: no
        /// `Content-Length` is added.
        static func respond<Body: AsyncSequence & Sendable>(
            settings: ServerSettings, service: any SystemOneService,
            method: HTTPRequest.Method, path: String, headers: [String: String] = [:],
            body: Body, logger: Logger? = nil
        ) async throws -> DirectResponse
        where Body.Element == ByteBuffer, Body.AsyncIterator: SendableMetatype {
            let responder = OpenJevApplication.router(settings: settings, service: service)
                .buildResponder()
            let request = Request(
                head: HTTPRequest(
                    method: method, scheme: "http", authority: "localhost", path: path,
                    headerFields: try fields(headers)),
                body: RequestBody(asyncSequence: body))
            let context = OpenJevRequestContext(
                source: ApplicationRequestContextSource(
                    channel: NIOAsyncTestingChannel(),
                    logger: logger ?? Logger(label: "OpenJevServerTests")))
            let response = try await responder.respond(to: request, context: context)
            let collector = BodyCollector()
            try await response.body.write(CollectingWriter(collector: collector))
            return DirectResponse(
                status: response.status, headers: response.headers, body: collector.buffer)
        }

        /// Header fields from string pairs, in name order.
        static func fields(_ headers: [String: String]) throws -> HTTPFields {
            var fields = HTTPFields()
            for (name, value) in headers.sorted(by: { $0.key < $1.key }) {
                fields.append(HTTPField(name: try #require(HTTPField.Name(name)), value: value))
            }
            return fields
        }

        /// A response header by name.
        static func header(_ response: some CheckedResponse, _ name: String) -> String? {
            HTTPField.Name(name).flatMap { response.headers[$0] }
        }

        /// The body as text.
        static func text(_ response: some CheckedResponse) -> String {
            String(buffer: response.body)
        }

        /// Checks upstream's per-response headers: equal request ids of the recorded format and a
        /// `server-timing` value with its three parts.
        static func expectServerHeaders(
            _ response: some CheckedResponse, _ label: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let id = header(response, "x-request-id")
            #expect(
                id.map(isRequestID) == true, "\(label): x-request-id \(String(describing: id))",
                sourceLocation: sourceLocation)
            #expect(
                header(response, "x-typesafe-request-id") == id,
                "\(label): x-typesafe-request-id", sourceLocation: sourceLocation)
            let timing = header(response, "server-timing")
            #expect(
                timing.flatMap(ServerTimingValue.init) != nil,
                "\(label): server-timing \(String(describing: timing))",
                sourceLocation: sourceLocation)
        }

        /// Checks status, body bytes, `content-type`, `retry-after` (present exactly when it was
        /// recorded) and the per-response headers against a recorded response.
        static func expectRecorded(
            _ response: some CheckedResponse, _ recorded: JSONValue, _ label: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            let expected = try #require(recorded["response"], "\(label): no response")
            #expect(
                Int(response.status.code) == expected["status"]?.intValue, "\(label): status",
                sourceLocation: sourceLocation)
            #expect(
                text(response) == expected["body_text"]?.stringValue, "\(label): body",
                sourceLocation: sourceLocation)
            for name in ["content-type", "retry-after"] {
                #expect(
                    header(response, name) == expected["headers"]?[name]?.stringValue,
                    "\(label): \(name)", sourceLocation: sourceLocation)
            }
            expectServerHeaders(response, label, sourceLocation: sourceLocation)
        }

        /// A recorded request's headers.
        static func headers(_ value: JSONValue?) -> [String: String] {
            var headers: [String: String] = [:]
            for (name, value) in value?.objectValue ?? [:] {
                headers[name] = value.stringValue
            }
            return headers
        }

        /// `req_` followed by 32 lowercase hex characters.
        static func isRequestID(_ text: String) -> Bool {
            guard text.hasPrefix("req_") else { return false }
            let hex = text.dropFirst(4)
            return hex.count == 32 && hex.allSatisfy { "0123456789abcdef".contains($0) }
        }
    }

    /// What the checks read from a response, whichever way it was sent.
    protocol CheckedResponse {
        var status: HTTPResponse.Status { get }
        var headers: HTTPFields { get }
        var body: ByteBuffer { get }
    }

    extension TestResponse: CheckedResponse {}

    /// A response from ``ServerHarness/respond(settings:service:method:path:headers:body:logger:)``.
    struct DirectResponse: CheckedResponse {
        var status: HTTPResponse.Status
        var headers: HTTPFields
        var body: ByteBuffer
    }

    /// Collects the bytes a response body writes.
    final class BodyCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var collected = ByteBuffer()

        /// Everything written so far.
        var buffer: ByteBuffer {
            lock.withLock { collected }
        }

        func append(_ buffer: ByteBuffer) {
            var buffer = buffer
            lock.withLock { _ = collected.writeBuffer(&buffer) }
        }
    }

    /// A response body writer into a ``BodyCollector``.
    struct CollectingWriter: ResponseBodyWriter {
        let collector: BodyCollector

        mutating func write(_ buffer: ByteBuffer) async throws {
            collector.append(buffer)
        }

        consuming func finish(_ trailingHeaders: HTTPFields?) async throws {}
    }

    /// A request body delivered in chunks, counting how many were read.
    struct ChunkedBody: AsyncSequence, Sendable {
        typealias Element = ByteBuffer

        let chunks: [ByteBuffer]
        let reads = ReadCounter()

        /// The bytes in chunks of `size`.
        init(_ bytes: [UInt8], chunkSize size: Int) {
            chunks = stride(from: 0, to: bytes.count, by: size).map { start in
                ByteBuffer(bytes: bytes[start..<Swift.min(start + size, bytes.count)])
            }
        }

        struct AsyncIterator: AsyncIteratorProtocol {
            let chunks: [ByteBuffer]
            let reads: ReadCounter
            var index = 0

            mutating func next() async throws -> ByteBuffer? {
                guard index < chunks.count else { return nil }
                reads.increment()
                defer { index += 1 }
                return chunks[index]
            }
        }

        func makeAsyncIterator() -> AsyncIterator {
            AsyncIterator(chunks: chunks, reads: reads)
        }
    }

    /// A request body that must never be read: reading it records a test issue.
    struct UnreadableBody: AsyncSequence, Sendable {
        typealias Element = ByteBuffer

        let reads = ReadCounter()

        struct AsyncIterator: AsyncIteratorProtocol {
            let reads: ReadCounter

            mutating func next() async throws -> ByteBuffer? {
                reads.increment()
                Issue.record("the request body was read")
                return nil
            }
        }

        func makeAsyncIterator() -> AsyncIterator {
            AsyncIterator(reads: reads)
        }
    }

    /// A count of reads, shared between a body and the test that checks it.
    final class ReadCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0

        /// The reads so far.
        var count: Int {
            lock.withLock { value }
        }

        func increment() {
            lock.withLock { value += 1 }
        }
    }

    /// Log lines captured from the application's loggers.
    final class LogRecorder: @unchecked Sendable {
        /// One captured line.
        struct Line: Sendable, Equatable {
            var level: Logger.Level
            var message: String
        }

        private let lock = NSLock()
        private var captured: [Line] = []

        /// Everything logged so far, in order.
        var lines: [Line] {
            lock.withLock { captured }
        }

        /// A logger whose lines go to this recorder, at every level.
        var logger: Logger {
            var logger = Logger(label: "OpenJevServerTests") { _ in
                RecordingHandler(recorder: self)
            }
            logger.logLevel = .trace
            return logger
        }

        func append(_ line: Line) {
            lock.withLock { captured.append(line) }
        }
    }

    /// A log handler that hands every event to a ``LogRecorder``.
    struct RecordingHandler: LogHandler {
        let recorder: LogRecorder
        var metadata: Logger.Metadata = [:]
        var logLevel: Logger.Level = .trace

        subscript(metadataKey key: String) -> Logger.Metadata.Value? {
            get { metadata[key] }
            set { metadata[key] = newValue }
        }

        func log(event: LogEvent) {
            recorder.append(LogRecorder.Line(level: event.level, message: "\(event.message)"))
        }
    }

    /// A parsed `server-timing` value, `model;dur=A, server;dur=B, total;dur=C`, each with one
    /// decimal.
    struct ServerTimingValue: Equatable {
        var model: Double
        var server: Double
        var total: Double

        init?(_ text: String) {
            let parts = text.components(separatedBy: ", ")
            let names = ["model", "server", "total"]
            guard parts.count == 3 else { return nil }
            var values: [Double] = []
            for (part, name) in zip(parts, names) {
                let prefix = "\(name);dur="
                guard part.hasPrefix(prefix) else { return nil }
                let number = part.dropFirst(prefix.count)
                let pieces = number.split(separator: ".", omittingEmptySubsequences: false)
                guard pieces.count == 2, pieces[1].count == 1, let value = Double(number) else {
                    return nil
                }
                values.append(value)
            }
            (model, server, total) = (values[0], values[1], values[2])
        }
    }

    extension WireFixtures {
        /// ``missingMessageText`` as a test comment.
        static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
    }

    extension PolicyFixtures {
        /// ``missingMessageText`` as a test comment.
        static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
    }
#endif
