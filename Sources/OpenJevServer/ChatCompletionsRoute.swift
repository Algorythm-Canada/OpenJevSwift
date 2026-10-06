// A port of upstream OpenJev (razorback16/openjev at dcd2094), the `chat_completions` route of
// `openjev/chat.py` and the `StreamingResponse` its `MlxGenerator.stream` returns, with
// `watch_client` replaced by the server's own connection watch (issue #37). Apache-2.0. See
// THIRD_PARTY.md.

#if canImport(Hummingbird)
    import HTTPTypes
    import Hummingbird
    import Logging
    import NIOCore
    import OpenJevCore

    /// `POST /v1/chat/completions` over ``/OpenJevCore/ChatCompletions``.
    ///
    /// The body is read as Starlette's `request.json()` reads it, whatever its content type, and
    /// everything upstream refuses is refused before an answer starts. A whole reply is generated
    /// while the client is there, as a decision is. A streamed reply is admitted and waits for its
    /// turn before its 200 is sent, then writes its events as the generation emits them; a client
    /// that goes away, or reads so slowly that 64 pieces wait for it, stops the generation at its
    /// next block (``/OpenJevCore/ChatCompletionStream``).
    struct ChatCompletionsRoute: Sendable {
        /// The settings: the body cap.
        let settings: ServerSettings
        /// The generation service.
        let chat: ChatCompletions
        /// The server's connections, or `nil` when requests are not watched.
        let connections: ConnectionRegistry?

        /// Upstream's `StreamingResponse(media_type="text/event-stream")`, to which Starlette adds
        /// the charset.
        static let eventStreamContentType = "text/event-stream; charset=utf-8"

        /// The answer to a request.
        ///
        /// - Throws: A ``/OpenJevCore/ChatCompletionError`` for every refusal before the answer
        ///   starts, which the headers middleware renders, the 503 for a generator that failed
        ///   included, which is logged at error level; ``ClientDisconnected`` when the client went
        ///   away before the answer was ready.
        func respond(_ request: Request, context: OpenJevRequestContext) async throws -> Response {
            let log = RefusalLog(context: context)
            let reader = RequestBodyReader(maxBodyBytes: settings.maxBodyBytes)
            let bytes = try await reader.bytes(of: request)
            let body: JSONValue
            do {
                // request.json() is json.loads(body): no content type is required, and an empty
                // body is not JSON.
                body = try reader.parse(bytes.readableBytesView)
            } catch {
                throw ChatCompletionError.notJSON
            }
            let prepared: PreparedChatCompletion
            do throws(ChatCompletionError) {
                prepared = try await chat.prepare(body)
            } catch {
                // The rest are the client's; a 503 is the generator failing, logged as one that
                // fails later is.
                if error.status == 503 {
                    log.failure(status: error.status, error.message)
                }
                throw error
            }
            let watch = connections?.watch(for: context.channel)
            let chat = chat
            if prepared.request.stream {
                let stream = try await answer(log: log) {
                    try await ClientConnection.cancellingOnDisconnect(watch) {
                        try await chat.stream(prepared)
                    }
                }
                return Response(
                    status: .ok, headers: [.contentType: Self.eventStreamContentType],
                    body: ResponseBody { writer in
                        try await Self.write(stream, to: &writer, watch: watch, log: log)
                    })
            }
            let reply = try await answer(log: log) {
                try await ClientConnection.cancellingOnDisconnect(watch) {
                    try await chat.complete(prepared)
                }
            }
            return WireResponses.json(status: .ok, bytes: try WireEncoder().bytes(reply))
        }

        /// The value of `body`, with a failure of the generator turned into its 503.
        ///
        /// - Throws: ``ClientDisconnected`` and ``/OpenJevCore/ChatCompletionError`` as they come;
        ///   anything else as ``/OpenJevCore/ChatCompletionError/backendUnavailable(_:)`` naming
        ///   its type, logged at error level. A request the server cancelled while stopping is a
        ///   `CancellationError` there, as a decision's is.
        private func answer<T>(
            log: RefusalLog, _ body: () async throws -> T
        ) async throws -> T {
            do {
                return try await body()
            } catch let error as ClientDisconnected {
                throw error
            } catch let error as ChatCompletionError {
                throw error
            } catch {
                let failure = ChatCompletionError.backendUnavailable(
                    String(describing: type(of: error)))
                log.failure(status: failure.status, failure.message)
                throw failure
            }
        }

        /// Runs `stream`, writing each event as it comes, while a sibling task watches the
        /// connection: a client that goes away cancels the stream.
        ///
        /// A complete stream, and one whose reader fell behind, end the response properly; the
        /// latter has no `[DONE]`, as upstream's has none. A client that went away gets nothing
        /// more. A generation that fails after the stream started can no longer change the
        /// status: the connection is closed without the response's end, as uvicorn closes it
        /// upstream, and the failure is logged.
        static func write(
            _ stream: ChatCompletionStream, to writer: inout any ResponseBodyWriter,
            watch: ConnectionWatch?, log: RefusalLog
        ) async throws {
            let ending: ChatCompletionStream.Ending
            do {
                ending = try await withThrowingTaskGroup(of: Void.self) { group in
                    if let watch {
                        group.addTask {
                            await watch.waitUntilClosed()
                            if watch.isClosed {
                                stream.cancel()
                            }
                        }
                    }
                    defer { group.cancelAll() }
                    return try await stream.run { event in
                        try await writer.write(ByteBuffer(string: event))
                    }
                }
            } catch {
                // The watch also closes when the server cancels the request, which it does when a
                // shutdown runs out of time: that is the server stopping, and the request was cut
                // short, so the error goes on (D-049).
                if !Task.isCancelled, watch?.isClosed == true {
                    // Nobody is listening: no finish chunk, no [DONE], nothing to report.
                    return
                }
                if let failure = error as? ChatCompletionStream.GenerationFailed {
                    log.failure(
                        status: 200,
                        "the stream broke off: inference backend unavailable: "
                            + String(describing: type(of: failure.error)))
                }
                throw error
            }
            switch ending {
            case .completed, .readerFellBehind:
                try await writer.finish(nil)
            }
        }
    }

    extension ChatCompletionsConfiguration {
        /// The settings upstream's generator reads: `OPENJEV_GEN_MAX_INFLIGHT`,
        /// `OPENJEV_GEN_MAX_QUEUE` and `OPENJEV_GEN_MAX_TOKENS`.
        public init(_ settings: ServerSettings) {
            self.init(
                maxInflight: settings.genMaxInflight, maxQueue: settings.genMaxQueue,
                maxTokens: settings.genMaxTokens)
        }
    }
#endif
