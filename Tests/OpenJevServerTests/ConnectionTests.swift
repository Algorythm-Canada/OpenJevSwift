#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import NIOCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// How the server follows its connections and requests (issue #37): no decision starts for a
    /// client already gone, and a cancellation cuts a request short when it reaches the request
    /// before the connection has taken its whole answer (D-049).
    @Suite("Connections and requests in flight")
    struct ConnectionTests {
        @Test("No work starts for a client already gone or a request already cancelled")
        func noWorkForAGoneClient() async throws {
            let started = ReadCounter()
            let gone = ConnectionWatch()
            gone.close()
            await #expect(throws: ClientDisconnected.self) {
                try await ClientConnection.cancellingOnDisconnect(gone) {
                    started.increment()
                    return 1
                }
            }
            #expect(started.count == 0)

            let open = ConnectionWatch()
            let cancelled = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await ClientConnection.cancellingOnDisconnect(open) {
                    started.increment()
                    return 1
                }
            }
            await #expect(throws: CancellationError.self) { try await cancelled.value }
            #expect(started.count == 0)

            let answer = try await ClientConnection.cancellingOnDisconnect(open) {
                started.increment()
                return 7
            }
            #expect(answer == 7)
            #expect(started.count == 1)
        }

        @Test("A request the server is cancelled while answering is cut short, and gets no answer")
        func answeredAfterTheCancellation() async throws {
            let gate = ReadGate()
            let inFlight = RequestsInFlight()
            let router = Router(context: OpenJevRequestContext.self)
            // As the decision route answers a read the server cancelled: with a 503.
            router.get("/held") { _, _ -> Response in
                do {
                    try await gate.pass()
                    return Response(status: .ok)
                } catch {
                    return Response(status: .serviceUnavailable)
                }
            }
            try await serving(router, noting: inFlight) { port, cancel in
                let client = TestClient(host: "127.0.0.1", port: port)
                client.connect()
                async let response = client.get("/held")
                await gate.waitForArrivals(1)
                cancel()
                // A write on a cancelled task is not sent, so the connection closes unanswered.
                let answered = try? await response
                #expect(answered == nil)
                try? await client.shutdown()
            }
            #expect(gate.cancellations == 1)
            #expect(inFlight.cutShort)
        }

        @Test("A cancellation in the middle of an answer's write cuts its request short")
        func cancelledWhileWriting() async throws {
            let gate = ReadGate()
            let inFlight = RequestsInFlight()
            let router = Router(context: OpenJevRequestContext.self)
            // The route returns at once; its body is written only once the gate opens, as a slow
            // client would hold up a large answer.
            router.get("/slow") { _, _ in
                Response(
                    status: .ok,
                    body: ResponseBody { writer in
                        try await gate.pass()
                        try await writer.write(ByteBuffer(string: "done"))
                        try await writer.finish(nil)
                    })
            }
            try await serving(router, noting: inFlight) { port, cancel in
                let client = TestClient(host: "127.0.0.1", port: port)
                client.connect()
                async let response = client.get("/slow")
                await gate.waitForArrivals(1)
                #expect(!inFlight.cutShort)
                cancel()
                _ = try? await response
                try? await client.shutdown()
            }
            #expect(gate.cancellations == 1)
            #expect(inFlight.cutShort)
        }

        @Test("A request answered before the cancellation is not cut short, however late it ends")
        func answeredBeforeTheCancellation() async throws {
            let inFlight = RequestsInFlight()
            let release = OneShot<Void>()
            let endedCancelled = OneShot<Bool>()
            let router = Router(context: OpenJevRequestContext.self)
            // The whole answer is written first; then the request's task goes on until the test
            // has cancelled the server, as a task on a loaded machine can.
            router.get("/answered") { _, _ in
                Response(
                    status: .ok,
                    body: ResponseBody { writer in
                        try await writer.write(ByteBuffer(string: "done"))
                        try await writer.finish(nil)
                        await release.value
                        endedCancelled.resolve(Task.isCancelled)
                    })
            }
            try await serving(router, noting: inFlight) { port, cancel in
                // Nothing here throws, so the held request is always let go and the server can
                // stop, even when an expectation fails.
                let client = TestClient(host: "127.0.0.1", port: port)
                client.connect()
                let response = try? await client.get("/answered")
                #expect(response?.status == .ok)
                #expect(response?.body.map { String(buffer: $0) } == "done")
                try? await client.shutdown()
                cancel()
                release.resolve(())
                // The cancellation reached the request's task before it ended. Only the route
                // answers 200, so a request that never reached it is not waited for.
                if response?.status == .ok {
                    #expect(await endedCancelled.value)
                }
            }
            #expect(!inFlight.cutShort)
        }

        @Test("A request whose answer fails with no cancellation is not cut short")
        func failedWithoutTheCancellation() async throws {
            let inFlight = RequestsInFlight()
            let router = Router(context: OpenJevRequestContext.self)
            // The answer fails on its own, as a write to a client that has gone can; the server is
            // cancelled only afterwards, with nothing in flight.
            router.get("/failing") { _, _ in
                Response(status: .ok, body: ResponseBody { _ in throw AnswerFailure() })
            }
            try await serving(router, noting: inFlight) { port, cancel in
                let client = TestClient(host: "127.0.0.1", port: port)
                client.connect()
                // The connection closes once the responder has thrown, so this returns after it.
                let answered = try? await client.get("/failing")
                #expect(answered == nil)
                try? await client.shutdown()
                cancel()
            }
            #expect(!inFlight.cutShort)
        }

        /// An answer's own failure.
        private struct AnswerFailure: Error {}

        /// Serves `router` on an ephemeral port of 127.0.0.1 through the server's HTTP/1 channel,
        /// which notes in `inFlight` the requests a cancellation cuts short, while `body` runs with
        /// the port and a way to cancel the server; then cancels the server and waits for it to
        /// end.
        private func serving(
            _ router: Router<OpenJevRequestContext>, noting inFlight: RequestsInFlight,
            _ body: (_ port: Int, _ cancel: () -> Void) async throws -> Void
        ) async throws {
            let (ports, portContinuation) = AsyncStream.makeStream(of: Int.self)
            let application = Application(
                router: router,
                server: OpenJevApplication.server(
                    connections: ConnectionRegistry(), inFlight: inFlight),
                configuration: ApplicationConfiguration(address: .hostname("127.0.0.1", port: 0)),
                onServerRunning: { channel in
                    portContinuation.yield(channel.localAddress?.port ?? 0)
                },
                logger: LiveServer.quietLogger)
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask { try await application.run() }
                var iterator = ports.makeAsyncIterator()
                let port = try #require(await iterator.next())
                try await body(port) { group.cancelAll() }
                group.cancelAll()
                // A cancelled server can end by throwing CancellationError.
                while (try? await group.next()) != nil {}
            }
        }
    }
#endif
