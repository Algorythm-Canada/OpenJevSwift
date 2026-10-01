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
    /// client already gone, and a request counts as in flight until its response is written.
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

        @Test(
            "A request counts as in flight until its response is written, after its route returns")
        func countsThroughTheWrite() async throws {
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
                let client = TestClient(host: "127.0.0.1", port: port)
                client.connect()
                async let response = client.get("/slow")
                await gate.waitForArrivals(1)
                #expect(inFlight.count == 1)
                inFlight.serverCancelled()
                #expect(inFlight.cutShort)
                gate.open()
                let answered = try await response
                #expect(answered.status == .ok)
                #expect(answered.body.map { String(buffer: $0) } == "done")
                #expect(try await eventually { inFlight.count == 0 })
                try await client.shutdown()
                group.cancelAll()
            }
        }
    }
#endif
