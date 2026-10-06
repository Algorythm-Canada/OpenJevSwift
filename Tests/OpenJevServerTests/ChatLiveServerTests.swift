#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import Logging
    import NIOCore
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// The chat route over real connections (issue #53): a client that goes away stops its
    /// generation at the next block and gives its place back, whether it was streaming, waiting
    /// for a whole reply or waiting for its turn.
    @Suite("Chat completions over real connections")
    struct ChatLiveServerTests {
        /// A `POST /v1/chat/completions` of `body`.
        static func post(_ body: JSONValue) throws -> TestClient.Request {
            TestClient.Request(
                "/v1/chat/completions", method: .post, authority: "localhost",
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: try WireEncoder().bytes(json: body)))
        }

        /// The first generation emits up to 200 pieces with a block-sized pause between them,
        /// counting them in `emitted`, and stops when asked or cancelled; every later one is
        /// upstream's stub reply.
        static func slowFirst(emitted: ReadCounter) -> StubTextGenerator {
            let calls = ReadCounter()
            return StubTextGenerator(generate: { call, emit in
                calls.increment()
                guard calls.count == 1 else {
                    return try await StubTextGenerator.upstreamReply()(call, emit)
                }
                for index in 0..<200 {
                    if Task.isCancelled {
                        break
                    }
                    emitted.increment()
                    if !emit("t\(index) ", 1000 + index) {
                        return TextGeneration(
                            generated: [1000], promptTokens: call.prompt.count,
                            finishReason: .cancelled)
                    }
                    try? await Task.sleep(for: .milliseconds(80))
                }
                return TextGeneration(
                    generated: [1000], promptTokens: call.prompt.count, finishReason: .cancelled)
            })
        }

        /// The runtime cannot be interrupted, only asked to stop at its next block: a client that
        /// closes its connection mid-stream must reach it as a false from `emit`, and the slot
        /// must come back either way.
        @Test("test_a_disconnected_client_stops_generation")
        func disconnectedClientStopsGeneration() async throws {
            let emitted = ReadCounter()
            let generator = Self.slowFirst(emitted: emitted)
            // One place: a slot that did not come back would refuse the next request with a 529.
            let settings = try ServerSettings(
                host: "127.0.0.1", port: 0, genMaxInflight: 1, genMaxQueue: 0)
            let service = try await ChatHarness.service(settings, generator: generator)
            try await LiveServer.run(service: service, settings: settings) { server in
                let client = server.client()
                try await client.executeAndDontWaitForResponse(
                    Self.post(ChatHarness.chat(["stream": true])))
                #expect(try await eventually { emitted.count >= 3 })
                try await client.shutdown()
                #expect(try await eventually { generator.running == 0 })
                #expect(
                    emitted.count < 200, "the generation ran to completion after the client left")
                // The stream gives its place back once its run has ended, a moment after the
                // generation stopped; until then the next request is the 529. One whose place
                // never came back would be refused until the limit.
                let next = server.client()
                #expect(
                    try await eventually {
                        (try? await next.execute(Self.post(ChatHarness.chat)))?.status == .ok
                    })
                try await next.shutdown()
            }
        }

        @Test("A client that leaves before its whole reply cancels the generation: a 499")
        func disconnectedWholeReply() async throws {
            let emitted = ReadCounter()
            let generator = Self.slowFirst(emitted: emitted)
            let settings = try ServerSettings(
                host: "127.0.0.1", port: 0, genMaxInflight: 1, genMaxQueue: 0)
            let service = try await ChatHarness.service(settings, generator: generator)
            let recorder = LogRecorder()
            try await LiveServer.run(service: service, settings: settings, logger: recorder.logger)
            {
                server in
                let client = server.client()
                try await client.executeAndDontWaitForResponse(Self.post(ChatHarness.chat))
                #expect(try await eventually { emitted.count >= 2 })
                try await client.shutdown()
                #expect(try await eventually { generator.running == 0 })
                #expect(emitted.count < 200)
                #expect(
                    try await eventually {
                        recorder.lines.contains {
                            $0.level == .info
                                && $0.message.hasPrefix("POST /v1/chat/completions 499 ")
                        }
                    })
                let next = server.client()
                #expect(try await next.execute(Self.post(ChatHarness.chat)).status == .ok)
                try await next.shutdown()
            }
            // A client that went away is not a backend failure.
            let failures = recorder.lines.filter { $0.level >= .warning }
            #expect(failures.isEmpty, "\(failures)")
        }

        /// A generation held until the test opens `gate`, then upstream's stub reply.
        static func gated(_ gate: ReadGate) -> StubTextGenerator {
            StubTextGenerator(generate: { call, emit in
                try await gate.pass()
                return try await StubTextGenerator.upstreamReply()(call, emit)
            })
        }

        @Test("A graceful shutdown lets a stream in flight finish, [DONE] last, then releases")
        func gracefulShutdownFinishesAStream() async throws {
            let gate = ReadGate()
            let backend = StubGeneratingBackend(generation: Self.gated(gate))
            let service = try await DecisionBackendProvider { _ in backend }.makeService(
                settings: try ServerSettings())
            try await LiveServer.run(service: service, shutdownTimeout: .seconds(10)) { server in
                let client = server.client()
                async let answer = client.execute(Self.post(ChatHarness.chat(["stream": true])))
                await gate.waitForArrivals(1)
                await server.triggerGracefulShutdown()
                gate.open()
                let response = try await answer
                #expect(response.status == .ok)
                let body = response.body.map { String(buffer: $0) } ?? ""
                #expect(body.hasSuffix("data: [DONE]\n\n"))
                let ended = await server.waitUntilEnded()
                #expect(throws: Never.self) { try ended.get() }
                #expect(backend.reads.closeCount == 1)
                try? await client.shutdown()
            }
        }

        /// The server cancels a stream still running when the shutdown's time is up: its
        /// generation stops, the request counts as cut short, and the model is still released.
        @Test("A shutdown that runs out of time cuts a stream off and stops its generation")
        func shutdownTimeoutCutsAStreamOff() async throws {
            let gate = ReadGate()
            let backend = StubGeneratingBackend(generation: Self.gated(gate))
            let service = try await DecisionBackendProvider { _ in backend }.makeService(
                settings: try ServerSettings())
            try await LiveServer.run(service: service, shutdownTimeout: .milliseconds(200)) {
                server in
                let client = server.client()
                try await client.executeAndDontWaitForResponse(
                    Self.post(ChatHarness.chat(["stream": true])))
                await gate.waitForArrivals(1)
                await server.triggerGracefulShutdown()
                let ended = await server.waitUntilEnded()
                #expect(throws: ShutdownInterrupted.self) { try ended.get() }
                #expect(gate.cancellations == 1)
                #expect(backend.generation.running == 0)
                #expect(backend.reads.closeCount == 1)
                try? await client.shutdown()
            }
        }

        /// Writes `stream` as the route does, on a task that is cancelled first when `cancelled`,
        /// with a connection watch that has closed.
        static func writeWithClosedWatch(
            _ stream: ChatCompletionStream, cancelled: Bool
        ) async throws {
            let watch = ConnectionWatch()
            watch.close()
            let task = Task {
                if cancelled {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
                var writer: any ResponseBodyWriter = CollectingWriter(collector: BodyCollector())
                try await ChatCompletionsRoute.write(
                    stream, to: &writer, watch: watch,
                    log: RefusalLog(logger: LiveServer.quietLogger, requestID: ""))
            }
            try await task.value
        }

        /// When a shutdown runs out of time the server cancels the request and closes the
        /// connection's input, so the watch closes too; the request must still end by throwing on
        /// its cancelled task, or the server would not count it as cut short (D-049). Which of the
        /// two reaches the stream first is a race on a live server, so it is pinned here.
        @Test("A stream the server cancels counts as cut short, even with its watch closed")
        func serverCancellationIsNotAClientLeaving() async throws {
            let chat = ChatCompletions(generator: Self.gated(ReadGate()))
            let streamed = ChatHarness.chat(["stream": true])
            let cut = try await chat.stream(try await chat.prepare(streamed))
            await #expect(throws: CancellationError.self) {
                try await Self.writeWithClosedWatch(cut, cancelled: true)
            }
            // A client that left, with the task not cancelled, ends the stream quietly.
            let left = try await chat.stream(try await chat.prepare(streamed))
            await #expect(throws: Never.self) {
                try await Self.writeWithClosedWatch(left, cancelled: false)
            }
            #expect(chat.running == 0 && chat.freeSlots == 8)
        }

        /// The HTTP side of `test_a_cancelled_wait_for_a_chat_slot_leaks_no_capacity`: a request
        /// waiting for its turn has no answer yet, and a client that leaves then gives its place
        /// back. With one slot and one place in the queue, a place that leaked would refuse the
        /// next request with a 529.
        @Test("A client that leaves while waiting for its turn gives its place back")
        func disconnectedWait() async throws {
            let gate = ReadGate()
            let generator = StubTextGenerator(generate: { call, emit in
                try await gate.pass()
                return try await StubTextGenerator.upstreamReply()(call, emit)
            })
            let settings = try ServerSettings(
                host: "127.0.0.1", port: 0, genMaxInflight: 1, genMaxQueue: 1)
            let service = try await ChatHarness.service(settings, generator: generator)
            let recorder = LogRecorder()
            try await LiveServer.run(service: service, settings: settings, logger: recorder.logger)
            {
                server in
                let holder = server.client()
                async let held = holder.execute(Self.post(ChatHarness.chat(["stream": true])))
                await gate.waitForArrivals(1)
                let waiting = server.client()
                try await waiting.executeAndDontWaitForResponse(Self.post(ChatHarness.chat))
                // Its prompt is rendered, then it waits for the slot the holder has.
                #expect(try await eventually { generator.renderedPrompts.count == 2 })
                try await Task.sleep(for: .milliseconds(200))
                try await waiting.shutdown()
                #expect(
                    try await eventually {
                        recorder.lines.contains {
                            $0.message.hasPrefix("POST /v1/chat/completions 499 ")
                        }
                    })
                let next = server.client()
                async let third = next.execute(Self.post(ChatHarness.chat))
                #expect(try await eventually { generator.renderedPrompts.count == 3 })
                gate.open()
                #expect(try await held.status == .ok)
                #expect(try await third.status == .ok)
                #expect(gate.arrivals == 2, "the request that left never generated")
                try await holder.shutdown()
                try await next.shutdown()
            }
        }
    }
#endif
