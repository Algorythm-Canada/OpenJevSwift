#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import Logging
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import ServiceLifecycle
    import Testing

    /// ``DecisionServer`` over real connections on an ephemeral port (issue #37): a client that
    /// goes away, the graceful shutdown with a request in flight, and a shutdown that runs out of
    /// time.
    @Suite(
        "Live server", .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    struct LiveServerTests {
        /// The README quickstart, which the policy recording's `plain` case holds.
        private func quickstart() throws -> JSONValue {
            try #require(PolicyFixtures.policyCase(named: "plain")["request"])
        }

        @Test("A client that goes away cancels its decision's read and frees its place")
        func clientDisconnect() async throws {
            let gate = ReadGate()
            let stub = StubBackend(gate: gate)
            let settings = try ServerSettings(host: "127.0.0.1", port: 0, maxQueue: 1)
            let service = try await ServerHarness.diffusionService(settings, backend: stub)
            let recorder = LogRecorder()
            let request = try quickstart()
            try await LiveServer.run(
                service: service, settings: settings, logger: recorder.logger
            ) { server in
                let client = server.client()
                try await client.executeAndDontWaitForResponse(LiveServer.post(request))
                await gate.waitForArrivals(1)
                try await client.shutdown()
                #expect(try await eventually { gate.cancellations == 1 })
                #expect(
                    try await eventually {
                        recorder.lines.contains {
                            $0.level == .info && $0.message.hasPrefix("POST /v1/systemone 499 ")
                        }
                    })
                // The place the cancelled request held is free: with maxQueue 1, the next
                // request is answered.
                gate.open()
                let next = server.client()
                let response = try await next.execute(LiveServer.post(request))
                #expect(response.status == .ok)
                try await next.shutdown()
            }
            // A client that went away is not a backend failure.
            let failures = recorder.lines.filter { $0.level >= .warning }
            #expect(failures.isEmpty, "\(failures)")
        }

        @Test("A graceful shutdown answers the request in flight, refuses new ones, then releases")
        func gracefulShutdown() async throws {
            let gate = ReadGate()
            let stub = StubBackend(gate: gate)
            let service = try await ServerHarness.diffusionService(ServerSettings(), backend: stub)
            let request = try quickstart()
            let clock = ContinuousClock()
            try await LiveServer.run(service: service, shutdownTimeout: .seconds(10)) { server in
                let client = server.client()
                async let answer = client.execute(LiveServer.post(request))
                await gate.waitForArrivals(1)
                let started = clock.now
                await server.triggerGracefulShutdown()
                // Once the listener has closed, a new connection is refused, while the first
                // request is still in flight.
                let refused = try await eventually {
                    let late = TestClient(host: "127.0.0.1", port: server.port)
                    late.connect()
                    do {
                        _ = try await late.get("/health")
                        try? await late.shutdown()
                        return false
                    } catch {
                        try? await late.shutdown()
                        return true
                    }
                }
                #expect(refused)
                #expect(gate.arrivals == 1)
                #expect(stub.closeCount == 0)
                gate.open()
                let response = try await answer
                #expect(response.status == .ok)
                #expect(response.headers[.init("x-request-id")!] != nil)
                let ended = await server.waitUntilEnded()
                #expect(throws: Never.self) { try ended.get() }
                #expect(clock.now - started < .seconds(10))
                #expect(stub.closeCount == 1)
                try? await client.shutdown()
            }
        }

        @Test("A shutdown that runs out of time cancels the read in flight and still releases")
        func shutdownTimeout() async throws {
            let gate = ReadGate()
            let stub = StubBackend(gate: gate)
            let service = try await ServerHarness.diffusionService(ServerSettings(), backend: stub)
            let request = try quickstart()
            let clock = ContinuousClock()
            try await LiveServer.run(service: service, shutdownTimeout: .milliseconds(200)) {
                server in
                let client = server.client()
                try await client.executeAndDontWaitForResponse(LiveServer.post(request))
                await gate.waitForArrivals(1)
                let started = clock.now
                await server.triggerGracefulShutdown()
                let ended = await server.waitUntilEnded()
                #expect(throws: ShutdownInterrupted.self) { try ended.get() }
                #expect(clock.now - started < .seconds(3))
                #expect(gate.cancellations == 1)
                #expect(stub.closeCount == 1)
                try? await client.shutdown()
            }
        }

        @Test("An address already in use stops the server with the error and releases the model")
        func addressInUse() async throws {
            let first = StubBackend()
            let service = try await ServerHarness.diffusionService(ServerSettings(), backend: first)
            try await LiveServer.run(service: service) { server in
                let second = StubBackend()
                let settings = try ServerSettings(host: "127.0.0.1", port: server.port)
                let other = DecisionServer(
                    settings: settings,
                    service: try await ServerHarness.diffusionService(settings, backend: second),
                    logger: Logger(label: "OpenJevServerTests"))
                let group = ServiceGroup(
                    configuration: ServiceGroupConfiguration(
                        services: [other], logger: Logger(label: "OpenJevServerTests")))
                await #expect(throws: (any Error).self) { try await group.run() }
                #expect(second.closeCount == 1)
            }
            #expect(first.closeCount == 1)
        }
    }
#endif
