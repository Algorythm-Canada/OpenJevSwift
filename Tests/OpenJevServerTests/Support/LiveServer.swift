#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import Logging
    import OpenJevCore
    @testable import OpenJevServer
    import ServiceLifecycle
    import Testing

    /// A ``DecisionServer`` listening on an ephemeral port of 127.0.0.1, inside a service group the
    /// test controls, for the tests that need real connections: a client that goes away, and a
    /// graceful shutdown with a request in flight.
    struct LiveServer: Sendable {
        /// The port the server listens on.
        let port: Int
        /// The group the server runs in.
        let group: ServiceGroup
        private let ended: OneShot<Result<Void, any Error>>

        /// Runs `body` with the server over `service` and shuts the server down afterwards,
        /// gracefully, unless `body` did. `shutdownTimeout` is the group's
        /// `maximumGracefulShutdownDuration`.
        static func run<Value: Sendable>(
            service: any SystemOneService, settings: ServerSettings? = nil,
            logger: Logger = LiveServer.quietLogger,
            shutdownTimeout: Duration = .seconds(10),
            _ body: @Sendable (LiveServer) async throws -> Value
        ) async throws -> Value {
            let (ports, portContinuation) = AsyncStream.makeStream(of: Int.self)
            let server = DecisionServer(
                settings: try settings ?? ServerSettings(host: "127.0.0.1", port: 0),
                service: service, logger: logger,
                onServerRunning: { port in portContinuation.yield(port) })
            var configuration = ServiceGroupConfiguration(services: [server], logger: logger)
            configuration.maximumGracefulShutdownDuration = shutdownTimeout
            let group = ServiceGroup(configuration: configuration)
            let ended = OneShot<Result<Void, any Error>>()
            return try await withThrowingTaskGroup(of: Void.self) { tasks in
                tasks.addTask {
                    let result: Result<Void, any Error>
                    do {
                        try await group.run()
                        result = .success(())
                    } catch {
                        result = .failure(error)
                    }
                    ended.resolve(result)
                    portContinuation.finish()
                }
                var iterator = ports.makeAsyncIterator()
                guard let port = await iterator.next() else {
                    throw HarnessError("the server ended before it listened: \(await ended.value)")
                }
                let live = LiveServer(port: port, group: group, ended: ended)
                do {
                    let value = try await body(live)
                    await group.triggerGracefulShutdown()
                    _ = await ended.value
                    return value
                } catch {
                    await group.triggerGracefulShutdown()
                    _ = await ended.value
                    throw error
                }
            }
        }

        /// A logger that writes only critical lines, so Hummingbird's own error lines on a
        /// cancelled or refused server do not clutter the test log.
        static var quietLogger: Logger {
            var logger = Logger(label: "OpenJevServerTests")
            logger.logLevel = .critical
            return logger
        }

        /// Starts the graceful shutdown, as SIGTERM does for `openjev serve`.
        func triggerGracefulShutdown() async {
            await group.triggerGracefulShutdown()
        }

        /// How the group's `run()` ended, once it has.
        func waitUntilEnded() async -> Result<Void, any Error> {
            await ended.value
        }

        /// A client connected to the server, which the test shuts down.
        func client() -> TestClient {
            let client = TestClient(
                host: "127.0.0.1", port: port,
                configuration: TestClient.Configuration(timeout: .seconds(20)))
            client.connect()
            return client
        }

        /// A `POST /v1/systemone` of a wire request.
        static func post(_ request: JSONValue) throws -> TestClient.Request {
            TestClient.Request(
                "/v1/systemone", method: .post, authority: "localhost",
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: try WireEncoder().bytes(json: request)))
        }
    }

    /// A value set once, which any number of tasks wait for.
    final class OneShot<Value: Sendable>: @unchecked Sendable {
        // Guarded by `lock`.
        private let lock = NSLock()
        private var result: Value?
        private var waiters: [CheckedContinuation<Value, Never>] = []

        /// Sets the value and wakes the waiters; later calls do nothing.
        func resolve(_ value: Value) {
            let woken = lock.withLock { () -> [CheckedContinuation<Value, Never>] in
                guard result == nil else { return [] }
                result = value
                defer { waiters.removeAll() }
                return waiters
            }
            for waiter in woken {
                waiter.resume(returning: value)
            }
        }

        /// The value, once it is set.
        var value: Value {
            get async {
                await withCheckedContinuation { continuation in
                    let ready = lock.withLock { () -> Value? in
                        if let result {
                            return result
                        }
                        waiters.append(continuation)
                        return nil
                    }
                    if let ready {
                        continuation.resume(returning: ready)
                    }
                }
            }
        }
    }

    /// A harness failure, with what went wrong.
    struct HarnessError: Error, CustomStringConvertible {
        var description: String

        init(_ description: String) {
            self.description = description
        }
    }

    /// Polls `condition` every millisecond until it holds, for up to `limit`.
    func eventually(
        within limit: Duration = .seconds(5), _ condition: @Sendable () async -> Bool
    ) async throws -> Bool {
        let clock = ContinuousClock()
        let deadline = clock.now + limit
        while clock.now < deadline {
            if await condition() {
                return true
            }
            try await Task.sleep(for: .milliseconds(1))
        }
        return await condition()
    }
#endif
