#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// The queue bound and the model time over HTTP (issue #37): upstream's 529 with
    /// `retry-after: 1`, and `server-timing` against the time a stub spends, as
    /// `test_model_time_reaches_the_response_from_parallel_reads` and
    /// `test_server_timing_counts_the_read` check it upstream.
    @Suite(
        "Capacity and model time over HTTP",
        .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    struct ServerCapacityTests {
        /// The recorded request of a policy case.
        private func request(_ name: String) throws -> JSONValue {
            try #require(PolicyFixtures.policyCase(named: name)["request"])
        }

        /// The parsed `server-timing` of a response.
        private func timing(_ response: TestResponse) throws -> ServerTimingValue {
            try #require(
                ServerHarness.header(response, "server-timing").flatMap(ServerTimingValue.init))
        }

        /// A duration in milliseconds.
        private func milliseconds(_ duration: Duration) -> Double {
            ServerTiming.milliseconds(duration)
        }

        /// Every call time of a stub, summed.
        private func spent(_ times: [Duration]) -> Double {
            milliseconds(times.reduce(Duration.zero, +))
        }

        /// How far below a time its `server-timing` value can be: half the one decimal it is
        /// written with.
        private let rounding = 0.05

        // MARK: The queue bound

        @Test("A request past the queue bound is the 529 with retry-after 1; the one inside ends")
        func overloaded() async throws {
            let gate = ReadGate()
            let settings = try ServerSettings(maxQueue: 1)
            let service = try await ServerHarness.diffusionService(
                settings, backend: StubBackend(gate: gate))
            let request = try request("plain")
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                try await withThrowingTaskGroup(of: TestResponse.self) { group in
                    group.addTask { try await ServerHarness.post(client, request) }
                    await gate.waitForArrivals(1)
                    let refused = try await ServerHarness.post(client, request)
                    #expect(refused.status.code == 529)
                    #expect(ServerHarness.header(refused, "retry-after") == "1")
                    #expect(
                        ServerHarness.text(refused)
                            == #"{"detail":{"error_type":"overloaded_error","message":"#
                            + #""OpenJev is at capacity. Retry shortly."}}"#)
                    ServerHarness.expectServerHeaders(refused, "529")
                    #expect(try timing(refused).model == 0)
                    gate.open()
                    let first = try #require(try await group.next())
                    #expect(first.status == .ok)
                }
                // The place is free again.
                #expect(try await ServerHarness.post(client, request).status == .ok)
            }
        }

        /// Upstream's `waiting >= max_queue` as written (D-027 item 5): 0 refuses every request,
        /// so the issue's "maxQueue 0 refuses the second" is maxQueue 1 above.
        @Test("A queue bound of 0 refuses the first request too, as upstream's does")
        func closedQueue() async throws {
            let settings = try ServerSettings(maxQueue: 0)
            let service = try await ServerHarness.diffusionService(settings)
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                let response = try await ServerHarness.post(client, try request("plain"))
                #expect(response.status.code == 529)
                #expect(ServerHarness.header(response, "retry-after") == "1")
            }
        }

        @Test("The encoder engine's 529 names its model")
        func encoderOverloaded() async throws {
            let gate = ReadGate()
            let settings = try ServerSettings(maxQueue: 1, warmup: false)
            let service = try await ServerHarness.encoderService(
                StubQuestionReadBackend(gate: gate), settings: settings)
            let request = try request("plain")
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                try await withThrowingTaskGroup(of: TestResponse.self) { group in
                    group.addTask { try await ServerHarness.post(client, request) }
                    await gate.waitForArrivals(1)
                    let refused = try await ServerHarness.post(client, request)
                    #expect(refused.status.code == 529)
                    #expect(ServerHarness.header(refused, "retry-after") == "1")
                    #expect(
                        ServerHarness.text(refused).contains(
                            #""message":"laya-1.0 is at capacity. Retry shortly.""#))
                    gate.open()
                    #expect(try #require(try await group.next()).status == .ok)
                }
            }
        }

        // MARK: Model time

        // The bounds on `model` hold however long the semaphore and the hops around each call
        // take: the engine times each call around the stub's own timing of it, so `model` is at
        // least the stub's call times, less `rounding`, and every call lies inside the request,
        // so `model` is at most `total` when the calls run one after another. A call counted
        // twice adds at least the stub's delay, which the upper bound catches whenever the
        // request spends less than that outside its calls.

        @Test("A serial read's model time is the stub's time and server is the rest")
        func serialRead() async throws {
            let stub = StubBackend(delay: .milliseconds(40))
            let service = try await ServerHarness.diffusionService(ServerSettings(), backend: stub)
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(client, try request("plain"))
                #expect(response.status == .ok)
                let timing = try timing(response)
                #expect(stub.callTimes.count == 1)
                #expect(timing.model >= spent(stub.callTimes) - rounding, "\(timing)")
                #expect(timing.model >= 40)
                // server is what remains of total, each written with one decimal.
                #expect(abs(timing.server - max(0, timing.total - timing.model)) <= 0.15)
                #expect(timing.total >= timing.model)
            }
        }

        @Test(
            "Parallel groups sum their reads' time, so model exceeds total",
            .timeLimit(.minutes(1)))
        func parallelGroups() async throws {
            // Two groups of sixteen samples: 32 reads at once. Each read waits in the stub until
            // all 32 are there and then takes 100 ms, so the reads overlap however late the
            // scheduler starts each one, and their times add up to at least 3.1 s (31 reads'
            // delay) more than the span from the first read's start to the last one's end. model
            // exceeds total whenever the request spends less than that outside its reads. That
            // time is the scheduler's, which no stub controls; under heavy load it has reached a
            // few hundred milliseconds. Reads that ran one at a time would never fill a round,
            // and the time limit would fail the test.
            let reads = 32
            let stub = StubBackend(delay: .milliseconds(100), barrier: ReadBarrier(parties: reads))
            let service = try await ServerHarness.diffusionService(ServerSettings(), backend: stub)
            var object = try #require(try request("parallel_groups_30_nouls").objectValue)
            object["samples"] = 16
            let request = JSONValue.object(object)
            try await ServerHarness.withClient(service: service) { client in
                // A first request loads the fixture tokenizer and fills the template cache, so
                // the measured one spends little time outside its reads.
                #expect(try await ServerHarness.post(client, request).status == .ok)
                let before = stub.callTimes.count
                let response = try await ServerHarness.post(client, request)
                #expect(response.status == .ok)
                let timing = try timing(response)
                let times = Array(stub.callTimes.dropFirst(before))
                #expect(times.count == reads)
                #expect(timing.model >= spent(times) - rounding, "\(timing)")
                // Each read lies inside the request, so the reads add up to at most one total
                // each.
                #expect(
                    timing.model <= Double(reads) * (timing.total + rounding) + rounding,
                    "\(timing)")
                #expect(timing.model > timing.total, "\(timing)")
                #expect(timing.server == 0)
            }
        }

        @Test("A request refused after a thought reports the thought's time")
        func refusalAfterAThought() async throws {
            // The read after the recorded thought is 191 tokens, one past this limit.
            let stub = StubBackend(maxPromptTokens: 190, delay: .milliseconds(30))
            let service = try await ServerHarness.diffusionService(ServerSettings(), backend: stub)
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(client, try request("think_256"))
                #expect(response.status == .badRequest)
                #expect(
                    ServerHarness.text(response)
                        == #"{"detail":"the request is 191 tokens; the limit is 190"}"#)
                let timing = try timing(response)
                #expect(stub.thinks.count == 1 && stub.reads.isEmpty)
                #expect(timing.model >= 30)
                #expect(timing.model >= spent(stub.callTimes) - rounding, "\(timing)")
                #expect(timing.model <= timing.total, "\(timing)")
            }
        }

        @Test("A sequential request whose second read is refused reports both reads")
        func refusalAfterAFirstRead() async throws {
            let stub = StubBackend(
                delay: .milliseconds(30), failure: BackendRefusal(reason: "no"),
                succeedingCalls: 1)
            let service = try await ServerHarness.diffusionService(ServerSettings(), backend: stub)
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(
                    client, try request("sequential_24_nouls"))
                #expect(response.status == .badRequest)
                #expect(
                    ServerHarness.text(response)
                        == #"{"detail":"the model rejected this request: no"}"#)
                let timing = try timing(response)
                #expect(stub.callTimes.count == 2)
                #expect(timing.model >= 60)
                #expect(timing.model >= spent(stub.callTimes) - rounding, "\(timing)")
                #expect(timing.model <= timing.total, "\(timing)")
            }
        }

        @Test("An encoder request whose second batch fails reports both batches in its 503")
        func failureAfterAFirstBatch() async throws {
            let stub = StubQuestionReadBackend(
                delay: .milliseconds(20), failure: .throwing("boom"), succeedingBatches: 1)
            let settings = try ServerSettings(warmup: false)
            let service = try await ServerHarness.encoderService(stub, settings: settings)
            var questions = JSONObject()
            for index in 0..<20 {
                questions["q\(index)"] = ["type": "noul"]
            }
            let body: JSONValue = [
                "state": "s", "model": "laya-1.0", "questions": .object(questions),
            ]
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                let response = try await ServerHarness.post(client, body)
                #expect(response.status == .serviceUnavailable)
                let timing = try timing(response)
                #expect(stub.calls.count == 2)
                #expect(timing.model >= 40)
                #expect(timing.model >= spent(stub.callTimes) - rounding, "\(timing)")
                #expect(timing.model <= timing.total, "\(timing)")
            }
        }

        @Test("A request refused before any backend call reports no model time")
        func refusalBeforeAnyRead() async throws {
            let stub = StubBackend(delay: .milliseconds(30))
            let service = try await ServerHarness.diffusionService(ServerSettings(), backend: stub)
            try await ServerHarness.withClient(service: service) { client in
                let request: JSONValue = [
                    "state": "s", "model": "gpt-4", "questions": ["a": ["type": "noul"]],
                ]
                let response = try await ServerHarness.post(client, request)
                #expect(response.status == .badRequest)
                #expect(try timing(response).model == 0)
                #expect(stub.callTimes.isEmpty)
            }
        }
    }
#endif
