import Foundation
import OpenJevTestSupport
import Testing

@testable import OpenJevCore

/// The in-flight semaphore under cancellation, the model time the engines record per request,
/// refusals and failures included, and the release hook (issue #37).
@Suite("Capacity and model time")
struct CapacityTests {
    // MARK: The in-flight semaphore

    @Test("A waiter cancelled while it waits takes no permit and returns none")
    func cancelledWaiterReleasesNothing() async throws {
        let semaphore = AsyncSemaphore(permits: 1)
        try await semaphore.wait()
        #expect(semaphore.availablePermits == 0)
        let waiter = Task { try await semaphore.wait() }
        try await Self.until { semaphore.waitingCount == 1 }
        waiter.cancel()
        await #expect(throws: CancellationError.self) { try await waiter.value }
        // The cancelled waiter left the queue and gave nothing back: the permit is still held.
        #expect(semaphore.waitingCount == 0)
        #expect(semaphore.availablePermits == 0)

        semaphore.signal()
        #expect(semaphore.availablePermits == 1)
        // One permit, so one more waiter gets it and the next waits.
        try await semaphore.wait()
        let blocked = Task { try await semaphore.wait() }
        try await Self.until { semaphore.waitingCount == 1 }
        #expect(semaphore.availablePermits == 0)
        semaphore.signal()
        try await blocked.value
        #expect(semaphore.availablePermits == 0)
        semaphore.signal()
        #expect(semaphore.availablePermits == 1)
    }

    @Test("A task cancelled before it asks for a permit gets none, even when one is free")
    func cancelledBeforeWaiting() async throws {
        let semaphore = AsyncSemaphore(permits: 2)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await semaphore.withPermit { "ran" }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(semaphore.availablePermits == 2)
    }

    @Test("withPermit returns its permit when the body throws")
    func withPermitReleasesOnThrow() async throws {
        struct Boom: Error {}
        let semaphore = AsyncSemaphore(permits: 1)
        await #expect(throws: Boom.self) {
            try await semaphore.withPermit { throw Boom() }
        }
        #expect(semaphore.availablePermits == 1)
    }

    @Test("Many waiters, half of them cancelled, leave exactly the permits they started with")
    func cancellationStress() async throws {
        let semaphore = AsyncSemaphore(permits: 3)
        let gate = ReadGate()
        let tasks = (0..<60).map { _ in
            Task {
                try await semaphore.withPermit { try await gate.pass() }
            }
        }
        try await Self.until { semaphore.waitingCount == 57 }
        for (index, task) in tasks.enumerated() where index.isMultiple(of: 2) {
            task.cancel()
        }
        gate.open()
        var finished = 0
        for task in tasks {
            if (try? await task.value) != nil {
                finished += 1
            }
        }
        #expect(finished == 30)
        #expect(semaphore.waitingCount == 0)
        #expect(semaphore.availablePermits == 3)
    }

    // MARK: Model time

    /// The quickstart of the policy recording, with options.
    private func request(_ name: String = "plain") throws -> SystemOneRequest {
        try PolicyFixtures.request(named: name)
    }

    @Test(
        "The recorder of a request gets exactly the decision's model time",
        .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    func recorderMatchesTheDecision() async throws {
        let stub = StubBackend(delay: .milliseconds(5))
        let engine = try DecisionEngine(backend: stub)
        var request = try request()
        request.samples = 3
        let recorder = ModelTimeRecorder()
        let decision = try await ModelTimeRecorder.$current.withValue(recorder) {
            try await engine.decide(request)
        }
        #expect(stub.reads.count == 3)
        #expect(recorder.total == decision.modelTime)
        #expect(decision.modelTime >= .milliseconds(15))

        let encoder = EncoderDecisionEngine(
            backend: StubQuestionReadBackend(delay: .milliseconds(5)),
            configuration: EncoderEngineConfiguration(batchSize: 1))
        let encoderRecorder = ModelTimeRecorder()
        let encoderDecision = try await ModelTimeRecorder.$current.withValue(encoderRecorder) {
            try await encoder.decide(try self.request())
        }
        #expect(encoderRecorder.total == encoderDecision.modelTime)
        #expect(encoderDecision.modelTime >= .milliseconds(15))
    }

    @Test(
        "A request refused after its thought reports the thought's time",
        .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    func refusalAfterAThought() async throws {
        // The recorded read after the thought is 191 tokens: the thought prompt fits 190, the
        // read does not, so the engine refuses after the backend has thought.
        let stub = StubBackend(maxPromptTokens: 190, delay: .milliseconds(20))
        let engine = try DecisionEngine(backend: stub)
        let recorder = ModelTimeRecorder()
        let error = await #expect(throws: SchemaError.self) {
            try await ModelTimeRecorder.$current.withValue(recorder) {
                try await engine.decide(try self.request("think_256"))
            }
        }
        #expect(error?.message == "the request is 191 tokens; the limit is 190")
        #expect(stub.thinks.count == 1)
        #expect(stub.reads.isEmpty)
        let thought = try #require(stub.callTimes.first)
        #expect(recorder.total >= thought)
        #expect(recorder.total - thought < .milliseconds(5))
    }

    @Test(
        "A sequential request whose second read is refused reports both reads",
        .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    func refusalAfterAFirstRead() async throws {
        let stub = StubBackend(
            delay: .milliseconds(20), failure: BackendRefusal(reason: "no"), succeedingCalls: 1)
        let engine = try DecisionEngine(backend: stub)
        let recorder = ModelTimeRecorder()
        await #expect(throws: BackendRefusal.self) {
            try await ModelTimeRecorder.$current.withValue(recorder) {
                try await engine.decide(try self.request("sequential_24_nouls"))
            }
        }
        #expect(stub.reads.count == 2)
        let spent = stub.callTimes.reduce(Duration.zero, +)
        #expect(stub.callTimes.count == 2)
        #expect(recorder.total >= spent)
        #expect(recorder.total - spent < .milliseconds(5))
    }

    @Test("An encoder request whose second batch fails reports both batches")
    func failureAfterAFirstBatch() async throws {
        let stub = StubQuestionReadBackend(
            delay: .milliseconds(10), failure: .throwing("boom"), succeedingBatches: 1)
        let engine = EncoderDecisionEngine(backend: stub)
        let questions = OrderedMap<Question>(
            uniqueKeysWithValues: (0..<20).map {
                ("q\($0)", Question.noul(instructions: nil, criteria: nil))
            })
        let request = SystemOneRequest(model: "laya-1.0", state: "s", questions: questions)
        let recorder = ModelTimeRecorder()
        await #expect(throws: StubQuestionReadBackend.StubError.self) {
            try await ModelTimeRecorder.$current.withValue(recorder) {
                try await engine.decide(request)
            }
        }
        #expect(stub.calls.count == 2)
        let spent = stub.callTimes.reduce(Duration.zero, +)
        #expect(recorder.total >= spent)
        #expect(recorder.total - spent < .milliseconds(5))
    }

    // MARK: Cancellation

    @Test(
        "A cancelled request cancels its read and frees its queue slot and its permit",
        .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    func cancellationReachesTheRead() async throws {
        let gate = ReadGate()
        let stub = StubBackend(gate: gate)
        let engine = try DecisionEngine(
            backend: stub, configuration: EngineConfiguration(maxInflight: 1, maxQueue: 1))
        let request = try request()
        let recorder = ModelTimeRecorder()
        let first = Task {
            try await ModelTimeRecorder.$current.withValue(recorder) {
                try await engine.decide(request)
            }
        }
        await gate.waitForArrivals(1)
        first.cancel()
        await #expect(throws: CancellationError.self) { try await first.value }
        #expect(gate.cancellations == 1)
        // The cancelled read still counts as model time.
        #expect(recorder.total > .zero)
        // The queue slot and the only permit are free: the next request runs.
        gate.open()
        let next = try await engine.decide(request)
        #expect(next.inputTokens == 123)
    }

    @Test(
        "An encoder request cancelled during a batch starts no further batch",
        .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    func encoderCancellation() async throws {
        let gate = ReadGate()
        let stub = StubQuestionReadBackend(gate: gate)
        let engine = EncoderDecisionEngine(
            backend: stub, configuration: EncoderEngineConfiguration(batchSize: 1, maxQueue: 1))
        let request = try request()
        let task = Task { try await engine.decide(request) }
        await gate.waitForArrivals(1)
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(stub.calls.count == 1)
        gate.open()
        _ = try await engine.decide(request)
        #expect(stub.calls.count == 4)
    }

    // MARK: Release

    @Test(
        "close() reaches a backend that releases its model, and nothing else",
        .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
    func release() async throws {
        let stub = StubBackend()
        let engine = try DecisionEngine(backend: stub)
        await engine.close()
        #expect(stub.closeCount == 1)
        let encoderStub = StubQuestionReadBackend()
        await EncoderDecisionEngine(backend: encoderStub).close()
        #expect(encoderStub.closeCount == 1)
    }

    /// Polls `condition` every millisecond for up to 60 seconds, and records an issue at the
    /// caller's line if it never holds.
    ///
    /// Only a failure waits that long: a condition that holds returns at once. The limit is
    /// generous because Swift Testing starts every test at once, so on a small CI runner a new
    /// task can wait seconds for a thread. On the Linux job, 60 tasks once took more than five
    /// seconds to reach the semaphore.
    static func until(
        sourceLocation: SourceLocation = #_sourceLocation, _ condition: () -> Bool
    ) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(60)
        while !condition() {
            guard clock.now < deadline else {
                Issue.record(
                    "the condition did not hold within 60 seconds", sourceLocation: sourceLocation)
                return
            }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
}
