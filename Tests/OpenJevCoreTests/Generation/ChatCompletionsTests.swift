import Foundation
import OpenJevTestSupport
import Testing

@testable import OpenJevCore

/// A count shared between a stub and its test.
final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    var count: Int {
        lock.withLock { value }
    }

    func increment() {
        lock.withLock { value += 1 }
    }
}

/// The capacity count of a chat service each time its stub renders a prompt.
final class RenderObserver: @unchecked Sendable {
    private let lock = NSLock()
    private var service: ChatCompletions?
    private var seen: [Int] = []

    /// The service whose count is recorded; set once it exists, and cleared to break the cycle
    /// through its generator.
    var chat: ChatCompletions? {
        get { lock.withLock { service } }
        set { lock.withLock { service = newValue } }
    }

    /// The counts recorded, in order.
    var counts: [Int] {
        lock.withLock { seen }
    }

    /// Records the service's count now, or -1 without a service.
    func record() {
        let running = chat?.running ?? -1
        lock.withLock { seen.append(running) }
    }
}

/// Upstream's `MlxGenerator` semantics over a stub generator, without HTTP: the capacity bound
/// and the slots, the whole reply, and the stream's guarantees as upstream's
/// `tests/test_mlx_backend.py` pins them, driven the way those tests drive `gen.stream` (issue #53).
@Suite("Chat completions over a stub generator")
struct ChatCompletionsTests {
    /// Upstream's `CHAT` request.
    static let chat: JSONValue = [
        "model": "diffusiongemma-26b", "messages": [["role": "user", "content": "Which city?"]],
    ]

    /// `CHAT` with `stream: true`.
    static let streamed: JSONValue = [
        "model": "diffusiongemma-26b", "messages": [["role": "user", "content": "Which city?"]],
        "stream": true,
    ]

    /// The ids of the synthetic prompt of `CHAT`.
    static func chatPromptTokens() throws -> Int {
        try StubTextGenerator.syntheticPrompt(
            messages: [["role": "user", "content": "Which city?"]], thinking: false
        ).count
    }

    /// The text of the content events, joined, and whether any event was `[DONE]`.
    static func content(_ events: [String]) throws -> (text: String, done: Bool) {
        var text = ""
        var done = false
        for event in events {
            #expect(event.hasPrefix("data: ") && event.hasSuffix("\n\n"), "\(event)")
            let payload = String(event.dropFirst(6).dropLast(2))
            if payload == "[DONE]" {
                done = true
                continue
            }
            let value = try JSONParser().parse(payload)
            text += value["choices"]?[0]?["delta"]?["content"]?.stringValue ?? ""
        }
        return (text, done)
    }

    /// A generation of `count` pieces, `prefix` and the index, with `pause` between them; it
    /// stops when `emit` says so and counts the pieces it emitted in `emitted`.
    static func counting(
        _ count: Int, prefix: String, pause: Duration? = nil, emitted: Counter
    ) -> StubTextGenerator.Body {
        { call, emit in
            for index in 0..<count {
                // The real generator stops at its next block when its task is cancelled.
                if Task.isCancelled {
                    return TextGeneration(
                        generated: [1000], promptTokens: call.prompt.count,
                        finishReason: .cancelled)
                }
                emitted.increment()
                if !emit("\(prefix)\(index)", 1000 + index) {
                    return TextGeneration(
                        generated: [1000], promptTokens: call.prompt.count,
                        finishReason: .cancelled)
                }
                if let pause {
                    // A cancelled sleep ends early; the next emit then says to stop.
                    try? await Task.sleep(for: pause)
                }
            }
            return TextGeneration(
                generated: Array(1000..<(1000 + count)), promptTokens: call.prompt.count,
                finishReason: .stop)
        }
    }

    /// Polls `condition` every millisecond for up to a minute.
    static func until(
        sourceLocation: SourceLocation = #_sourceLocation, _ condition: () -> Bool
    ) async throws {
        try await CapacityTests.until(sourceLocation: sourceLocation, condition)
    }

    // MARK: The stream

    /// 64 pieces of slack, then the reply ends: gap-free, and without `[DONE]`.
    @Test("test_a_slow_reader_cancels_rather_than_loses_chunks")
    func slowReader() async throws {
        let emitted = Counter()
        let generator = StubTextGenerator(
            generate: Self.counting(500, prefix: " ", emitted: emitted))
        let chat = ChatCompletions(generator: generator)
        let stream = try await chat.stream(try await chat.prepare(Self.streamed))
        var events: [String] = []
        let ending = try await stream.run { event in
            events.append(event)
            if events.count == 10 {
                // Let the generation outrun the reader and fill the queue.
                try await Task.sleep(for: .milliseconds(200))
            }
        }
        #expect(ending == .readerFellBehind)
        let (text, done) = try Self.content(events)
        #expect(!done, "a cancelled reply must not look complete")
        let numbers = text.split(separator: " ").compactMap { Int($0) }
        #expect(!numbers.isEmpty && numbers.count < 500)
        // A prefix of the reply with nothing missing: the piece that found the queue full ended
        // the reply instead of being dropped from the middle of it.
        #expect(numbers == Array(0..<numbers.count))
        #expect(numbers.count >= ChatCompletionStream.bufferCapacity)
        #expect(emitted.count < 500)
        #expect(generator.running == 0)
        #expect(chat.running == 0 && chat.freeSlots == 8)
    }

    @Test("A cancelled stream stops its generation at the next emit and writes nothing more")
    func cancelledStream() async throws {
        let emitted = Counter()
        let generator = StubTextGenerator(
            generate: Self.counting(200, prefix: "t", pause: .milliseconds(20), emitted: emitted))
        let chat = ChatCompletions(generator: generator)
        let stream = try await chat.stream(try await chat.prepare(Self.streamed))
        let written = TextParts()
        let pieces = Counter()
        await #expect(throws: CancellationError.self) {
            _ = try await stream.run { event in
                written.append(event)
                pieces.increment()
                if pieces.count == 3 {
                    stream.cancel()
                }
            }
        }
        #expect(emitted.count < 200, "the generation ran on after the stream was cancelled")
        #expect(!written.joined().contains("[DONE]"))
        #expect(!written.joined().contains("finish_reason\": \"stop"))
        #expect(generator.running == 0)
        #expect(chat.running == 0 && chat.freeSlots == 8)
    }

    @Test("Cancelling the task that runs a stream stops the generation and frees the slot")
    func cancelledTask() async throws {
        let emitted = Counter()
        let generator = StubTextGenerator(
            generate: Self.counting(200, prefix: "t", pause: .milliseconds(20), emitted: emitted))
        let chat = ChatCompletions(generator: generator)
        let stream = try await chat.stream(try await chat.prepare(Self.streamed))
        let started = Counter()
        let run = Task {
            try await stream.run { _ in started.increment() }
        }
        try await Self.until { started.count >= 3 }
        run.cancel()
        await #expect(throws: CancellationError.self) { _ = try await run.value }
        #expect(emitted.count < 200)
        #expect(generator.running == 0)
        #expect(chat.running == 0 && chat.freeSlots == 8)
    }

    /// The last segment, flushed with the end, is never lost.
    @Test("test_a_one_token_reply_still_streams")
    func oneToken() async throws {
        let chat = ChatCompletions(
            generator: StubTextGenerator(generate: StubTextGenerator.oneToken()))
        for _ in 0..<20 {
            let stream = try await chat.stream(try await chat.prepare(Self.streamed))
            var events: [String] = []
            let ending = try await stream.run { events.append($0) }
            #expect(
                ending
                    == .completed(
                        TextGeneration(
                            generated: [1000], promptTokens: try Self.chatPromptTokens(),
                            finishReason: .stop)))
            let (text, done) = try Self.content(events)
            #expect(text == "7")
            #expect(done)
        }
    }

    @Test("The events come in upstream's order: role, pieces, finish, usage, [DONE]")
    func eventOrder() async throws {
        let chat = ChatCompletions(
            generator: StubTextGenerator(), configuration: .init(),
            identity: { ChatCompletionIdentity(id: "chatcmpl-x", created: 7) })
        let stream = try await chat.stream(try await chat.prepare(Self.streamed))
        var events: [String] = []
        _ = try await stream.run { events.append($0) }
        let prefix =
            #"data: {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 7, "#
            + #""model": "diffusiongemma-26b", "#
        #expect(events.count == 9)
        #expect(
            events.first
                == prefix
                + #""choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "#
                + #""finish_reason": null, "logprobs": null}]}"# + "\n\n")
        #expect(
            events[6]
                == prefix
                + #""choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "#
                + #""logprobs": null}]}"# + "\n\n")
        let tokens = try Self.chatPromptTokens()
        #expect(
            events[7]
                == prefix
                + #""choices": [], "usage": {"prompt_tokens": \#(tokens), "completion_tokens": 4, "#
                + #""total_tokens": \#(tokens + 4)}}"# + "\n\n")
        #expect(events.last == "data: [DONE]\n\n")
        #expect(try Self.content(events).text == #"{"city": "Zurich"}"#)
    }

    @Test("A generation that fails mid-stream breaks the stream off without a finish chunk")
    func failingGeneration() async throws {
        struct Boom: Error {}
        let generator = StubTextGenerator(generate: { _, emit in
            _ = emit("partial", 1000)
            throw Boom()
        })
        let chat = ChatCompletions(generator: generator)
        let stream = try await chat.stream(try await chat.prepare(Self.streamed))
        var events: [String] = []
        await #expect(throws: ChatCompletionStream.GenerationFailed.self) {
            _ = try await stream.run { events.append($0) }
        }
        let (text, done) = try Self.content(events)
        #expect(text == "partial")
        #expect(!done)
        #expect(events.count == 2)
        #expect(chat.running == 0 && chat.freeSlots == 8)
    }

    @Test("A stream that is never run gives its place and its slot back")
    func unrunStream() async throws {
        let chat = ChatCompletions(
            generator: StubTextGenerator(), configuration: .init(maxInflight: 1, maxQueue: 0))
        do {
            let stream = try await chat.stream(try await chat.prepare(Self.streamed))
            #expect(chat.running == 1 && chat.freeSlots == 0)
            // The bound is reached: one more is refused before anything is counted.
            await #expect(throws: ChatCompletionError.overloaded) {
                _ = try await chat.prepare(Self.streamed)
            }
            withExtendedLifetime(stream) {}
        }
        #expect(chat.running == 0 && chat.freeSlots == 1)
        // And the place is usable again.
        let stream = try await chat.stream(try await chat.prepare(Self.streamed))
        _ = try await stream.run { _ in }
        #expect(chat.running == 0 && chat.freeSlots == 1)
    }

    // MARK: Capacity

    /// A wait cancelled before its slot gives its place back.
    @Test("test_a_cancelled_wait_for_a_chat_slot_leaks_no_capacity")
    func cancelledWait() async throws {
        let gate = ReadGate()
        let generator = StubTextGenerator(generate: { call, emit in
            try await gate.pass()
            return try await StubTextGenerator.upstreamReply()(call, emit)
        })
        let chat = ChatCompletions(
            generator: generator, configuration: .init(maxInflight: 1, maxQueue: 32))
        let holder = try await chat.stream(try await chat.prepare(Self.streamed))
        #expect(chat.running == 1 && chat.freeSlots == 0)
        var waits: [Task<ChatCompletionStream, any Error>] = []
        for _ in 0..<3 {
            let prepared = try await chat.prepare(Self.streamed)
            waits.append(Task { try await chat.stream(prepared) })
        }
        #expect(chat.running == 4)
        try await Self.until { chat.capacity.slots.waitingCount == 3 }
        for wait in waits {
            wait.cancel()
        }
        for wait in waits {
            await #expect(throws: CancellationError.self) { _ = try await wait.value }
        }
        #expect(chat.running == 1)
        gate.open()
        _ = try await holder.run { _ in }
        #expect(chat.running == 0 && chat.freeSlots == 1)
    }

    @Test("A whole reply is cancelled with its task, and its place comes back")
    func cancelledWholeReply() async throws {
        let emitted = Counter()
        let generator = StubTextGenerator(
            generate: Self.counting(200, prefix: "t", pause: .milliseconds(20), emitted: emitted))
        let chat = ChatCompletions(generator: generator)
        let prepared = try await chat.prepare(Self.chat)
        let reply = Task { try await chat.complete(prepared) }
        try await Self.until { emitted.count >= 2 }
        reply.cancel()
        // A whole reply's emit never asks to stop: the generation ends through its task.
        let ended = try await reply.value
        #expect(ended.finishReason == .cancelled)
        #expect(emitted.count < 200)
        #expect(chat.running == 0 && chat.freeSlots == 8)
    }

    @Test("The bound counts prepared requests: inflight plus queue, then the 529")
    func bound() async throws {
        let gate = ReadGate()
        let generator = StubTextGenerator(generate: { call, emit in
            try await gate.pass()
            return try await StubTextGenerator.upstreamReply()(call, emit)
        })
        let chat = ChatCompletions(
            generator: generator, configuration: .init(maxInflight: 1, maxQueue: 1))
        let first = Task { try await chat.complete(try await chat.prepare(Self.chat)) }
        await gate.waitForArrivals(1)
        // A prepared request holds its place: it waits for its turn and is never refused.
        let prepared = try await chat.prepare(Self.chat)
        #expect(chat.running == 2)
        await #expect(throws: ChatCompletionError.overloaded) {
            _ = try await chat.prepare(Self.chat)
        }
        #expect(generator.renderedPrompts.count == 2, "a refused request rendered its prompt")
        let second = Task { try await chat.complete(prepared) }
        gate.open()
        _ = try await first.value
        _ = try await second.value
        #expect(chat.running == 0 && chat.freeSlots == 1)
    }

    /// Upstream renders a prompt on its event loop, so nothing else runs between its check of
    /// the bound and its count; here prompts render concurrently, so a request is counted in
    /// before its prompt renders, and the bound bounds how many render at once.
    @Test("A request is counted in before its prompt renders, and refused before rendering")
    func countedBeforeRendering() async throws {
        let observer = RenderObserver()
        let chat = ChatCompletions(
            generator: StubTextGenerator(prompt: { messages, thinking in
                observer.record()
                return try StubTextGenerator.syntheticPrompt(messages: messages, thinking: thinking)
            }), configuration: .init(maxInflight: 1, maxQueue: 0))
        observer.chat = chat
        defer { observer.chat = nil }
        _ = try await chat.complete(try await chat.prepare(Self.chat))
        #expect(observer.counts == [1])
        let release = try Self.occupy(chat)
        await #expect(throws: ChatCompletionError.overloaded) {
            _ = try await chat.prepare(Self.chat)
        }
        release()
        #expect(observer.counts == [1], "a refused request rendered its prompt")
    }

    @Test("A request refused after it was counted in gives its place back")
    func refusedAfterCounting() async throws {
        struct Unencodable: Error {}
        let tokens = try Self.chatPromptTokens()
        let short = ChatCompletions(
            generator: StubTextGenerator(maxPromptTokens: tokens - 1),
            configuration: .init(maxInflight: 1, maxQueue: 0))
        let unencodable = ChatCompletions(
            generator: StubTextGenerator(encode: { _ in throw Unencodable() }),
            configuration: .init(maxInflight: 1, maxQueue: 0))
        let unrenderable = ChatCompletions(
            generator: StubTextGenerator(prompt: { _, _ in throw Unencodable() }),
            configuration: .init(maxInflight: 1, maxQueue: 0))
        var body = try #require(Self.chat.objectValue)
        body.updateValue(true, forKey: "max_tokens")
        let refusals: [(ChatCompletions, JSONValue, ChatCompletionError)] = [
            (short, .object(body), .maxTokens(true)),
            (short, Self.chat, .promptTooLong(tokens: tokens, limit: tokens - 1)),
            (unencodable, Self.chat(stop: "x"), .backendUnavailable("Unencodable")),
            (
                unrenderable, Self.chat,
                .invalidRequest(
                    "The messages could not be rendered with the model's chat template: "
                        + "Unencodable()")
            ),
        ]
        for (chat, request, refusal) in refusals {
            // Twice: with one place, a place kept by the first would make the second a 529.
            for _ in 0..<2 {
                await #expect(throws: refusal) {
                    _ = try await chat.prepare(request)
                }
            }
            #expect(chat.running == 0, "\(refusal)")
        }
    }

    @Test("A prepared request that is never answered gives its place back")
    func discardedPreparation() async throws {
        let chat = ChatCompletions(
            generator: StubTextGenerator(), configuration: .init(maxInflight: 1, maxQueue: 0))
        do {
            let prepared = try await chat.prepare(Self.chat)
            #expect(chat.running == 1 && prepared.prompt.count > 0)
        }
        #expect(chat.running == 0 && chat.freeSlots == 1)
        _ = try await chat.complete(try await chat.prepare(Self.chat))
        #expect(chat.running == 0 && chat.freeSlots == 1)
    }

    /// Python's sum of the two settings cannot overflow.
    @Test("A bound past Int.max is no bound, not a crash")
    func unboundedSettings() async throws {
        let chat = ChatCompletions(
            generator: StubTextGenerator(),
            configuration: .init(maxInflight: .max, maxQueue: .max))
        #expect(chat.capacity.limit == .max)
        let reply = try await chat.complete(try await chat.prepare(Self.chat))
        #expect(reply.content == #"{"city": "Zurich"}"#)
        #expect(chat.running == 0)
    }

    /// Counts one request in, as upstream's tests set `running`, and returns a function that
    /// counts it out again.
    static func occupy(_ chat: ChatCompletions) throws -> @Sendable () -> Void {
        try chat.capacity.admit()
        return { chat.capacity.leave() }
    }

    /// `CHAT` with a `stop`.
    static func chat(stop: JSONValue) -> JSONValue {
        var body = chat.objectValue ?? JSONObject()
        body.updateValue(stop, forKey: "stop")
        return .object(body)
    }

    // MARK: The whole reply and preparation

    @Test("A whole reply joins every emitted text, the last segment included, and bills usage")
    func wholeReply() async throws {
        let generator = StubTextGenerator()
        let chat = ChatCompletions(generator: generator)
        let reply = try await chat.complete(try await chat.prepare(Self.chat))
        #expect(reply.content == #"{"city": "Zurich"}"#)
        #expect(reply.finishReason == .stop)
        #expect(
            reply.usage
                == ChatCompletionUsage(
                    promptTokens: try Self.chatPromptTokens(), completionTokens: 4))
        let call = try #require(generator.calls.first)
        #expect(call.maxTokens == 1024)
        #expect(call.stopIDs == [])
        #expect(call.skipSpecialTokenIDs == [100, 45518, 107, 101])
        let prompt = try StubTextGenerator.syntheticPrompt(
            messages: [["role": "user", "content": "Which city?"]], thinking: false)
        #expect(call.prompt == prompt)
    }

    @Test("JSON mode reduces a whole reply to its JSON value; a stream is not reduced")
    func jsonMode() async throws {
        let generator = StubTextGenerator(
            generate: StubTextGenerator.upstreamReply(
                reply: ["Sure!\n```json\n", "{\"city\":", " \"Zurich\"}", "\n```"], tail: ""))
        let chat = ChatCompletions(generator: generator)
        var body = try #require(Self.chat.objectValue)
        body.updateValue(["type": "json_object"], forKey: "response_format")
        let reply = try await chat.complete(try await chat.prepare(.object(body)))
        #expect(reply.content == #"{"city": "Zurich"}"#)
        body.updateValue(true, forKey: "stream")
        let stream = try await chat.stream(try await chat.prepare(.object(body)))
        var events: [String] = []
        _ = try await stream.run { events.append($0) }
        #expect(try Self.content(events).text == "Sure!\n```json\n{\"city\": \"Zurich\"}\n```")
    }

    @Test("Only single-token stop strings become stop ids, read after the prompt's limit")
    func stops() async throws {
        let tokens = try Self.chatPromptTokens()
        let generator = StubTextGenerator(maxPromptTokens: tokens - 1)
        let chat = ChatCompletions(generator: generator)
        var body = try #require(Self.chat.objectValue)
        body.updateValue(["x", "yz", "\n"], forKey: "stop")
        // The limit's 400 comes before stop is read.
        let tooLong = ChatCompletionError.promptTooLong(tokens: tokens, limit: tokens - 1)
        await #expect(throws: tooLong) {
            _ = try await chat.prepare(.object(body))
        }
        let roomy = ChatCompletions(generator: StubTextGenerator())
        #expect(try await roomy.prepare(.object(body)).stopIDs == [120, 10])
        body.updateValue(5, forKey: "stop")
        await #expect(
            throws: ChatCompletionError.invalidRequest(
                "stop must be a string or an array of strings.")
        ) {
            _ = try await roomy.prepare(.object(body))
        }
        body.updateValue(["END": 1, "x": 2], forKey: "stop")
        #expect(try await roomy.prepare(.object(body)).stopIDs == [120])
    }

    @Test("A template that cannot render the messages is this port's 400")
    func renderingFailure() async throws {
        struct Broken: Error, CustomStringConvertible {
            var description: String { "no such filter" }
        }
        let chat = ChatCompletions(generator: StubTextGenerator(prompt: { _, _ in throw Broken() }))
        await #expect(
            throws: ChatCompletionError.invalidRequest(
                "The messages could not be rendered with the model's chat template: "
                    + "no such filter")
        ) {
            _ = try await chat.prepare(Self.chat)
        }
    }
}
