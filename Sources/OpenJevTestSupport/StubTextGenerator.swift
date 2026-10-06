// Stand-ins for upstream OpenJev's MLX runtime in its chat tests (razorback16/openjev at dcd2094,
// `tests/test_mlx_backend.py`: `StubRuntime`, `ReplayRuntime` and `OneTokenRuntime`), behind the
// core's TextGenerator. Apache-2.0. See THIRD_PARTY.md.

import Foundation
import OpenJevCore

/// A ``/OpenJevCore/TextGenerator`` without a model, for the chat route's tests.
///
/// A generation runs ``Body``, which emits what it likes and returns a result: the factories give
/// upstream's stub runtimes (``upstreamReply(reply:tail:)``, ``replay(clean:leaky:)``,
/// ``oneToken()``), and a test can pass its own. Every call is recorded, with the number of calls
/// still running, so a test can tell that a generation stopped. A prompt is rendered by
/// `prompt`, ``syntheticPrompt(messages:thinking:)`` unless the test replays recorded ones, and a
/// stop string is encoded by `encode`.
public final class StubTextGenerator: TextGenerator, @unchecked Sendable {
    /// One recorded generation: its arguments, as upstream's stubs record them.
    public struct Call: Sendable, Hashable {
        /// The prompt ids.
        public var prompt: [Int]
        /// The token budget.
        public var maxTokens: Int
        /// The stop ids.
        public var stopIDs: [Int]
        /// The ids the detokenizer was asked to skip.
        public var skipSpecialTokenIDs: [Int]
    }

    /// What a generation does: emit with the given closure, then return the result.
    public typealias Body =
        @Sendable (_ call: Call, _ emit: @Sendable (_ text: String, _ token: Int?) -> Bool)
        async throws -> TextGeneration

    /// Upstream's `StubRuntime.REPLY` and `TAIL`.
    public static let reply = ["{\"city\"", ": \"", "Zurich", "\""]
    /// The segment `StubRuntime` emits with no token when it stops.
    public static let tail = "}"
    /// The DiffusionGemma tokenizer's thought-channel markers, `[100, 45518, 107, 101]`.
    public static let markers = [100, 45518, 107, 101]
    /// The DiffusionGemma tokenizer's scaffold, the same four ids.
    public static let scaffold = [100, 45518, 107, 101]

    public let maxPromptTokens: Int
    public let thoughtChannelMarkerIDs: [Int]
    /// The DiffusionGemma runtime's block, 256, unless the test says otherwise.
    public let blockLength: Int
    private let prompt: @Sendable (_ messages: [JSONValue], _ thinking: Bool) throws -> [Int]
    private let encoding: @Sendable (_ text: String) throws -> [Int]
    private let body: Body

    private let lock = NSLock()
    private var recorded: [Call] = []
    private var active = 0
    private var promptRequests: [(messages: [JSONValue], thinking: Bool)] = []

    /// Creates a stub; by default it answers as upstream's `StubRuntime`.
    public init(
        maxPromptTokens: Int = 32768,
        markers: [Int] = StubTextGenerator.markers,
        blockLength: Int = 256,
        prompt: @escaping @Sendable (_ messages: [JSONValue], _ thinking: Bool) throws -> [Int] =
            StubTextGenerator.syntheticPrompt,
        encode: @escaping @Sendable (_ text: String) throws -> [Int] =
            StubTextGenerator.syntheticEncoding,
        generate body: @escaping Body = StubTextGenerator.upstreamReply()
    ) {
        self.maxPromptTokens = maxPromptTokens
        self.thoughtChannelMarkerIDs = markers
        self.blockLength = blockLength
        self.prompt = prompt
        self.encoding = encode
        self.body = body
    }

    /// Every generation so far, in call order.
    public var calls: [Call] {
        lock.withLock { recorded }
    }

    /// The generations running now.
    public var running: Int {
        lock.withLock { active }
    }

    /// Every prompt rendered so far: the messages and the thinking switch.
    public var renderedPrompts: [(messages: [JSONValue], thinking: Bool)] {
        lock.withLock { promptRequests }
    }

    public func generationPromptIDs(messages: [JSONValue], thinking: Bool) async throws -> [Int] {
        lock.withLock { promptRequests.append((messages, thinking)) }
        return try prompt(messages, thinking)
    }

    public func encode(_ text: String) throws -> [Int] {
        try encoding(text)
    }

    public func generate(
        prompt: [Int], maxTokens: Int, stopIDs: [Int], skipSpecialTokenIDs: [Int],
        emit: @Sendable (_ text: String, _ token: Int?) -> Bool
    ) async throws -> TextGeneration {
        let call = Call(
            prompt: prompt, maxTokens: maxTokens, stopIDs: stopIDs,
            skipSpecialTokenIDs: skipSpecialTokenIDs)
        lock.withLock {
            recorded.append(call)
            active += 1
        }
        defer { lock.withLock { active -= 1 } }
        return try await body(call, emit)
    }

    /// Upstream's `StubRuntime.generate`: each piece of `reply` as token 1000 and up, stopping
    /// before a token in the stop ids or once `maxTokens` are generated (`length`), then `tail`
    /// with no token; `stop` when the reply ran out. An `emit` that returns false cancels.
    public static func upstreamReply(
        reply: [String] = StubTextGenerator.reply, tail: String = StubTextGenerator.tail
    ) -> Body {
        { call, emit in
            var generated: [Int] = []
            var finish = TextGeneration.FinishReason.stop
            for (index, text) in reply.enumerated() {
                if generated.count >= call.maxTokens {
                    finish = .length
                    break
                }
                let token = 1000 + index
                if call.stopIDs.contains(token) {
                    break
                }
                generated.append(token)
                if !emit(text, token) {
                    return TextGeneration(
                        generated: generated, promptTokens: call.prompt.count,
                        finishReason: .cancelled)
                }
            }
            _ = emit(tail, nil)
            return TextGeneration(
                generated: generated, promptTokens: call.prompt.count, finishReason: finish)
        }
    }

    /// The segments upstream recorded from a reply that opened a thought channel of its own
    /// accord: the markers reach the detokenizer fused into a later segment.
    public static let leakyEmits: [(token: Int?, text: String)] = [
        (100, ""), (45518, ""), (107, ""), (101, ""), (34699, ""),
        (6819, "<|channel>thought\n<channel|>six"), (6589, " seven"), (10155, " eight"),
        (3595, " nine"), (nil, " ten"),
    ]

    /// The same reply with the markers skipped, as the real generator gives it.
    public static let cleanEmits: [(token: Int?, text: String)] = [
        (34699, ""), (6819, "six"), (6589, " seven"), (10155, " eight"), (3595, " nine"),
        (nil, " ten"),
    ]

    /// Upstream's `ReplayRuntime`: ``cleanEmits`` when the call skips the four ``markers``,
    /// ``leakyEmits`` otherwise, then `stop`.
    public static func replay(
        clean: [(token: Int?, text: String)] = StubTextGenerator.cleanEmits,
        leaky: [(token: Int?, text: String)] = StubTextGenerator.leakyEmits
    ) -> Body {
        { call, emit in
            let skipping = Set(markers).isSubset(of: call.skipSpecialTokenIDs)
            var generated: [Int] = []
            for (token, text) in skipping ? clean : leaky {
                guard let token else {
                    _ = emit(text, nil)
                    break
                }
                generated.append(token)
                if !emit(text, token) {
                    return TextGeneration(
                        generated: generated, promptTokens: call.prompt.count,
                        finishReason: .cancelled)
                }
            }
            return TextGeneration(
                generated: generated, promptTokens: call.prompt.count, finishReason: .stop)
        }
    }

    /// Upstream's `OneTokenRuntime`: one token with no text of its own, then `7` with no token.
    public static func oneToken() -> Body {
        { call, emit in
            _ = emit("", 1000)
            _ = emit("7", nil)
            return TextGeneration(
                generated: [1000], promptTokens: call.prompt.count, finishReason: .stop)
        }
    }

    /// A prompt without a tokenizer, its length growing with the messages: `<bos>` (2), then for
    /// each message `<|turn>` (105), one id per word of its text and `<turn|>` (106), then
    /// `<|turn>model\n` (105, 4368, 107), with `<|think|>` (98) first when `thinking`, and the
    /// scaffold.
    @Sendable public static func syntheticPrompt(
        messages: [JSONValue], thinking: Bool
    ) throws -> [Int] {
        var ids = [2]
        if thinking {
            ids.append(98)
        }
        for message in messages {
            ids.append(105)
            var texts: [String] = []
            switch message["content"] {
            case .string(let text)?:
                texts = [text]
            case .array(let parts)?:
                texts = parts.compactMap { $0["text"]?.stringValue }
            default:
                break
            }
            for word in texts.flatMap({ $0.split(whereSeparator: \.isWhitespace) }) {
                ids.append(5000 + word.unicodeScalars.count)
            }
            ids.append(106)
        }
        return ids + [105, 4368, 107] + scaffold
    }

    /// An encoding without a tokenizer: one id per Unicode scalar, its value. A stop of one
    /// character is one token; a longer one is dropped.
    @Sendable public static func syntheticEncoding(_ text: String) throws -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }
}
