// A port of upstream OpenJev (razorback16/openjev at dcd2094), the response shapes of
// `openjev/chat.py`: `MlxGenerator.complete`'s body, and `completion_id`, `finish_reason`, `usage`,
// `chunk` and `sse`. Apache-2.0. See THIRD_PARTY.md.

import Foundation

/// The id and creation time a reply carries: upstream's `completion_id()` and
/// `int(time.time())`.
public struct ChatCompletionIdentity: Sendable, Hashable {
    /// `chatcmpl-` and 24 lowercase hexadecimal characters.
    public var id: String
    /// Seconds since 1970, rounded down.
    public var created: Int

    /// Creates an identity.
    public init(id: String, created: Int) {
        self.id = id
        self.created = created
    }

    /// A fresh identity, as upstream draws one per reply: `"chatcmpl-" + secrets.token_hex(12)`
    /// from the system's random source, and the current time.
    public static func random() -> ChatCompletionIdentity {
        var generator = SystemRandomNumberGenerator()
        let bytes = (0..<12).map { _ in UInt8.random(in: .min ... .max, using: &generator) }
        let hex = bytes.map { byte in
            let digits = String(byte, radix: 16)
            return digits.count == 1 ? "0" + digits : digits
        }
        return ChatCompletionIdentity(
            id: "chatcmpl-" + hex.joined(), created: Int(Date().timeIntervalSince1970))
    }
}

/// The token counts a reply bills, upstream's `usage`.
public struct ChatCompletionUsage: Sendable, Hashable, WireEncodable {
    /// The prompt tokens the generation processed.
    public var promptTokens: Int
    /// The tokens generated, the stop token not included.
    public var completionTokens: Int

    /// Creates the counts.
    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    /// `{"prompt_tokens", "completion_tokens", "total_tokens"}`.
    public var json: JSONValue {
        [
            "prompt_tokens": JSONValue(promptTokens),
            "completion_tokens": JSONValue(completionTokens),
            "total_tokens": JSONValue(promptTokens + completionTokens),
        ]
    }
}

extension TextGeneration.FinishReason {
    /// The `finish_reason` a reply reports, upstream's `finish_reason`: `stop` for a stop, and
    /// `length` for anything else, a cancelled generation included.
    public var wireValue: String {
        self == .stop ? "stop" : "length"
    }
}

/// The body of a whole reply, `MlxGenerator.complete`'s dictionary, which FastAPI renders as
/// ``WireEncoder`` does.
public struct ChatCompletion: Sendable, Hashable, WireEncodable {
    /// The reply's id and creation time.
    public var identity: ChatCompletionIdentity
    /// The reply's text: everything the generation emitted, or in JSON mode its first JSON
    /// object or array (``ExtractJSON``).
    public var content: String
    /// Why the generation ended.
    public var finishReason: TextGeneration.FinishReason
    /// The billed tokens.
    public var usage: ChatCompletionUsage

    /// Creates a reply.
    public init(
        identity: ChatCompletionIdentity, content: String,
        finishReason: TextGeneration.FinishReason, usage: ChatCompletionUsage
    ) {
        self.identity = identity
        self.content = content
        self.finishReason = finishReason
        self.usage = usage
    }

    /// `{"id", "object": "chat.completion", "created", "model", "choices": [...], "usage"}`, in
    /// upstream's key order.
    public var json: JSONValue {
        [
            "id": .string(identity.id),
            "object": "chat.completion",
            "created": JSONValue(identity.created),
            "model": .string(ChatCompletionRequest.modelName),
            "choices": [
                [
                    "index": 0,
                    "finish_reason": .string(finishReason.wireValue),
                    "message": ["role": "assistant", "content": .string(content)],
                    "logprobs": nil,
                ]
            ],
            "usage": usage.json,
        ]
    }
}

/// The server-sent events of a streamed reply, upstream's `chunk` and `sse`: each event is
/// `data: ` and `json.dumps(payload, ensure_ascii=False)`, with Python's default `, ` and `: `
/// separators, and a blank line.
public enum ChatCompletionEvent {
    /// The first event: the assistant's role with empty content.
    public static func role(_ identity: ChatCompletionIdentity) -> String {
        chunk(identity, delta: ["role": "assistant", "content": ""], finishReason: nil)
    }

    /// A piece of the reply's text.
    public static func content(_ identity: ChatCompletionIdentity, _ text: String) -> String {
        chunk(identity, delta: ["content": .string(text)], finishReason: nil)
    }

    /// The event after the last piece: an empty delta with the finish reason.
    public static func finish(
        _ identity: ChatCompletionIdentity, _ reason: TextGeneration.FinishReason
    ) -> String {
        chunk(identity, delta: [:], finishReason: reason)
    }

    /// The usage event, which has no choices.
    public static func usage(
        _ identity: ChatCompletionIdentity, _ usage: ChatCompletionUsage
    ) -> String {
        event([
            "id": .string(identity.id), "object": "chat.completion.chunk",
            "created": JSONValue(identity.created),
            "model": .string(ChatCompletionRequest.modelName), "choices": [],
            "usage": usage.json,
        ])
    }

    /// The last event of a complete stream.
    public static let done = "data: [DONE]\n\n"

    /// `chunk(cid, created, delta, finish_reason)`.
    static func chunk(
        _ identity: ChatCompletionIdentity, delta: JSONObject,
        finishReason: TextGeneration.FinishReason?
    ) -> String {
        event([
            "id": .string(identity.id), "object": "chat.completion.chunk",
            "created": JSONValue(identity.created),
            "model": .string(ChatCompletionRequest.modelName),
            "choices": [
                [
                    "index": 0, "delta": .object(delta),
                    "finish_reason": finishReason.map { .string($0.wireValue) } ?? .null,
                    "logprobs": nil,
                ]
            ],
        ])
    }

    /// `sse(payload)`.
    static func event(_ payload: JSONValue) -> String {
        // A payload of strings and integers can always be written.
        let text = (try? PythonJSONWriter(options: .init(ensureASCII: false)).string(payload)) ?? ""
        return "data: " + text + "\n\n"
    }
}
