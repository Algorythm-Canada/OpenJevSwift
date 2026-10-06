// A port of upstream OpenJev (razorback16/openjev at dcd2094), `oai_error` and the refusals of
// `openjev/chat.py`: the route's checks, `Generator.normalize`'s `max_tokens` error, the prompt
// limit and the capacity bound. Apache-2.0. See THIRD_PARTY.md.

/// An error answer of `POST /v1/chat/completions`, in OpenAI's shape
/// `{"error": {"message", "type", "code"}}`, upstream's `oai_error`.
///
/// The static members build upstream's refusals with its exact messages. The members marked as
/// this port's own answer requests on which upstream's route raises and Starlette answers a bare
/// 500 (D-058).
public struct ChatCompletionError: Error, Sendable, Hashable, WireEncodable {
    /// The HTTP status code.
    public var status: Int
    /// The message.
    public var message: String
    /// `invalid_request_error`, `overloaded_error` or `api_error`.
    public var type: String
    /// `model_not_found` for an unknown model, otherwise `nil`, which is written as `null`.
    public var code: String?
    /// Headers the error adds, such as `retry-after`.
    public var headers: [WireError.Header]

    /// Creates an error answer.
    public init(
        status: Int, message: String, type: String = "invalid_request_error",
        code: String? = nil, headers: [WireError.Header] = []
    ) {
        self.status = status
        self.message = message
        self.type = type
        self.code = code
        self.headers = headers
    }

    /// `{"error": {"message", "type", "code"}}`.
    public var json: JSONValue {
        [
            "error": [
                "message": .string(message), "type": .string(type),
                "code": code.map(JSONValue.string) ?? .null,
            ]
        ]
    }

    /// A 400 `invalid_request_error` with any message.
    public static func invalidRequest(_ message: String) -> ChatCompletionError {
        ChatCompletionError(status: 400, message: message)
    }

    /// The 400 for a body `json.loads` refuses, empty bodies included.
    public static let notJSON = invalidRequest("The request body is not valid JSON.")

    /// The 400 for a body that is not an object, or whose `messages` is missing, not an array or
    /// empty.
    public static let messagesRequired = invalidRequest("messages must be a non-empty array.")

    /// The 400 for a `model` that is missing or not a string.
    public static let modelRequired = invalidRequest("model is required and must be a string.")

    /// The 404 `model_not_found` for a model the route does not serve:
    /// `Model {model!r} not found. Available: diffusiongemma-26b.`
    public static func modelNotFound(_ model: String) -> ChatCompletionError {
        ChatCompletionError(
            status: 404,
            message:
                "Model \(model.pythonRepr) not found. Available: \(ChatCompletionRequest.modelName).",
            code: "model_not_found")
    }

    /// The 400 for a `max_tokens` (or `max_completion_tokens`) that is not a positive integer, with
    /// the value as Python's `repr` writes it.
    public static func maxTokens(_ value: JSONValue) -> ChatCompletionError {
        invalidRequest("max_tokens must be a positive integer, got \(value.pythonRepr)")
    }

    /// The 400 for a prompt longer than `OPENJEV_MLX_MAX_PROMPT`, scaffold included.
    public static func promptTooLong(tokens: Int, limit: Int) -> ChatCompletionError {
        invalidRequest("the request is \(tokens) tokens; the limit is \(limit)")
    }

    /// The message of the 529.
    public static let overloadedMessage = "Text generation is at capacity. Retry shortly."

    /// The 529 `overloaded_error` with `retry-after: 2`, for a request past
    /// `OPENJEV_GEN_MAX_INFLIGHT` plus `OPENJEV_GEN_MAX_QUEUE`.
    public static let overloaded = ChatCompletionError(
        status: 529, message: overloadedMessage, type: "overloaded_error",
        headers: [WireError.Header(name: "retry-after", value: "2")])

    /// This port's 503 `api_error` with `retry-after: 2`, for a generation that failed before its
    /// answer started, naming the error's type as upstream's vLLM generator names the HTTP error of
    /// a backend that did not answer (`inference backend unavailable: {type(e).__name__}`).
    /// Upstream's MLX generator lets the error through, a bare 500.
    public static func backendUnavailable(_ name: String) -> ChatCompletionError {
        ChatCompletionError(
            status: 503, message: "inference backend unavailable: \(name)", type: "api_error",
            headers: [WireError.Header(name: "retry-after", value: "2")])
    }
}
