// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/chat.py`: the checks of the
// `chat_completions` route before its capacity bound, `Generator.normalize`, and the reading of
// `stop` and `chat_template_kwargs` in `MlxGenerator.stop_ids` and `prompt_ids`. Apache-2.0. See
// THIRD_PARTY.md.

/// A chat completion request as upstream's MLX generator reads it: the body's checks, then
/// `Generator.normalize`, which keeps only the fields a generation passes on, bounds
/// `max_tokens`, turns thinking off unless asked, forces streamed usage on and turns JSON mode
/// into an instruction.
public struct ChatCompletionRequest: Sendable, Hashable {
    /// Upstream's `GEN_MODEL`, the name a response carries.
    public static let modelName = "diffusiongemma-26b"

    /// Upstream's `MODEL_NAMES`, the names a request may use.
    public static let modelNames: Set<String> = [modelName, "diffusiongemma"]

    /// Upstream's `PASSTHROUGH`: the fields `normalize` keeps. Everything else (temperature, seed,
    /// penalties, `response_format`, `n`, reasoning switches) is dropped.
    public static let passthroughFields: Set<String> = [
        "messages", "max_tokens", "stop", "top_p", "top_k", "stream", "stream_options", "tools",
        "tool_choice", "logprobs", "top_logprobs", "chat_template_kwargs",
    ]

    /// Upstream's `DEFAULT_MAX_TOKENS`.
    public static let defaultMaxTokens = 1024

    /// The deepest the `messages` array may nest, counting itself as one level: a message is two,
    /// its tool calls three, and so on. The chat template recurses into tool call arguments, so the
    /// bound keeps that recursion within a task's stack; past it upstream's jinja2 raises
    /// `RecursionError` at a depth its stack decides, a bare 500 (D-058).
    public static let maximumNesting = 64

    /// Upstream's `JSON_INSTRUCTION`, appended to the system message in JSON mode.
    public static let jsonInstruction =
        "Reply with exactly one JSON object and nothing else: no prose, no code fences."

    /// What `normalize` returns, without the `model` it adds for vLLM: the passthrough fields in
    /// the body's order, `max_tokens` bounded, `chat_template_kwargs` with `enable_thinking` first,
    /// `stream_options` with `include_usage` when streaming, and in JSON mode the messages with the
    /// instruction. Fields `normalize` adds go after the body's.
    public var normalized: JSONObject
    /// The messages the prompt is rendered from, JSON mode's instruction included.
    public var messages: [JSONValue]
    /// The longest reply in tokens: `max_tokens`, else `max_completion_tokens`, else 1024, at most
    /// `OPENJEV_GEN_MAX_TOKENS`.
    public var maxTokens: Int
    /// Whether the prompt is rendered with thinking on: Python's truth value of
    /// `chat_template_kwargs.enable_thinking`, as `prompt_ids` reads it.
    public var thinking: Bool
    /// Whether the reply is streamed: Python's truth value of `stream`.
    public var stream: Bool
    /// Whether a stream ends with a usage chunk, upstream's `usage_wanted`. `normalize` forces it
    /// on for every stream.
    public var includeUsage: Bool
    /// Whether `response_format` asked for JSON (`json_object` or `json_schema`): the reply is
    /// then reduced to its first JSON object or array (``ExtractJSON``).
    public var jsonMode: Bool

    /// The route's checks before its capacity bound, in upstream's order: an object body with a
    /// non-empty `messages` array, then a string `model`, then a model this route serves.
    ///
    /// - Throws: ``ChatCompletionError/messagesRequired``, ``ChatCompletionError/modelRequired``
    ///   or ``ChatCompletionError/modelNotFound(_:)``.
    /// - Returns: The body as an object.
    public static func checked(_ body: JSONValue) throws(ChatCompletionError) -> JSONObject {
        guard case .object(let object) = body, case .array(let messages)? = object["messages"],
            !messages.isEmpty
        else {
            throw .messagesRequired
        }
        guard case .string(let model)? = object["model"] else {
            throw .modelRequired
        }
        guard modelNames.contains(model) else {
            throw .modelNotFound(model)
        }
        return object
    }

    /// Normalizes a checked body (``checked(_:)``) as upstream's `Generator.normalize` does, with
    /// `max_tokens` bounded by `maxTokensCap`, `OPENJEV_GEN_MAX_TOKENS`.
    ///
    /// Every message must be an object with a string `role`; upstream's template raises on any
    /// other message, a bare 500 (D-058). The check comes where upstream's first touch of the
    /// messages would raise: in JSON mode after `json_schema` is read, otherwise last.
    ///
    /// - Throws: ``ChatCompletionError/maxTokens(_:)`` for a `max_tokens` that is not a positive
    ///   integer, a bool included, as upstream refuses it; and this port's 400s for the requests
    ///   upstream crashes on: a `chat_template_kwargs`, a streaming `stream_options`, a
    ///   `response_format` or a `response_format.json_schema` that is true but not an object, and
    ///   a message that is not an object with a string `role`.
    public init(normalizing body: JSONObject, maxTokensCap: Int) throws(ChatCompletionError) {
        var out = JSONObject()
        for (key, value) in body where Self.passthroughFields.contains(key) {
            out.updateValue(value, forKey: key)
        }
        // A present max_tokens wins, null included; only an absent one falls back.
        let requested = out["max_tokens"] ?? body["max_completion_tokens"] ?? .null
        let maxTokens = try Self.maxTokens(requested, cap: maxTokensCap)
        out.updateValue(JSONValue(maxTokens), forKey: "max_tokens")

        var kwargs: JSONObject = ["enable_thinking": false]
        if let given = out["chat_template_kwargs"], given.isPythonTruthy {
            guard case .object(let object) = given else {
                throw .invalidRequest("chat_template_kwargs must be an object.")
            }
            for (key, value) in object {
                kwargs.updateValue(value, forKey: key)
            }
        }
        out.updateValue(.object(kwargs), forKey: "chat_template_kwargs")

        let stream = out["stream"]?.isPythonTruthy ?? false
        if stream {
            var options = JSONObject()
            if let given = out["stream_options"], given.isPythonTruthy {
                guard case .object(let object) = given else {
                    throw .invalidRequest("stream_options must be an object.")
                }
                options = object
            }
            options.updateValue(true, forKey: "include_usage")
            out.updateValue(.object(options), forKey: "stream_options")
        }

        var format = JSONObject()
        if let given = body["response_format"], given.isPythonTruthy {
            guard case .object(let object) = given else {
                throw .invalidRequest("response_format must be an object.")
            }
            format = object
        }
        let jsonMode = format["type"] == "json_object" || format["type"] == "json_schema"

        var messages = out["messages"]?.arrayValue ?? []
        if jsonMode {
            var note = Self.jsonInstruction
            var holder = JSONObject()
            if let given = format["json_schema"], given.isPythonTruthy {
                guard case .object(let object) = given else {
                    throw .invalidRequest("response_format.json_schema must be an object.")
                }
                holder = object
            }
            if let schema = holder["schema"], schema.isPythonTruthy {
                // The body was parsed from JSON, so it holds nothing json.dumps would refuse.
                let text = (try? PythonJSONWriter.modelText(schema)) ?? ""
                note += " It must match this JSON schema: " + text
            }
            try Self.checkRoles(messages)
            if case .object(var first)? = messages.first, first["role"] == "system",
                case .string(let content)? = first["content"]
            {
                first.updateValue(
                    .string(TextOf.pythonRightStripped(content) + "\n\n" + note),
                    forKey: "content")
                messages[0] = .object(first)
            } else {
                messages.insert(["role": "system", "content": .string(note)], at: 0)
            }
            out.updateValue(.array(messages), forKey: "messages")
        } else {
            try Self.checkRoles(messages)
        }

        self.normalized = out
        self.messages = messages
        self.maxTokens = maxTokens
        self.thinking = kwargs["enable_thinking"]?.isPythonTruthy ?? false
        self.stream = stream
        self.includeUsage =
            stream && out["stream_options"]?["include_usage"]?.isPythonTruthy == true
        self.jsonMode = jsonMode
    }

    /// The `stop` strings, as `stop_ids` reads them: `[stop]` for a string, else
    /// `list(stop or ())`, which for an object is its keys. Upstream reads them only once the
    /// prompt is within its limit, so a request refused for both gets the prompt's 400.
    ///
    /// - Throws: This port's 400 for a `stop` that is true but neither a string, an array of
    ///   strings nor an object, on which upstream's generation raises: a bare 500, or for a
    ///   stream a reply that breaks off after its first chunk.
    public func stopStrings() throws(ChatCompletionError) -> [String] {
        guard let value = normalized["stop"], value.isPythonTruthy else { return [] }
        switch value {
        case .string(let text):
            return [text]
        case .object(let object):
            return object.keys
        case .array(let elements):
            var strings: [String] = []
            for element in elements {
                guard case .string(let text) = element else {
                    throw .invalidRequest("stop must be a string or an array of strings.")
                }
                strings.append(text)
            }
            return strings
        default:
            throw .invalidRequest("stop must be a string or an array of strings.")
        }
    }

    /// `max_tokens` as `normalize` reads it: `null` is the default, anything but a positive
    /// integer is refused, and the rest is bounded by `cap`.
    static func maxTokens(_ value: JSONValue, cap: Int) throws(ChatCompletionError) -> Int {
        switch value {
        case .null:
            return min(defaultMaxTokens, cap)
        case .integer(let digits) where !digits.hasPrefix("-") && digits != "0":
            // An integer too large for Int is still a positive integer, bounded like any other.
            return min(Int(digits) ?? .max, cap)
        default:
            throw .maxTokens(value)
        }
    }

    /// Refuses a message that is not an object with a string `role`, and messages that nest
    /// deeper than ``maximumNesting``.
    static func checkRoles(_ messages: [JSONValue]) throws(ChatCompletionError) {
        for (index, message) in messages.enumerated() {
            guard case .object(let object) = message, case .string? = object["role"] else {
                throw .invalidRequest("messages[\(index)] must be an object with a string role.")
            }
        }
        let depth = nesting(of: .array(messages))
        if depth > maximumNesting {
            throw .invalidRequest(
                "messages nest \(depth) levels deep; the limit is \(maximumNesting).")
        }
    }

    /// How many arrays and objects deep `value` nests: 0 for a scalar, 1 for a container of
    /// scalars. Walked with an explicit stack.
    public static func nesting(of value: JSONValue) -> Int {
        var deepest = 0
        var pending: [(JSONValue, Int)] = [(value, 1)]
        while let (next, depth) = pending.popLast() {
            switch next {
            case .array(let elements):
                deepest = max(deepest, depth)
                pending.append(contentsOf: elements.map { ($0, depth + 1) })
            case .object(let object):
                deepest = max(deepest, depth)
                pending.append(contentsOf: object.values.map { ($0, depth + 1) })
            case .null, .bool, .integer, .float, .string:
                break
            }
        }
        return deepest
    }
}
