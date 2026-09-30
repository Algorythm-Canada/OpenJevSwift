// The bodies, statuses and messages follow upstream OpenJev (razorback16/openjev at dcd2094),
// `openjev/api.py`: `error`, `semantic_error`, the `RequestValidationError` handler
// `invalid_body`, the `systemone` route, `read_capped_body` and `check_auth`. Apache-2.0. See
// THIRD_PARTY.md.

/// One component of a validation error's `loc` path: an object key or an array index.
public enum LocComponent: Sendable, Hashable, CustomStringConvertible {
    /// An object key, or a pydantic union member label such as `str` or `dict[str,any]`.
    case key(String)
    /// An array index.
    case index(Int)

    /// The component as JSON: a string or an integer.
    public var json: JSONValue {
        switch self {
        case .key(let key): return .string(key)
        case .index(let index): return JSONValue(index)
        }
    }

    /// The key, or the index in decimal.
    public var description: String {
        switch self {
        case .key(let key): return key
        case .index(let index): return String(index)
        }
    }
}

extension LocComponent: ExpressibleByStringLiteral {
    /// Creates a key component.
    public init(stringLiteral value: String) {
        self = .key(value)
    }
}

extension LocComponent: ExpressibleByIntegerLiteral {
    /// Creates an index component.
    public init(integerLiteral value: Int) {
        self = .index(value)
    }
}

/// One entry of a 422 body's `detail` list, in FastAPI's shape.
///
/// Upstream's handler writes `type`, `loc`, `msg` and `input`, then `ctx` and `url` when pydantic
/// supplied them. With the FastAPI version the fixtures record, `url` never appears; the field
/// is kept so a body that has one still decodes and re-encodes.
public struct ValidationErrorItem: Sendable, Hashable, WireEncodable {
    /// pydantic's error type, such as `missing` or `less_than_equal`.
    public var type: String
    /// Where the error is, starting with `body`.
    public var loc: [LocComponent]
    /// pydantic's message.
    public var msg: String
    /// The offending value. ``WireError/validation422(_:)`` trims it as upstream does.
    public var input: JSONValue
    /// pydantic's context values, such as `{"le": 8}`, or `nil` when there are none.
    public var ctx: JSONValue?
    /// A documentation link, or `nil`.
    public var url: String?

    /// Creates an item.
    public init(
        type: String, loc: [LocComponent], msg: String, input: JSONValue,
        ctx: JSONValue? = nil, url: String? = nil
    ) {
        self.type = type
        self.loc = loc
        self.msg = msg
        self.input = input
        self.ctx = ctx
        self.url = url
    }

    /// `{"type", "loc", "msg", "input"}`, then `ctx` and `url` when set.
    public var json: JSONValue {
        var object: JSONObject = [
            "type": .string(type),
            "loc": .array(loc.map(\.json)),
            "msg": .string(msg),
            "input": input,
        ]
        if let ctx {
            object["ctx"] = ctx
        }
        if let url {
            object["url"] = .string(url)
        }
        return .object(object)
    }

    /// Decodes an item. `ctx` and `url` are optional; no other keys are allowed.
    public init(json: JSONValue) throws(WireDecodingError) {
        try self.init(json: json, path: [])
    }

    init(json: JSONValue, path: [LocComponent]) throws(WireDecodingError) {
        let object = try json.requireObject(at: path)
        var expected: Set<String> = ["type", "loc", "msg", "input"]
        if object["ctx"] != nil {
            expected.insert("ctx")
        }
        if object["url"] != nil {
            expected.insert("url")
        }
        try object.requireKeys(expected, at: path)
        type = try object.require("type", at: path).requireString(at: path + ["type"])
        var loc: [LocComponent] = []
        let components = try object.require("loc", at: path).requireArray(at: path + ["loc"])
        for (index, component) in components.enumerated() {
            if let key = component.stringValue {
                loc.append(.key(key))
            } else if let position = component.intValue {
                loc.append(.index(position))
            } else {
                throw WireDecodingError(
                    path: path + ["loc", .index(index)], reason: "expected a string or an integer")
            }
        }
        self.loc = loc
        msg = try object.require("msg", at: path).requireString(at: path + ["msg"])
        input = try object.require("input", at: path)
        ctx = object["ctx"]
        url = try object["url"].map { value throws(WireDecodingError) in
            try value.requireString(at: path + ["url"])
        }
    }
}

/// `{"detail": {"error_type": ..., "message": ...}}`, the body of most non-422 errors.
public struct TypedErrorBody: Sendable, Hashable, WireEncodable {
    /// `api_usage_error`, `authentication_error`, `permission_error`, `api_error` or
    /// `overloaded_error`.
    public var errorType: String
    /// The message.
    public var message: String

    /// Creates a body.
    public init(errorType: String, message: String) {
        self.errorType = errorType
        self.message = message
    }

    /// `{"detail": {"error_type", "message"}}`.
    public var json: JSONValue {
        ["detail": ["error_type": .string(errorType), "message": .string(message)]]
    }

    /// Decodes a body, requiring its exact key sets.
    public init(json: JSONValue) throws(WireDecodingError) {
        let object = try json.requireObject(at: [])
        try object.requireKeys(["detail"], at: [])
        let detail = try object.require("detail", at: []).requireObject(at: ["detail"])
        try detail.requireKeys(["error_type", "message"], at: ["detail"])
        errorType = try detail.require("error_type", at: ["detail"])
            .requireString(at: ["detail", "error_type"])
        message = try detail.require("message", at: ["detail"])
            .requireString(at: ["detail", "message"])
    }
}

/// `{"detail": "..."}`, the body of a 400 for a request whose shape is fine but whose meaning is
/// not.
public struct PlainDetailBody: Sendable, Hashable, WireEncodable {
    /// The reason.
    public var detail: String

    /// Creates a body.
    public init(detail: String) {
        self.detail = detail
    }

    /// `{"detail": ...}`.
    public var json: JSONValue {
        ["detail": .string(detail)]
    }

    /// Decodes a body, requiring its exact key set.
    public init(json: JSONValue) throws(WireDecodingError) {
        let object = try json.requireObject(at: [])
        try object.requireKeys(["detail"], at: [])
        detail = try object.require("detail", at: []).requireString(at: ["detail"])
    }
}

/// An error response: its HTTP status, its body and any headers it adds.
///
/// The static members build every row of the error table in `docs/02-jev-wire-api.md` with
/// upstream's exact messages. The request id and `server-timing` headers are the server's and
/// are not part of this value.
public struct WireError: Error, Sendable, Hashable, WireEncodable {
    /// The three body shapes.
    public enum Body: Sendable, Hashable, WireEncodable {
        /// `{"detail": [ValidationErrorItem, ...]}` (422).
        case validation([ValidationErrorItem])
        /// `{"detail": {"error_type", "message"}}`.
        case typed(TypedErrorBody)
        /// `{"detail": "..."}`.
        case plain(PlainDetailBody)

        /// The body as it goes on the wire.
        public var json: JSONValue {
            switch self {
            case .validation(let items): return ["detail": .array(items.map(\.json))]
            case .typed(let body): return body.json
            case .plain(let body): return body.json
            }
        }
    }

    /// A response header the error adds, such as `retry-after`.
    public struct Header: Sendable, Hashable {
        /// The lowercase header name.
        public var name: String
        /// The value.
        public var value: String

        /// Creates a header.
        public init(name: String, value: String) {
            self.name = name
            self.value = value
        }
    }

    /// The HTTP status code.
    public var status: Int
    /// The body.
    public var body: Body
    /// Headers the error adds.
    public var headers: [Header]

    /// Creates an error response.
    public init(status: Int, body: Body, headers: [Header] = []) {
        self.status = status
        self.body = body
        self.headers = headers
    }

    /// The body as it goes on the wire.
    public var json: JSONValue { body.json }

    /// A typed body with a status and headers, as upstream's `error` helper builds it.
    private static func typed(
        _ status: Int, _ errorType: String, _ message: String, headers: [Header] = []
    ) -> WireError {
        WireError(
            status: status, body: .typed(TypedErrorBody(errorType: errorType, message: message)),
            headers: headers)
    }

    /// 422: shape validation. Each item's `input` and `ctx` are trimmed with
    /// ``JSONValue/trimmed(depth:items:characters:)``, as upstream's handler does.
    public static func validation422(_ items: [ValidationErrorItem]) -> WireError {
        let trimmed = items.map { item in
            var item = item
            item.input = item.input.trimmed()
            item.ctx = item.ctx?.trimmed()
            return item
        }
        return WireError(status: 422, body: .validation(trimmed))
    }

    /// 400 `api_usage_error` with any message.
    public static func apiUsage400(_ message: String) -> WireError {
        typed(400, "api_usage_error", message)
    }

    /// 400 `api_usage_error "Unknown model: <name>"`, for a model this server does not serve.
    public static func unknownModel(_ name: String) -> WireError {
        apiUsage400("Unknown model: \(name)")
    }

    /// 400 `api_usage_error "Invalid request."`, which replaces the whole 422 when any question
    /// has an unknown type, as Jev answers it.
    public static let invalidRequest = apiUsage400("Invalid request.")

    /// 400 with a plain `detail`: a request the model cannot answer as asked (upstream's
    /// `semantic_error`).
    public static func semantic400(_ detail: String) -> WireError {
        WireError(status: 400, body: .plain(PlainDetailBody(detail: detail)))
    }

    /// 400 with a plain `detail` for a ``SchemaError``. The body is the error's message; its
    /// `loc` is for the log and is not sent, as upstream's `semantic_error` does.
    public static func semantic400(_ error: SchemaError) -> WireError {
        semantic400(error.message)
    }

    /// 400 with a plain `detail`, for an inference backend that refused a request with a 4xx.
    public static func modelRejected400(_ reason: String) -> WireError {
        semantic400("the model rejected this request: \(reason)")
    }

    /// 401 `authentication_error`, for a wrong API key.
    public static let authentication401 = typed(
        401, "authentication_error",
        "Cannot authenticate with the server. Please check your API key and try again.")

    /// 403 `authentication_error`, for a request without an `Authorization` header when a key
    /// is required.
    public static let authenticationMissing403 = typed(
        403, "authentication_error", "Must supply an API key! Check your request and try again.")

    /// 403 `permission_error`, for a wrong or missing origin secret.
    public static let permission403 = typed(
        403, "permission_error", "Direct access to this origin is not allowed.")

    /// 413 `api_usage_error`, for a body over the byte limit.
    public static func bodyTooLarge413(limit: Int) -> WireError {
        typed(413, "api_usage_error", "request body is larger than \(limit) bytes")
    }

    /// 503 `api_error` with `retry-after: 2`, for an unreachable inference backend. Upstream's
    /// reason is the Python exception's class name, such as `ConnectError`.
    public static func backendUnavailable503(_ reason: String) -> WireError {
        typed(
            503, "api_error", "inference backend unavailable: \(reason)",
            headers: [Header(name: "retry-after", value: "2")])
    }

    /// The message upstream's engine gives a full queue.
    public static let overloadedMessage = "OpenJev is at capacity. Retry shortly."

    /// 529 `overloaded_error` with `retry-after: 1`, for a full queue.
    public static func overloaded529(_ message: String = overloadedMessage) -> WireError {
        typed(529, "overloaded_error", message, headers: [Header(name: "retry-after", value: "1")])
    }
}
