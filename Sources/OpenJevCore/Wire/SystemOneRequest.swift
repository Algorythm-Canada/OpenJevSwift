// The shapes follow upstream OpenJev (razorback16/openjev at dcd2094), `openjev/api.py`, the
// pydantic models `NoulCriteria`, `NoulQuestion`, `ChoiceQuestion`, `ScoreQuestion`,
// `ImageObject` and `SystemOneRequest`. Apache-2.0. See THIRD_PARTY.md.

/// A description or instruction: a string, an object, an array or `null`, as Jev allows.
///
/// `nil` means the field was absent, and `.some(.null)` means it was sent as `null`. Upstream
/// treats both the same way, and a decode followed by an encode keeps the difference, as
/// pydantic's `model_dump(exclude_unset=True)` does.
public typealias Described = JSONValue?

/// Optional descriptions of a noul question's two outcomes.
///
/// On the wire this is `{"true": Described, "false": Described}`, both keys optional. Other keys
/// are ignored.
public struct NoulCriteria: Sendable, Hashable, WireEncodable {
    /// What a true outcome means, sent under the key `true`.
    public var whenTrue: Described
    /// What a false outcome means, sent under the key `false`.
    public var whenFalse: Described

    /// Creates criteria. Leave a description `nil` to omit its key.
    public init(whenTrue: Described = nil, whenFalse: Described = nil) {
        self.whenTrue = whenTrue
        self.whenFalse = whenFalse
    }

    /// The criteria with only the keys that are set, `true` before `false`.
    public var json: JSONValue {
        var object = JSONObject()
        if let whenTrue {
            object["true"] = whenTrue
        }
        if let whenFalse {
            object["false"] = whenFalse
        }
        return .object(object)
    }
}

/// One question of a request, keyed by its id in ``SystemOneRequest/questions``.
public enum Question: Sendable, Hashable, WireEncodable {
    /// A yes or no question, answered with the probability of yes.
    ///
    /// Upstream accepts `"criteria": null` and treats it as absent; so does this type, so a
    /// decoded `null` re-encodes without the key.
    case noul(instructions: Described, criteria: NoulCriteria?)
    /// A choice among named options. The order of `criteria` defines the label letters and the
    /// order of the answer's probabilities. Each value is a description or `null`.
    case choice(instructions: Described, criteria: JSONObject)
    /// A score over ordered levels. Each level is a string, an object or an array; `null` is not
    /// allowed here, unlike in the other criteria.
    case score(instructions: Described, criteria: [JSONValue])

    /// The wire tag: `noul`, `choice` or `score`.
    public var type: String {
        switch self {
        case .noul: return "noul"
        case .choice: return "choice"
        case .score: return "score"
        }
    }

    /// The instructions, or `nil` when absent.
    public var instructions: Described {
        switch self {
        case .noul(let instructions, _), .choice(let instructions, _),
            .score(let instructions, _):
            return instructions
        }
    }

    /// Decodes a question with the rules upstream applies to one entry of `questions`.
    ///
    /// - Throws: A ``WireError`` whose `loc` paths start at the question, without the
    ///   `["body", "questions", id]` prefix a whole request would give them.
    public init(json: JSONValue) throws(WireError) {
        self = try RequestValidator().validateQuestion(json)
    }

    /// The question as pydantic's `model_dump(exclude_unset=True)` gives it: `type`, then
    /// `instructions` and `criteria` when they are set.
    public var json: JSONValue {
        var object: JSONObject = ["type": .string(type)]
        if let instructions {
            object["instructions"] = instructions
        }
        switch self {
        case .noul(_, let criteria):
            if let criteria {
                object["criteria"] = criteria.json
            }
        case .choice(_, let criteria):
            object["criteria"] = .object(criteria)
        case .score(_, let criteria):
            object["criteria"] = .array(criteria)
        }
        return .object(object)
    }
}

/// An image sent with a request, ahead of the state.
///
/// This type holds the image as sent. Whether its content type, encoding and size are acceptable
/// is decided later, by the image checks of issue #18.
public enum ImageInput: Sendable, Hashable, WireEncodable {
    /// A string, which upstream expects to be `data:image/...;base64,...`.
    case dataURL(String)
    /// An object `{"content_type": ..., "base64": ...}`.
    case object(contentType: String, base64: String)

    /// The image as it was sent.
    public var json: JSONValue {
        switch self {
        case .dataURL(let url):
            return .string(url)
        case .object(let contentType, let base64):
            return ["content_type": .string(contentType), "base64": .string(base64)]
        }
    }
}

/// A `POST /v1/systemone` request body.
///
/// Decode one with ``init(json:)`` or ``RequestValidator``, which report errors exactly as
/// upstream does. Unknown top-level fields are ignored.
public struct SystemOneRequest: Sendable, Hashable, WireEncodable {
    /// The requested model name, as sent. Whether it is served is decided by the server.
    public var model: String
    /// The state: a string, an object or an array.
    public var state: JSONValue
    /// The questions in request order, which defines q1 to qN and the answer order.
    public var questions: OrderedMap<Question>
    /// Images to read ahead of the state (OpenJev extension).
    public var images: [ImageInput]?
    /// Denoise passes per read, 1 to 8 (OpenJev extension).
    public var steps: Int?
    /// A fixed number of noise draws, 1 to 32 (OpenJev extension).
    public var samples: Int?
    /// A thought token budget before the read, 0 to 4096 (OpenJev extension).
    public var think: Int?
    /// Read chunks in order, conditioning on earlier answers (OpenJev extension).
    public var sequential: Bool?

    /// Creates a request.
    public init(
        model: String,
        state: JSONValue,
        questions: OrderedMap<Question>,
        images: [ImageInput]? = nil,
        steps: Int? = nil,
        samples: Int? = nil,
        think: Int? = nil,
        sequential: Bool? = nil
    ) {
        self.model = model
        self.state = state
        self.questions = questions
        self.images = images
        self.steps = steps
        self.samples = samples
        self.think = think
        self.sequential = sequential
    }

    /// Decodes and validates a request body.
    ///
    /// - Throws: The ``WireError`` upstream would answer with: a 422 listing every shape
    ///   problem, or the 400 `api_usage_error` for an unknown question type.
    public init(json: JSONValue) throws(WireError) {
        self = try RequestValidator().validate(json)
    }

    /// The request as pydantic's `model_dump(exclude_unset=True)` gives it, in field order:
    /// `state`, `model`, `questions`, then each extension field that is set.
    ///
    /// An extension field sent as `null` is decoded as `nil` and so is not written back.
    public var json: JSONValue {
        var object: JSONObject = [
            "state": state,
            "model": .string(model),
            "questions": .object(
                JSONObject(uniqueKeysWithValues: questions.map { ($0.key, $0.value.json) })),
        ]
        if let images {
            object["images"] = .array(images.map(\.json))
        }
        if let steps {
            object["steps"] = JSONValue(steps)
        }
        if let samples {
            object["samples"] = JSONValue(samples)
        }
        if let think {
            object["think"] = JSONValue(think)
        }
        if let sequential {
            object["sequential"] = .bool(sequential)
        }
        return .object(object)
    }
}
