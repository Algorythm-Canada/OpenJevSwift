// Reproduces the validation upstream OpenJev (razorback16/openjev at dcd2094) gets from pydantic
// for the models in `openjev/api.py`, and the handler `invalid_body` that turns it into a 422 or,
// for an unknown question type, a 400. Apache-2.0. See THIRD_PARTY.md. The error types, messages,
// `loc` paths, `ctx` values and coercions are the ones Fixtures/wire/cases.json records for
// pydantic 2.13 in Python (lax) mode.

/// Checks a parsed `POST /v1/systemone` body and builds a ``SystemOneRequest`` from it.
///
/// It reports what upstream reports, in the same order: every error, field by field in the
/// model's order (`state`, `model`, `questions`, `images`, `steps`, `samples`, `think`,
/// `sequential`), each nested field in its own model's order and each object or array entry in
/// input order. When any question has an unknown `type`, the whole result is
/// ``WireError/invalidRequest`` instead. Unknown fields are ignored.
///
/// Values are coerced as pydantic's lax mode does. `steps`, `samples` and `think` accept
/// integers, `true` and `false`, integral floats and decimal strings (with surrounding
/// whitespace, a sign, single underscores between digits and a zero fraction such as `3.00`).
/// `sequential` accepts Booleans, `0` and `1` as integers or floats, and the strings `0`, `1`,
/// `f`, `t`, `n`, `y`, `no`, `yes`, `off`, `on`, `false` and `true` in any ASCII case.
///
/// Only shape is checked here. Upstream's later checks, which answer with other errors, belong
/// elsewhere: an unknown model is the server's (issue #38), and too many questions, empty choice
/// criteria, more than 255 options or 10 levels are the question schema builder's (issue #10).
/// The image content checks are issue #18's, and the 422 for a body that is not valid JSON is
/// the HTTP layer's (issue #35).
public struct RequestValidator: Sendable {
    /// Creates a validator.
    public init() {}

    /// Validates a request body.
    ///
    /// - Parameter body: The parsed body, or `nil` when the request had no body. A JSON `null`
    ///   body is treated like a missing one, as FastAPI does.
    /// - Throws: A 422 ``WireError`` listing every problem, or ``WireError/invalidRequest``.
    public func validate(_ body: JSONValue?) throws(WireError) -> SystemOneRequest {
        var run = Run()
        let request = run.request(body)
        try run.finish()
        guard let request else {
            preconditionFailure("a request with no errors was not built")
        }
        return request
    }

    /// Validates one question, with `loc` paths that start at the question.
    func validateQuestion(_ value: JSONValue) throws(WireError) -> Question {
        var run = Run()
        let question = run.question(value, at: [])
        try run.finish()
        guard let question else {
            preconditionFailure("a question with no errors was not built")
        }
        return question
    }
}

/// The union member labels pydantic puts in `loc` for `Union[str, dict[str, Any], list[Any]]`.
private let jsonContentMembers: [(label: String, type: String, msg: String)] = [
    ("str", "string_type", "Input should be a valid string"),
    ("dict[str,any]", "dict_type", "Input should be a valid dictionary"),
    ("list[any]", "list_type", "Input should be a valid list"),
]

/// pydantic's message for a model field given something other than an object.
private let modelAttributesMessage =
    "Input should be a valid dictionary or object to extract fields from"

/// The longest integer string pydantic parses, in bytes after trimming.
private let maximumIntegerText = 4300

/// One validation pass, collecting errors as it goes.
private struct Run {
    var errors: [ValidationErrorItem] = []
    var sawUnknownQuestionType = false

    /// Throws the collected result, if there is one.
    func finish() throws(WireError) {
        if sawUnknownQuestionType {
            throw WireError.invalidRequest
        }
        if !errors.isEmpty {
            throw WireError.validation422(errors)
        }
    }

    mutating func fail(
        _ type: String, _ loc: [LocComponent], _ msg: String, _ input: JSONValue,
        ctx: JSONValue? = nil
    ) {
        errors.append(ValidationErrorItem(type: type, loc: loc, msg: msg, input: input, ctx: ctx))
    }

    mutating func missing(_ loc: [LocComponent], in container: JSONValue) {
        fail("missing", loc, "Field required", container)
    }

    // MARK: The request

    mutating func request(_ body: JSONValue?) -> SystemOneRequest? {
        let loc: [LocComponent] = ["body"]
        guard let body, !body.isNull else {
            missing(loc, in: .null)
            return nil
        }
        guard let object = body.objectValue else {
            fail("model_attributes_type", loc, modelAttributesMessage, body)
            return nil
        }
        let start = errors.count

        var state: JSONValue?
        if let value = object["state"] {
            state = jsonContent(value, at: loc + ["state"])
        } else {
            missing(loc + ["state"], in: body)
        }

        var model: String?
        if let value = object["model"] {
            model = value.stringValue
            if model == nil {
                fail("string_type", loc + ["model"], "Input should be a valid string", value)
            }
        } else {
            missing(loc + ["model"], in: body)
        }

        var questions: OrderedMap<Question>?
        if let value = object["questions"] {
            questions = questionMap(value, at: loc + ["questions"])
        } else {
            missing(loc + ["questions"], in: body)
        }

        let images = object["images"].flatMap { self.images($0, at: loc + ["images"]) }
        let steps = object["steps"].flatMap { self.integer($0, at: loc + ["steps"], in: 1...8) }
        let samples = object["samples"].flatMap {
            self.integer($0, at: loc + ["samples"], in: 1...32)
        }
        let think = object["think"].flatMap {
            self.integer($0, at: loc + ["think"], in: 0...4096)
        }
        let sequential = object["sequential"].flatMap {
            self.boolean($0, at: loc + ["sequential"])
        }

        guard errors.count == start, !sawUnknownQuestionType, let state, let model, let questions
        else {
            return nil
        }
        return SystemOneRequest(
            model: model, state: state, questions: questions, images: images, steps: steps,
            samples: samples, think: think, sequential: sequential)
    }

    // MARK: Unions of strings, objects and arrays

    /// `Union[str, dict[str, Any], list[Any]]`: the value, or one error per member.
    mutating func jsonContent(_ value: JSONValue, at loc: [LocComponent]) -> JSONValue? {
        switch value {
        case .string, .object, .array:
            return value
        default:
            for member in jsonContentMembers {
                fail(member.type, loc + [.key(member.label)], member.msg, value)
            }
            return nil
        }
    }

    /// ``Described``: like ``jsonContent(_:at:)`` but `null` is allowed too.
    mutating func described(_ value: JSONValue, at loc: [LocComponent]) -> JSONValue? {
        value.isNull ? value : jsonContent(value, at: loc)
    }

    // MARK: Questions

    /// `dict[str, Question]` with at least one entry.
    mutating func questionMap(_ value: JSONValue, at loc: [LocComponent]) -> OrderedMap<Question>? {
        guard let object = value.objectValue else {
            fail("dict_type", loc, "Input should be a valid dictionary", value)
            return nil
        }
        guard !object.isEmpty else {
            fail(
                "too_short", loc, "Dictionary should have at least 1 item after validation, not 0",
                value, ctx: ["field_type": "Dictionary", "min_length": 1, "actual_length": 0])
            return nil
        }
        var questions = OrderedMap<Question>()
        var complete = true
        for (id, entry) in object {
            if let question = self.question(entry, at: loc + [.key(id)]) {
                questions.updateValue(question, forKey: id)
            } else {
                complete = false
            }
        }
        return complete ? questions : nil
    }

    /// The question union, discriminated by `type`.
    mutating func question(_ value: JSONValue, at loc: [LocComponent]) -> Question? {
        guard let object = value.objectValue else {
            fail("model_attributes_type", loc, modelAttributesMessage, value)
            return nil
        }
        guard let tag = object["type"] else {
            fail(
                "union_tag_not_found", loc, "Unable to extract tag using discriminator 'type'",
                value, ctx: ["discriminator": "'type'"])
            return nil
        }
        let kind = tag.stringValue ?? ""
        guard kind == "noul" || kind == "choice" || kind == "score" else {
            // pydantic reports union_tag_invalid, and upstream's handler answers any request
            // that has one with the generic 400, so the item itself never reaches the wire.
            sawUnknownQuestionType = true
            return nil
        }
        let loc = loc + [.key(kind)]
        let start = errors.count

        var instructions: Described = nil
        if let value = object["instructions"] {
            instructions = described(value, at: loc + ["instructions"])
        }

        let criteriaLoc = loc + ["criteria"]
        let criteria = object["criteria"]
        switch kind {
        case "noul":
            let parsed = criteria.flatMap { noulCriteria($0, at: criteriaLoc) }
            return errors.count == start ? .noul(instructions: instructions, criteria: parsed) : nil
        case "choice":
            guard let criteria else {
                missing(criteriaLoc, in: value)
                return nil
            }
            let parsed = choiceCriteria(criteria, at: criteriaLoc)
            guard errors.count == start, let parsed else { return nil }
            return .choice(instructions: instructions, criteria: parsed)
        default:
            guard let criteria else {
                missing(criteriaLoc, in: value)
                return nil
            }
            let parsed = scoreCriteria(criteria, at: criteriaLoc)
            guard errors.count == start, let parsed else { return nil }
            return .score(instructions: instructions, criteria: parsed)
        }
    }

    /// `NoulCriteria | None`. `null` gives `nil`; unknown keys are ignored.
    mutating func noulCriteria(_ value: JSONValue, at loc: [LocComponent]) -> NoulCriteria? {
        if value.isNull {
            return nil
        }
        guard let object = value.objectValue else {
            fail("model_attributes_type", loc, modelAttributesMessage, value)
            return nil
        }
        var criteria = NoulCriteria()
        if let value = object["true"] {
            criteria.whenTrue = described(value, at: loc + ["true"])
        }
        if let value = object["false"] {
            criteria.whenFalse = described(value, at: loc + ["false"])
        }
        return criteria
    }

    /// `dict[str, Described]`, in input order.
    mutating func choiceCriteria(_ value: JSONValue, at loc: [LocComponent]) -> JSONObject? {
        guard let object = value.objectValue else {
            fail("dict_type", loc, "Input should be a valid dictionary", value)
            return nil
        }
        for (name, description) in object {
            _ = described(description, at: loc + [.key(name)])
        }
        return object
    }

    /// `list[Union[str, dict, list]]` with at least one level.
    mutating func scoreCriteria(_ value: JSONValue, at loc: [LocComponent]) -> [JSONValue]? {
        guard let levels = value.arrayValue else {
            fail("list_type", loc, "Input should be a valid list", value)
            return nil
        }
        guard !levels.isEmpty else {
            fail(
                "too_short", loc, "List should have at least 1 item after validation, not 0", value,
                ctx: ["field_type": "List", "min_length": 1, "actual_length": 0])
            return nil
        }
        for (index, level) in levels.enumerated() {
            _ = jsonContent(level, at: loc + [.index(index)])
        }
        return levels
    }

    // MARK: Images

    /// `list[Union[str, ImageObject]] | None`.
    mutating func images(_ value: JSONValue, at loc: [LocComponent]) -> [ImageInput]? {
        if value.isNull {
            return nil
        }
        guard let items = value.arrayValue else {
            fail("list_type", loc, "Input should be a valid list", value)
            return nil
        }
        var images: [ImageInput] = []
        for (index, item) in items.enumerated() {
            if let image = self.image(item, at: loc + [.index(index)]) {
                images.append(image)
            }
        }
        return images.count == items.count ? images : nil
    }

    /// `Union[str, ImageObject]`: the string member's error comes first, then the object's.
    mutating func image(_ value: JSONValue, at loc: [LocComponent]) -> ImageInput? {
        if let url = value.stringValue {
            return .dataURL(url)
        }
        if let object = value.objectValue, let contentType = object["content_type"]?.stringValue,
            let base64 = object["base64"]?.stringValue
        {
            return .object(contentType: contentType, base64: base64)
        }
        fail("string_type", loc + ["str"], "Input should be a valid string", value)
        let objectLoc = loc + ["ImageObject"]
        guard let object = value.objectValue else {
            fail("model_attributes_type", objectLoc, modelAttributesMessage, value)
            return nil
        }
        for field in ["content_type", "base64"] {
            if let fieldValue = object[field] {
                if fieldValue.stringValue == nil {
                    fail(
                        "string_type", objectLoc + [.key(field)], "Input should be a valid string",
                        fieldValue)
                }
            } else {
                missing(objectLoc + [.key(field)], in: value)
            }
        }
        return nil
    }

    // MARK: Integers and Booleans

    /// `int | None` with bounds, coerced as pydantic's lax mode does.
    mutating func integer(_ value: JSONValue, at loc: [LocComponent], in bounds: ClosedRange<Int>)
        -> Int?
    {
        let number: LaxInteger
        switch value {
        case .null:
            return nil
        case .bool(let flag):
            number = .exact(flag ? 1 : 0)
        case .integer(let digits):
            number = LaxInteger(normalizedDigits: digits)
        case .float(let float):
            guard float.rounded(.towardZero) == float else {
                fail(
                    "int_from_float", loc,
                    "Input should be a valid integer, got a number with a fractional part", value)
                return nil
            }
            // pydantic converts only floats strictly inside the 64-bit range.
            guard float > -0x1p63, float < 0x1p63 else {
                failIntegerSize(loc, value)
                return nil
            }
            number = .exact(Int(float))
        case .string(let text):
            let trimmed = trimmingUnicodeWhitespace(text)
            guard trimmed.utf8.count <= maximumIntegerText else {
                failIntegerSize(loc, value)
                return nil
            }
            guard let parsed = LaxInteger(text: trimmed) else {
                fail(
                    "int_parsing", loc,
                    "Input should be a valid integer, unable to parse string as an integer", value)
                return nil
            }
            number = parsed
        case .array, .object:
            fail("int_type", loc, "Input should be a valid integer", value)
            return nil
        }
        switch number.compare(to: bounds) {
        case .below:
            fail(
                "greater_than_equal", loc,
                "Input should be greater than or equal to \(bounds.lowerBound)", value,
                ctx: ["ge": JSONValue(bounds.lowerBound)])
            return nil
        case .above:
            fail(
                "less_than_equal", loc,
                "Input should be less than or equal to \(bounds.upperBound)", value,
                ctx: ["le": JSONValue(bounds.upperBound)])
            return nil
        case .inside(let result):
            return result
        }
    }

    mutating func failIntegerSize(_ loc: [LocComponent], _ value: JSONValue) {
        fail(
            "int_parsing_size", loc,
            "Unable to parse input string as an integer, exceeded maximum size", value)
    }

    /// `bool | None`, coerced as pydantic's lax mode does.
    mutating func boolean(_ value: JSONValue, at loc: [LocComponent]) -> Bool? {
        let parsingMessage = "Input should be a valid boolean, unable to interpret input"
        switch value {
        case .null:
            return nil
        case .bool(let flag):
            return flag
        case .integer(let digits):
            switch digits {
            case "0": return false
            case "1": return true
            default:
                fail("bool_parsing", loc, parsingMessage, value)
                return nil
            }
        case .float(let float):
            if float == 0 {
                return false
            }
            if float == 1 {
                return true
            }
            // An integral float is judged as an integer; any other float is not a Boolean.
            if float.rounded(.towardZero) == float {
                fail("bool_parsing", loc, parsingMessage, value)
            } else {
                fail("bool_type", loc, "Input should be a valid boolean", value)
            }
            return nil
        case .string(let text):
            switch text.lowercasedASCII() {
            case "0", "f", "n", "no", "off", "false": return false
            case "1", "t", "y", "on", "yes", "true": return true
            default:
                fail("bool_parsing", loc, parsingMessage, value)
                return nil
            }
        case .array, .object:
            fail("bool_type", loc, "Input should be a valid boolean", value)
            return nil
        }
    }
}

/// An integer pydantic accepted, which may not fit in `Int`.
private enum LaxInteger {
    /// A value that fits.
    case exact(Int)
    /// A value too large in magnitude for `Int`, with its sign.
    case huge(negative: Bool)

    enum Placement {
        case below
        case inside(Int)
        case above
    }

    /// An integer from normalized digits with an optional leading `-`.
    init(normalizedDigits digits: String) {
        if let value = Int(digits) {
            self = .exact(value)
        } else {
            self = .huge(negative: digits.hasPrefix("-"))
        }
    }

    /// Parses trimmed text as pydantic does: an optional sign, ASCII digits with single
    /// underscores between them, and optionally a `.` followed by one or more zeros.
    init?(text: String) {
        var body = Substring(text)
        if let point = body.firstIndex(of: ".") {
            let fraction = body[body.index(after: point)...]
            guard !fraction.isEmpty, fraction.allSatisfy({ $0 == "0" }) else { return nil }
            body = body[..<point]
        }
        var negative = false
        if let sign = body.first, sign == "+" || sign == "-" {
            negative = sign == "-"
            body = body.dropFirst()
        }
        guard let first = body.first, let last = body.last, first != "_", last != "_",
            !body.contains("__")
        else {
            return nil
        }
        var digits = ""
        for character in body where character != "_" {
            guard character.isASCII, character.isNumber else { return nil }
            digits.append(character)
        }
        let significant = digits.drop { $0 == "0" }
        if significant.isEmpty {
            self = .exact(0)
        } else {
            self.init(normalizedDigits: (negative ? "-" : "") + significant)
        }
    }

    func compare(to bounds: ClosedRange<Int>) -> Placement {
        switch self {
        case .exact(let value) where value < bounds.lowerBound: return .below
        case .exact(let value) where value > bounds.upperBound: return .above
        case .exact(let value): return .inside(value)
        case .huge(let negative): return negative ? .below : .above
        }
    }
}

/// The text without leading and trailing Unicode `White_Space` scalars, as Rust's `str::trim`
/// removes them.
private func trimmingUnicodeWhitespace(_ text: String) -> String {
    let scalars = text.unicodeScalars
    guard let start = scalars.firstIndex(where: { !$0.properties.isWhitespace }) else {
        return ""
    }
    let end = scalars.lastIndex(where: { !$0.properties.isWhitespace })!
    return String(scalars[start...end])
}

extension String {
    /// The text with ASCII letters lowercased and everything else unchanged.
    fileprivate func lowercasedASCII() -> String {
        String(
            String.UnicodeScalarView(
                unicodeScalars.map { scalar in
                    ("A"..."Z").contains(scalar)
                        ? Unicode.Scalar(scalar.value + 32)! : scalar
                }))
    }
}
