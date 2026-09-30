import OpenJevCore
import Testing

/// The quickstart questions, reused by the tests below.
private let quickstartQuestions: JSONValue = [
    "department": [
        "type": "choice", "instructions": "Which team should handle this",
        "criteria": [
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        ],
    ],
    "frustration": [
        "type": "score", "instructions": "How frustrated the customer appears",
        "criteria": [
            "Calm, just stating facts", "Frustrated but civil", "Very angry, strong language",
        ],
    ],
    "is_urgent": [
        "type": "noul", "instructions": "The message conveys urgency or time-sensitivity",
    ],
]

/// A valid body with some fields replaced or removed.
private func body(
    _ changes: [(String, JSONValue?)] = [], questions: JSONValue = quickstartQuestions
) -> JSONValue {
    var object: JSONObject = ["state": "x", "model": "jev-latest", "questions": questions]
    for (key, value) in changes {
        object[key] = value
    }
    return .object(object)
}

/// The validation error for a body, or `nil` when it is accepted.
private func rejection(_ value: JSONValue?) -> WireError? {
    do {
        _ = try RequestValidator().validate(value)
        return nil
    } catch {
        return error
    }
}

/// The items of a 422, or an empty list for any other outcome.
private func items(_ value: JSONValue?) -> [ValidationErrorItem] {
    guard case .validation(let items)? = rejection(value)?.body else { return [] }
    return items
}

/// Each item's type and loc, for compact expectations.
private func summary(_ value: JSONValue?) -> [String] {
    items(value).map { item in
        "\(item.type) " + item.loc.map(\.description).joined(separator: ".")
    }
}

@Suite("RequestValidator")
struct RequestValidatorTests {
    @Test("A valid body is accepted and keeps its order")
    func acceptsQuickstart() throws {
        let request = try RequestValidator().validate(body())
        #expect(request.model == "jev-latest")
        #expect(request.state == "x")
        #expect(request.questions.keys == ["department", "frustration", "is_urgent"])
        guard case .choice(_, let criteria)? = request.questions["department"] else {
            Issue.record("department is not a choice")
            return
        }
        #expect(criteria.keys == ["billing", "technical", "sales"])
        #expect(request.steps == nil && request.images == nil && request.sequential == nil)
    }

    @Test("A missing or null body is one missing error at body")
    func missingBody() {
        #expect(summary(nil) == ["missing body"])
        #expect(summary(.null) == ["missing body"])
        #expect(items(nil).first?.input == .null)
        #expect(summary([1]) == ["model_attributes_type body"])
    }

    @Test("Missing fields are reported in model order with the body as input")
    func missingFields() {
        #expect(
            summary([:]) == ["missing body.state", "missing body.model", "missing body.questions"])
        let value = body([("state", nil)])
        #expect(summary(value) == ["missing body.state"])
        #expect(items(value).first?.input == value)
    }

    @Test("state accepts strings, objects and arrays and reports each union member otherwise")
    func stateUnion() {
        #expect(rejection(body([("state", ["a": 1])])) == nil)
        #expect(rejection(body([("state", [1, 2])])) == nil)
        #expect(
            summary(body([("state", 5)])) == [
                "string_type body.state.str", "dict_type body.state.dict[str,any]",
                "list_type body.state.list[any]",
            ])
        #expect(summary(body([("state", .null)])).count == 3)
    }

    @Test("model must be a string")
    func modelType() {
        #expect(summary(body([("model", 5)])) == ["string_type body.model"])
        #expect(rejection(body([("model", "gpt-4")])) == nil)  // served models are #38's
    }

    @Test("questions must be a non-empty object")
    func questionsShape() {
        #expect(summary(body([("questions", [])])) == ["dict_type body.questions"])
        let empty = items(body([("questions", [:])]))
        #expect(empty.map(\.type) == ["too_short"])
        #expect(
            empty.first?.ctx == ["field_type": "Dictionary", "min_length": 1, "actual_length": 0])
        #expect(
            empty.first?.msg == "Dictionary should have at least 1 item after validation, not 0")
    }

    @Test("A question needs an object with a type tag")
    func questionTag() {
        #expect(
            summary(body(questions: ["q": "x"])) == ["model_attributes_type body.questions.q"])
        let untagged = items(body(questions: ["q": ["instructions": "x"]]))
        #expect(untagged.map(\.type) == ["union_tag_not_found"])
        #expect(untagged.first?.ctx == ["discriminator": "'type'"])
    }

    @Test("An unknown question type collapses every error into Invalid request")
    func unknownType() {
        for tag: JSONValue in ["nope", 5, .null] {
            let value = body([("state", nil)], questions: ["q": ["type": tag]])
            #expect(rejection(value) == .invalidRequest)
        }
        #expect(WireError.invalidRequest.status == 400)
    }

    /// Upstream's handler logged these problems for this body, in this order, with FastAPI
    /// 0.142.1 and pydantic 2.13.5.
    @Test("problems(_:) lists what upstream logs, unknown question types included")
    func problems() {
        let value: JSONValue = [
            "model": 5,
            "questions": [
                "q": ["type": "nope"], "r": ["type": 5], "s": ["type": .null],
                "t": ["type": "choice"],
            ],
            "steps": 99,
        ]
        #expect(rejection(value) == .invalidRequest)
        #expect(
            RequestValidator().problems(value).map(\.description) == [
                "body.state: missing", "body.model: string_type",
                "body.questions.q: union_tag_invalid", "body.questions.r: union_tag_invalid",
                "body.questions.s: union_tag_invalid", "body.questions.t.choice.criteria: missing",
                "body.steps: less_than_equal",
            ])
        let mistyped = body([("model", 5), ("steps", 0)])
        #expect(
            RequestValidator().problems(mistyped)
                == items(mistyped).map { ValidationProblem(loc: $0.loc, type: $0.type) })
        #expect(RequestValidator().problems(body()).isEmpty)
    }

    @Test("instructions and descriptions take the union member labels in loc")
    func describedUnion() {
        let value = body(questions: ["q": ["type": "noul", "instructions": 5]])
        #expect(
            summary(value) == [
                "string_type body.questions.q.noul.instructions.str",
                "dict_type body.questions.q.noul.instructions.dict[str,any]",
                "list_type body.questions.q.noul.instructions.list[any]",
            ])
        #expect(rejection(body(questions: ["q": ["type": "noul", "instructions": .null]])) == nil)
    }

    @Test("noul criteria is an optional object of true and false")
    func noulCriteria() {
        #expect(
            summary(body(questions: ["q": ["type": "noul", "criteria": "x"]])) == [
                "model_attributes_type body.questions.q.noul.criteria"
            ])
        #expect(
            summary(body(questions: ["q": ["type": "noul", "criteria": ["true": 5]]])).first
                == "string_type body.questions.q.noul.criteria.true.str")
        #expect(rejection(body(questions: ["q": ["type": "noul", "criteria": ["yes": 5]]])) == nil)
        #expect(rejection(body(questions: ["q": ["type": "noul", "criteria": .null]])) == nil)
    }

    @Test("choice criteria is a required object of descriptions")
    func choiceCriteria() {
        #expect(
            summary(body(questions: ["q": ["type": "choice"]])) == [
                "missing body.questions.q.choice.criteria"
            ])
        #expect(
            summary(body(questions: ["q": ["type": "choice", "criteria": ["a"]]])) == [
                "dict_type body.questions.q.choice.criteria"
            ])
        #expect(
            summary(body(questions: ["q": ["type": "choice", "criteria": ["a": 5]]])).first
                == "string_type body.questions.q.choice.criteria.a.str")
        // An empty choice is a semantic 400 from the schema builder (#10), not a shape error.
        #expect(rejection(body(questions: ["q": ["type": "choice", "criteria": [:]]])) == nil)
    }

    @Test("score criteria is a required non-empty list with integer indices in loc")
    func scoreCriteria() {
        #expect(
            summary(body(questions: ["q": ["type": "score"]])) == [
                "missing body.questions.q.score.criteria"
            ])
        let empty = items(body(questions: ["q": ["type": "score", "criteria": []]]))
        #expect(empty.map(\.type) == ["too_short"])
        #expect(empty.first?.ctx == ["field_type": "List", "min_length": 1, "actual_length": 0])
        let level = items(body(questions: ["q": ["type": "score", "criteria": ["ok", .null]]]))
        #expect(level.first?.loc == ["body", "questions", "q", "score", "criteria", 1, "str"])
    }

    @Test("steps, samples and think enforce their bounds with ge and le in ctx")
    func integerBounds() {
        let cases: [(String, JSONValue, String?)] = [
            ("steps", 0, "greater_than_equal"), ("steps", 1, nil), ("steps", 8, nil),
            ("steps", 9, "less_than_equal"), ("samples", 0, "greater_than_equal"),
            ("samples", 32, nil), ("samples", 33, "less_than_equal"),
            ("think", -1, "greater_than_equal"), ("think", 0, nil), ("think", 4096, nil),
            ("think", 4097, "less_than_equal"),
        ]
        for (field, value, type) in cases {
            let found = items(body([(field, value)]))
            #expect(found.map(\.type) == (type.map { [$0] } ?? []), "\(field) \(value)")
            #expect(found.first?.loc == (type == nil ? nil : ["body", .key(field)]))
        }
        #expect(items(body([("steps", 9)])).first?.ctx == ["le": 8])
        #expect(items(body([("think", -1)])).first?.ctx == ["ge": 0])
    }

    @Test("Integer fields coerce as pydantic's lax mode does")
    func integerCoercion() throws {
        let accepted: [(JSONValue, Int)] = [
            ("3", 3), (" 3 ", 3), ("3.00", 3), ("+3", 3), ("0003", 3), ("1_0.0", 10), (true, 1),
            (2.0, 2),
        ]
        for (value, expected) in accepted {
            let request = try? RequestValidator().validate(body([("think", value)]))
            #expect(request?.think == expected, "\(value)")
        }
        let rejected: [(JSONValue, String)] = [
            ("3.", "int_parsing"), ("1__0", "int_parsing"), ("0x10", "int_parsing"),
            ("\u{663}", "int_parsing"), (2.5, "int_from_float"), (1e20, "int_parsing_size"),
            (.string(String(repeating: "1", count: 4301)), "int_parsing_size"), ([], "int_type"),
            (false, "greater_than_equal"), ("1_0", "less_than_equal"),
        ]
        for (value, type) in rejected {
            #expect(items(body([("steps", value)])).map(\.type) == [type], "\(value)")
        }
    }

    @Test("sequential coerces as pydantic's lax mode does")
    func booleanCoercion() throws {
        let accepted: [(JSONValue, Bool)] = [
            ("yes", true), ("OFF", false), ("t", true), ("0", false), (1, true), (0.0, false),
            (true, true),
        ]
        for (value, expected) in accepted {
            let request = try? RequestValidator().validate(body([("sequential", value)]))
            #expect(request?.sequential == expected, "\(value)")
        }
        let rejected: [(JSONValue, String)] = [
            ("maybe", "bool_parsing"), (" yes", "bool_parsing"), (2, "bool_parsing"),
            (2.0, "bool_parsing"), (0.5, "bool_type"), ([:], "bool_type"),
        ]
        for (value, type) in rejected {
            #expect(items(body([("sequential", value)])).map(\.type) == [type], "\(value)")
        }
    }

    @Test("images take strings or objects and report both union members otherwise")
    func images() throws {
        let request = try RequestValidator().validate(
            body([
                (
                    "images",
                    [
                        "data:image/png;base64,AA==",
                        ["content_type": "image/png", "base64": "AA=="],
                    ]
                )
            ]))
        #expect(
            request.images == [
                .dataURL("data:image/png;base64,AA=="),
                .object(contentType: "image/png", base64: "AA=="),
            ])
        #expect(summary(body([("images", "x")])) == ["list_type body.images"])
        #expect(
            summary(body([("images", [5])])) == [
                "string_type body.images.0.str", "model_attributes_type body.images.0.ImageObject",
            ])
        #expect(
            summary(body([("images", [[:]])])) == [
                "string_type body.images.0.str",
                "missing body.images.0.ImageObject.content_type",
                "missing body.images.0.ImageObject.base64",
            ])
    }

    @Test("Errors across fields are all reported, in model order")
    func everyError() {
        let value = body([("state", nil), ("sequential", "maybe"), ("steps", 9)], questions: [:])
        #expect(
            summary(value) == [
                "missing body.state", "too_short body.questions", "less_than_equal body.steps",
                "bool_parsing body.sequential",
            ])
    }

    @Test("Unknown top-level fields are ignored")
    func unknownFields() {
        #expect(rejection(body([("extra", ["anything": [1]])])) == nil)
    }

    @Test("The 422 trims each input as upstream does")
    func trimming() {
        var deep: JSONValue = ["a": .null]
        for _ in 0..<1000 {
            deep = ["a": deep]
        }
        let found = items(body(questions: ["q": deep]))
        #expect(found.first?.input == ["a": ["a": ["a": ["a": "..."]]]])
    }
}
