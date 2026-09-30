import OpenJevCore
import Testing

@Suite("Wire types")
struct WireTypesTests {
    private let encoder = WireEncoder()

    @Test("Choice criteria keep their order through decode and encode")
    func criteriaOrder() throws {
        let text = #"{"type":"choice","criteria":{"zeta":"z","alpha":null,"mid":{"b":1,"a":2}}}"#
        let question = try Question(json: JSONParser().parse(text))
        guard case .choice(_, let criteria) = question else {
            Issue.record("not a choice")
            return
        }
        #expect(criteria.keys == ["zeta", "alpha", "mid"])
        #expect(try encoder.string(question) == text)
    }

    @Test("Answers keep their order through decode and encode")
    func answerOrder() throws {
        let answers: OrderedMap<Answer> = ["z": .noul(0.5), "a": .noul(1.0), "m": .noul(0.0)]
        let response = SystemOneResponse(
            model: "openjev-0.1", answers: answers, usage: Usage(inputTokens: 3, outputTokens: 0))
        let text = try encoder.string(response)
        #expect(
            text
                == #"{"model":"openjev-0.1","answers":{"z":{"type":"noul","noul":0.5},"a":{"type":"noul","noul":1.0},"m":{"type":"noul","noul":0.0}},"usage":{"input_tokens":3,"output_tokens":0}}"#
        )
        let decoded = try SystemOneResponse(json: JSONParser().parse(text))
        #expect(decoded == response)
        #expect(decoded.answers.keys == ["z", "a", "m"])
    }

    @Test("Question ids are looked up by Unicode scalars, as in JSONObject")
    func scalarKeys() throws {
        let composed = "caf\u{E9}"
        let decomposed = "cafe\u{301}"
        let request = try RequestValidator().validate([
            "state": "x", "model": "jev-latest",
            "questions": .object(
                JSONObject(uniqueKeysWithValues: [
                    (composed, ["type": "noul"] as JSONValue),
                    (decomposed, ["type": "score", "criteria": ["a"]] as JSONValue),
                ])),
        ])
        #expect(request.questions.count == 2)
        #expect(request.questions[composed]?.type == "noul")
        #expect(request.questions[decomposed]?.type == "score")
        #expect(request.questions.index(forKey: decomposed) == 1)
        #expect(request.questions["cafe"] == nil)
    }

    @Test("ImageInput writes both forms")
    func imageForms() throws {
        #expect(
            try encoder.string(ImageInput.dataURL("data:image/png;base64,AA=="))
                == #""data:image/png;base64,AA==""#)
        #expect(
            try encoder.string(ImageInput.object(contentType: "image/jpeg", base64: "AA=="))
                == #"{"content_type":"image/jpeg","base64":"AA=="}"#)
    }

    @Test("Answers have exact key sets and Python float layout")
    func answerShapes() throws {
        #expect(try encoder.string(Answer.noul(1.0)) == #"{"type":"noul","noul":1.0}"#)
        let choice = Answer.choice(
            choice: "b", probabilities: ["a": 0.25, "b": 0.75], confidence: 0.1875)
        #expect(
            try encoder.string(choice)
                == #"{"type":"choice","choice":"b","probabilities":{"a":0.25,"b":0.75},"confidence":0.1875}"#
        )
        let score = Answer.score(
            score: 0.75, legend: ["a", ["k": 1]], probabilities: [0.25, 0.75], confidence: 0.2)
        #expect(
            try encoder.string(score)
                == #"{"type":"score","score":0.75,"legend":{"0":"a","1":{"k":1}},"probabilities":{"0":0.25,"1":0.75},"confidence":0.2}"#
        )
        #expect(throws: WireDecodingError.self) {
            try Answer(json: ["type": "noul", "noul": 0.5, "confidence": 1.0])
        }
        #expect(throws: WireDecodingError.self) {
            try Answer(json: [
                "type": "score", "score": 0.0, "legend": ["1": "a"], "probabilities": ["1": 1.0],
                "confidence": 1.0,
            ])
        }
    }

    @Test("Non-finite numbers are refused, as FastAPI refuses them")
    func nonFinite() {
        #expect(throws: JSONWriteError.nonFiniteNumber(.infinity)) {
            try encoder.bytes(Answer.noul(.infinity))
        }
    }

    @Test("Every row of the error table has its status, body and headers")
    func errorTable() throws {
        // (error, status, error_type or nil for a plain detail, message, retry-after)
        let rows: [(WireError, Int, String?, String, String?)] = [
            (.unknownModel("gpt-4"), 400, "api_usage_error", "Unknown model: gpt-4", nil),
            (.invalidRequest, 400, "api_usage_error", "Invalid request.", nil),
            (.semantic400("at most 8 images"), 400, nil, "at most 8 images", nil),
            (.modelRejected400("bad"), 400, nil, "the model rejected this request: bad", nil),
            (.unparsableBody400, 400, nil, "There was an error parsing the body", nil),
            (
                .authentication401, 401, "authentication_error",
                "Cannot authenticate with the server. Please check your API key and try again.",
                nil
            ),
            (
                .authenticationMissing403, 403, "authentication_error",
                "Must supply an API key! Check your request and try again.", nil
            ),
            (
                .permission403, 403, "permission_error",
                "Direct access to this origin is not allowed.", nil
            ),
            (
                .bodyTooLarge413(limit: 512), 413, "api_usage_error",
                "request body is larger than 512 bytes", nil
            ),
            (
                .backendUnavailable503("ConnectError"), 503, "api_error",
                "inference backend unavailable: ConnectError", "2"
            ),
            (
                .overloaded529(), 529, "overloaded_error", "OpenJev is at capacity. Retry shortly.",
                "1"
            ),
        ]
        for (error, status, errorType, message, retryAfter) in rows {
            let text =
                errorType.map { #"{"detail":{"error_type":"\#($0)","message":"\#(message)"}}"# }
                ?? #"{"detail":"\#(message)"}"#
            #expect(error.status == status)
            #expect(try encoder.string(error) == text)
            #expect(error.headers.first { $0.name == "retry-after" }?.value == retryAfter)
        }
        let invalid = WireError.jsonInvalid422(message: "Expecting value", position: 10)
        #expect(invalid.status == 422)
        #expect(
            try encoder.string(invalid)
                == #"{"detail":[{"type":"json_invalid","loc":["body",10],"msg":"JSON decode error","#
                + #""input":{},"ctx":{"error":"Expecting value"}}]}"#)
    }

    @Test("A 422 item writes ctx and url only when set, and validation422 trims")
    func validationItems() throws {
        let long = String(repeating: "y", count: 600)
        let error = WireError.validation422([
            ValidationErrorItem(
                type: "missing", loc: ["body", "model"], msg: "Field required",
                input: ["state": .string(long)]),
            ValidationErrorItem(
                type: "less_than_equal", loc: ["body", "steps"],
                msg: "Input should be less than or equal to 8", input: 9, ctx: ["le": 8]),
        ])
        let text = try encoder.string(error)
        let trimmed = String(repeating: "y", count: 500) + "..."
        #expect(
            text
                == #"{"detail":[{"type":"missing","loc":["body","model"],"msg":"Field required","input":{"state":""#
                + trimmed
                + #""}},{"type":"less_than_equal","loc":["body","steps"],"msg":"Input should be less than or equal to 8","input":9,"ctx":{"le":8}}]}"#
        )
        #expect(error.status == 422)
        let item = try ValidationErrorItem(json: [
            "type": "t", "loc": ["body", 0], "msg": "m", "input": .null,
            "url": "https://example.com",
        ])
        #expect(item.loc == ["body", 0])
        #expect(
            try encoder.string(item)
                == #"{"type":"t","loc":["body",0],"msg":"m","input":null,"url":"https://example.com"}"#
        )
    }

    @Test("Usage and ModelInfo round trip through their wire form")
    func smallTypes() throws {
        let info = ModelInfo(name: "openjev-0.1", description: "d", releaseDate: "2026-09-18")
        #expect(
            try encoder.string(info)
                == #"{"name":"openjev-0.1","description":"d","release_date":"2026-09-18"}"#)
        #expect(try ModelInfo(json: info.json) == info)
        let usage = Usage(inputTokens: 1, outputTokens: 2)
        #expect(try Usage(json: usage.json) == usage)
        #expect(try encoder.string(HealthResponse.ok) == #"{"status":"ok"}"#)
    }

    @Test("A request re-encodes in pydantic's field order and omits unset fields")
    func requestEncoding() throws {
        let request = try RequestValidator().validate([
            "sequential": "yes", "questions": ["q": ["criteria": ["b": "B"], "type": "choice"]],
            "model": "m", "steps": "3", "state": ["b": 1, "a": 2],
        ])
        #expect(
            try encoder.string(request)
                == #"{"state":{"b":1,"a":2},"model":"m","questions":{"q":{"type":"choice","criteria":{"b":"B"}}},"steps":3,"sequential":true}"#
        )
    }
}
