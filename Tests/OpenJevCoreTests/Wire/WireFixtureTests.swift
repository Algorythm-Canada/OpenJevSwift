import Foundation
import OpenJevCore
import Testing

/// Compares the wire types with upstream's recorded exchanges in Fixtures/wire.
///
/// Every response body in cases.json is rebuilt from the wire types and compared byte for byte.
/// Every request that reaches upstream's body validation is run through ``RequestValidator``:
/// a recorded 422 or unknown-question-type 400 must be reproduced byte for byte, and any other
/// outcome means validation passed, so the validator must accept the body.
///
/// Parts of those exchanges belong to later issues and are asserted here only as far as the
/// bodies go: authentication (401, 403) is #36, the body cap (413), bodies that are not JSON and
/// non-JSON content types are #35, and the unknown model 400, the semantic 400s and the backend
/// 503 come from the server and engine (#35, #38, #10, #18).
@Suite(
    "Wire fixtures", .enabled(if: WireFixtures.exists("cases.json"), WireFixtures.missingMessage))
struct WireFixtureTests {
    private let encoder = WireEncoder()

    @Test("Every recorded response body is rebuilt byte for byte")
    func responseBodies() throws {
        let cases = try #require(WireFixtures.load("cases.json")["cases"]?.arrayValue)
        #expect(cases.count > 150)
        var failures: [String] = []
        for fixture in cases {
            let name = fixture["name"]?.stringValue ?? "?"
            do {
                if let problem = try checkResponse(fixture) {
                    failures.append("\(name): \(problem)")
                }
            } catch {
                failures.append("\(name): \(error)")
            }
        }
        #expect(failures.isEmpty, "\(failures.count) failures: \(failures.prefix(10))")
    }

    @Test("Request validation reproduces every recorded 422 and unknown-type 400")
    func requestValidation() throws {
        let cases = try #require(WireFixtures.load("cases.json")["cases"]?.arrayValue)
        var failures: [String] = []
        var reproduced = 0
        var accepted = 0
        for fixture in cases {
            let name = fixture["name"]?.stringValue ?? "?"
            do {
                switch try checkValidation(fixture) {
                case .reproduced: reproduced += 1
                case .accepted: accepted += 1
                case .notOwned: break
                case .failed(let problem): failures.append("\(name): \(problem)")
                }
            } catch {
                failures.append("\(name): \(error)")
            }
        }
        #expect(failures.isEmpty, "\(failures.count) failures: \(failures.prefix(10))")
        #expect(reproduced >= 92, "only \(reproduced) error bodies reproduced")
        #expect(accepted >= 70, "only \(accepted) valid bodies accepted")
    }

    @Test(
        "Every recorded answer is written byte for byte",
        .enabled(if: WireFixtures.exists("answers.json"), WireFixtures.missingMessage))
    func answers() throws {
        let fixture = try WireFixtures.load("answers.json")
        let rows = try #require(fixture["answers"]?.arrayValue)
        #expect(rows.count >= 15)
        for row in rows {
            let name = row["name"]?.stringValue ?? "?"
            let expected = try #require(row["body_text"]?.stringValue)
            let decoded = try Answer(json: JSONParser().parse(expected))
            #expect(try encoder.string(decoded) == expected, "\(name)")

            // Built from the question and the probabilities; the argmax, score and confidence
            // come from the recording because computing them is issue #16's.
            let question = try Question(json: #require(row["question"]))
            let probabilities = try #require(row["probabilities"]?.arrayValue).compactMap(
                \.doubleValue)
            let built: Answer
            switch (question, decoded) {
            case (.noul, _):
                built = .noul(probabilities[0])
            case (.choice(_, let criteria), .choice(let choice, _, let confidence)):
                built = .choice(
                    choice: choice,
                    probabilities: OrderedMap(
                        uniqueKeysWithValues: Array(zip(criteria.keys, probabilities))),
                    confidence: confidence)
            case (.score(_, let criteria), .score(let score, _, _, let confidence)):
                built = .score(
                    score: score, legend: criteria, probabilities: probabilities,
                    confidence: confidence)
            default:
                Issue.record("\(name): the question and the answer have different types")
                continue
            }
            #expect(try encoder.string(built) == expected, "\(name)")
        }
    }

    @Test(
        "The full response keeps the request's question order",
        .enabled(if: WireFixtures.exists("answers.json"), WireFixtures.missingMessage))
    func fullResponse() throws {
        let fixture = try #require(WireFixtures.load("answers.json")["response"])
        let expected = try #require(fixture["body_text"]?.stringValue)
        let response = try SystemOneResponse(json: JSONParser().parse(expected))
        #expect(try encoder.string(response) == expected)
        let questions = try #require(fixture["questions"]?.objectValue)
        #expect(response.answers.keys == questions.keys)
        #expect(response.model == "openjev-0.1")
        #expect(response.usage == Usage(inputTokens: 123, outputTokens: 0))
    }

    @Test(
        "Every recorded request decodes and re-encodes to its compact bytes",
        .enabled(if: WireFixtures.exists("requests.json"), WireFixtures.missingMessage))
    func requests() throws {
        let rows = try #require(WireFixtures.load("requests.json")["requests"]?.arrayValue)
        #expect(rows.count == 10)
        for row in rows {
            let name = row["name"]?.stringValue ?? "?"
            let compact = try #require(row["compact"]?.stringValue)
            let fromDefault = try SystemOneRequest(
                json: JSONParser().parse(#require(row["default"]?.stringValue)))
            let fromCompact = try SystemOneRequest(json: JSONParser().parse(compact))
            #expect(try encoder.string(fromDefault) == compact, "\(name)")
            #expect(fromDefault == fromCompact, "\(name)")
        }
    }

    @Test(
        "Every recorded model listing decodes and re-encodes identically",
        .enabled(if: WireFixtures.exists("models.json"), WireFixtures.missingMessage))
    func models() throws {
        let listings = try #require(WireFixtures.load("models.json")["listings"]?.arrayValue)
        #expect(listings.count >= 6)
        for listing in listings {
            let backend = listing["backend"]?.stringValue ?? "?"
            let expected = try #require(listing["body_text"]?.stringValue)
            let response = try ModelsResponse(json: JSONParser().parse(expected))
            #expect(try encoder.string(response) == expected, "\(backend)")
            let version = try #require(listing["model_version"]?.stringValue)
            let accepted = try #require(listing["accepted_names"]?.arrayValue).compactMap(
                \.stringValue)
            #expect(accepted.contains(version), "\(backend)")
            #expect(response.models.contains { $0.name == version }, "\(backend)")
        }
    }

    // MARK: Checks

    private enum ValidationOutcome {
        case reproduced
        case accepted
        case notOwned
        case failed(String)
    }

    /// Rebuilds the recorded response body from the wire types. Returns a problem, or `nil`.
    private func checkResponse(_ fixture: JSONValue) throws -> String? {
        let request = try #require(fixture["request"])
        let response = try #require(fixture["response"])
        let status = try #require(response["status"]?.intValue)
        let expected = try #require(response["body_text"]?.stringValue)
        let path = try #require(request["path"]?.stringValue)
        let body = try JSONParser().parse(expected)
        let rebuilt: String
        var headers: [WireError.Header] = []
        var rebuiltStatus = status

        switch (status, path) {
        case (200, "/health"):
            #expect(try HealthResponse(json: body) == .ok)
            rebuilt = try encoder.string(HealthResponse.ok)
        case (200, "/v1/models"):
            rebuilt = try encoder.string(ModelsResponse(json: body))
        case (200, "/v1/systemone"):
            rebuilt = try encoder.string(SystemOneResponse(json: body))
        case (422, _):
            // The recorded items are already trimmed, so they are wrapped as they are.
            let items = try #require(body["detail"]?.arrayValue).map {
                try ValidationErrorItem(json: $0)
            }
            rebuilt = try encoder.string(WireError(status: 422, body: .validation(items)))
        default:
            guard let error = try namedError(for: body, fixture: fixture) else {
                return "no wire error constructor matches this body"
            }
            rebuilt = try encoder.string(error)
            headers = error.headers
            rebuiltStatus = error.status
        }
        if rebuilt != expected {
            return "wrote \(rebuilt)"
        }
        if rebuiltStatus != status {
            return "status \(rebuiltStatus), recorded \(status)"
        }
        let retryAfter = response["headers"]?["retry-after"]?.stringValue
        if headers.first(where: { $0.name == "retry-after" })?.value != retryAfter {
            return "retry-after \(headers), recorded \(retryAfter ?? "none")"
        }
        return nil
    }

    /// The ``WireError`` constructor for a recorded non-422 error body.
    private func namedError(for body: JSONValue, fixture: JSONValue) throws -> WireError? {
        if let detail = body["detail"]?.stringValue {
            return .semantic400(detail)
        }
        let typed = try TypedErrorBody(json: body)
        let status = fixture["response"]?["status"]?.intValue
        switch (status, typed.errorType, typed.message) {
        case (400, "api_usage_error", "Invalid request."):
            return .invalidRequest
        case (400, "api_usage_error", let message) where message.hasPrefix("Unknown model: "):
            let text = try #require(fixture["request"]?["body_text"]?.stringValue)
            let model = try #require(JSONParser().parse(text)["model"]?.stringValue)
            return .unknownModel(model)
        case (401, _, _):
            return .authentication401
        case (403, "authentication_error", _):
            return .authenticationMissing403
        case (403, "permission_error", _):
            return .permission403
        case (413, _, _):
            let limit = try #require(fixture["settings"]?["max_body_bytes"]?.intValue)
            return .bodyTooLarge413(limit: limit)
        case (503, _, _):
            return .backendUnavailable503("ConnectError")
        default:
            return nil
        }
    }

    /// Runs a recorded request through the validator where upstream's body validation ran.
    private func checkValidation(_ fixture: JSONValue) throws -> ValidationOutcome {
        let request = try #require(fixture["request"])
        let response = try #require(fixture["response"])
        let status = try #require(response["status"]?.intValue)
        let expected = try #require(response["body_text"]?.stringValue)
        guard request["method"]?.stringValue == "POST",
            request["path"]?.stringValue == "/v1/systemone",
            ![401, 403, 413].contains(status)
        else {
            return .notOwned  // no body validation took place (#35, #36)
        }
        let contentType = request["headers"]?["content-type"]?.stringValue ?? ""
        guard contentType.hasPrefix("application/json") || contentType.hasSuffix("+json") else {
            return .notOwned  // FastAPI does not parse the body as JSON (#35)
        }
        let bytes = try #require(try WireFixtures.bodyBytes(of: request))
        let body: JSONValue?
        if bytes.isEmpty {
            body = nil
        } else {
            do {
                body = try JSONParser().parse(bytes)
            } catch {
                // Invalid JSON: the 422 with Python's decoder message is #35's to reproduce.
                let recorded = try JSONParser().parse(expected)
                let invalid =
                    recorded["detail"]?[0]?["type"]?.stringValue == "json_invalid"
                    || recorded["detail"]?.stringValue == "There was an error parsing the body"
                return invalid ? .notOwned : .failed("the parser rejected a body upstream read")
            }
        }

        let recordedBody = try JSONParser().parse(expected)
        let isInvalidRequest =
            status == 400 && recordedBody == WireError.invalidRequest.json
        do {
            _ = try RequestValidator().validate(body)
            if status == 422 || isInvalidRequest {
                return .failed("accepted a body upstream rejected with \(expected)")
            }
            return .accepted
        } catch {
            guard status == 422 || isInvalidRequest else {
                return .failed("rejected a body upstream accepted: \(try encoder.string(error))")
            }
            let written = try encoder.string(error)
            if written != expected {
                return .failed("wrote \(written)")
            }
            if error.status != status {
                return .failed("status \(error.status), recorded \(status)")
            }
            return .reproduced
        }
    }
}
