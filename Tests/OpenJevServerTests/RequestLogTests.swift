#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import Logging
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// The request log (issue #40): one info line per request with the method, the path, the
    /// status, the milliseconds and the request id, and never a body, a header value or a query.
    @Suite("Request log")
    struct RequestLogTests {
        @Test("Every request gets one line, refusals included, and nothing it should not show")
        func lines() async throws {
            let recorder = LogRecorder()
            let settings = try ServerSettings(apiKey: "sk-HEADER-VALUE")
            let service = try await ServerHarness.encoderService(
                StubQuestionReadBackend(), settings: try ServerSettings(warmup: false))
            let body: JSONValue = [
                "state": "SECRET STATE", "model": "laya-1.0",
                "questions": ["a": ["type": "noul", "instructions": "SECRET INSTRUCTIONS"]],
            ]
            let responses = try await ServerHarness.withClient(
                settings: settings, service: service, logger: recorder.logger
            ) { client in
                let health = try await ServerHarness.send(client, .get, "/health?token=QUERY")
                let denied = try await ServerHarness.send(client, .get, "/v1/models?key=QUERY")
                let answered = try await ServerHarness.send(
                    client, .post, "/v1/systemone",
                    headers: [
                        "authorization": "Bearer sk-HEADER-VALUE",
                        "content-type": "application/json",
                    ],
                    body: try WireEncoder().bytes(json: body))
                let missing = try await ServerHarness.send(client, .get, "/nope")
                return [health, denied, answered, missing]
            }
            #expect(responses.map(\.status.code) == [200, 403, 200, 404])
            let lines = recorder.lines.filter { $0.level == .info }
            #expect(lines.count == 4)
            let expected = zip(
                [
                    "GET /health 200", "GET /v1/models 403", "POST /v1/systemone 200",
                    "GET /nope 404",
                ], responses)
            for (line, (prefix, response)) in zip(lines, expected) {
                let id = try #require(ServerHarness.header(response, "x-request-id"))
                #expect(line.message.hasPrefix(prefix + " "), "\(line.message)")
                #expect(line.message.hasSuffix("ms " + id), "\(line.message)")
                let fields = line.message.split(separator: " ")
                #expect(fields.count == 5, "\(line.message)")
                let milliseconds = fields[3].dropLast(2)
                #expect(fields[3].hasSuffix("ms") && Double(milliseconds) != nil)
                #expect(milliseconds.split(separator: ".").last?.count == 1)
            }
            for line in recorder.lines {
                for secret in ["QUERY", "SECRET", "HEADER-VALUE", "token=", "key="] {
                    #expect(!line.message.contains(secret), "\(line.message)")
                }
            }
        }

        @Test("A line is method, path, status, milliseconds with one decimal and the request id")
        func format() {
            #expect(
                RequestLogMiddleware.line(
                    method: "POST", path: "/v1/systemone", status: 200,
                    elapsed: .microseconds(41_840), requestID: "req_abc")
                    == "POST /v1/systemone 200 41.8ms req_abc")
            #expect(
                RequestLogMiddleware.line(
                    method: "GET", path: "/health", status: 499, elapsed: .zero, requestID: nil)
                    == "GET /health 499 0.0ms -")
        }
    }
#endif
