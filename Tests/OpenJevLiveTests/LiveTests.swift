// A port of upstream OpenJev's `tests/test_live.py` (razorback16/openjev at dcd2094), case by
// case and under its test names. Apache-2.0. See THIRD_PARTY.md.

import Foundation
import OpenJevCore
import Testing

/// End to end against a running OpenJev server, whichever backend and implementation and
/// wherever it runs (issue #41):
///
///     OPENJEV_LIVE_URL=http://127.0.0.1:8080 swift test --filter OpenJevLiveTests
///
/// The DiffusionGemma tests run when the server lists `openjev-latest`, and `test_encoder` runs
/// for whichever of `laya-1.0`, `verdict-1.4`, `clm-v0.1` and `jevk5-0.2` it lists, as upstream's
/// do. `test_unknown_model` runs against every server. Every answer's headers and shapes are
/// checked too (``JevContract``). `test_think`, `test_chat` and `test_chat_stream` wait for the
/// issues that bring their features to this server, so they skip; their bodies are upstream's
/// checks, ready for then. `LiveSettings` lists the variables.
@Suite(
    "test_live.py", .serialized, .enabled(if: LiveSettings.configured, LiveSettings.unsetMessage))
struct LiveTests {
    /// Upstream's `QUESTIONS`, the README example's three questions.
    static let questions: JSONValue = [
        "urgent": [
            "type": "noul", "instructions": "Does the customer need a reply within the hour?",
        ],
        "team": [
            "type": "choice", "instructions": "Which team should handle it?",
            "criteria": [
                "outage": "service down", "billing": "charges, refunds",
                "feature": "requests, how-to",
            ],
        ],
        "tone": [
            "type": "score", "instructions": "How upset is the customer?",
            "criteria": ["calm", "annoyed", "furious"],
        ],
    ]

    /// Upstream's `STATE`.
    static let state = "Everything is down and we have a demo with our biggest client at noon."

    /// The DiffusionGemma model, upstream's `dgemma` fixture's condition.
    static let diffusionGemma = "openjev-latest"

    /// Upstream's `ask`: posts a decision request, requires a 200 and `server-timing`, and
    /// returns the body. The request id and Jev's shapes are checked as well.
    @discardableResult
    static func ask(
        _ questions: JSONValue = questions, state: JSONValue = .string(state),
        model: String = diffusionGemma, extra: [(String, JSONValue)] = [],
        sourceLocation: SourceLocation = #_sourceLocation
    ) async throws -> JSONValue {
        let client = try LiveClient.make()
        var body: JSONObject = ["model": .string(model), "state": state, "questions": questions]
        for (key, value) in extra {
            body.updateValue(value, forKey: key)
        }
        let response = try await client.post("/v1/systemone", json: .object(body))
        try #require(
            response.status == 200, "\(response.status): \(response.text)",
            sourceLocation: sourceLocation)
        JevContract.checkHeaders(
            response, gateway: client.settings.gateway, sourceLocation: sourceLocation)
        let answer = try response.json()
        let thought = extra.contains { $0.0 == "think" && $0.1 != 0 }
        JevContract.checkBody(
            answer, questions: questions.objectValue ?? [:], requested: model,
            listed: try await ModelListing.shared.names(from: client), thought: thought,
            sourceLocation: sourceLocation)
        return answer
    }

    @Test(
        "test_readme_example",
        .enabled("the server at OPENJEV_LIVE_URL does not list openjev-latest") {
            try await ModelListing.lists(diffusionGemma)
        })
    func readmeExample() async throws {
        let answers = try #require(try await Self.ask()["answers"])
        let urgent = answers["urgent"]
        let team = answers["team"]
        let tone = answers["tone"]
        #expect((urgent?["noul"]?.doubleValue ?? 0) > 0.5, "\(JevContract.text(urgent))")
        #expect(team?["choice"]?.stringValue == "outage", "\(JevContract.text(team))")
        #expect((tone?["score"]?.doubleValue ?? 0) > 1.0, "\(JevContract.text(tone))")
        let probabilities = try #require(team?["probabilities"]?.objectValue)
        #expect(abs(JevContract.total(probabilities) - 1) < 1e-6)
    }

    @Test(
        "test_image",
        .enabled("the server at OPENJEV_LIVE_URL does not list openjev-latest") {
            try await ModelListing.lists(diffusionGemma)
        })
    func image() async throws {
        let body = try await Self.ask(
            [
                "hotdog": ["type": "noul", "instructions": "The photo shows a hot dog"],
                "cat": ["type": "noul", "instructions": "The photo shows a cat"],
            ], state: "Look at the photo.", extra: [("images", [try Self.hotdog()])])
        let answers = body["answers"]
        #expect(
            (answers?["hotdog"]?["noul"]?.doubleValue ?? 0) > 0.8
                && (answers?["cat"]?["noul"]?.doubleValue ?? 1) < 0.2,
            "\(JevContract.text(answers))")
        // The image's tokens are counted.
        #expect((body["usage"]?["input_tokens"]?.intValue ?? 0) > 200)
    }

    /// Upstream's `HOTDOG`: `tests/data/hotdog.jpg` as a data URL, read from the pinned checkout
    /// (`make upstream`), since no image of that size is committed here.
    static func hotdog() throws -> JSONValue {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let file = root.appendingPathComponent("Upstream/openjev/tests/data/hotdog.jpg")
        let data = try Data(contentsOf: file)
        return .string("data:image/jpeg;base64," + data.base64EncodedString())
    }

    /// One of `test_read_options`' parameters: an upstream read option and its value.
    struct ReadOption: Sendable, CustomTestStringConvertible {
        let field: String
        let value: JSONValue

        var testDescription: String { "\(field) \(JevContract.text(value))" }

        /// Upstream's `[{"steps": 4}, {"samples": 4}, {"sequential": True}]`.
        static let all = [
            ReadOption(field: "steps", value: 4), ReadOption(field: "samples", value: 4),
            ReadOption(field: "sequential", value: true),
        ]
    }

    @Test(
        "test_read_options",
        .enabled("the server at OPENJEV_LIVE_URL does not list openjev-latest") {
            try await ModelListing.lists(diffusionGemma)
        }, arguments: ReadOption.all)
    func readOptions(_ option: ReadOption) async throws {
        let body = try await Self.ask(extra: [(option.field, option.value)])
        #expect(body["answers"]?["team"]?["choice"]?.stringValue == "outage")
    }

    @Test(
        "test_think",
        .disabled(Waiting.think),
        .enabled("the server at OPENJEV_LIVE_URL does not list openjev-latest") {
            try await ModelListing.lists(diffusionGemma)
        })
    func think() async throws {
        let body = try await Self.ask(extra: [("think", 256)])
        #expect((body["usage"]?["output_tokens"]?.intValue ?? 0) > 0)
        #expect(body["answers"]?["team"]?["choice"]?.stringValue == "outage")
    }

    @Test(
        "test_255_options",
        .enabled("the server at OPENJEV_LIVE_URL does not list openjev-latest") {
            try await ModelListing.lists(diffusionGemma)
        })
    func options255() async throws {
        var criteria = JSONObject()
        for index in 0..<255 {
            criteria.updateValue(.string("option number \(index)"), forKey: "opt\(index)")
        }
        let body = try await Self.ask([
            "pick": [
                "type": "choice", "instructions": "Which option fits?",
                "criteria": .object(criteria),
            ]
        ])
        let probabilities = try #require(body["answers"]?["pick"]?["probabilities"]?.objectValue)
        #expect(probabilities.count == 255)
        #expect(abs(JevContract.total(probabilities) - 1) < 1e-3)
    }

    @Test(
        "test_many_questions_in_chunks",
        .enabled("the server at OPENJEV_LIVE_URL does not list openjev-latest") {
            try await ModelListing.lists(diffusionGemma)
        })
    func manyQuestionsInChunks() async throws {
        var questions = JSONObject()
        for index in 0..<30 {
            questions.updateValue(
                ["type": "noul", "instructions": .string("Is the number \(index) even?")],
                forKey: "q\(index)")
        }
        let body = try await Self.ask(.object(questions), state: "Answer about numbers.")
        // In the questions' order too, which ask's shape check requires.
        #expect(body["answers"]?.objectValue?.count == 30)
    }

    @Test("test_unknown_model")
    func unknownModel() async throws {
        let client = try LiveClient.make()
        let response = try await client.post(
            "/v1/systemone",
            json: ["model": "nope", "state": .string(Self.state), "questions": Self.questions])
        #expect(response.status == 400, "\(response.text)")
        // Jev's shape for a model it does not serve, as Fixtures/wire records it.
        let detail = try response.json()["detail"]
        #expect(detail?["error_type"]?.stringValue == "api_usage_error", "\(response.text)")
        #expect(detail?["message"]?.stringValue == "Unknown model: nope", "\(response.text)")
        JevContract.checkHeaders(response, gateway: client.settings.gateway)
    }

    @Test(
        "test_concurrent_reads",
        .enabled("the server at OPENJEV_LIVE_URL does not list openjev-latest") {
            try await ModelListing.lists(diffusionGemma)
        })
    func concurrentReads() async throws {
        let client = try LiveClient.make()
        let total = 64
        // Upstream's ThreadPoolExecutor(32): at most 32 requests in flight.
        let width = 32
        let request: @Sendable (Int) -> JSONValue = { index in
            [
                "model": .string(Self.diffusionGemma),
                "state": .string("\(Self.state) (ticket \(index))"),
                "questions": Self.questions,
            ]
        }
        let responses = try await withThrowingTaskGroup(of: (Int, LiveResponse).self) { group in
            var results = [LiveResponse?](repeating: nil, count: total)
            var started = 0
            while started < min(width, total) {
                let index = started
                group.addTask {
                    (index, try await client.post("/v1/systemone", json: request(index)))
                }
                started += 1
            }
            while let (index, response) = try await group.next() {
                results[index] = response
                if started < total {
                    let next = started
                    group.addTask {
                        (next, try await client.post("/v1/systemone", json: request(next)))
                    }
                    started += 1
                }
            }
            return results.compactMap { $0 }
        }
        #expect(responses.map(\.status) == Array(repeating: 200, count: total))
        let listed = try await ModelListing.shared.names(from: client)
        for (index, response) in responses.enumerated() {
            JevContract.checkHeaders(response, gateway: client.settings.gateway)
            if response.status == 200 {
                JevContract.checkBody(
                    try response.json(), questions: Self.questions.objectValue ?? [:],
                    requested: Self.diffusionGemma, listed: listed)
            } else {
                Issue.record("ticket \(index): \(response.status): \(response.text)")
            }
        }
        // 64 answers, 64 request ids.
        #expect(Set(responses.compactMap { $0.header("x-request-id") }).count == total)
    }

    @Test(
        "test_chat",
        .disabled(Waiting.chat),
        .enabled("the server at OPENJEV_LIVE_URL does not serve text generation") {
            try await ModelListing.lists("diffusiongemma-26b")
        })
    func chat() async throws {
        let response = try await LiveClient.make().post(
            "/v1/chat/completions",
            json: [
                "model": "diffusiongemma-26b", "max_tokens": 64,
                "messages": [["role": "user", "content": "What is 2+2? Answer with one number."]],
            ])
        try #require(response.status == 200, "\(response.status): \(response.text)")
        let content = try response.json()["choices"]?[0]?["message"]?["content"]?.stringValue
        #expect(content?.contains("4") == true, "\(response.text)")
    }

    @Test(
        "test_chat_stream",
        .disabled(Waiting.chat),
        .enabled("the server at OPENJEV_LIVE_URL does not serve text generation") {
            try await ModelListing.lists("diffusiongemma-26b")
        })
    func chatStream() async throws {
        let response = try await LiveClient.make().post(
            "/v1/chat/completions",
            json: [
                "model": "diffusiongemma-26b", "max_tokens": 64, "stream": true,
                "messages": [["role": "user", "content": "Name a colour of the sky."]],
            ])
        try #require(response.status == 200, "\(response.status): \(response.text)")
        // Python's splitlines also splits on \r, which an event stream may end its lines with.
        let lines = response.text.split(omittingEmptySubsequences: false) {
            $0 == "\n" || $0 == "\r\n" || $0 == "\r"
        }.filter { $0.hasPrefix("data: ") }
        #expect(lines.last == "data: [DONE]", "\(response.text)")
        var text = ""
        for line in lines.dropLast() {
            let chunk = try JSONParser().parse(String(line.dropFirst("data: ".count)))
            guard let choices = chunk["choices"]?.arrayValue, !choices.isEmpty else { continue }
            text += choices[0]["delta"]?["content"]?.stringValue ?? ""
        }
        #expect(!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "\(response.text)")
    }

    /// Upstream's `test_encoder` for one of its four models.
    static func encoder(_ model: String) async throws {
        let answers = try #require(
            try await ask(state: "I was charged twice this month.", model: model)["answers"])
        let team = answers["team"]
        #expect(team?["choice"]?.stringValue == "billing", "\(JevContract.text(team))")
        let probabilities = try #require(team?["probabilities"]?.objectValue)
        #expect(abs(JevContract.total(probabilities) - 1) < 1e-6)
    }

    @Test(
        "test_encoder[laya-1.0]",
        .enabled("the server at OPENJEV_LIVE_URL does not list laya-1.0") {
            try await ModelListing.lists("laya-1.0")
        })
    func encoderLaya() async throws {
        try await Self.encoder("laya-1.0")
    }

    @Test(
        "test_encoder[verdict-1.4]",
        .enabled("the server at OPENJEV_LIVE_URL does not list verdict-1.4") {
            try await ModelListing.lists("verdict-1.4")
        })
    func encoderVerdict() async throws {
        try await Self.encoder("verdict-1.4")
    }

    @Test(
        "test_encoder[clm-v0.1]",
        .enabled("the server at OPENJEV_LIVE_URL does not list clm-v0.1") {
            try await ModelListing.lists("clm-v0.1")
        })
    func encoderCLM() async throws {
        try await Self.encoder("clm-v0.1")
    }

    @Test(
        "test_encoder[jevk5-0.2]",
        .enabled("the server at OPENJEV_LIVE_URL does not list jevk5-0.2") {
            try await ModelListing.lists("jevk5-0.2")
        })
    func encoderJevK5() async throws {
        try await Self.encoder("jevk5-0.2")
    }
}

/// The skip comments of the tests that wait for a feature of this server. Each names the issue,
/// and `OPENJEV_LIVE_URL`, which CI's test log check requires of a skipped live test.
enum Waiting {
    static let think = Comment(
        rawValue: "waits for the think option (#52); it then runs against OPENJEV_LIVE_URL when "
            + "the server lists openjev-latest")
    static let chat = Comment(
        rawValue: "waits for /v1/chat/completions (#53); it then runs against OPENJEV_LIVE_URL "
            + "when the server lists diffusiongemma-26b")
}
