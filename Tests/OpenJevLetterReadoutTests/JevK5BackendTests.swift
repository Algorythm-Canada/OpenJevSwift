import Foundation
import OpenJevCore
import OpenJevLetterReadout
import OpenJevServer
import Testing

/// ``JevK5Backend`` over upstream's stub model, through the server as upstream's
/// `tests/test_encoders.py` runs `JevK5Engine` through its API: the three JevK5 tests that apply
/// to a model in process, and what the port adds (the passes one at a time, the first error in
/// question order, the prompt limit's two refusals, the letter check and the calibration file).
///
/// Upstream's fourth test, `test_jevk5_unreachable_vllm_is_a_503`, has no counterpart: there is no
/// vLLM server to be unreachable. A model that throws is answered with the 503 every backend
/// failure gets, which ``modelFailureIsA503()`` checks.
@Suite("JevK5 backend")
struct JevK5BackendTests {
    /// A backend over the stub at upstream's test temperature, 2.0.
    static func backend(
        model: StubLetterModel = StubLetterModel(),
        tokenizer: StubLetterTokenizer = StubLetterTokenizer()
    ) throws -> JevK5Backend {
        try JevK5Backend(model: model, tokenizer: tokenizer, temperature: 2.0)
    }

    @Test("Matches upstream's test_jevk5_reads_letters_under_its_temperature")
    func readsLettersUnderItsTemperature() async throws {
        let model = StubLetterModel()
        let tokenizer = StubLetterTokenizer()
        let service = try await JevK5Server.service(
            try Self.backend(model: model, tokenizer: tokenizer))
        let response = try await JevK5Server.send(
            service, .post, "/v1/systemone", body: UpstreamRequest.body)
        #expect(response.status == 200, "\(response.body)")
        let answers = try #require(response.body["answers"])
        // B is the second option: billing, "annoyed", and false for a noul (A is true).
        #expect(answers["team"]?["choice"]?.stringValue == "billing")
        let team = try #require(answers["team"]?["probabilities"]?.objectValue)
        #expect(team.keys == ["outage", "billing", "feature"])
        expectClose(
            team.values.compactMap(\.doubleValue), UpstreamRequest.softmax([-2.0, -0.5, -3.0]))
        let tone = try #require(answers["tone"]?["probabilities"]?["1"]?.doubleValue)
        expectClose([tone], [UpstreamRequest.softmax([-2.0, -0.5, -3.0])[1]])
        let urgent = try #require(answers["urgent"]?["noul"]?.doubleValue)
        expectClose([urgent], [UpstreamRequest.softmax([-2.0, -0.5])[0]])
        // One pass per question.
        #expect(response.body["usage"]?["input_tokens"]?.intValue == 300)
        #expect(model.passes.map(\.letterIDs.count).sorted() == [2, 3, 3])
        // jevk5's prompt: the chat template with thinking off, the decision as JSON.
        let prompts = tokenizer.prompts
        #expect(prompts.count == 3)
        for prompt in prompts {
            #expect(prompt.hasPrefix("<|im_start|>system\nApply the supplied criterion"))
            #expect(prompt.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
            #expect(prompt.contains("\"evidence\": \"I was charged twice this month.\""))
        }
    }

    @Test("Matches upstream's test_jevk5_reads_more_than_16_options_in_passes")
    func readsMoreThan16OptionsInPasses() async throws {
        let model = StubLetterModel()
        let service = try await JevK5Server.service(try Self.backend(model: model))
        var criteria = JSONObject()
        for index in 0..<20 {
            criteria["o\(index)"] = .string("option \(index)")
        }
        let body: JSONValue = [
            "state": "I was charged twice this month.", "model": "jevk5-0.2",
            "questions": [
                "c": ["type": "choice", "instructions": "Which?", "criteria": .object(criteria)]
            ],
        ]
        let response = try await JevK5Server.send(service, .post, "/v1/systemone", body: body)
        #expect(response.status == 200, "\(response.body)")
        let probabilities = try #require(
            response.body["answers"]?["c"]?["probabilities"]?.objectValue)
        #expect(probabilities.keys == (0..<20).map { "o\($0)" })
        #expect(abs(probabilities.values.compactMap(\.doubleValue).reduce(0, +) - 1) <= 1e-6)
        // Two groups of 10, then a final.
        #expect(model.passes.count == 3)
        #expect(response.body["usage"]?["input_tokens"]?.intValue == 300)
    }

    @Test("Matches upstream's test_jevk5_model_rejection_is_a_400")
    func modelRejectionIsA400() async throws {
        // Every prompt is one token over what a pass holds: vLLM's refusal, upstream's 400.
        let tokenizer = StubLetterTokenizer(promptTokens: 16_384)
        let service = try await JevK5Server.service(try Self.backend(tokenizer: tokenizer))
        let response = try await JevK5Server.send(
            service, .post, "/v1/systemone", body: UpstreamRequest.body)
        #expect(response.status == 400)
        let detail = try #require(response.body["detail"]?.stringValue)
        #expect(detail.contains("maximum context length"))
        #expect(
            detail
                == "the model rejected this request: This model's maximum context length is "
                + "16384 tokens. However, you requested 1 output tokens and your prompt contains "
                + "at least 16384 input tokens, for a total of at least 16385 tokens. Please "
                + "reduce the length of the input prompt or the number of requested output "
                + "tokens. (parameter=input_tokens, value=16384)")
    }

    @Test("A model that throws is the 503 of any backend failure")
    func modelFailureIsA503() async throws {
        let model = StubLetterModel(failures: [1: JevK5ModelError("the GPU is gone")])
        let service = try await JevK5Server.service(try Self.backend(model: model))
        let response = try await JevK5Server.send(
            service, .post, "/v1/systemone", body: UpstreamRequest.body)
        #expect(response.status == 503)
    }

    @Test("GET /v1/models lists jevk5-0.2 as upstream does")
    func modelsListing() async throws {
        let service = try await JevK5Server.service(try Self.backend())
        let response = try await JevK5Server.send(service, .get, "/v1/models")
        #expect(response.status == 200)
        let models = try #require(response.body["models"]?.arrayValue)
        #expect(models.count == 1)
        #expect(models.first?["name"]?.stringValue == "jevk5-0.2")
        #expect(models.first?["description"]?.stringValue == KnownEncoderModels.jevk5.description)
        #expect(models.first?["release_date"]?.stringValue == "2026-09-25")
    }

    @Test("The questions of a batch run concurrently and their passes one at a time")
    func passesRunOneAtATime() async throws {
        let model = StubLetterModel(delay: 0.02)
        let backend = try Self.backend(model: model)
        var questions = OrderedMap<Question>()
        for index in 0..<8 {
            _ = questions.updateValue(
                .noul(instructions: .string("question \(index)"), criteria: nil),
                forKey: "q\(index)")
        }
        let schema = try EncoderQuestionSchemaBuilder(maxChoices: 255).build(questions)
        let result = try await backend.readBatch(
            state: .string("state"), stateText: "state", questions: schema.questions)
        #expect(result.probabilities.count == 8)
        #expect(result.inputTokens == 800)
        #expect(model.passes.count == 8)
        #expect(model.maxConcurrentPasses == 1)
        #expect(await backend.passCount == 8)
    }

    @Test("When several questions fail, the first in question order is reported")
    func firstErrorInQuestionOrder() async throws {
        // Prompts are numbered as they are tokenized, which the concurrent questions do in any
        // order; every question fails here, each with its own prompt's number. The first
        // question is tokenized last, so its pass fails last: the batch must still report its
        // error, not the first one to happen.
        struct Numbered: Error, Equatable { var prompt: Int }
        let failures = Dictionary(uniqueKeysWithValues: (1...4).map { ($0, Numbered(prompt: $0)) })
        let model = StubLetterModel(failures: failures)
        let tokenizer = StubLetterTokenizer(slowText: "question 0", slowSeconds: 0.3)
        let backend = try Self.backend(model: model, tokenizer: tokenizer)
        var questions = OrderedMap<Question>()
        for index in 0..<4 {
            _ = questions.updateValue(
                .noul(instructions: .string("question \(index)"), criteria: nil),
                forKey: "q\(index)")
        }
        let schema = try EncoderQuestionSchemaBuilder(maxChoices: 255).build(questions)
        do {
            _ = try await backend.readBatch(
                state: .string("state"), stateText: "state", questions: schema.questions)
            Issue.record("the batch did not fail")
        } catch let error as Numbered {
            // The first question's prompt holds "question 0", and it was the last to fail.
            let first = try #require(tokenizer.prompts.firstIndex { $0.contains("question 0") })
            #expect(first == tokenizer.prompts.count - 1)
            #expect(error.prompt == first + 1)
        }
    }

    @Test("A pass up to 16,383 tokens is read; one more is refused")
    func tokenLimitBoundary() async throws {
        let questions = try EncoderQuestionSchemaBuilder(maxChoices: 255).build([
            "n": .noul(instructions: .string("x"), criteria: nil)
        ]).questions
        let fits = try Self.backend(tokenizer: StubLetterTokenizer(promptTokens: 16_383))
        let read = try await fits.readBatch(state: "s", stateText: "s", questions: questions)
        #expect(read.inputTokens == 16_383)
        let over = try Self.backend(tokenizer: StubLetterTokenizer(promptTokens: 16_384))
        await #expect(throws: BackendRefusal.self) {
            _ = try await over.readBatch(state: "s", stateText: "s", questions: questions)
        }
        #expect(fits.maxPromptTokens == 16_383)
    }

    @Test("vLLM's two refusals, word for word")
    func refusalMessages() {
        let limit = JevK5PromptLimit(maxCharactersPerToken: 128)
        #expect(limit.maxInputTokens == 16_383)
        #expect(limit.maxInputCharacters == 2_097_024)
        #expect(limit.characterRefusal(characters: 2_097_024) == nil)
        #expect(
            limit.characterRefusal(characters: 2_097_025)?.reason
                == "This model's maximum context length is 16384 tokens. However, you requested "
                + "1 output tokens and your prompt contains 2097025 characters (more than 2097024 "
                + "characters, which is the upper bound for 16383 input tokens). Please reduce the "
                + "length of the input prompt or the number of requested output tokens. "
                + "(parameter=input_text, value=2097025)")
        #expect(limit.tokenRefusal(tokens: 16_383) == nil)
        // vLLM truncates at one token past the bound, so every refused prompt reads the same.
        let expected =
            "This model's maximum context length is 16384 tokens. However, you requested 1 output "
            + "tokens and your prompt contains at least 16384 input tokens, for a total of at least "
            + "16385 tokens. Please reduce the length of the input prompt or the number of "
            + "requested output tokens. (parameter=input_tokens, value=16384)"
        #expect(limit.tokenRefusal(tokens: 16_384)?.reason == expected)
        #expect(limit.tokenRefusal(tokens: 50_000)?.reason == expected)
    }

    @Test("A prompt over the character bound is refused before it is tokenized")
    func characterBoundComesFirst() async throws {
        let tokenizer = StubLetterTokenizer()
        let backend = try JevK5Backend(
            model: StubLetterModel(), tokenizer: tokenizer, temperature: 2.0,
            limit: JevK5PromptLimit(maxCharactersPerToken: 0))
        let questions = try EncoderQuestionSchemaBuilder(maxChoices: 255).build([
            "n": .noul(instructions: .string("x"), criteria: nil)
        ]).questions
        do {
            _ = try await backend.readBatch(state: "s", stateText: "s", questions: questions)
            Issue.record("the prompt was read")
        } catch let refusal as BackendRefusal {
            #expect(refusal.reason.contains("characters (more than 0 characters"))
        }
        #expect(tokenizer.prompts.isEmpty)
    }

    @Test("Every letter must be one token, with upstream's message")
    func letterCheck() {
        struct TwoTokenA: LetterReadoutTokenizing {
            let maxCharactersPerToken = 128
            func encode(_ text: String) -> [Int] {
                text == "A" ? [1, 2] : [Int(text.unicodeScalars.first!.value)]
            }
        }
        #expect(throws: JevK5LoadError.self) {
            _ = try JevK5Backend(model: StubLetterModel(), tokenizer: TwoTokenA(), temperature: 1)
        }
        do {
            _ = try JevK5Backend(model: StubLetterModel(), tokenizer: TwoTokenA(), temperature: 1)
        } catch {
            // The initializer throws JevK5LoadError only.
            #expect(
                error.description
                    == "every answer letter must be one token, got [[1, 2], [66], [67], [68], "
                    + "[69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80]]")
        }
    }

    @Test("A temperature that is not a positive number is refused")
    func temperatureCheck() {
        for temperature in [0.0, -1.0, .infinity, .nan] {
            #expect(throws: JevK5LoadError.self) {
                _ = try JevK5Backend(
                    model: StubLetterModel(), tokenizer: StubLetterTokenizer(),
                    temperature: temperature)
            }
        }
    }

    @Test("jevk5_config.json gives the temperature, as upstream reads it")
    func calibrationFile() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("jevk5-calibration-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        func calibration(_ text: String) throws -> JevK5Calibration {
            let url = folder.appendingPathComponent("jevk5_config.json")
            try Data(text.utf8).write(to: url)
            return try JevK5Calibration(contentsOf: url)
        }
        #expect(try calibration(#"{"temperature": 1.532}"#).temperature == 1.532)
        // An integer is a number too, and other keys are ignored, as upstream ignores them.
        #expect(
            try calibration(#"{"temperature": 2, "knockout_temperature": 0.93}"#).temperature == 2)
        for text in [#"{}"#, #"{"temperature": "1.5"}"#, #"{"temperature": 0}"#, "[1.5]", "nope"] {
            #expect(throws: JevK5LoadError.self, "\(text)") { _ = try calibration(text) }
        }
    }

    /// Upstream's `pytest.approx`: a relative tolerance of 1e-6.
    private func expectClose(
        _ actual: [Double], _ expected: [Double], sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(actual.count == expected.count, sourceLocation: sourceLocation)
        for (a, e) in zip(actual, expected) {
            #expect(
                abs(a - e) <= 1e-6 * abs(e), "\(a) against \(e)", sourceLocation: sourceLocation)
        }
    }
}
