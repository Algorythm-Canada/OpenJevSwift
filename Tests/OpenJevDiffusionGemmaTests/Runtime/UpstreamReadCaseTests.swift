import CoreGraphics
import Foundation
import ImageIO
import MLX
import OpenJevCore
import OpenJevTestSupport
import Testing
import UniformTypeIdentifiers

@testable import OpenJevDiffusionGemma

/// Upstream's tests/test_mlx_model.py `STATES`: each state with its urgent answer, team and
/// tone level, under the README questions.
private let upstreamStates: [(state: String, urgent: Bool, team: String, tone: Int)] = [
    ("Everything is down and we have a demo with our biggest client at noon.", true, "outage", 2),
    (
        "I was charged twice this month. Not urgent, just let me know when it's refunded. Thanks!",
        false, "billing", 0
    ),
    ("Love the product. Any chance you could add a dark mode at some point?", false, "feature", 0),
]

/// A request over `state` with `questions` (JSON object text) and `extra` fields (JSON members
/// without braces, such as `"steps": 4`).
private func ask(_ state: String, _ questions: String = readmeQuestions, _ extra: String = "")
    throws -> SystemOneRequest
{
    let quoted = String(decoding: try JSONEncoder().encode(state), as: UTF8.self)
    let tail = extra.isEmpty ? "" : ", \(extra)"
    return try liveRequest(
        #"{"state": \#(quoted), "model": "openjev-latest", "questions": \#(questions)\#(tail)}"#)
}

/// Upstream's tests/test_mlx_model.py `COLOUR`.
private let colourQuestions =
    #"{"colour": {"type": "choice", "instructions": "What colour fills the picture?", "#
    + #""criteria": {"red": "the image is red", "blue": "the image is blue", "#
    + #""green": "the image is green"}}}"#

/// The JSON member `"images": [...]` of `urls`.
private func images(_ urls: [String]) -> String {
    #""images": ["# + urls.map { "\"\($0)\"" }.joined(separator: ", ") + "]"
}

/// Upstream's `solid_png`: a 64 by 64 PNG of one colour, as a data URL.
private func solidPNG(_ rgb: (Int, Int, Int), size: Int = 64) throws -> String {
    var pixels = [UInt8](repeating: 255, count: size * size * 4)
    for index in 0..<(size * size) {
        pixels[index * 4] = UInt8(rgb.0)
        pixels[index * 4 + 1] = UInt8(rgb.1)
        pixels[index * 4 + 2] = UInt8(rgb.2)
    }
    let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
    let context = try #require(
        CGContext(
            data: &pixels, width: size, height: size, bitsPerComponent: 8,
            bytesPerRow: size * 4, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    let image = try #require(context.makeImage())
    let data = NSMutableData()
    let destination = try #require(
        CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil)
    try #require(CGImageDestinationFinalize(destination))
    return "data:image/png;base64,\((data as Data).base64EncodedString())"
}

/// True when two maps have the same ids and the same float32 logprobs, bit for bit.
private func identical(_ a: ReadOutput, _ b: ReadOutput) -> Bool {
    a.slots.count == b.slots.count
        && zip(a.slots, b.slots).allSatisfy { x, y in
            x.map(\.tokenID) == y.map(\.tokenID)
                && x.map { Float($0.logprob).bitPattern } == y.map { Float($0.logprob).bitPattern }
        }
}

extension MLXTests {
    /// The read cases of upstream's tests/test_mlx_model.py that the earlier live suites did not
    /// cover, named after upstream's, through ``DecisionEngine`` over ``DiffusionGemmaRuntime``.
    /// The README example, same request same answer and the cached prefill are in
    /// RuntimeLiveTests and ReadOracleTests; the think and chat cases wait for their
    /// milestones and are listed at the end as disabled tests.
    @Suite(
        "upstream's test_mlx_model.py read cases",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage))
    struct UpstreamReadCaseTests {
        @Test("test_many_questions_chunk_and_run_in_sequence")
        func manyQuestionsChunkAndRunInSequence() async throws {
            let live = try await LiveCheckpoint.shared()
            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            let questions =
                "{"
                + (0..<24).map {
                    #""k\#($0)": {"type": "noul", "instructions": "The message mentions the number \#($0)"}"#
                }.joined(separator: ", ") + "}"
            let state = "The numbers I care about are 3, 11 and 20."
            for extra in ["", #""sequential": true"#] {
                let before = await live.runtime.statistics()
                let decision = try await engine.decide(ask(state, questions, extra))
                let after = await live.runtime.statistics()
                print(
                    "24 questions\(extra.isEmpty ? "" : ", sequential"): \(after.reads - before.reads) reads, "
                        + "\(decision.inputTokens) input tokens")
                #expect(decision.answers.count == 24)
                // The billed figures measured on 2026-10-01 (docs/09): two groups, the second
                // one's prompt longer by the first group's answers when sequential.
                #expect(decision.inputTokens == (extra.isEmpty ? 657 : 1_147))
                for (key, answer) in decision.answers {
                    guard case .noul(let p) = answer else {
                        Issue.record("\(key): \(answer)")
                        continue
                    }
                    #expect((0...1).contains(p), "\(key)")
                }
                // More than one read: the 24 questions do not fit one canvas.
                #expect(after.reads - before.reads > 1)
            }
        }

        @Test("test_more_steps_still_answer_and_cost_no_more_prompt")
        func moreStepsStillAnswerAndCostNoMorePrompt() async throws {
            let live = try await LiveCheckpoint.shared()
            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            var last: (SystemOneRequest, Decision)?
            for expected in upstreamStates {
                let one = try await engine.decide(ask(expected.state))
                let before = await live.runtime.statistics()
                let request = try ask(expected.state, readmeQuestions, #""steps": 4"#)
                let four = try await engine.decide(request)
                let after = await live.runtime.statistics()
                last = (request, four)
                print("\(expected.state) steps 1: \(one.answers); steps 4: \(four.answers)")
                #expect(one.inputTokens == four.inputTokens, "\(expected.state)")
                #expect(one.outputTokens == four.outputTokens)
                // Every step shares the one prefill steps 1 made.
                #expect(after.prefillMisses == before.prefillMisses, "\(expected.state)")
                #expect(after.prefillHits > before.prefillHits)
                guard case .noul(let urgent)? = four.answers["urgent"],
                    case .choice(let team, _, let confidence)? = four.answers["team"],
                    case .score(let tone, _, _, _)? = four.answers["tone"]
                else {
                    Issue.record("unexpected answer types: \(four.answers)")
                    continue
                }
                #expect((urgent > 0.9) == expected.urgent, "\(expected.state) urgent \(urgent)")
                #expect(team == expected.team && confidence > 0.9, "\(expected.state) \(team)")
                #expect(abs(tone - Double(expected.tone)) < 0.25, "\(expected.state) tone \(tone)")
            }
            // And it stays deterministic: the same answers and usage (the model time differs).
            let (request, four) = try #require(last)
            let again = try await engine.decide(request)
            #expect(again.answers == four.answers)
            #expect(
                again.inputTokens == four.inputTokens && again.outputTokens == four.outputTokens)
        }

        @Test("test_steps_hold_the_template_and_reuse_one_prefill")
        func stepsHoldTheTemplateAndReuseOnePrefill() async throws {
            let live = try await LiveCheckpoint.shared()
            let runtime = live.runtime
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let schema = try engine.schemaBuilder.build(readmeRequest(state: "").questions)
            let resolved = try engine.resolver.resolve(schema.questions, format: schema.format)
            let system = SystemText.render(schema.questions, format: schema.format, chunked: false)
            let state = "The invoice is wrong."
            let prompt = try runtime.tokenizer.chatPromptIDs(
                system: system, user: state, thinking: false)
            let canvas = CanvasBuilder.build(
                template: resolved.template, slots: resolved.slots, seed: 7,
                geometry: engine.configuration.geometry)
            let seen = canvas.tokens
            func canvasRead(steps: Int) -> CanvasRead {
                CanvasRead(
                    prompt: .tokens(prompt), systemText: system, stateText: state,
                    template: resolved.template, slots: resolved.slots, canvas: canvas,
                    steps: steps, seed: 7)
            }

            func run(steps: Int) async throws -> (ReadOutput, misses: Int, cached: Int) {
                await runtime.removeCachedPrefills()
                let before = await runtime.statistics()
                let output = try await runtime.modelRead(canvasRead(steps: steps)).output
                let after = await runtime.statistics()
                return (output, after.prefillMisses - before.prefillMisses, after.cachedPrefills)
            }
            let one = try await run(steps: 1)
            let four = try await run(steps: 4)
            // One prefill however many steps.
            #expect(one.misses == 1 && four.misses == 1)
            #expect(one.cached == 1 && four.cached == 1)
            #expect(canvas.tokens == seen, "the step loop wrote back into the caller's canvas")
            #expect(!identical(one.0, four.0), "four steps returned the single pass's logprobs")
            #expect(four.0.written.count == 3)

            // Four steps again on the cached prefill: a hit, no prefill, the same bits.
            let before = await runtime.statistics()
            let again = try await runtime.modelRead(canvasRead(steps: 4)).output
            let after = await runtime.statistics()
            #expect(after.prefillHits == before.prefillHits + 1)
            #expect(after.prefillMisses == before.prefillMisses)
            #expect(identical(again, four.0))

            // read() as it was before the step loop: one decoder pass, no self-conditioning, a
            // log-softmax per slot. steps 1 must be exactly that.
            let model = live.loaded.model
            let cache = try model.prefill(promptIDs: prompt)
            let ids = MLXArray(canvas.tokens.map(Int32.init)).reshaped(1, canvas.tokens.count)
            let logits = model.decoderLogits(
                canvas: ids, cache: cache, conditioning: nil,
                masks: model.decoderMasks(canvasLength: canvas.tokens.count, cache: cache))
            let single = ReadOutput(
                slots: resolved.slots.map {
                    DiffusionGemmaModel.slotLogprobs(
                        row: logits[0, $0.position], labelIDs: $0.labelIDs,
                        topK: DiffusionGemmaRuntime.topK)
                }, written: [], promptTokens: cache.promptTokens)
            #expect(identical(single, one.0))
            #expect(single.promptTokens == one.0.promptTokens)
        }

        @Test("steps 1 answers and bills as a request without the field (#43)")
        func stepsOneIsTheDefault() async throws {
            let live = try await LiveCheckpoint.shared()
            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            let requests = [
                try readmeRequest(state: upstreamStates[0].state), try quickstartRequest(),
            ]
            for request in requests {
                var explicit = request
                explicit.steps = 1
                let plain = try await engine.decide(request)
                let one = try await engine.decide(explicit)
                // Answers and usage; the model time differs from run to run.
                #expect(one.answers == plain.answers)
                #expect(one.inputTokens == plain.inputTokens)
                #expect(one.outputTokens == plain.outputTokens)
            }
        }

        @Test("steps 8 reads on one prefill, bills it once and holds the template (#43)")
        func eightSteps() async throws {
            let live = try await LiveCheckpoint.shared()
            let runtime = live.runtime
            let engine = try DecisionEngine(backend: runtime, configuration: .default)

            /// The request at `steps`, on a fresh prefill cache: its decision, its reads, its
            /// prefill misses and hits, and its prompt length.
            func decide(_ request: SystemOneRequest, steps: Int) async throws
                -> (decision: Decision, reads: Int, misses: Int, hits: Int, promptTokens: Int)
            {
                var request = request
                request.steps = steps
                let schema = try engine.schemaBuilder.build(request.questions)
                let system = SystemText.render(
                    schema.questions, format: schema.format, chunked: false)
                let prompt = try runtime.tokenizer.chatPromptIDs(
                    system: system, user: StateText.render(request.state), thinking: false)
                await runtime.removeCachedPrefills()
                let before = await runtime.statistics()
                let decision = try await engine.decide(request)
                let after = await runtime.statistics()
                return (
                    decision, after.reads - before.reads,
                    after.prefillMisses - before.prefillMisses,
                    after.prefillHits - before.prefillHits, prompt.count
                )
            }

            for expected in upstreamStates {
                let request = try ask(expected.state)
                let one = try await decide(request, steps: 1)
                let eight = try await decide(request, steps: 8)
                print(
                    "\(expected.state) steps 8: \(eight.reads) reads, \(eight.misses) prefills, "
                        + "\(eight.hits) hits, \(eight.decision.inputTokens) input tokens; "
                        + "\(eight.decision.answers)")
                // One prefill for all steps and every read; the prompt billed once.
                #expect(eight.misses == 1, "\(expected.state)")
                #expect(eight.hits == eight.reads - 1, "\(expected.state)")
                #expect(eight.decision.inputTokens == eight.promptTokens)
                #expect(eight.decision.inputTokens == one.decision.inputTokens)
                #expect(eight.decision.outputTokens == 0)
                guard case .noul(let urgent)? = eight.decision.answers["urgent"],
                    case .choice(let team, _, let confidence)? = eight.decision.answers["team"],
                    case .score(let tone, _, _, _)? = eight.decision.answers["tone"]
                else {
                    Issue.record("unexpected answer types: \(eight.decision.answers)")
                    continue
                }
                // upstream's thresholds for these states (test_mlx_model.py STATES).
                #expect((urgent > 0.9) == expected.urgent, "\(expected.state) urgent \(urgent)")
                #expect(team == expected.team && confidence > 0.9, "\(expected.state) \(team)")
                #expect(abs(tone - Double(expected.tone)) < 0.25, "\(expected.state) tone \(tone)")
            }

            let quickstart = try quickstartRequest()
            let one = try await decide(quickstart, steps: 1)
            let eight = try await decide(quickstart, steps: 8)
            print(
                "quickstart steps 1: \(one.reads) reads, \(one.decision.answers); steps 8: "
                    + "\(eight.reads) reads, \(eight.misses) prefills, \(eight.hits) hits, "
                    + "\(eight.decision.inputTokens) input tokens, \(eight.decision.answers)")
            #expect(eight.misses == 1)
            #expect(eight.hits == eight.reads - 1)
            #expect(eight.decision.inputTokens == eight.promptTokens)
            #expect(eight.decision.inputTokens == one.decision.inputTokens)
            #expect(eight.decision.answers.keys == quickstart.questions.keys)

            // The template holds: the step loop replayed with the model's own decoder passes
            // changes only slot positions, and the runtime's eight-step read is that replay, bit
            // for bit.
            let schema = try engine.schemaBuilder.build(readmeRequest(state: "").questions)
            let resolved = try engine.resolver.resolve(schema.questions, format: schema.format)
            let system = SystemText.render(schema.questions, format: schema.format, chunked: false)
            let state = upstreamStates[0].state
            let prompt = try runtime.tokenizer.chatPromptIDs(
                system: system, user: state, thinking: false)
            let canvas = CanvasBuilder.build(
                template: resolved.template, slots: resolved.slots, seed: 11,
                geometry: engine.configuration.geometry)
            let read = try await runtime.modelRead(
                CanvasRead(
                    prompt: .tokens(prompt), systemText: system, stateText: state,
                    template: resolved.template, slots: resolved.slots, canvas: canvas, steps: 8,
                    seed: 11)
            ).output
            #expect(read.written.count == 7)

            let model = live.loaded.model
            let cache = try model.prefill(promptIDs: prompt)
            let length = canvas.tokens.count
            let ids = MLXArray(canvas.tokens.map(Int32.init)).reshaped(1, length)
            let masks = model.decoderMasks(canvasLength: length, cache: cache)
            let slotPositions = resolved.slots.map(\.position)
            let positions = MLXArray(slotPositions.map(Int32.init))
            let isSlot = Set(slotPositions)
            var conditioning: MLXArray?
            var written: [[Int]] = []
            var changedTemplatePositions = 0
            var changedSlotPositions = 0
            var previous = canvas.tokens
            for _ in 1..<8 {
                let logits = model.decoderLogits(
                    canvas: ids, cache: cache, conditioning: conditioning, masks: masks)
                ids[0, positions] = argMax(logits[0, positions], axis: -1).asType(ids.dtype)
                conditioning = logits
                eval(ids, logits)
                let now = ids[0].asArray(Int32.self).map(Int.init)
                for index in now.indices where now[index] != previous[index] {
                    if isSlot.contains(index) {
                        changedSlotPositions += 1
                    } else {
                        changedTemplatePositions += 1
                    }
                }
                #expect(
                    now.indices.allSatisfy { isSlot.contains($0) || now[$0] == canvas.tokens[$0] })
                written.append(slotPositions.map { now[$0] })
                previous = now
            }
            let logits = model.decoderLogits(
                canvas: ids, cache: cache, conditioning: conditioning, masks: masks)
            let replay = ReadOutput(
                slots: resolved.slots.map {
                    DiffusionGemmaModel.slotLogprobs(
                        row: logits[0, $0.position], labelIDs: $0.labelIDs,
                        topK: DiffusionGemmaRuntime.topK)
                }, written: written, promptTokens: cache.promptTokens)
            print(
                "steps 8 template replay: \(changedSlotPositions) slot writes changed a token, "
                    + "\(changedTemplatePositions) template positions changed, written \(written)")
            #expect(changedTemplatePositions == 0)
            #expect(replay.written == read.written)
            #expect(identical(replay, read))
        }

        @Test("test_the_prompt_cache_is_bounded_in_tokens")
        func thePromptCacheIsBoundedInTokens() async throws {
            let live = try await LiveCheckpoint.shared()
            let runtime = live.runtime
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let refund =
                #"{"refund": {"type": "noul", "instructions": "The customer wants a refund"}}"#
            await runtime.removeCachedPrefills()
            let before = await runtime.statistics()
            var tokens: [Int] = []
            for index in 0..<12 {
                let state =
                    "Order \(index) arrived broken. "
                    + String(repeating: "Please help. ", count: 600)
                tokens.append(try await engine.decide(ask(state, refund)).inputTokens)
            }
            let after = await runtime.statistics()
            let budget = runtime.configuration.promptCacheTokens
            print(
                "12 long prompts (\(tokens.min() ?? 0) to \(tokens.max() ?? 0) tokens): "
                    + "\(after.cachedPrefills) cached, \(after.cachedPrefillTokens) tokens of "
                    + "\(budget), \(after.prefillMisses - before.prefillMisses) prefills")
            #expect(after.prefillMisses - before.prefillMisses == 12)
            #expect((1..<12).contains(after.cachedPrefills))
            #expect(after.cachedPrefillTokens <= budget)
            #expect(budget == PrefillCacheDefaults.tokens)
        }

        /// A solid colour is the least ambiguous thing an image can say, so a wrong or hedged
        /// answer here is a real failure, not flakiness.
        @Test("test_the_model_reads_an_image")
        func theModelReadsAnImage() async throws {
            let live = try await LiveCheckpoint.shared()
            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            for (colour, rgb) in [("red", (255, 0, 0)), ("blue", (0, 0, 255))] {
                let decision = try await engine.decide(
                    ask("What colour is this?", colourQuestions, images([try solidPNG(rgb)])))
                guard case .choice(let choice, let probabilities, _) = decision.answers["colour"]
                else {
                    Issue.record("no choice: \(decision.answers)")
                    continue
                }
                let p = probabilities[colour] ?? 0
                print("\(colour): \(choice) at \(p), \(decision.inputTokens) input tokens")
                #expect(choice == colour && p > 0.9, "\(colour): \(choice) \(p)")
                #expect(decision.inputTokens > 100)
            }
        }

        /// Mirrors test_a_cached_prefill_reads_the_same: the decoder pass must leave an image
        /// prompt's cache as it found it, or re-reads would drift.
        @Test("test_an_image_prefill_reads_the_same_cold_or_reused")
        func anImagePrefillReadsTheSameColdOrReused() async throws {
            let live = try await LiveCheckpoint.shared()
            let runtime = live.runtime
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let image = images([try solidPNG((255, 0, 0))])
            let request = try ask("What colour is this?", colourQuestions, image)
            await runtime.removeCachedPrefills()
            let cold = try await engine.decide(request)
            #expect(
                await runtime.statistics().cachedPrefills > 0, "the image prefill was not cached")
            let before = await runtime.statistics()
            // This one hits the cached vision pass.
            let reused = try await engine.decide(request)
            let after = await runtime.statistics()
            #expect(after.prefillMisses == before.prefillMisses)
            #expect(try PolicyFixtures.body(of: reused) == PolicyFixtures.body(of: cold))
            let samples = try ask(
                "What colour is this?", colourQuestions, image + #", "samples": 3"#)
            #expect(
                try PolicyFixtures.body(of: await engine.decide(samples))
                    == PolicyFixtures.body(of: await engine.decide(samples)))
        }

        // The cases that wait for later milestones. Each runs with OPENJEV_TEST_MODEL once the
        // feature it needs exists; until then the backend refuses it.

        @Test(
            "test_think_answers_and_is_billed",
            .disabled("needs think (#52); runs with OPENJEV_TEST_MODEL once #52 lands"))
        func thinkAnswersAndIsBilled() {}

        @Test(
            "test_think_works_with_sequential",
            .disabled("needs think (#52); runs with OPENJEV_TEST_MODEL once #52 lands"))
        func thinkWorksWithSequential() {}

        @Test(
            "test_think_still_gets_its_thought",
            .disabled("needs think (#52); runs with OPENJEV_TEST_MODEL once #52 lands"))
        func thinkStillGetsItsThought() {}

        /// Why the chat cases skip until the model generates text.
        static let waitingForGeneration = Comment(
            rawValue: "needs the model's generation (#51) behind the chat routes (#53's "
                + "follow-up); runs with OPENJEV_TEST_MODEL then")

        /// The chat service over the checkpoint, through the backend's ``TextGenerator``
        /// conformance, which the model's generation brings (#51, then #53's follow-up wires the
        /// `mlx` backend's routes).
        static func chatService() async throws -> ChatCompletions {
            let live = try await LiveCheckpoint.shared()
            let engine = try DecisionEngine(backend: live.runtime, configuration: .default)
            let generator = try #require(
                engine.textGenerator, "DiffusionGemmaRuntime does not generate text yet")
            return ChatCompletions(generator: generator)
        }

        /// Upstream's `chat()` request: one short question, 64 tokens at most.
        static let chatRequest: JSONValue = [
            "model": "diffusiongemma-26b", "max_tokens": 64,
            "messages": [
                [
                    "role": "user",
                    "content": "What is the capital of France? Answer in one short sentence.",
                ]
            ],
        ]

        /// `body` with `stream: true` and `stream_options.include_usage`, the stream's text and
        /// whether it carried a usage event.
        static func streamed(
            _ chat: ChatCompletions, _ body: JSONValue
        ) async throws -> (text: String, usage: Bool) {
            var object = try #require(body.objectValue)
            object.updateValue(true, forKey: "stream")
            object.updateValue(["include_usage": true], forKey: "stream_options")
            let stream = try await chat.stream(try await chat.prepare(.object(object)))
            var text = ""
            var usage = false
            _ = try await stream.run { event in
                let payload = String(event.dropFirst("data: ".count).dropLast(2))
                guard payload != "[DONE]" else { return }
                let chunk = try JSONParser().parse(payload)
                usage = usage || chunk["usage"] != nil
                text += chunk["choices"]?[0]?["delta"]?["content"]?.stringValue ?? ""
            }
            return (text, usage)
        }

        @Test(
            "test_chat_completion_generates_text",
            .disabled(Self.waitingForGeneration))
        func chatCompletionGeneratesText() async throws {
            let chat = try await Self.chatService()
            let reply = try await chat.complete(try await chat.prepare(Self.chatRequest))
            #expect(reply.content.contains("Paris"), "\(reply.content)")
            #expect([.stop, .length].contains(reply.finishReason))
            #expect(reply.usage.completionTokens > 0)
        }

        /// Greedy generation, same prompt: the streamed pieces join to the whole reply.
        @Test(
            "test_chat_stream_matches_the_whole_reply",
            .disabled(Self.waitingForGeneration))
        func chatStreamMatchesTheWholeReply() async throws {
            let chat = try await Self.chatService()
            let whole = try await chat.complete(try await chat.prepare(Self.chatRequest)).content
            let (streamed, usage) = try await Self.streamed(chat, Self.chatRequest)
            #expect(streamed == whole)
            #expect(usage, "include_usage asked for, none sent")
        }

        @Test(
            "test_chat_json_mode_returns_one_object",
            .disabled(Self.waitingForGeneration))
        func chatJSONModeReturnsOneObject() async throws {
            let chat = try await Self.chatService()
            let body: JSONValue = [
                "model": "diffusiongemma-26b", "max_tokens": 128,
                "response_format": ["type": "json_object"],
                "messages": [
                    ["role": "user", "content": #"Give the capital of France as {"city": ...}."#]
                ],
            ]
            let reply = try await chat.complete(try await chat.prepare(body))
            // The reply is exactly one JSON value, no prose or fences.
            #expect(throws: Never.self) { _ = try JSONParser().parse(reply.content) }
        }

        /// No reply may show the thought channel's markers, on either path. Emptiness is counted
        /// rather than asserted per reply: the checkpoint returns an empty generation for an
        /// identical greedy prompt about once in thirty, upstream measured.
        @Test(
            "test_no_reply_leaks_the_thought_channel",
            .disabled(Self.waitingForGeneration))
        func noReplyLeaksTheThoughtChannel() async throws {
            let chat = try await Self.chatService()
            let prompts = [
                "Count: one two three four five", "Name one prime number",
                "What colour is the sky?", "Say hello.", "Give one European capital.",
            ]
            var replies: [String] = []
            for _ in 0..<3 {
                for prompt in prompts {
                    let body: JSONValue = [
                        "model": "diffusiongemma-26b", "max_tokens": 40,
                        "messages": [["role": "user", "content": .string(prompt)]],
                    ]
                    let whole = try await chat.complete(try await chat.prepare(body)).content
                    #expect(!whole.contains("channel"), "\(prompt): \(whole)")
                    let (streamed, _) = try await Self.streamed(chat, body)
                    #expect(!streamed.contains("channel"), "\(prompt): \(streamed)")
                    replies += [whole, streamed]
                }
            }
            let nonEmpty = replies.filter {
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            #expect(Double(nonEmpty.count) >= 0.6 * Double(replies.count), "\(replies)")
        }
    }
}
