import Foundation
import MLX
import OpenJevCore
import Testing

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
    /// RuntimeLiveTests and ReadOracleTests; the image, think and chat cases wait for their
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

        // The cases that wait for later milestones. Each runs with OPENJEV_TEST_MODEL once the
        // feature it needs exists; until then the backend refuses it.

        @Test(
            "test_the_model_reads_an_image",
            .disabled("needs images (#48); runs with OPENJEV_TEST_MODEL once #48 lands"))
        func theModelReadsAnImage() {}

        @Test(
            "test_an_image_prefill_reads_the_same_cold_or_reused",
            .disabled("needs images (#48); runs with OPENJEV_TEST_MODEL once #48 lands"))
        func anImagePrefillReadsTheSameColdOrReused() {}

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

        @Test(
            "test_chat_completion_generates_text",
            .disabled(
                "needs /v1/chat/completions (#53); runs with OPENJEV_TEST_MODEL once #53 lands"))
        func chatCompletionGeneratesText() {}

        @Test(
            "test_chat_stream_matches_the_whole_reply",
            .disabled(
                "needs /v1/chat/completions (#53); runs with OPENJEV_TEST_MODEL once #53 lands"))
        func chatStreamMatchesTheWholeReply() {}

        @Test(
            "test_chat_json_mode_returns_one_object",
            .disabled(
                "needs /v1/chat/completions (#53); runs with OPENJEV_TEST_MODEL once #53 lands"))
        func chatJSONModeReturnsOneObject() {}

        @Test(
            "test_no_reply_leaks_the_thought_channel",
            .disabled(
                "needs /v1/chat/completions (#53); runs with OPENJEV_TEST_MODEL once #53 lands"))
        func noReplyLeaksTheThoughtChannel() {}
    }
}
