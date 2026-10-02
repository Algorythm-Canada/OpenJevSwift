import Foundation
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// Every read an instrumented runtime makes, each with its prompt's token ids.
///
/// `@unchecked Sendable` because the runtime actor makes every call, one at a time; the lock only
/// guards the test's reads of what was recorded.
private final class CallRecorder: @unchecked Sendable {
    struct Read {
        /// The prompt the read's prefill was made from.
        var prompt: [Int]
        var canvas: [Int]
        var steps: Int
        /// The read through `slot_distribution`, as the engine sees it.
        var result: ReadResult
    }

    private let lock = NSLock()
    private var promptOfCache: [ObjectIdentifier: [Int]] = [:]
    /// Every cache made, kept alive so that no identifier is reused after an eviction.
    private var caches: [PromptCache] = []
    private var recordedReads: [Read] = []

    var reads: [Read] { lock.withLock { recordedReads } }

    func prefilled(_ ids: [Int], _ cache: PromptCache) {
        lock.withLock {
            caches.append(cache)
            promptOfCache[ObjectIdentifier(cache)] = ids
        }
    }

    func read(
        canvas: [Int], slots: [SlotRequest], cache: PromptCache, steps: Int, output: ReadOutput
    ) {
        lock.withLock {
            recordedReads.append(
                Read(
                    prompt: promptOfCache[ObjectIdentifier(cache)] ?? [], canvas: canvas,
                    steps: steps, result: output.readResult(for: slots)))
        }
    }

    /// The reads grouped by prompt, in the order the prompts were first read.
    var readsByPrompt: [(prompt: [Int], reads: [Read])] {
        var out: [(prompt: [Int], reads: [Read])] = []
        for read in reads {
            if let index = out.firstIndex(where: { $0.prompt == read.prompt }) {
                out[index].reads.append(read)
            } else {
                out.append((read.prompt, [read]))
            }
        }
        return out
    }
}

/// A runtime over the shared checkpoint's model whose calls are recorded, built with the
/// internal ``DiffusionGemmaRuntime/init(tokenizer:configuration:calls:setCacheLimit:)``: its own
/// prefill cache and statistics, the one 16 GB model.
private func instrumentedRuntime() async throws -> (DiffusionGemmaRuntime, CallRecorder) {
    let live = try await LiveCheckpoint.shared()
    let recorder = CallRecorder()
    let model = live.loaded.model
    let runtime = DiffusionGemmaRuntime(
        tokenizer: live.runtime.tokenizer, configuration: live.runtime.configuration,
        calls: .init(
            prefill: { ids in
                let cache = try model.prefill(promptIDs: ids)
                recorder.prefilled(ids, cache)
                return cache
            },
            read: { canvas, slots, cache, steps, topK in
                let output = try model.read(
                    canvas: canvas, slots: slots, cache: cache, steps: steps, topK: topK)
                recorder.read(
                    canvas: canvas, slots: slots, cache: cache, steps: steps, output: output)
                return output
            }),
        setCacheLimit: { _ in })
    return (runtime, recorder)
}

/// `count` noul questions `k0`, `k1`, ... asking whether the message mentions the number.
private func numberQuestions(_ count: Int) -> String {
    "{"
        + (0..<count).map {
            #""k\#($0)": {"type": "noul", "instructions": "The message mentions the number \#($0)"}"#
        }.joined(separator: ", ") + "}"
}

/// The state of upstream's test_many_questions_chunk_and_run_in_sequence.
private let numbersState = "The numbers I care about are 3, 11 and 20."

/// A request over `state` with `questions` (JSON object text) and `extra` fields (JSON members
/// without braces).
private func request(_ state: String, _ questions: String, _ extra: String = "") throws
    -> SystemOneRequest
{
    let quoted = String(decoding: try JSONEncoder().encode(state), as: UTF8.self)
    let tail = extra.isEmpty ? "" : ", \(extra)"
    return try liveRequest(
        #"{"state": \#(quoted), "model": "openjev-latest", "questions": \#(questions)\#(tail)}"#)
}

private func entropies(_ read: CallRecorder.Read) -> String {
    "[" + read.result.slots.map { String(format: "%.3f", $0.entropy) }.joined(separator: ", ")
        + "]"
}

extension MLXTests {
    /// samples, the automatic re-read policy (#44) and sequential mode (#45) through
    /// ``DecisionEngine`` over a ``DiffusionGemmaRuntime`` on the checkpoint, with
    /// ``EngineConfiguration/default`` (autoThreshold 0.1, autoMax 4). Each test builds a fresh
    /// instrumented runtime, so its prefill cache starts empty.
    @Suite(
        "samples, re-reads and sequential on the checkpoint",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage))
    struct ReadPolicyLiveTests {
        @Test("An uncertain read is read autoMax times on one prefill and billed once (#44)")
        func automaticRereads() async throws {
            let (runtime, recorder) = try await instrumentedRuntime()
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let quickstart = try quickstartRequest()
            let decision = try await engine.decide(quickstart)
            let statistics = await runtime.statistics()
            let reads = recorder.reads
            for (k, read) in reads.enumerated() {
                print("quickstart read \(k): entropies \(entropies(read))")
            }
            print("quickstart under the automatic policy: \(decision.answers)")
            let first = try #require(reads.first)
            let largest = first.result.slots.map(\.entropy).max() ?? 0
            #expect(largest > engine.configuration.autoThreshold, "entropy \(largest)")
            #expect(statistics.reads == engine.configuration.autoMax)
            #expect(reads.count == 4)
            // One prompt, one prefill, three hits; four distinct canvases (seeds seed + 7919 k).
            #expect(Set(reads.map(\.prompt)).count == 1)
            #expect(statistics.prefillMisses == 1 && statistics.prefillHits == 3)
            #expect(Set(reads.map(\.canvas)).count == 4)
            // Billed as one read: the prompt's tokens once.
            #expect(decision.inputTokens == first.result.promptTokens)
            #expect(decision.inputTokens == first.prompt.count)
            #expect(decision.outputTokens == 0)
        }

        @Test("samples N is N billed reads; samples 1 one read whatever the entropy (#44)")
        func samples() async throws {
            let (runtime, recorder) = try await instrumentedRuntime()
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let quickstart = try quickstartRequest()

            func decide(samples: Int) async throws
                -> (decision: Decision, reads: Int, misses: Int, hits: Int)
            {
                var request = quickstart
                request.samples = samples
                await runtime.removeCachedPrefills()
                let before = await runtime.statistics()
                let decision = try await engine.decide(request)
                let after = await runtime.statistics()
                return (
                    decision, after.reads - before.reads,
                    after.prefillMisses - before.prefillMisses,
                    after.prefillHits - before.prefillHits
                )
            }

            let one = try await decide(samples: 1)
            let largest = recorder.reads.first?.result.slots.map(\.entropy).max() ?? 0
            print("samples 1: entropy \(largest), \(one.decision.inputTokens) input tokens")
            // The quickstart's first read is above the threshold, and samples 1 still reads once.
            #expect(largest > engine.configuration.autoThreshold)
            #expect(one.reads == 1 && one.misses == 1 && one.hits == 0)
            let promptTokens = one.decision.inputTokens

            let four = try await decide(samples: 4)
            print("samples 4: \(four.reads) reads, \(four.decision.inputTokens) input tokens")
            #expect(four.reads == 4)
            #expect(four.decision.inputTokens == 4 * promptTokens)
            #expect(four.misses == 1 && four.hits == 3)

            let eight = try await decide(samples: 8)
            print(
                "samples 8: \(eight.reads) reads, \(eight.misses) prefill misses, \(eight.hits) "
                    + "hits, \(eight.decision.inputTokens) input tokens")
            #expect(eight.reads == 8)
            #expect(eight.misses == 1 && eight.hits == 7)
            #expect(eight.decision.inputTokens == 8 * promptTokens)
        }

        @Test("One uncertain question re-reads every question of its group, per group (#44)")
        func rereadsPerGroup() async throws {
            let (runtime, recorder) = try await instrumentedRuntime()
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let decision = try await engine.decide(request(numbersState, numberQuestions(24)))
            let groups = recorder.readsByPrompt
            #expect(groups.count == 2)
            var billed = 0
            // The groups read at once, so their order here is the order they reached the model.
            for (k, group) in groups.enumerated() {
                let first = try #require(group.reads.first)
                let slotEntropies = first.result.slots.map(\.entropy)
                let uncertain = slotEntropies.filter { $0 > engine.configuration.autoThreshold }
                print(
                    "24 questions, a group of \(first.result.slots.count) questions: "
                        + "\(uncertain.count) above the threshold, \(group.reads.count) reads; "
                        + "entropies \(entropies(first))")
                let expected = uncertain.isEmpty ? 1 : engine.configuration.autoMax
                #expect(group.reads.count == expected, "prompt \(k)")
                // Every re-read reads every question of the group.
                #expect(
                    group.reads.allSatisfy {
                        $0.result.slots.count == first.result.slots.count
                    })
                billed += first.result.promptTokens
            }
            #expect(decision.inputTokens == billed)
            #expect(decision.inputTokens == 657)
        }

        @Test("sequential: one read per group in series, carrying the earlier answers (#45)")
        func sequentialGroups() async throws {
            let (runtime, recorder) = try await instrumentedRuntime()
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let tokenizer = runtime.tokenizer
            // samples 1 so the count is one read per group whatever the entropies.
            let sequential = try request(
                numbersState, numberQuestions(40), #""sequential": true, "samples": 1"#)
            let schema = try engine.schemaBuilder.build(sequential.questions)
            let format = schema.format
            let groups = try ReadGrouping.groups(
                schema.questions, format: format, geometry: engine.configuration.geometry,
                scaffold: engine.tokens.scaffold, tokenizer: tokenizer)
            #expect(groups.count >= 3)
            let system = SystemText.render(schema.questions, format: format, chunked: false)
            let chat = try tokenizer.chatPromptIDs(
                system: system, user: StateText.render(sequential.state), thinking: false)
            let base = chat + engine.tokens.scaffold

            let decision = try await engine.decide(sequential)
            let statistics = await runtime.statistics()
            let reads = recorder.reads
            #expect(statistics.reads == groups.count)
            #expect(reads.count == groups.count)
            // Every extended prefix is its own cache key: one miss per group, no hit.
            #expect(statistics.prefillMisses == groups.count && statistics.prefillHits == 0)
            #expect(Set(reads.map(\.prompt)).count == groups.count)

            // Group 0 reads the plain chat prompt under the full question list; group k the base
            // prompt and the answer lines of groups 0 to k-1, the argmax label of each question.
            var lines: [String] = []
            for (k, (read, group)) in zip(reads, groups).enumerated() {
                #expect(read.result.slots.count == group.count, "group \(k)")
                if k == 0 {
                    #expect(read.prompt == chat)
                } else {
                    let joined = lines.joined(separator: format.join)
                    #expect(read.prompt.starts(with: base), "group \(k)")
                    #expect(
                        read.prompt
                            == base + (try tokenizer.encode(joined, addSpecialTokens: false)),
                        "group \(k)")
                    let continuation = try tokenizer.decode(
                        Array(read.prompt.dropFirst(base.count)), skipSpecialTokens: false)
                    for (j, line) in lines.enumerated() {
                        #expect(
                            continuation.contains(line), "group \(k) lacks group \(j)'s answers")
                    }
                    print(
                        "sequential group \(k): \(read.prompt.count) prompt tokens, base "
                            + "\(base.count), continuation \(continuation.debugDescription)")
                }
                let chosen = read.result.slots.map {
                    ReadDivergence.firstLargest($0.probabilities)
                }
                lines.append(AnswerText.render(group, labelIndices: chosen, format: format))
            }
            // Billed per group: the sum of the groups' prompts.
            let billed = reads.reduce(0) { $0 + $1.result.promptTokens }
            print(
                "40 questions, sequential: \(groups.count) groups of "
                    + "\(groups.map(\.count)), \(reads.count) reads, prompts "
                    + "\(reads.map(\.prompt.count)), \(decision.inputTokens) input tokens")
            #expect(decision.inputTokens == billed)
            #expect(decision.inputTokens == reads.reduce(0) { $0 + $1.prompt.count })

            // The same request again hits every group's prefill and answers the same.
            let before = await runtime.statistics()
            let again = try await engine.decide(sequential)
            let after = await runtime.statistics()
            #expect(after.prefillHits - before.prefillHits == groups.count)
            #expect(after.prefillMisses == before.prefillMisses)
            #expect(again.answers == decision.answers)
            #expect(again.inputTokens == decision.inputTokens)
        }

        @Test("sequential with a single group answers and bills as the plain request (#45)")
        func sequentialSingleGroup() async throws {
            let (runtime, recorder) = try await instrumentedRuntime()
            let engine = try DecisionEngine(backend: runtime, configuration: .default)
            let plain = try quickstartRequest()
            var sequential = plain
            sequential.sequential = true
            let a = try await engine.decide(plain)
            let plainPrompts = recorder.reads.map(\.prompt)
            let b = try await engine.decide(sequential)
            let sequentialPrompts = recorder.reads.dropFirst(plainPrompts.count).map(\.prompt)
            #expect(b.answers == a.answers)
            #expect(b.inputTokens == a.inputTokens && b.outputTokens == a.outputTokens)
            #expect(sequentialPrompts == plainPrompts)
        }
    }
}
