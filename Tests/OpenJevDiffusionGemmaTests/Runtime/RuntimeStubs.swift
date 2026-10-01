import Foundation
import OpenJevCore

@testable import OpenJevDiffusionGemma

/// A tokenizer for the model-free runtime tests: the chat prompt is one id per character of the
/// state, which no test reads back.
struct StubRuntimeTokenizer: DecisionTokenizer {
    func encode(_ text: String, addSpecialTokens: Bool) throws -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(_ ids: [Int], skipSpecialTokens: Bool) throws -> String {
        String(String.UnicodeScalarView(ids.compactMap { Unicode.Scalar(UInt32($0)) }))
    }

    func chatPromptIDs(system: String, user: String, thinking: Bool) throws -> [Int] {
        try encode(user, addSpecialTokens: false)
    }
}

/// What the stub model was asked, for the runtime tests.
final class StubModelLog: @unchecked Sendable {
    struct Read: Equatable {
        var canvas: [Int]
        var slots: [SlotRequest]
        var steps: Int
        var topK: Int
        var promptTokens: Int
    }

    private let lock = NSLock()
    private var prefillCalls: [[Int]] = []
    private var readCalls: [Read] = []
    private var cacheLimits: [Int] = []

    var prefills: [[Int]] { lock.withLock { prefillCalls } }
    var reads: [Read] { lock.withLock { readCalls } }
    var limits: [Int] { lock.withLock { cacheLimits } }
    var touched: Bool { !prefills.isEmpty || !reads.isEmpty }

    func prefilled(_ ids: [Int]) { lock.withLock { prefillCalls.append(ids) } }
    func read(_ read: Read) { lock.withLock { readCalls.append(read) } }
    func limit(_ bytes: Int) { lock.withLock { cacheLimits.append(bytes) } }
}

extension DiffusionGemmaRuntime {
    /// A runtime over a stub model that needs no MLX: the prefill is a ``PromptCache`` without
    /// layers, and each slot's map gives the first label log(0.75) and the others the rest.
    static func stub(
        configuration: Configuration = .default, log: StubModelLog = StubModelLog()
    ) -> DiffusionGemmaRuntime {
        let calls = ModelCalls(
            prefill: { ids in
                log.prefilled(ids)
                return PromptCache(layers: [], offset: ids.count, promptTokens: ids.count)
            },
            read: { canvas, slots, cache, steps, topK in
                log.read(
                    .init(
                        canvas: canvas, slots: slots, steps: steps, topK: topK,
                        promptTokens: cache.promptTokens))
                let maps = slots.map { slot in
                    slot.labelIDs.enumerated().map { index, id in
                        (
                            tokenID: id,
                            logprob: index == 0
                                ? Foundation.log(0.75)
                                : Foundation.log(0.25 / Double(slot.labelIDs.count - 1))
                        )
                    }
                }
                return ReadOutput(slots: maps, written: [], promptTokens: cache.promptTokens)
            })
        return DiffusionGemmaRuntime(
            tokenizer: StubRuntimeTokenizer(), configuration: configuration, calls: calls,
            setCacheLimit: { log.limit($0) })
    }
}

/// A read of `prompt` over a 16-token canvas with two slots.
func stubCanvasRead(prompt: ReadPrompt, steps: Int = 1) -> CanvasRead {
    let slots = [
        ResolvedTemplate.Slot(position: 3, labelIDs: [10, 11]),
        ResolvedTemplate.Slot(position: 7, labelIDs: [20, 21, 22]),
    ]
    let template = Array(1...9)
    let canvas = SeededCanvas(
        tokens: template + [106] + [Int](repeating: 0, count: 6), noise: [500, 600])
    return CanvasRead(
        prompt: prompt, systemText: "system", stateText: "state", template: template,
        slots: slots, canvas: canvas, steps: steps, seed: 42)
}
