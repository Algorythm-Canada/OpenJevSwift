import Foundation
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// What `emit` received, in order.
private final class Emitted: @unchecked Sendable {
    private let lock = NSLock()
    private var calls: [(text: String, token: Int?)] = []
    /// The call after which `emit` returns false, counting from 1; nil never.
    let stopAfter: Int?

    init(stopAfter: Int? = nil) {
        self.stopAfter = stopAfter
    }

    var pieces: [(text: String, token: Int?)] { lock.withLock { calls } }
    var tokens: [Int?] { pieces.map(\.token) }
    var text: String { pieces.map(\.text).joined() }

    func emit(_ text: String, _ token: Int?) -> Bool {
        lock.withLock {
            calls.append((text, token))
            return stopAfter.map { calls.count < $0 } ?? true
        }
    }
}

/// ``DiffusionGemmaRuntime/generate(prompt:maxTokens:stopIDs:skipSpecialTokenIDs:emit:)`` and
/// `think` over a stub model whose blocks are scripted: the block loop's sizing, commits, stops,
/// streaming and cancellation, upstream's `MlxRuntime.generate` contract. No weights and no MLX
/// evaluation.
@Suite("Generation on the runtime over a stub model")
struct GenerationRuntimeTests {
    @Test("Canvases are 256 while 256 or more tokens remain, then max(remaining, 64)")
    func canvasSizing() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(log: log)
        let emitted = Emitted()
        let result = try await runtime.generate(
            prompt: [1, 2, 3], maxTokens: 600, stopIDs: [], skipSpecialTokenIDs: [],
            emit: emitted.emit)
        #expect(log.canvases == [256, 256, 88])
        #expect(result.finishReason == .length)
        #expect(result.generated.count == 600)
        #expect(result.promptTokens == 3)
        // Each block but the last is committed whole, after the one before it.
        #expect(log.commits.map(\.offset) == [3, 259])
        #expect(log.commits.map(\.tokens.count) == [256, 256])
        #expect(log.commits.allSatisfy { $0.promptTokens == 3 })
        // One emit per token, then the buffered tail.
        #expect(emitted.tokens.count == 601 && emitted.tokens.last == .some(nil))

        let shortLog = StubModelLog()
        let cut = try await DiffusionGemmaRuntime.stub(log: shortLog).generate(
            prompt: [1], maxTokens: 40, stopIDs: [], skipSpecialTokenIDs: [],
            emit: { _, _ in true })
        #expect(shortLog.canvases == [64])
        #expect(cut.generated.count == 40 && cut.finishReason == .length)
        #expect(shortLog.commits.isEmpty)
    }

    @Test("maxTokens 0 generates the checkpoint's max_new_tokens, as mlx-vlm does")
    func zeroMaxTokens() async throws {
        let log = StubModelLog()
        let result = try await DiffusionGemmaRuntime.stub(log: log).generate(
            prompt: [1], maxTokens: 0, stopIDs: [], skipSpecialTokenIDs: [],
            emit: { _, _ in true })
        #expect(log.canvases == [256])
        #expect(result.generated.count == 256 && result.finishReason == .length)
    }

    @Test("An EOS id or a stop id ends the reply at the token, which is not returned")
    func stops() async throws {
        for (stopIDs, block, expected) in [
            ([Int](), [10, 11, 106, 12], [10, 11]),
            ([Int](), [10, 1], [10]),
            ([Int](), [50, 10], []),
            ([11], [10, 11, 12], [10]),
        ] {
            let log = StubModelLog()
            let runtime = DiffusionGemmaRuntime.stub(log: log, blocks: { _, _ in block })
            let emitted = Emitted()
            let result = try await runtime.generate(
                prompt: [1, 2], maxTokens: 600, stopIDs: stopIDs, skipSpecialTokenIDs: [],
                emit: emitted.emit)
            #expect(result.generated == expected, "\(block)")
            #expect(result.finishReason == .stop)
            #expect(log.canvases == [256] && log.commits.isEmpty)
            #expect(emitted.text == expected.map { " w\($0)" }.joined())
        }
    }

    @Test("Each token's emit carries the text the detokenizer released; the tail comes with nil")
    func streaming() async throws {
        let runtime = DiffusionGemmaRuntime.stub(blocks: { _, _ in [10, 11, 12, 1] })
        let emitted = Emitted()
        _ = try await runtime.generate(
            prompt: [1], maxTokens: 64, stopIDs: [], skipSpecialTokenIDs: [],
            emit: emitted.emit)
        #expect(emitted.pieces.map(\.text) == ["", " w10", " w11", " w12"])
        #expect(emitted.tokens == [10, 11, 12, nil])
    }

    @Test("Skipped ids are generated and returned but leave no text")
    func skipped() async throws {
        let runtime = DiffusionGemmaRuntime.stub(blocks: { _, _ in [10, 100, 11, 101, 12, 1] })
        let emitted = Emitted()
        let result = try await runtime.generate(
            prompt: [1], maxTokens: 64, stopIDs: [], skipSpecialTokenIDs: [100, 101],
            emit: emitted.emit)
        #expect(result.generated == [10, 100, 11, 101, 12])
        #expect(emitted.text == " w10 w11 w12")
        #expect(emitted.tokens == [10, 100, 11, 101, 12, nil])
    }

    @Test("emit returning false ends the reply after that token, with no tail and no next block")
    func emitCancels() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(log: log, blocks: { _, _ in [10, 11, 12] })
        let emitted = Emitted(stopAfter: 2)
        let result = try await runtime.generate(
            prompt: [1], maxTokens: 600, stopIDs: [], skipSpecialTokenIDs: [],
            emit: emitted.emit)
        #expect(result.generated == [10, 11])
        #expect(result.finishReason == .cancelled)
        #expect(emitted.tokens == [10, 11])
        #expect(log.canvases == [256] && log.commits.isEmpty)
    }

    @Test("A cancelled task ends the reply before the next block")
    func taskCancels() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(
            log: log,
            blocks: { index, _ in
                // The denoise runs on the calling task: cancel it during the first block.
                withUnsafeCurrentTask { $0?.cancel() }
                return [10 + index]
            })
        let emitted = Emitted()
        let result = try await Task {
            try await runtime.generate(
                prompt: [1], maxTokens: 600, stopIDs: [], skipSpecialTokenIDs: [],
                emit: emitted.emit)
        }.value
        #expect(log.canvases == [256])
        #expect(result.finishReason == .cancelled)
        #expect(result.generated.count == 256 && result.generated.first == 10)
        #expect(emitted.tokens.count == 256 && !emitted.tokens.contains(nil))
    }

    @Test("A generation reuses the prefill a read cached for its prompt and caches none of its own")
    func prefillSharing() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(log: log, blocks: { _, _ in [10, 1] })
        _ = try await runtime.read(stubCanvasRead(prompt: .tokens([7, 8, 9])))
        _ = try await runtime.generate(
            prompt: [7, 8, 9], maxTokens: 64, stopIDs: [], skipSpecialTokenIDs: [],
            emit: { _, _ in true })
        #expect(log.prefills == [[7, 8, 9]])
        _ = try await runtime.generate(
            prompt: [4, 5], maxTokens: 64, stopIDs: [], skipSpecialTokenIDs: [],
            emit: { _, _ in true })
        #expect(log.prefills == [[7, 8, 9], [4, 5]])
        let statistics = await runtime.statistics()
        #expect(statistics.cachedPrefills == 1)
        #expect(statistics.prefillHits == 0 && statistics.prefillMisses == 1)
    }

    @Test("A prompt over the cap is refused with upstream's message before the model is touched")
    func promptCap() async {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(
            configuration: .init(maxPromptTokens: 4), log: log)
        let error = await #expect(throws: SchemaError.self) {
            try await runtime.think(prompt: [1, 2, 3, 4, 5], budget: 8, stopIDs: [101])
        }
        #expect(error?.message == "the request is 5 tokens; the limit is 4")
        #expect(!log.touched && log.canvases.isEmpty)
    }

    @Test("think generates up to the budget, stops at the close id, skips nothing")
    func think() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(
            log: log, blocks: { _, _ in [7, 100, 8, 101, 9] })
        let thought = try await runtime.think(prompt: [1, 2, 3], budget: 16, stopIDs: [101])
        #expect(thought == ThoughtGeneration(generated: [7, 100, 8], promptTokens: 3))
        #expect(log.canvases == [64])
        let capped = try await DiffusionGemmaRuntime.stub(blocks: { _, _ in [7, 8, 9] })
            .think(prompt: [1], budget: 2, stopIDs: [101])
        #expect(capped == ThoughtGeneration(generated: [7, 8], promptTokens: 1))
    }
}
