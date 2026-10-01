import Foundation
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// ``DiffusionGemmaRuntime`` over a stub model: what it refuses before the model runs, what it
/// declares, and how a ``CanvasRead`` reaches the model and comes back. No weights and no MLX.
@Suite("DiffusionGemma runtime over a stub model")
struct RuntimeTests {
    @Test("A prompt over the cap is refused with upstream's message before the model is touched")
    func promptCap() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(
            configuration: .init(maxPromptTokens: 8), log: log)
        let error = await #expect(throws: SchemaError.self) {
            try await runtime.read(stubCanvasRead(prompt: .tokens(Array(1...9))))
        }
        #expect(error?.message == "the request is 9 tokens; the limit is 8")
        #expect(!log.touched)
        // At the cap the read runs.
        _ = try await runtime.read(stubCanvasRead(prompt: .tokens(Array(1...8))))
        #expect(log.prefills == [Array(1...8)])
    }

    @Test("The default prompt cap is 32,768 tokens")
    func defaultCap() {
        #expect(DiffusionGemmaRuntime.stub().maxPromptTokens == 32_768)
    }

    @Test("An image prompt is refused as unsupported before the model is touched")
    func imagePrompt() async {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(log: log)
        let read = stubCanvasRead(
            prompt: .image(systemText: "system", stateText: "state", images: []))
        await #expect(throws: DiffusionGemmaRuntimeError.unsupported("images")) {
            try await runtime.read(read)
        }
        #expect(!log.touched)
        #expect(
            DiffusionGemmaRuntimeError.unsupported("images").description.contains("vision"))
    }

    @Test("Steps, samples and sequential are on; think and images are off; the name is openjev-0.1")
    func capabilities() {
        let runtime = DiffusionGemmaRuntime.stub()
        #expect(
            runtime.capabilities
                == BackendCapabilities(
                    steps: true, samples: true, think: false, sequential: true, images: false))
        #expect(runtime.modelName == "openjev-0.1")
        #expect(runtime.modelName == ServedModels.diffusionGemmaVersion)
    }

    @Test("think throws unsupported until milestone 5")
    func think() async {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(log: log)
        await #expect(throws: DiffusionGemmaRuntimeError.unsupported("think")) {
            try await runtime.think(prompt: [1, 2, 3], budget: 16, stopIDs: [4])
        }
        #expect(!log.touched)
    }

    @Test("A CanvasRead reaches the model as its canvas, SlotRequests, steps and top-k 20")
    func mapping() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(log: log)
        let request = stubCanvasRead(prompt: .tokens([5, 6, 7, 8, 9]), steps: 3)
        let result = try await runtime.read(request)
        let read = try #require(log.reads.first)
        #expect(read.canvas == request.canvas.tokens)
        #expect(
            read.slots == [
                SlotRequest(position: 3, labelIDs: [10, 11]),
                SlotRequest(position: 7, labelIDs: [20, 21, 22]),
            ])
        #expect(read.steps == 3)
        #expect(read.topK == 20)
        #expect(result.promptTokens == 5)
        #expect(result.slots.count == 2)
        #expect(abs(result.slots[0].probabilities[0] - 0.75) < 1e-12)
        #expect(abs(result.slots[1].probabilities[2] - 0.125) < 1e-12)
    }

    @Test("A repeated prompt prefills once and reports the same prompt tokens")
    func cachedPrefill() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(log: log)
        let first = try await runtime.read(stubCanvasRead(prompt: .tokens([1, 2, 3])))
        let second = try await runtime.read(stubCanvasRead(prompt: .tokens([1, 2, 3])))
        _ = try await runtime.read(stubCanvasRead(prompt: .tokens([4, 5])))
        #expect(log.prefills == [[1, 2, 3], [4, 5]])
        #expect(first.promptTokens == 3 && second.promptTokens == 3)
        let statistics = await runtime.statistics()
        #expect(statistics.reads == 3)
        #expect(statistics.prefillHits == 1 && statistics.prefillMisses == 2)
        #expect(statistics.cachedPrefills == 2 && statistics.cachedPrefillTokens == 5)
        let state = await runtime.prefillCacheState
        #expect(state.keys == [.tokens([1, 2, 3]), .tokens([4, 5])])
    }

    @Test("The engine refuses think with \"openjev-0.1 does not support think\" before any read")
    func engineRefusals() async throws {
        let log = StubModelLog()
        let engine = try DecisionEngine(backend: DiffusionGemmaRuntime.stub(log: log))
        let think = try SystemOneRequest(
            json: JSONParser().parse(
                #"{"state": "s", "model": "jev-latest", "think": 8, "questions": "#
                    + #"{"q": {"type": "noul"}}}"#))
        let error = await #expect(throws: SchemaError.self) { try await engine.decide(think) }
        #expect(error?.message == "openjev-0.1 does not support think")
        #expect(!log.touched)
    }
}
