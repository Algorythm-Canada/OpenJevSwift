import Foundation
import MLX
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// What the runtime's model did for one reply: each block and each commit.
private final class BlockRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [(canvasLength: Int, block: DenoisedBlock)] = []
    private var commitOffsets: [Int] = []

    var blocks: [(canvasLength: Int, block: DenoisedBlock)] { lock.withLock { recorded } }
    var commits: [Int] { lock.withLock { commitOffsets } }

    func reset() {
        lock.withLock {
            recorded = []
            commitOffsets = []
        }
    }
    func denoised(_ length: Int, _ block: DenoisedBlock) {
        lock.withLock { recorded.append((length, block)) }
    }
    func committed(_ offset: Int) { lock.withLock { commitOffsets.append(offset) } }
}

/// What `emit` received.
private final class Pieces: @unchecked Sendable {
    private let lock = NSLock()
    private var calls: [GenerationOracle.Piece] = []
    var pieces: [GenerationOracle.Piece] { lock.withLock { calls } }
    func emit(_ text: String, _ token: Int?) -> Bool {
        lock.withLock { calls.append(.init(text: text, token: token)) }
        return true
    }
}

/// A runtime over the shared checkpoint's model with its own generation seed, whose blocks and
/// commits are recorded. Built with the internal ``DiffusionGemmaRuntime/init(tokenizer:configuration:calls:setCacheLimit:)``,
/// so the 16 GB model is the shared one.
private func recordingRuntime(seed: UInt64) async throws -> (DiffusionGemmaRuntime, BlockRecorder) {
    let live = try await LiveCheckpoint.shared()
    let model = live.loaded.model
    let tokenizer = try #require(live.runtime.tokenizer as? SwiftTransformersTokenizer)
    let policy = try DiffusionGenerationPolicy(configuration: live.loaded.configuration)
    let recorder = BlockRecorder()
    var configuration = live.runtime.configuration
    configuration.generationSeed = seed
    let runtime = DiffusionGemmaRuntime(
        tokenizer: tokenizer, configuration: configuration,
        calls: .init(
            prefill: { try model.prefill(promptIDs: $0) },
            read: { canvas, slots, cache, steps, topK in
                try model.read(canvas: canvas, slots: slots, cache: cache, steps: steps, topK: topK)
            },
            imagePrompt: nil,
            generation: .init(
                policy: policy,
                denoise: { cache, length, random in
                    let block = try model.denoiseBlock(
                        cache: cache, canvasLength: length, policy: policy, random: random)
                    recorder.denoised(length, block)
                    return block
                },
                commit: { cache, tokens in
                    recorder.committed(cache.offset)
                    return try model.updateCache(cache, tokens: tokens)
                },
                tokenText: { tokenizer.token(of: $0) })))
    return (runtime, recorder)
}

/// Installs the oracle's RoPE table in the full-attention layers when the exact tier is on
/// (`OPENJEV_MLX_METALLIB` taken), runs `body`, and restores the model's own tables.
private func withTier<T>(
    _ model: DiffusionGemmaModel, _ body: (_ exact: Bool) async throws -> T
) async throws -> T {
    let exact = MetalLibrary.override != nil
    let fullLayers = model.decoder.layers.filter { $0.layerType == .fullAttention }
    var restore: [MLXArray] = []
    if exact {
        let bits = try ModelFixtures.oracle().rope.float32Bits
        let table = MLXArray(bits.map { Float(bitPattern: $0) })
        for layer in fullLayers {
            restore.append(try #require(layer.selfAttention.fullAttentionFrequencies))
            layer.selfAttention.fullAttentionFrequencies = table
        }
    }
    defer {
        for (layer, table) in zip(fullLayers, restore) {
            layer.selfAttention.fullAttentionFrequencies = table
        }
    }
    return try await body(exact)
}

/// Whether `answer` is the recorded one bit for bit: a noul's probability; a choice's label, every
/// probability and its confidence; a score's value, every probability and its confidence.
private func sameAnswer(_ answer: Answer, _ recorded: JSONValue) -> Bool {
    switch answer {
    case .noul(let p):
        return recorded["type"]?.stringValue == "noul" && recorded["noul"]?.doubleValue == p
    case .choice(let choice, let probabilities, let confidence):
        guard recorded["type"]?.stringValue == "choice",
            recorded["choice"]?.stringValue == choice,
            recorded["confidence"]?.doubleValue == confidence,
            let recordedProbabilities = recorded["probabilities"]?.objectValue,
            recordedProbabilities.count == probabilities.count
        else { return false }
        return probabilities.allSatisfy {
            recorded["probabilities"]?[$0.key]?.doubleValue == $0.value
        }
    case .score(let score, _, let probabilities, let confidence):
        guard recorded["type"]?.stringValue == "score",
            recorded["score"]?.doubleValue == score,
            recorded["confidence"]?.doubleValue == confidence
        else { return false }
        return probabilities.indices.allSatisfy {
            recorded["probabilities"]?[String($0)]?.doubleValue == probabilities[$0]
        } && recorded["probabilities"]?.objectValue?.count == probabilities.count
    }
}

/// How far a reply agrees with the oracle's: the leading blocks equal, and the first token that
/// differs.
private struct Agreement: CustomStringConvertible {
    var name: String
    var blocks: (equal: Int, of: Int)
    var tokens: (equal: Int, of: Int)
    var finish: (port: String, oracle: String)
    var whole: Bool

    var description: String {
        "| \(name) | \(tokens.of) | \(blocks.of) | \(whole ? "whole reply" : "\(blocks.equal) of \(blocks.of) blocks, first \(tokens.equal) tokens") | \(finish.port) (oracle \(finish.oracle)) |"
    }
}

extension MLXTests {
    /// The runtime's greedy replies against upstream's (Fixtures/generation/generation.json).
    ///
    /// In the exact tier (D-014: `OPENJEV_MLX_METALLIB` set to the oracle wheel's metallib, the
    /// oracle's RoPE table installed) every reply must agree token for token, block for block,
    /// with the same steps, finish reason and emitted text. In the native tier the kernels round
    /// differently in the last bit, which can flip a near-tied argmax and change the rest of a
    /// reply; the agreement is printed as a table row per reply, and only #51's three short
    /// prompts (a short answer, a list, a JSON reply) are held to the whole reply.
    @Suite(
        "Generation against the mlx-vlm oracle",
        .enabled(if: ModelFixtures.checkpointAvailable, ModelFixtures.missingCheckpointMessage))
    struct GenerationOracleTests {
        @Test("The policy is the one mlx-vlm derives from the checkpoint")
        func policy() async throws {
            let live = try await LiveCheckpoint.shared()
            let oracle = try GenerationOracle.load()
            let policy = try DiffusionGenerationPolicy(configuration: live.loaded.configuration)
            #expect(Set(policy.eosTokenIDs) == Set(oracle.settings.eosTokenIDs))
            #expect(policy.maxCanvasLength == oracle.settings.canvasLength)
            #expect(policy.minCanvasLength == oracle.settings.minCanvasLength)
            #expect(policy.maxDenoisingSteps == oracle.settings.maxDenoisingSteps)
            #expect(
                Double(policy.samplerThreshold)
                    == Double(Float(oracle.settings.confidenceThreshold)))
            #expect(policy.stopping == .init(stabilityThreshold: 1, confidenceThreshold: 0.005))
            #expect(policy.tMin == 0.4 && policy.tMax == 0.8 && policy.temperature == 0)
            #expect(live.runtime.capabilities.think)
        }

        @Test("Each recorded reply, token for token in the exact tier")
        func replies() async throws {
            let live = try await LiveCheckpoint.shared()
            let oracle = try GenerationOracle.load()
            var rows: [Agreement] = []
            try await withTier(live.loaded.model) { exact in
                for generation in oracle.generations {
                    let (runtime, recorder) = try await recordingRuntime(seed: generation.seed)
                    let pieces = Pieces()
                    let result = try await runtime.generate(
                        prompt: generation.prompt, maxTokens: generation.maxTokens,
                        stopIDs: generation.stopIDs,
                        skipSpecialTokenIDs: generation.skipSpecialTokenIDs, emit: pieces.emit)
                    let blocks = recorder.blocks
                    let equalBlocks = zip(blocks, generation.blocks).prefix {
                        $0.0.block.tokens == $0.1.finalCanvas
                    }.count
                    let equalTokens = zip(result.generated, generation.generated).prefix {
                        $0 == $1
                    }.count
                    let whole =
                        result.generated == generation.generated
                        && result.finishReason.rawValue == generation.finishReason
                    rows.append(
                        Agreement(
                            name: generation.name, blocks: (equalBlocks, generation.blocks.count),
                            tokens: (equalTokens, generation.generated.count),
                            finish: (result.finishReason.rawValue, generation.finishReason),
                            whole: whole))
                    #expect(result.promptTokens == generation.promptTokens)
                    #expect(
                        blocks.first.map { $0.canvasLength }
                            == generation.blocks.first?.canvasLength)
                    // The canvases come from the reply's seed (`generationSeed`), as
                    // `mx.random.seed` gives them: the first block's in both tiers, and each later
                    // one while the blocks before it drew as many canvases as the oracle's.
                    for (index, (block, recorded)) in zip(blocks, generation.blocks).enumerated() {
                        let drewAlike = zip(blocks, generation.blocks).prefix(index).allSatisfy {
                            $0.0.block.steps == $0.1.steps && $0.0.block.tokens == $0.1.finalCanvas
                        }
                        guard exact || drewAlike else { break }
                        #expect(
                            block.block.initialCanvas == recorded.initialCanvas,
                            "\(generation.name) block \(index): initial canvas")
                    }
                    // #51's three prompts are held to the whole reply in both tiers.
                    if ["short_answer", "list", "json"].contains(generation.name) {
                        #expect(whole, "\(generation.name)")
                    }
                    if exact {
                        #expect(whole, "\(generation.name)")
                        #expect(blocks.map(\.canvasLength) == generation.blocks.map(\.canvasLength))
                        #expect(blocks.map(\.block.steps) == generation.blocks.map(\.steps))
                        #expect(
                            blocks.map(\.block.ending.rawValue) == generation.blocks.map(\.ended))
                        #expect(blocks.map(\.block.tokens) == generation.blocks.map(\.finalCanvas))
                        #expect(pieces.pieces == generation.pieces, "\(generation.name)")
                    }
                }
                print("generation agreement (\(exact ? "exact" : "native") tier):")
                print("| reply | tokens | blocks | agreement | finish |")
                for row in rows {
                    print(row)
                }
            }
        }

        @Test("MlxEngine.think's passes: the prompt, the thought and the billing")
        func think() async throws {
            let live = try await LiveCheckpoint.shared()
            let runtime = live.runtime
            let oracle = try GenerationOracle.load()
            let file = try JSONParser().parse(Data(contentsOf: GenerationOracle.url))
            let tokens = try EngineTokens(tokenizer: runtime.tokenizer)
            #expect(tokens.thoughtOpen == oracle.settings.thoughtOpen)
            #expect(tokens.thoughtClose == oracle.settings.thoughtClose)
            try await withTier(live.loaded.model) { exact in
                for (index, record) in oracle.think.enumerated() {
                    let body = try #require(file["think"]?[index]?["body"])
                    let request = try SystemOneRequest(json: body)
                    for thought in record.thoughts {
                        // The engine's think prompt is upstream's, in both tiers.
                        let prompt =
                            try runtime.tokenizer.chatPromptIDs(
                                system: thought.system, user: thought.user, thinking: true)
                            + tokens.thoughtOpen
                        #expect(prompt == thought.prompt, "\(record.name)")
                        let generated = try await runtime.think(
                            prompt: thought.prompt, budget: thought.budget,
                            stopIDs: thought.stopIDs)
                        #expect(generated.promptTokens == thought.promptTokens)
                        print(
                            "think \(record.name): \(generated.generated.count) ids, oracle "
                                + "\(thought.generated.count), first \(zip(generated.generated, thought.generated).prefix { $0 == $1 }.count) agree"
                        )
                        if exact {
                            #expect(generated.generated == thought.generated, "\(record.name)")
                        }
                    }
                    let engine = try DecisionEngine(backend: runtime, configuration: .default)
                    let decision = try await engine.decide(request)
                    print(
                        "think \(record.name): \(decision.outputTokens) thought tokens and "
                            + "\(decision.inputTokens) input tokens; oracle \(record.outputTokens) "
                            + "and \(record.inputTokens)")
                    // The thought is billed as output, and the input is the thought pass plus
                    // the reads after it, as upstream bills them.
                    let thoughts = record.thoughts.map(\.thoughtTokens).reduce(0, +)
                    #expect(record.outputTokens == thoughts)
                    if exact {
                        #expect(decision.outputTokens == record.outputTokens)
                        #expect(decision.inputTokens == record.inputTokens)
                        // The reads after the thought are the oracle's too: every answer, its
                        // probabilities and its confidence, bit for bit.
                        let answers = try #require(file["think"]?[index]?["answers"])
                        #expect(decision.answers.count == answers.objectValue?.count)
                        for (key, answer) in decision.answers {
                            let recorded = try #require(answers[key], "\(key)")
                            #expect(
                                sameAnswer(answer, recorded),
                                "\(record.name) \(key): \(answer) against \(recorded)")
                        }
                    } else {
                        #expect((1...request.think!).contains(decision.outputTokens))
                    }
                }
            }
        }

        @Test("Committed blocks leave the shared prefill's tensors unchanged")
        func commitsCopy() async throws {
            let live = try await LiveCheckpoint.shared()
            let model = live.loaded.model
            let oracle = try GenerationOracle.load()
            /// Every layer's offset and key and value digests.
            func digests(_ cache: PromptCache) -> [String] {
                cache.layers.map { layer in
                    let digest = layer.digest
                    return "\(layer.offset) \(digest?.keys.shape ?? []) "
                        + "\(digest?.keys.sha256 ?? "-") \(digest?.values.sha256 ?? "-")"
                }
            }
            // A short prompt and one past the sliding window, whose commit trims.
            for name in ["short_answer", "long_prompt"] {
                let prompt = try #require(oracle.generations.first { $0.name == name }).prompt
                let cache = try model.prefill(promptIDs: prompt)
                let before = digests(cache)
                // Two continuations of the prefill, and a second block on the first.
                let first = try model.updateCache(
                    cache, tokens: [Int](repeating: 236761, count: 64))
                let second = try model.updateCache(cache, tokens: [Int](repeating: 818, count: 64))
                _ = try model.updateCache(first, tokens: [Int](repeating: 529, count: 64))
                #expect(digests(cache) == before, "\(name): the prefill changed")
                #expect(digests(first) != digests(second), "\(name)")
                #expect(first.layers.allSatisfy { $0.offset == prompt.count + 64 }, "\(name)")
            }
            let prompt = try #require(oracle.generations.first).prompt
            let cache = try model.prefill(promptIDs: prompt)
            let tokens = [Int](repeating: 236761, count: 64)
            let extended = try model.updateCache(cache, tokens: tokens)
            #expect(extended.offset == cache.offset + 64)
            #expect(extended.promptTokens == cache.promptTokens)
            #expect(extended.layers.allSatisfy { $0.offset == prompt.count + 64 })
            #expect(throws: ReadInputError.emptyCanvas) { try model.updateCache(cache, tokens: []) }
        }
    }
}
