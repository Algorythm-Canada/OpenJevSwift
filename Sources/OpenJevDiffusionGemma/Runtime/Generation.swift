// Text generation on the DiffusionGemma runtime (issues #51 and #52): upstream OpenJev's
// `MlxRuntime.generate` and `MlxEngine.think` (razorback16/openjev at dcd2094,
// openjev/mlx_backend.py lines 210 to 257 and 274 to 288, Apache-2.0, see THIRD_PARTY.md) over
// the block loop of mlx-vlm 0.6.15's `stream_diffusion_generate` (generate/diffusion.py lines
// 729 to 1066, adapted from mlx-vlm, Copyright © 2025 Prince Canuma, MIT).

import Foundation
import MLX
import OpenJevCore

/// What one generation produced.
public struct GenerationResult: Sendable, Hashable {
    /// Why a generation ended.
    public enum FinishReason: String, Sendable, Hashable {
        /// An EOS id of the model or one of the caller's stop ids; the stop id is not in
        /// ``GenerationResult/generated``.
        case stop
        /// `maxTokens` tokens.
        case length
        /// `emit` returned false, or the calling task was cancelled.
        case cancelled
    }

    /// The generated ids, without the stop id.
    public var generated: [Int]
    /// The prompt tokens processed, the prompt's length.
    public var promptTokens: Int
    /// Why it ended.
    public var finishReason: FinishReason

    /// Creates a result.
    public init(generated: [Int], promptTokens: Int, finishReason: FinishReason) {
        self.generated = generated
        self.promptTokens = promptTokens
        self.finishReason = finishReason
    }
}

extension DiffusionGemmaRuntime {
    /// The model operations generation drives. The real runtime binds them to a
    /// ``DiffusionGemmaModel``; the model-free tests bind them to a stub.
    struct GenerationCalls {
        /// The checkpoint's generation settings.
        var policy: DiffusionGenerationPolicy
        /// One block over a cache, ``DiffusionGemmaModel/denoiseBlock(cache:canvasLength:policy:random:)``.
        var denoise:
            (_ cache: PromptCache, _ canvasLength: Int, _ random: MLXRandom.RandomState) throws
                -> DenoisedBlock
        /// A committed block appended to a cache, ``DiffusionGemmaModel/updateCache(_:tokens:)``.
        var commit: (_ cache: PromptCache, _ tokens: [Int]) throws -> PromptCache
        /// The vocabulary entry of an id, for the detokenizer.
        var tokenText: (Int) -> String?
    }

    /// Generates text after `prompt`, greedily, as upstream's `MlxRuntime.generate` runs mlx-vlm's
    /// `stream_diffusion_generate` with `temperature=0.0`.
    ///
    /// The reply is written a block at a time: a canvas of 256 positions while 256 or more tokens
    /// remain, then one of `max(remaining, 64)`, each denoised by the checkpoint's policy
    /// (``DiffusionGemmaModel/denoiseBlock(cache:canvasLength:policy:random:)``) and, unless the
    /// reply ended in it, appended to the cache before the next block
    /// (``DiffusionGemmaModel/updateCache(_:tokens:)``). The prompt's prefill is the one reads
    /// cached for the same prompt when there is one; a generation's own prefill is not cached,
    /// as upstream caches none, and its blocks go to a copy, never to the cached prefill.
    ///
    /// The tokens of a block are taken in order. The first of the model's EOS ids (1, 106, 50) or
    /// of `stopIDs` ends the reply with ``GenerationResult/FinishReason/stop`` and is not
    /// returned; the `maxTokens`-th token ends it with ``GenerationResult/FinishReason/length``.
    /// Every other token goes through the streaming detokenizer, `skipSpecialTokenIDs` left out
    /// before they enter its buffer, and `emit` is called once per token with the text the
    /// detokenizer released (often `""`: a word's text comes with the token after it). At the end
    /// `emit` is called once more with the last buffered text and a nil token, when there is any.
    /// `emit` returning false ends the reply after that token, and a cancelled calling task ends
    /// it before the next block, both with ``GenerationResult/FinishReason/cancelled`` and no
    /// final call.
    ///
    /// The canvases are drawn from MLX's generator seeded with
    /// ``Configuration/generationSeed`` (0) for every reply, so a prompt always gets the same
    /// reply, the one Fixtures/generation records; upstream leaves MLX's generator unseeded, so
    /// its replies vary from one process to the next.
    ///
    /// The whole reply runs inside the actor, as upstream holds its one MLX thread for it.
    ///
    /// - Parameters:
    ///   - prompt: the prompt's token ids.
    ///   - maxTokens: the most tokens to generate; 0 is the checkpoint's `max_new_tokens` (256),
    ///     as in mlx-vlm.
    ///   - stopIDs: ids that end the reply besides the model's EOS ids.
    ///   - skipSpecialTokenIDs: ids left out of the text, such as a chat reply's thought-channel
    ///     markers; they are still generated and returned.
    ///   - emit: receives each token's text and the token, then the final text and nil; returns
    ///     false to stop.
    /// - Throws: ``/OpenJevCore/SchemaError`` `"the request is {n} tokens; the limit is {max}"`
    ///   for a prompt longer than ``maxPromptTokens``; ``DiffusionGemmaRuntimeError/unsupported(_:)``
    ///   for a runtime made without generation; ``ReadInputError`` for a prompt the model refuses.
    public func generate(
        prompt: [Int], maxTokens: Int, stopIDs: [Int], skipSpecialTokenIDs: [Int],
        emit: @Sendable (_ text: String, _ token: Int?) -> Bool
    ) async throws -> GenerationResult {
        guard let generation = calls.generation else {
            throw DiffusionGemmaRuntimeError.unsupported("generation")
        }
        if prompt.count > maxPromptTokens {
            throw SchemaError(
                "the request is \(prompt.count) tokens; the limit is \(maxPromptTokens)")
        }
        let clock = ContinuousClock()
        let start = clock.now
        defer { modelTime += clock.now - start }

        let policy = generation.policy
        let limit = maxTokens == 0 ? policy.maxNewTokens : maxTokens
        let stops = Set(policy.eosTokenIDs + stopIDs)
        let skipped = Set(skipSpecialTokenIDs)
        var cache = try prefills.peek(.tokens(prompt)) ?? calls.prefill(prompt)
        let random = MLXRandom.RandomState(seed: configuration.generationSeed)
        var detokenizer = StreamingDetokenizer(tokenText: generation.tokenText)
        var ids: [Int] = []
        var count = 0
        var committed: [Int]?
        var finish = GenerationResult.FinishReason.length
        blocks: while count < limit {
            if Task.isCancelled {
                return GenerationResult(
                    generated: ids, promptTokens: prompt.count, finishReason: .cancelled)
            }
            if let committed {
                cache = try generation.commit(cache, committed)
            }
            let block = try generation.denoise(
                cache, policy.canvasLength(remaining: limit - count), random)
            for token in block.tokens {
                count += 1
                if stops.contains(token) {
                    finish = .stop
                    break blocks
                }
                detokenizer.add(token, skipping: skipped)
                ids.append(token)
                if !emit(detokenizer.lastSegment(), token) {
                    return GenerationResult(
                        generated: ids, promptTokens: prompt.count, finishReason: .cancelled)
                }
                if count >= limit {
                    finish = .length
                    break blocks
                }
            }
            committed = block.tokens
        }
        detokenizer.finalize()
        let tail = detokenizer.lastSegment()
        if !tail.isEmpty {
            _ = emit(tail, nil)
        }
        return GenerationResult(generated: ids, promptTokens: prompt.count, finishReason: finish)
    }

    /// Upstream's `MlxEngine.think` on the runtime: ``generate(prompt:maxTokens:stopIDs:skipSpecialTokenIDs:emit:)``
    /// of up to `budget` tokens after `prompt` (the chat prompt with thinking on and the
    /// thought-open marker, which ``/OpenJevCore/DecisionEngine`` builds), stopping at `stopIDs`
    /// (the thought-close marker), skipping nothing, since the markers are what a thought is
    /// cut at. The engine cuts the ids at the first close marker and bills the prompt tokens
    /// as input and the thought as output.
    ///
    /// - Throws: ``/OpenJevCore/SchemaError`` `"the request is {n} tokens; the limit is {max}"`
    ///   for a prompt longer than ``maxPromptTokens``, as `MlxEngine.think` checks it;
    ///   ``DiffusionGemmaRuntimeError/unsupported(_:)`` for a runtime made without generation,
    ///   whose ``capabilities`` flag `think` off so the engine never calls it.
    public func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws
        -> ThoughtGeneration
    {
        guard calls.generation != nil else {
            throw DiffusionGemmaRuntimeError.unsupported("think")
        }
        let result = try await generate(
            prompt: prompt, maxTokens: budget, stopIDs: stopIDs, skipSpecialTokenIDs: [],
            emit: { _, _ in true })
        return ThoughtGeneration(generated: result.generated, promptTokens: result.promptTokens)
    }
}
