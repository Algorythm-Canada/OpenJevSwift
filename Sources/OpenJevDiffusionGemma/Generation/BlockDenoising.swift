// One block of text generation (issues #50 and #51): the settings mlx-vlm 0.6.15's
// generate/diffusion.py `stream_diffusion_generate` derives from the checkpoint (lines 568 to 654)
// and its denoising loop for one canvas (lines 776 to 1010), adapted from mlx-vlm, Copyright ©
// 2025 Prince Canuma, MIT, as upstream OpenJev's `MlxRuntime.generate` runs it: greedy
// (`temperature=0.0`), the default `confidence-threshold` sampler, no static cache, no compile.

import MLX

/// The generation settings mlx-vlm derives from the checkpoint for `stream_diffusion_generate`
/// as upstream calls it.
///
/// For the pinned checkpoint: canvases of 64 to 256 positions, up to 48 denoising steps, the
/// linear temperature schedule `0.4 + 0.4 × step / 48`, from 0.8 on the first step down to
/// 0.4 + 0.4/48 on the 48th, the stable-and-confident rule (stability 1,
/// mean entropy below 0.005), the `confidence-threshold` sampler at 0.9, greedy, and the EOS ids
/// 1, 106 and 50.
public struct DiffusionGenerationPolicy: Sendable, Hashable {
    /// The model's `canvas_length`, the largest canvas (256).
    public var maxCanvasLength: Int
    /// mlx-vlm's `DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH` (64), the smallest canvas.
    public var minCanvasLength: Int
    /// `max_denoising_steps` (48), the step cap of a block.
    public var maxDenoisingSteps: Int
    /// `t_min` (0.4), the floor the schedule approaches: the last of 48 steps runs at
    /// `tMin + (tMax − tMin) / 48`.
    public var tMin: Double
    /// `t_max` (0.8), the temperature the schedule starts from.
    public var tMax: Double
    /// `(tMin, tMax)`, as ``DiffusionSampler/linearTemperature(step:maxSteps:schedule:)`` takes it.
    public var temperatureSchedule: (tMin: Double, tMax: Double) { (tMin, tMax) }
    /// The stable-and-confident rule, nil when the checkpoint names neither threshold.
    public var stopping: DiffusionSampler.StoppingRule?
    /// The `confidence-threshold` sampler's threshold, mlx-vlm's
    /// `DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD` (0.9).
    public var samplerThreshold: Float
    /// The sampling temperature: 0, greedy, as upstream calls mlx-vlm.
    public var temperature: Float
    /// The ids that end a reply besides the caller's stop ids: the configuration's `eos_token_id`
    /// and the generation configuration's (1, 106, 50).
    public var eosTokenIDs: [Int]
    /// `max_new_tokens` (256), the reply length when the caller asks for 0 tokens.
    public var maxNewTokens: Int
    /// The vocabulary size, the bound of the random canvas ids.
    public var vocabularySize: Int

    /// The settings `stream_diffusion_generate` derives from `configuration`.
    ///
    /// - Throws: ``DiffusionGemmaRuntimeError/unsupported(_:)`` for a sampler configuration other
    ///   than `EntropyBoundSamplerConfig`, which mlx-vlm refuses with `NotImplementedError`.
    public init(configuration: DiffusionGemmaConfiguration) throws {
        let generation = configuration.generation
        if let name = generation?.sampler?.className, name != "EntropyBoundSamplerConfig" {
            throw DiffusionGemmaRuntimeError.unsupported("the diffusion sampler \(name)")
        }
        maxCanvasLength = configuration.canvasLength
        minCanvasLength = min(maxCanvasLength, Self.defaultMinCanvasLength)
        // `int(generation_config.get("max_denoising_steps") or 48)`: 0 is the default too.
        let steps = generation?.maxDenoisingSteps ?? 0
        maxDenoisingSteps = steps == 0 ? Self.defaultMaxDenoisingSteps : steps
        tMin = generation?.tMin ?? 0.4
        tMax = generation?.tMax ?? 0.8
        if generation?.confidenceThreshold != nil || generation?.stabilityThreshold != nil {
            stopping = DiffusionSampler.StoppingRule(
                stabilityThreshold: generation?.stabilityThreshold ?? 1,
                confidenceThreshold: generation?.confidenceThreshold ?? 0.005)
        } else {
            stopping = nil
        }
        samplerThreshold = Self.defaultConfidenceThreshold
        temperature = 0
        var eos = configuration.eosTokenIDs
        for id in generation?.eosTokenIDs ?? [] where !eos.contains(id) {
            eos.append(id)
        }
        eosTokenIDs = eos
        maxNewTokens = generation?.maxNewTokens ?? 256
        vocabularySize = configuration.text.vocabSize
    }

    /// A policy with the given settings; every default is the pinned checkpoint's.
    public init(
        maxCanvasLength: Int = 256, minCanvasLength: Int = defaultMinCanvasLength,
        maxDenoisingSteps: Int = defaultMaxDenoisingSteps, tMin: Double = 0.4,
        tMax: Double = 0.8,
        stopping: DiffusionSampler.StoppingRule? = DiffusionSampler.StoppingRule(),
        samplerThreshold: Float = defaultConfidenceThreshold, temperature: Float = 0,
        eosTokenIDs: [Int] = [1, 106, 50], maxNewTokens: Int = 256,
        vocabularySize: Int = 262_144
    ) {
        self.maxCanvasLength = maxCanvasLength
        self.minCanvasLength = minCanvasLength
        self.maxDenoisingSteps = maxDenoisingSteps
        self.tMin = tMin
        self.tMax = tMax
        self.stopping = stopping
        self.samplerThreshold = samplerThreshold
        self.temperature = temperature
        self.eosTokenIDs = eosTokenIDs
        self.maxNewTokens = maxNewTokens
        self.vocabularySize = vocabularySize
    }

    /// `DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH`.
    public static let defaultMinCanvasLength = 64
    /// `DEFAULT_DIFFUSION_MAX_DENOISING_STEPS`.
    public static let defaultMaxDenoisingSteps = 48
    /// `DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD`.
    public static let defaultConfidenceThreshold: Float = 0.9

    /// The canvas of the next block, `min(max, max(remaining, min))`: a full canvas while 256 or
    /// more tokens remain, then a partial one of at least 64.
    public func canvasLength(remaining: Int) -> Int {
        min(maxCanvasLength, max(remaining, minCanvasLength))
    }
}

/// One denoised block: its final canvas and how it got there.
public struct DenoisedBlock: Sendable, Hashable {
    /// Why a block's denoising ended.
    public enum Ending: String, Sendable, Hashable {
        /// The `confidence-threshold` sampler had accepted every position.
        case allRevealed = "all_revealed"
        /// The argmax canvas was stable and its mean entropy below the threshold.
        case stableAndConfident = "stable_and_confident"
        /// The step cap.
        case maxDenoisingSteps = "max_denoising_steps"
    }

    /// The random canvas the block started from.
    public var initialCanvas: [Int]
    /// The final canvas, the last step's argmax.
    public var tokens: [Int]
    /// The decoder passes it took.
    public var steps: Int
    /// Why it ended.
    public var ending: Ending

    /// Creates a block.
    public init(initialCanvas: [Int] = [], tokens: [Int], steps: Int, ending: Ending) {
        self.initialCanvas = initialCanvas
        self.tokens = tokens
        self.steps = steps
        self.ending = ending
    }
}

extension DiffusionGemmaModel {
    /// One block of `stream_diffusion_generate` over `cache`: a random canvas of `canvasLength`
    /// ids, then up to ``DiffusionGenerationPolicy/maxDenoisingSteps`` decoder passes. Each pass
    /// divides the logits by the step's temperature and takes their argmax; the
    /// `confidence-threshold` sampler accepts the unrevealed positions whose probability is at
    /// least 0.9 (at least the most probable one), re-noises the others with fresh random ids,
    /// and the next pass is conditioned on these logits. The block ends when every position is
    /// accepted, when the argmax canvas is stable and confident, or on the last step, which stops
    /// after its argmax; the block is that last argmax.
    ///
    /// Draws from `random` in mlx-vlm's order: the initial canvas, then one canvas of noise in
    /// every pass that does not stop at its argmax (the 48th, `cur_step == 1`).
    ///
    /// - Throws: ``ReadInputError/cacheLayerMismatch(cacheLayers:modelLayers:)``.
    public func denoiseBlock(
        cache: PromptCache, canvasLength: Int, policy: DiffusionGenerationPolicy,
        random: MLXRandom.RandomState
    ) throws -> DenoisedBlock {
        guard cache.layers.count == decoder.layers.count else {
            throw ReadInputError.cacheLayerMismatch(
                cacheLayers: cache.layers.count, modelLayers: decoder.layers.count)
        }
        let vocabulary = policy.vocabularySize
        var current = DiffusionSampler.initialCanvas(
            length: canvasLength, vocabularySize: vocabulary, random: random)
        let initialCanvas = current
        var revealed = MLXArray.zeros(current.shape, dtype: .bool)
        var draft = current
        var argmaxCanvas = current
        var conditioning: MLXArray?
        let masks = decoderMasks(canvasLength: canvasLength, cache: cache)
        var history: [MLXArray] = []
        var steps = 0
        var ending = DenoisedBlock.Ending.maxDenoisingSteps
        for step in stride(from: policy.maxDenoisingSteps, through: 1, by: -1) {
            steps += 1
            var logits = decoderLogits(
                canvas: current, cache: cache, conditioning: conditioning, masks: masks)
            if let temperature = DiffusionSampler.linearTemperature(
                step: step, maxSteps: policy.maxDenoisingSteps,
                schedule: policy.temperatureSchedule)
            {
                logits = logits / temperature
            }
            argmaxCanvas = argMax(logits, axis: -1).asType(.int32)
            if step == 1 {
                break
            }
            let denoised =
                policy.temperature <= 0
                ? argmaxCanvas
                : DiffusionSampler.sampleCanvas(
                    logits, temperature: policy.temperature, random: random)
            let confidence = DiffusionSampler.tokenProbability(logits, tokenIDs: denoised)
            let acceptance = DiffusionSampler.confidenceTransferMask(
                confidence: confidence, unrevealed: logicalNot(revealed),
                threshold: policy.samplerThreshold, forceAll: step == 1)
            let accepted = which(acceptance, denoised, draft)
            current = which(
                logicalOr(revealed, acceptance), accepted,
                DiffusionSampler.initialCanvas(
                    length: canvasLength, vocabularySize: vocabulary, random: random))
            revealed = logicalOr(revealed, acceptance)
            draft = which(acceptance, accepted, draft)
            if all(revealed).item(Bool.self) {
                ending = .allRevealed
                break
            }
            if DiffusionSampler.stableAndConfident(
                canvas: argmaxCanvas, logits: logits, history: &history, rule: policy.stopping)
            {
                ending = .stableAndConfident
                break
            }
            // With a quantized embedding the conditioning is the processed logits themselves
            // (`diffusion_self_conditioning`), which the next pass turns into soft embeddings.
            conditioning = logits
        }
        eval(argmaxCanvas, initialCanvas)
        let tokens = argmaxCanvas[0].asArray(Int32.self).map(Int.init)
        return DenoisedBlock(
            initialCanvas: initialCanvas[0].asArray(Int32.self).map(Int.init), tokens: tokens,
            steps: steps, ending: ending)
    }
}
