// The DiffusionGemma runtime (issue #29): upstream OpenJev's MlxRuntime and MlxEngine
// (razorback16/openjev at dcd2094, openjev/mlx_backend.py lines 76 to 208 and 260 to 303,
// Apache-2.0, see THIRD_PARTY.md) as one actor that conforms to the core's DecisionBackend.

import Foundation
import MLX
import OpenJevCore

/// Why the runtime refused a call.
public enum DiffusionGemmaRuntimeError: Error, Sendable, Hashable, CustomStringConvertible {
    /// A feature that arrives with a later milestone: `think` (generation, milestone 5) or
    /// `images` (the vision milestone). ``DiffusionGemmaRuntime/capabilities`` flags both off, so
    /// the engine refuses such requests before they get here.
    case unsupported(String)

    public var description: String {
        switch self {
        case .unsupported("think"):
            return "the DiffusionGemma runtime does not support think yet; generation arrives "
                + "with milestone 5"
        case .unsupported("images"):
            return "the DiffusionGemma runtime does not read image prompts yet; images arrive "
                + "with the vision milestone"
        case .unsupported(let feature):
            return "the DiffusionGemma runtime does not support \(feature)"
        }
    }
}

/// DiffusionGemma on MLX behind the core's ``DecisionBackend``.
///
/// The actor owns the model, the tokenizer and the prefill cache, and every MLX evaluation runs
/// inside it, one at a time, as upstream runs them on its one MLX thread (R14). Only values
/// cross its boundary: ``CanvasRead`` in, ``ReadResult`` out. Concurrent callers queue on the
/// actor, so concurrency buys prefill cache reuse and overlap of the engine's CPU work, not GPU
/// parallelism.
///
/// ```swift
/// let runtime = try await DiffusionGemmaRuntime.load(.fourBit)
/// let engine = try DecisionEngine(backend: runtime, configuration: .default)
/// ```
public actor DiffusionGemmaRuntime: DecisionBackend {
    /// Upstream's `TOPK`: the tokens kept per slot beside the labels.
    public static let topK = 20

    /// The model operations the runtime drives. The real runtime binds them to a
    /// ``DiffusionGemmaModel``; the model-free tests bind them to a stub.
    struct ModelCalls {
        var prefill: (_ promptIDs: [Int]) throws -> PromptCache
        var read:
            (
                _ canvas: [Int], _ slots: [SlotRequest], _ cache: PromptCache, _ steps: Int,
                _ topK: Int
            ) throws -> ReadOutput
    }

    public nonisolated let tokenizer: any DecisionTokenizer
    /// The runtime's settings.
    public nonisolated let configuration: Configuration
    /// ``Configuration/maxPromptTokens``.
    public nonisolated var maxPromptTokens: Int { configuration.maxPromptTokens }
    /// Steps, samples and sequential reads; not `think` until milestone 5, nor images until the
    /// vision milestone, so the engine answers `"openjev-0.1 does not support think"` and its
    /// image refusal.
    public nonisolated let capabilities = BackendCapabilities(
        steps: true, samples: true, think: false, sequential: true, images: false)
    /// `openjev-0.1`, ``ServedModels/diffusionGemmaVersion``.
    public nonisolated let modelName = ServedModels.diffusionGemmaVersion

    /// The model the live tests share with the model-level suites, so the 16 GB checkpoint loads
    /// once per test process. Those suites are serialized under one parent and never overlap a
    /// read; nothing else reads it.
    nonisolated(unsafe) let sharedLoadedModel: DiffusionGemmaModel.LoadedModel?

    private let calls: ModelCalls
    private let setCacheLimit: @Sendable (Int) -> Void
    private var prefills: PrefillCache<PromptCache>
    private var reads = 0
    private var modelTime = Duration.zero
    /// What loading took; nil for a runtime made without ``load(_:configuration:cache:token:resolver:progress:)``.
    public private(set) var loadReport: LoadReport?

    /// A runtime over `calls`, for the model-free tests.
    init(
        tokenizer: any DecisionTokenizer, configuration: Configuration,
        calls: sending ModelCalls,
        setCacheLimit: @escaping @Sendable (Int) -> Void = { Memory.cacheLimit = $0 }
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.calls = calls
        self.setCacheLimit = setCacheLimit
        sharedLoadedModel = nil
        prefills = PrefillCache(
            entryBudget: configuration.promptCacheEntries,
            tokenBudget: configuration.promptCacheTokens)
    }

    /// A runtime over a loaded model.
    init(
        tokenizer: any DecisionTokenizer, configuration: Configuration,
        loaded: sending DiffusionGemmaModel.LoadedModel
    ) {
        let model = loaded.model
        self.tokenizer = tokenizer
        self.configuration = configuration
        calls = ModelCalls(
            prefill: { try model.prefill(promptIDs: $0) },
            read: { canvas, slots, cache, steps, topK in
                try model.read(canvas: canvas, slots: slots, cache: cache, steps: steps, topK: topK)
            })
        setCacheLimit = { Memory.cacheLimit = $0 }
        sharedLoadedModel = loaded
        prefills = PrefillCache(
            entryBudget: configuration.promptCacheEntries,
            tokenBudget: configuration.promptCacheTokens)
    }

    // MARK: Loading

    /// Resolves `source`, loads its tokenizer and weights, applies the cache limit and warms up.
    ///
    /// - Parameters:
    ///   - source: the checkpoint, ``ModelSource/fourBit`` by default.
    ///   - configuration: the runtime's settings.
    ///   - cache: the Hugging Face cache a Hub source is kept in.
    ///   - token: the Hub access token, `HF_TOKEN` (``HubCacheLocation/token(environment:)``).
    ///   - resolver: the downloader, the public Hub unless a test serves its own.
    ///   - progress: each ``LoadStage`` in order.
    /// - Throws: ``ModelResolverError``, ``TokenizerFilesError``, the tokenizer's and the weight
    ///   loader's errors, or the warm-up read's.
    public static func load(
        _ source: ModelSource = .fourBit, configuration: Configuration = .default,
        cache: HubCacheLocation = .standard, token: String? = nil,
        resolver: ModelResolver = ModelResolver(),
        progress: (@Sendable (LoadStage) -> Void)? = nil
    ) async throws -> DiffusionGemmaRuntime {
        let clock = ContinuousClock()
        let start = clock.now
        let resolution = try await resolver.resolution(
            of: source, cache: cache, token: token, progress: { progress?(.resolving($0)) })
        let resolveTime = clock.now - start

        progress?(.loadingTokenizer)
        let tokenizer = try await SwiftTransformersTokenizer.load(
            from: TokenizerFiles(directory: resolution.directory))
        let loaded = try await DiffusionGemmaModel.load(
            from: resolution.directory, progress: { progress?(.loadingWeights($0)) })
        let modelMetrics = loaded.metrics
        let runtime = DiffusionGemmaRuntime(
            tokenizer: tokenizer, configuration: configuration, loaded: loaded)
        try await runtime.prepare(
            resolution: resolution, resolveTime: resolveTime,
            tokenizerMetrics: tokenizer.loadMetrics, modelMetrics: modelMetrics,
            progress: progress)
        progress?(.ready)
        return runtime
    }

    /// The steps of loading that run inside the actor: the cache limit, the warm-up and the
    /// report.
    private func prepare(
        resolution: ModelResolver.Resolution, resolveTime: Duration,
        tokenizerMetrics: LoadMetrics, modelMetrics: DiffusionGemmaModel.LoadMetrics,
        progress: (@Sendable (LoadStage) -> Void)?
    ) throws {
        progress?(.applyingCacheLimit)
        applyConfiguredCacheLimit()
        var warmUpTime: Duration?
        if configuration.warmUp {
            progress?(.warmingUp)
            warmUpTime = try warmUp()
        }
        loadReport = LoadReport(
            directory: resolution.directory, resolveTime: resolveTime,
            downloadedBytes: resolution.downloadedBytes,
            downloadedFiles: resolution.downloadedFiles, tokenizerMetrics: tokenizerMetrics,
            modelMetrics: modelMetrics, warmUpTime: warmUpTime, memory: memoryReport())
    }

    // MARK: Memory

    /// Upstream's `set_cache_limit(gb)`: nil leaves MLX alone, 0 disables MLX's buffer pool,
    /// anything else caps it at `gb × 1024³` bytes.
    public func setCacheLimit(gb: Double?) {
        guard let gb else { return }
        setCacheLimit(Int(gb * 1024 * 1024 * 1024))
    }

    /// Applies ``Configuration/cacheLimitGB``, as loading does.
    func applyConfiguredCacheLimit() {
        setCacheLimit(gb: configuration.cacheLimitGB)
    }

    /// MLX's active, cache and peak bytes and the process's resident bytes now.
    public func memoryReport() -> MemoryReport {
        MemoryReport.current()
    }

    /// The reads run, the prefill cache's hits and contents, and the time spent in the model.
    public func statistics() -> ReadStatistics {
        ReadStatistics(
            reads: reads, prefillHits: prefills.hits, prefillMisses: prefills.misses,
            cachedPrefills: prefills.count, cachedPrefillTokens: prefills.tokenCount,
            modelTime: modelTime)
    }

    /// Empties the prefill cache; MLX frees the caches on its next allocation or `clearCache`.
    public func removeCachedPrefills() {
        prefills.removeAll()
    }

    /// The prefill cache's keys, oldest first, and its token total, for the tests.
    var prefillCacheState: (keys: [PrefillKey], tokens: Int, entryBudget: Int, tokenBudget: Int) {
        (prefills.keys, prefills.tokenCount, prefills.entryBudget, prefills.tokenBudget)
    }

    // MARK: DecisionBackend

    /// One read, upstream's `MlxEngine.one_read` and `MlxRuntime.read`: the prompt cap, the
    /// prefill or a cached one, `steps` decoder passes over the canvas, and each slot's top 20
    /// and labels through `slot_distribution`.
    ///
    /// - Throws: ``SchemaError`` `"the request is {n} tokens; the limit is {max}"` before anything
    ///   runs when the prompt is longer than ``maxPromptTokens``;
    ///   ``DiffusionGemmaRuntimeError/unsupported(_:)`` for an image prompt; ``ReadInputError``
    ///   for a canvas or slots the model refuses.
    public func read(_ read: CanvasRead) async throws -> ReadResult {
        let (output, slots) = try modelRead(read)
        return output.readResult(for: slots)
    }

    /// ``read(_:)`` before `slot_distribution`: the model's maps and the slots they are for.
    func modelRead(_ read: CanvasRead) throws -> (output: ReadOutput, slots: [SlotRequest]) {
        guard case .tokens(let ids) = read.prompt else {
            throw DiffusionGemmaRuntimeError.unsupported("images")
        }
        if ids.count > maxPromptTokens {
            throw SchemaError(
                "the request is \(ids.count) tokens; the limit is \(maxPromptTokens)")
        }
        let slots = read.slots.map { SlotRequest(position: $0.position, labelIDs: $0.labelIDs) }
        let clock = ContinuousClock()
        let start = clock.now
        defer { modelTime += clock.now - start }
        let calls = calls
        let cache = try prefills.value(for: .tokens(ids), tokens: ids.count) {
            try calls.prefill(ids)
        }.value
        let output = try calls.read(read.canvas.tokens, slots, cache, read.steps, Self.topK)
        reads += 1
        return (output, slots)
    }

    /// Throws ``DiffusionGemmaRuntimeError/unsupported(_:)`` until generation arrives with
    /// milestone 5. ``capabilities`` flags `think` off, so the engine never calls it.
    public func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws
        -> ThoughtGeneration
    {
        throw DiffusionGemmaRuntimeError.unsupported("think")
    }

    // MARK: Warm-up

    /// The warm-up request's state, upstream's `warmup.STATE`.
    static let warmUpState =
        "Checkout has been down for every customer since 9:02 and we are losing orders."

    /// The warm-up read's question, the first of upstream's `warmup.questions`.
    static let warmUpQuestion = ReadQuestion(
        key: "q0", id: "q1", kind: .noul,
        instructions: "The customer needs a reply within the hour",
        choices: [(name: "yes", description: ""), (name: "no", description: "")],
        labels: ["yes", "no"], legend: nil)

    /// Runs one small read on the model, outside the prefill cache, so the first user does not
    /// pay kernel compilation: one noul question over upstream's warm-up state, its prompt from
    /// the chat template and its canvas from ``CanvasBuilder`` with seed 0. No engine is needed.
    ///
    /// - Returns: the time it took.
    @discardableResult
    public func warmUp() throws -> Duration {
        let clock = ContinuousClock()
        let start = clock.now
        let system = SystemText.render([Self.warmUpQuestion], format: .lines, chunked: false)
        let ids = try tokenizer.chatPromptIDs(
            system: system, user: Self.warmUpState, thinking: false)
        let resolver = TemplateResolver(
            tokenizer: tokenizer, tokens: try EngineTokens(tokenizer: tokenizer))
        let template = try resolver.resolve([Self.warmUpQuestion], format: .lines)
        let canvas = CanvasBuilder.build(
            template: template.template, slots: template.slots, seed: 0, geometry: .standard)
        let slots = template.slots.map {
            SlotRequest(position: $0.position, labelIDs: $0.labelIDs)
        }
        let cache = try calls.prefill(ids)
        _ = try calls.read(canvas.tokens, slots, cache, 1, Self.topK)
        return clock.now - start
    }
}
