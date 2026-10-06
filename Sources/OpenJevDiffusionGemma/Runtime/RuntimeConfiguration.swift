// The settings upstream OpenJev's MlxEngine applies to its MlxRuntime (razorback16/openjev at
// dcd2094, openjev/mlx_backend.py lines 45 to 50, 129 to 148 and 260 to 271, and
// openjev/config.py lines 39 to 49, Apache-2.0, see THIRD_PARTY.md), and the runtime's
// diagnostics.

import Foundation
import MLX

extension DiffusionGemmaRuntime {
    /// The runtime's settings, with upstream's defaults.
    ///
    /// The library never reads the environment (D-013). The CLI maps `ServerSettings` onto this
    /// initializer: `mlxMaxPrompt` to ``maxPromptTokens``, `mlxPromptCache` to
    /// ``promptCacheEntries``, `mlxCacheLimitGB` to ``cacheLimitGB`` and `warmup` to ``warmUp``.
    public struct Configuration: Sendable, Hashable {
        /// The most prompt tokens one read may carry, `OPENJEV_MLX_MAX_PROMPT` (32,768).
        public var maxPromptTokens: Int
        /// The most prefills cached, `OPENJEV_MLX_PROMPT_CACHE` (12). 0 turns the cache off.
        public var promptCacheEntries: Int
        /// The most prompt tokens the cached prefills may hold, upstream's `PROMPT_CACHE_TOKENS`
        /// (16,384), which upstream does not expose as a setting.
        public var promptCacheTokens: Int
        /// MLX's buffer pool limit in GB, `OPENJEV_MLX_CACHE_LIMIT_GB`. nil leaves MLX alone; 0
        /// disables the pool, the worst allocator churn rather than a way back to the default;
        /// any other value sets `Memory.cacheLimit` to that many GiB (`gb × 1024³` bytes, as
        /// upstream's `set_cache_limit` computes it).
        public var cacheLimitGB: Double?
        /// Whether loading runs one small read so the first user does not pay kernel compilation,
        /// `OPENJEV_WARMUP` (on).
        public var warmUp: Bool
        /// The seed of MLX's generator for each reply's random canvases (0). Every reply draws
        /// from a generator seeded with it, so a prompt gets the same reply each time, in any
        /// process; upstream leaves MLX's generator unseeded, so its replies vary from one
        /// process to the next. Fixtures/generation records the replies with seed 0.
        public var generationSeed: UInt64

        /// Creates a configuration; every argument defaults to upstream's value.
        public init(
            maxPromptTokens: Int = 32_768,
            promptCacheEntries: Int = PrefillCacheDefaults.entries,
            promptCacheTokens: Int = PrefillCacheDefaults.tokens,
            cacheLimitGB: Double? = nil,
            warmUp: Bool = true,
            generationSeed: UInt64 = 0
        ) {
            self.maxPromptTokens = maxPromptTokens
            self.promptCacheEntries = promptCacheEntries
            self.promptCacheTokens = promptCacheTokens
            self.cacheLimitGB = cacheLimitGB
            self.warmUp = warmUp
            self.generationSeed = generationSeed
        }

        /// Upstream's defaults.
        public static let `default` = Configuration()

        /// The byte count ``cacheLimitGB`` asks MLX for, or nil when MLX is left alone. 0 stays
        /// 0, which is not the same as unset.
        ///
        /// - Throws: ``DiffusionGemmaRuntimeError/invalidCacheLimit(_:)`` for a value that is not
        ///   a finite number of GB, 0 or more, that fits in an `Int` of bytes; `nan` and `inf`
        ///   are among them, which `OPENJEV_MLX_CACHE_LIMIT_GB` accepts as upstream's does.
        public func cacheLimitBytes() throws(DiffusionGemmaRuntimeError) -> Int? {
            try Self.bytes(gb: cacheLimitGB)
        }

        /// `gb × 1024³` as an `Int`, nil for nil.
        static func bytes(gb: Double?) throws(DiffusionGemmaRuntimeError) -> Int? {
            guard let gb else { return nil }
            let bytes = gb * 1024 * 1024 * 1024
            guard gb.isFinite, gb >= 0, bytes < Double(Int.max) else {
                throw DiffusionGemmaRuntimeError.invalidCacheLimit(gb)
            }
            return Int(bytes)
        }
    }

    /// The memory MLX and the process hold, for diagnostics and the memory risk (R4).
    public struct MemoryReport: Sendable, Hashable, CustomStringConvertible {
        /// MLX's live arrays in bytes (`Memory.activeMemory`).
        public var activeBytes: Int
        /// MLX's buffer pool in bytes (`Memory.cacheMemory`).
        public var cacheBytes: Int
        /// MLX's peak active memory in bytes since the process started or the peak was reset.
        public var peakBytes: Int
        /// The process's resident size in bytes (`task_info`).
        public var residentBytes: Int
        /// The process's peak resident size in bytes (`getrusage`).
        public var peakResidentBytes: Int

        /// Creates a report from figures in bytes.
        public init(
            activeBytes: Int, cacheBytes: Int, peakBytes: Int, residentBytes: Int,
            peakResidentBytes: Int
        ) {
            self.activeBytes = activeBytes
            self.cacheBytes = cacheBytes
            self.peakBytes = peakBytes
            self.residentBytes = residentBytes
            self.peakResidentBytes = peakResidentBytes
        }

        /// The figures now.
        static func current() -> MemoryReport {
            let usage = ResourceUsage.current()
            return MemoryReport(
                activeBytes: Memory.activeMemory, cacheBytes: Memory.cacheMemory,
                peakBytes: Memory.peakMemory, residentBytes: usage.residentBytes,
                peakResidentBytes: usage.peakResidentBytes)
        }

        /// The five figures in GiB, for logs.
        public var description: String {
            func gib(_ bytes: Int) -> String {
                String(format: "%.2f GiB", Double(bytes) / Double(1 << 30))
            }
            return "MLX active \(gib(activeBytes)), cache \(gib(cacheBytes)), peak "
                + "\(gib(peakBytes)); resident \(gib(residentBytes)), peak resident "
                + "\(gib(peakResidentBytes))"
        }
    }

    /// What the runtime has read since it was made, for diagnostics.
    public struct ReadStatistics: Sendable, Hashable {
        /// The reads run.
        public var reads: Int
        /// The prefill cache's hits.
        public var prefillHits: Int
        /// Its misses: the prompts that were prefilled.
        public var prefillMisses: Int
        /// The prefills cached now.
        public var cachedPrefills: Int
        /// The prompt tokens they hold.
        public var cachedPrefillTokens: Int
        /// The time spent in the model, prefills included.
        public var modelTime: Duration

        /// The share of prefill lookups that hit, 0 before the first read.
        public var hitRate: Double {
            let lookups = prefillHits + prefillMisses
            return lookups == 0 ? 0 : Double(prefillHits) / Double(lookups)
        }
    }

    /// The stages ``DiffusionGemmaRuntime/load(_:configuration:cache:token:resolver:progress:)`` reports,
    /// in order, for the CLI to render.
    public enum LoadStage: Sendable, Hashable {
        /// Resolving the model source; for a Hub source, checking the snapshot and downloading
        /// what it lacks.
        case resolving(ModelResolver.Progress)
        /// Loading the tokenizer from the snapshot.
        case loadingTokenizer
        /// Loading the weights, with the model loader's own stage.
        case loadingWeights(DiffusionGemmaModel.LoadStage)
        /// Applying ``Configuration/cacheLimitGB``.
        case applyingCacheLimit
        /// Running the warm-up read.
        case warmingUp
        /// Loaded and ready.
        case ready
    }

    /// What loading took, for the CLI to print.
    public struct LoadReport: Sendable, Hashable {
        /// The directory the model was loaded from.
        public var directory: URL
        /// The time resolving the source took, downloads included.
        public var resolveTime: Duration
        /// The bytes downloaded while resolving; 0 for a local directory or a complete snapshot.
        public var downloadedBytes: Int
        /// The files downloaded while resolving.
        public var downloadedFiles: Int
        /// What loading the tokenizer cost.
        public var tokenizerMetrics: LoadMetrics
        /// What loading the weights cost.
        public var modelMetrics: DiffusionGemmaModel.LoadMetrics
        /// The warm-up read's time, nil when ``Configuration/warmUp`` is off.
        public var warmUpTime: Duration?
        /// The memory after loading and warming up.
        public var memory: MemoryReport
    }
}
