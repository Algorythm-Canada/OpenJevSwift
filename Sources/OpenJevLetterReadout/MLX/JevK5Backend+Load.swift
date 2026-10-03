// The loading half of upstream OpenJev's `JevK5Engine.load` and `jevk5_temperature`
// (razorback16/openjev at dcd2094, `openjev/encoders.py`), with the model in process on MLX
// instead of a vLLM server. Apache-2.0. See THIRD_PARTY.md.

import Foundation
import MLX
import OpenJevDiffusionGemma

extension JevK5Backend {
    /// Loads JevK5 to run on MLX: resolves the checkpoint, reads its calibration temperature from
    /// `jevk5_config.json`, loads the tokenizer (checking that every letter is one token) and the
    /// weights, and caps MLX's buffer pool when asked.
    ///
    /// This is the server's backend for `OPENJEV_BACKEND=jevk5`, with `OPENJEV_JEVK5_MODEL` as
    /// the source and `OPENJEV_MLX_CACHE_LIMIT_GB` as `cacheLimitGB`:
    ///
    /// ```swift
    /// QuestionReadBackendProvider { settings in
    ///     try await JevK5Backend.load(
    ///         JevK5ModelFiles.source(setting: settings.jevk5Model),
    ///         cacheLimitGB: settings.mlxCacheLimitGB)
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - source: the checkpoint, a converted folder or Hub repository; by default
    ///     ``JevK5Checkpoint/platformDefault``'s repository, the 8-bit conversion on macOS and the
    ///     4-bit one on iOS, which is refused until it is published (D-052).
    ///   - cache: the Hugging Face cache a Hub source is kept in.
    ///   - token: the Hub access token, `HF_TOKEN`.
    ///   - cacheLimitGB: the MLX buffer pool's ceiling in GB, as upstream's `set_cache_limit`
    ///     takes it: nil leaves MLX alone, 0 disables the pool. It is process-wide.
    ///   - resolver: the downloader, the public Hub unless a test serves its own.
    /// - Throws: ``JevK5LoadError``, ``/OpenJevDiffusionGemma/ModelResolverError``, and the
    ///   tokenizer's and the weight loader's errors.
    public static func load(
        _ source: ModelSource = JevK5Checkpoint.platformDefault.hubSource,
        cache: HubCacheLocation = .standard, token: String? = nil, cacheLimitGB: Double? = nil,
        resolver: ModelResolver = ModelResolver()
    ) async throws -> JevK5Backend {
        if let cacheLimitGB {
            let bytes = cacheLimitGB * 1024 * 1024 * 1024
            guard cacheLimitGB.isFinite, cacheLimitGB >= 0, bytes < Double(Int.max) else {
                throw JevK5LoadError.invalidCacheLimit(cacheLimitGB)
            }
            Memory.cacheLimit = Int(bytes)
        }
        let directory = try await JevK5ModelFiles.resolve(
            source, cache: cache, token: token, resolver: resolver)
        let calibration = try JevK5Calibration(
            contentsOf: directory.appendingPathComponent("jevk5_config.json"))
        let tokenizer = try await JevK5Tokenizer.load(directory: directory)
        let model = try await Qwen35LetterReadoutModel.load(directory: directory)
        return try JevK5Backend(
            model: model, tokenizer: tokenizer, temperature: calibration.temperature)
    }
}
