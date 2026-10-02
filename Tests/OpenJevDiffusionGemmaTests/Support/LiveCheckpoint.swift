import Foundation

@testable import OpenJevDiffusionGemma

/// The real checkpoint, loaded once per process through ``DiffusionGemmaRuntime/load(_:configuration:cache:token:resolver:progress:)``
/// from its directory, warm-up included.
///
/// `@unchecked Sendable` because the model holds MLX arrays: the suites that use it are nested in
/// the serialized ``MLXTests``, so one test touches it at a time. CheckpointTests, the read
/// suites and the runtime suites share it, so the 16 GB load happens once per test process.
final class LiveCheckpoint: @unchecked Sendable {
    /// The runtime, which owns the model.
    let runtime: DiffusionGemmaRuntime
    /// The runtime's model, for the suites that drive the model directly.
    let loaded: DiffusionGemmaModel.LoadedModel
    /// What loading took.
    let report: DiffusionGemmaRuntime.LoadReport

    private init(
        runtime: DiffusionGemmaRuntime, loaded: DiffusionGemmaModel.LoadedModel,
        report: DiffusionGemmaRuntime.LoadReport
    ) {
        self.runtime = runtime
        self.loaded = loaded
        self.report = report
    }

    /// MLX's buffer pool limit for the run, `OPENJEV_MLX_CACHE_LIMIT_GB` as the server reads it;
    /// unset leaves MLX alone. A limit leaves every read bit-identical (spike #22) and keeps a long
    /// run's pool, which otherwise grows to the peak working set, from swapping (D-048).
    static let cacheLimitGB = ProcessInfo.processInfo.environment["OPENJEV_MLX_CACHE_LIMIT_GB"]
        .flatMap(Double.init)

    private static let loading = Task { () throws -> LiveCheckpoint in
        MetalLibrary.configure()
        let runtime = try await DiffusionGemmaRuntime.load(
            .directory(ModelFixtures.checkpointDirectory),
            configuration: .init(cacheLimitGB: cacheLimitGB, warmUp: true))
        guard let loaded = runtime.sharedLoadedModel, let report = await runtime.loadReport
        else { throw CocoaError(.featureUnsupported) }
        return LiveCheckpoint(runtime: runtime, loaded: loaded, report: report)
    }

    static func shared() async throws -> LiveCheckpoint {
        try await loading.value
    }
}
