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

    private static let loading = Task { () throws -> LiveCheckpoint in
        MetalLibrary.configure()
        let runtime = try await DiffusionGemmaRuntime.load(
            .directory(ModelFixtures.checkpointDirectory),
            configuration: .init(cacheLimitGB: nil, warmUp: true))
        guard let loaded = runtime.sharedLoadedModel, let report = await runtime.loadReport
        else { throw CocoaError(.featureUnsupported) }
        return LiveCheckpoint(runtime: runtime, loaded: loaded, report: report)
    }

    static func shared() async throws -> LiveCheckpoint {
        try await loading.value
    }
}
