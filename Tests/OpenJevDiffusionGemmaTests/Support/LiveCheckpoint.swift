import Foundation
import OpenJevDiffusionGemma

/// The real checkpoint, loaded once per process.
///
/// `@unchecked Sendable` because the model holds MLX arrays: the suites that use it are nested in
/// the serialized ``MLXTests``, so one test touches it at a time. CheckpointTests and ReadTests
/// share it, so the 16 GB load happens once per test process.
final class LiveCheckpoint: @unchecked Sendable {
    let loaded: DiffusionGemmaModel.LoadedModel

    init(_ loaded: DiffusionGemmaModel.LoadedModel) {
        self.loaded = loaded
    }

    private static let loading = Task { () throws -> LiveCheckpoint in
        MetalLibrary.configure()
        let loaded = try await DiffusionGemmaModel.load(from: ModelFixtures.checkpointDirectory)
        return LiveCheckpoint(loaded)
    }

    static func shared() async throws -> LiveCheckpoint {
        try await loading.value
    }
}
