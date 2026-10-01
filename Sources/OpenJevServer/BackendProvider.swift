// A port of upstream OpenJev (razorback16/openjev at dcd2094), the `lifespan` of `create_app` in
// `openjev/api.py`, which loads the engine the settings name, and the `Settings` fields `Engine`
// and `EncoderEngine` read. Apache-2.0. See THIRD_PARTY.md.

import OpenJevCore

/// Loads the model the settings name and returns the service that answers for it, as upstream's
/// `lifespan` builds `app.state.engine`.
///
/// The server calls it once, before it binds. Tests provide a stub-backed service; the CLI
/// provides the DiffusionGemma runtime or an encoder backend.
public protocol BackendProvider: Sendable {
    /// Loads the model and returns the service that answers `POST /v1/systemone` and
    /// `GET /v1/models`.
    func makeService(settings: ServerSettings) async throws -> any SystemOneService
}

/// A provider that loads a ``DecisionBackend`` and serves it through a ``DecisionEngine``
/// configured from the settings.
public struct DecisionBackendProvider: BackendProvider {
    private let load: @Sendable (ServerSettings) async throws -> any DecisionBackend

    /// Creates a provider from a function that loads the backend.
    public init(load: @escaping @Sendable (ServerSettings) async throws -> any DecisionBackend) {
        self.load = load
    }

    /// Loads the backend and builds the engine over it.
    public func makeService(settings: ServerSettings) async throws -> any SystemOneService {
        let backend = try await load(settings)
        return try DecisionEngine(backend: backend, configuration: EngineConfiguration(settings))
    }
}

/// A provider that loads a ``QuestionReadBackend`` and serves it through an
/// ``EncoderDecisionEngine`` configured from the settings, warmed up when `OPENJEV_WARMUP` asks.
public struct QuestionReadBackendProvider: BackendProvider {
    private let load: @Sendable (ServerSettings) async throws -> any QuestionReadBackend
    private let willWarmUp: @Sendable () -> Void

    /// Creates a provider from a function that loads the backend. `willWarmUp` runs once the
    /// backend has loaded, just before the warm-up read, and not at all without one; the CLI
    /// prints its phase there.
    public init(
        load: @escaping @Sendable (ServerSettings) async throws -> any QuestionReadBackend,
        willWarmUp: @escaping @Sendable () -> Void = {}
    ) {
        self.load = load
        self.willWarmUp = willWarmUp
    }

    /// Loads the backend, builds the engine over it and runs upstream's warm-up read when the
    /// settings ask for it.
    public func makeService(settings: ServerSettings) async throws -> any SystemOneService {
        let backend = try await load(settings)
        let engine = EncoderDecisionEngine(
            backend: backend, configuration: EncoderEngineConfiguration(settings))
        if settings.warmup {
            willWarmUp()
            try await engine.warmUp()
        }
        return engine
    }
}

extension EngineConfiguration {
    /// The settings `Engine` reads: the canvas, the re-read policy, the read and queue bounds and
    /// the image limits. The rest keep their defaults.
    ///
    /// - Throws: ``CanvasGeometryError``, which validated settings never cause.
    public init(_ settings: ServerSettings) throws {
        self.init(
            geometry: try CanvasGeometry(canvas: settings.canvas, step: settings.canvasStep),
            autoThreshold: settings.autoThreshold,
            autoMax: settings.autoMax,
            maxInflight: settings.maxInflight,
            maxQueue: settings.maxQueue,
            imageLimits: ImageLimits(
                maxImages: settings.maxImages, maxImageBytes: settings.maxImageBytes))
    }
}

extension EncoderEngineConfiguration {
    /// The settings `EncoderEngine` reads: the batch size, the queue bound and the warm-up. Reads
    /// stay one at a time, upstream's one model thread.
    public init(_ settings: ServerSettings) {
        self.init(
            batchSize: settings.encoderBatch, maxQueue: settings.maxQueue,
            warmUp: settings.warmup)
    }
}
