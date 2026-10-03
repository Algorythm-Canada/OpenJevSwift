// The backends `OPENJEV_BACKEND` selects, as `create_app` in upstream OpenJev's `openjev/api.py`
// (razorback16/openjev at dcd2094) picks its engine. Apache-2.0. See THIRD_PARTY.md.

import OpenJevCore
import OpenJevServer

#if canImport(OpenJevEncoders)
    import OpenJevEncoders
#endif
#if canImport(OpenJevDiffusionGemma)
    import OpenJevDiffusionGemma
#endif
#if canImport(OpenJevLetterReadout)
    import OpenJevLetterReadout
#endif

/// The backends `OPENJEV_BACKEND` may name, in upstream's order: `mlx`, then the encoder models.
///
/// Upstream also has `vllm` and `clm`, which this port does not: `vllm` because it has no vLLM
/// backend (D-030), `clm` until issue #59. They are unknown here, as any other name is. `mlx` and
/// `jevk5` need MLX, and `verdict` and `laya` need Core ML.
struct BackendRegistry: Sendable {
    /// One backend.
    struct Backend: Sendable {
        /// What kind of engine serves it, which decides the settings `serve` prints.
        enum Kind: Sendable {
            /// ``DecisionEngine`` over a ``DecisionBackend``: DiffusionGemma.
            case diffusion
            /// ``EncoderDecisionEngine`` over a Core ML ``QuestionReadBackend``: Verdict, Laya.
            case encoder
            /// ``EncoderDecisionEngine`` over JevK5's letter readout on MLX.
            case letterReadout
        }

        /// How to load the backend, or why this build cannot.
        enum Availability: Sendable {
            /// The provider that loads it.
            case available(MakeProvider)
            /// This build cannot load it; the message says why and what to do.
            case unavailable(String)
        }

        /// Makes a backend's provider from the environment, which the encoder store reads
        /// (`OPENJEV_ENCODER_MODELS`), and a function to call before the warm-up read.
        typealias MakeProvider =
            @Sendable (_ environment: [String: String], _ willWarmUp: @escaping WillWarmUp) ->
            any BackendProvider
        /// Runs before the warm-up read.
        typealias WillWarmUp = @Sendable () -> Void

        /// The name `OPENJEV_BACKEND` gives.
        var name: String
        /// The served model's name, for the phase lines.
        var modelName: String
        /// The engine kind.
        var kind: Kind
        /// What `GET /v1/models` lists for it, when that needs no model; `nil` when only the
        /// loaded backend knows.
        var servedModels: ServedModels?
        /// How to load it.
        var availability: Availability

        /// The backend's provider.
        ///
        /// - Throws: ``CommandFailure`` with ``ExitStatus/backendUnavailable`` when this build
        ///   cannot load it.
        func provider(
            environment: [String: String], willWarmUp: @escaping @Sendable () -> Void = {}
        ) throws(CommandFailure) -> any BackendProvider {
            switch availability {
            case .available(let make):
                return make(environment, willWarmUp)
            case .unavailable(let message):
                throw CommandFailure(.backendUnavailable, message: message)
            }
        }

        /// Loads the backend's service.
        ///
        /// - Throws: ``CommandFailure`` with ``ExitStatus/backendUnavailable`` when this build
        ///   cannot load it or the load fails, naming the error.
        func load(
            settings: ServerSettings, environment: [String: String],
            willWarmUp: @escaping @Sendable () -> Void = {}
        ) async throws(CommandFailure) -> any SystemOneService {
            let provider = try provider(environment: environment, willWarmUp: willWarmUp)
            do {
                return try await provider.makeService(settings: settings)
            } catch {
                throw CommandFailure(
                    .backendUnavailable,
                    message: "\(modelName) failed to load (OPENJEV_BACKEND=\(name)): \(error)")
            }
        }
    }

    /// Every backend, in the order the unknown-backend message lists them.
    var backends: [Backend]

    /// The backend named `name`.
    ///
    /// - Throws: ``CommandFailure`` with ``ExitStatus/invalidSettings`` and upstream's message for
    ///   a name no backend has.
    func backend(named name: String) throws(CommandFailure) -> Backend {
        guard let backend = backends.first(where: { $0.name == name }) else {
            throw CommandFailure(
                .invalidSettings,
                message: ServerSettingsError.unknownBackend(name, known: backends.map(\.name))
                    .message)
        }
        return backend
    }

    /// The registry with `backend` added at the end, or in the place of one with its name.
    func adding(_ backend: Backend) -> BackendRegistry {
        var registry = self
        registry.backends.removeAll { $0.name == backend.name }
        registry.backends.append(backend)
        return registry
    }

    /// The backends of this build: `mlx`, then `laya`, `verdict` and `jevk5`.
    static let standard = BackendRegistry(backends: [
        Backend(
            name: "mlx", modelName: ServedModels.diffusionGemmaVersion, kind: .diffusion,
            servedModels: .diffusionGemma, availability: mlx),
        Backend(
            name: "laya", modelName: KnownEncoderModels.laya.name, kind: .encoder,
            servedModels: .encoder(KnownEncoderModels.laya), availability: laya),
        Backend(
            name: "verdict", modelName: KnownEncoderModels.verdict.name, kind: .encoder,
            servedModels: .encoder(KnownEncoderModels.verdict), availability: verdict),
        Backend(
            name: "jevk5", modelName: KnownEncoderModels.jevk5.name, kind: .letterReadout,
            servedModels: .encoder(KnownEncoderModels.jevk5), availability: jevk5),
    ])

    /// DiffusionGemma on MLX (D-039): the checkpoint `OPENJEV_MLX_MODEL` names, a directory or a
    /// Hub repository resolved in the Hugging Face cache the environment names, with `HF_TOKEN`.
    /// The runtime warms itself up when `OPENJEV_WARMUP` asks.
    private static var mlx: Backend.Availability {
        #if canImport(OpenJevDiffusionGemma)
            return .available { environment, willWarmUp in
                DecisionBackendProvider { settings in
                    try await DiffusionGemmaRuntime.load(
                        ModelSource(setting: settings.mlxModel),
                        configuration: .init(
                            maxPromptTokens: settings.mlxMaxPrompt,
                            promptCacheEntries: settings.mlxPromptCache,
                            cacheLimitGB: settings.mlxCacheLimitGB, warmUp: settings.warmup),
                        cache: HubCacheLocation(environment: environment),
                        token: HubCacheLocation.token(environment: environment),
                        progress: { stage in
                            if case .warmingUp = stage {
                                willWarmUp()
                            }
                        })
                }
            }
        #else
            return .unavailable(
                "OPENJEV_BACKEND=mlx: DiffusionGemma runs on MLX, which needs Apple silicon; "
                    + "serve it from a Mac")
        #endif
    }

    /// Verdict on Core ML (D-011, D-034): the package from the store that `OPENJEV_ENCODER_MODELS`
    /// points at a folder of converted packages, or from the release downloads (D-033), with at
    /// most `OPENJEV_ENCODER_FUNCTIONS` functions loaded (D-042).
    private static var verdict: Backend.Availability {
        #if canImport(OpenJevEncoders)
            return .available { environment, willWarmUp in
                QuestionReadBackendProvider(
                    load: { settings in
                        try await VerdictBackend.load(
                            from: EncoderPackageStore(environment: environment),
                            functionCapacity: settings.encoderFunctions)
                    }, willWarmUp: willWarmUp)
            }
        #else
            return .unavailable(
                "OPENJEV_BACKEND=verdict: Verdict runs on Core ML, which this platform does not "
                    + "have; serve it from a Mac with macOS 15 or later")
        #endif
    }

    /// Laya on Core ML (D-037): on a Mac its multifunction package, from the store and with
    /// `OPENJEV_ENCODER_FUNCTIONS` as Verdict's.
    private static var laya: Backend.Availability {
        #if canImport(OpenJevEncoders)
            return .available { environment, willWarmUp in
                QuestionReadBackendProvider(
                    load: { settings in
                        try await LayaBackend.load(
                            from: EncoderPackageStore(environment: environment),
                            functionCapacity: settings.encoderFunctions)
                    }, willWarmUp: willWarmUp)
            }
        #else
            return .unavailable(
                "OPENJEV_BACKEND=laya: Laya runs on Core ML, which this platform does not have; "
                    + "serve it from a Mac with macOS 15 or later")
        #endif
    }

    /// JevK5 on MLX (D-052): the converted checkpoint `OPENJEV_JEVK5_MODEL` names, a folder or a
    /// Hub repository resolved in the Hugging Face cache the environment names, with `HF_TOKEN`,
    /// and MLX's buffer pool capped by `OPENJEV_MLX_CACHE_LIMIT_GB`. The engine warms it up with
    /// upstream's warm-up questions when `OPENJEV_WARMUP` asks.
    private static var jevk5: Backend.Availability {
        #if canImport(OpenJevLetterReadout)
            return .available { environment, willWarmUp in
                QuestionReadBackendProvider(
                    load: { settings in
                        try await JevK5Backend.load(
                            JevK5ModelFiles.source(setting: settings.jevk5Model),
                            cache: HubCacheLocation(environment: environment),
                            token: HubCacheLocation.token(environment: environment),
                            cacheLimitGB: settings.mlxCacheLimitGB)
                    }, willWarmUp: willWarmUp)
            }
        #else
            return .unavailable(
                "OPENJEV_BACKEND=jevk5: JevK5 runs on MLX, which needs Apple silicon; serve it "
                    + "from a Mac")
        #endif
    }
}
