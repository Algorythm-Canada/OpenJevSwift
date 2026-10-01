// The backends `OPENJEV_BACKEND` selects, as `create_app` in upstream OpenJev's `openjev/api.py`
// (razorback16/openjev at dcd2094) picks its engine. Apache-2.0. See THIRD_PARTY.md.

import OpenJevCore
import OpenJevServer

#if canImport(OpenJevEncoders)
    import OpenJevEncoders
#endif

/// The backends `OPENJEV_BACKEND` may name, in upstream's order: `mlx`, then the encoder models.
///
/// Upstream also has `vllm`, `clm` and `jevk5`, which this port does not: `vllm` because it has no
/// vLLM backend (D-030), `clm` and `jevk5` until issues #59 and #55. They are unknown here, as any
/// other name is. `mlx` and `laya` are known but not built yet, and `verdict` needs Core ML.
struct BackendRegistry: Sendable {
    /// One backend.
    struct Backend: Sendable {
        /// What kind of engine serves it, which decides the settings `serve` prints.
        enum Kind: Sendable {
            /// ``DecisionEngine`` over a ``DecisionBackend``: DiffusionGemma.
            case diffusion
            /// ``EncoderDecisionEngine`` over a ``QuestionReadBackend``.
            case encoder
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

    /// The issue page of this repository's issue `number`.
    static func issue(_ number: Int) -> String {
        "https://github.com/Algorythm-Canada/OpenJevSwift/issues/\(number)"
    }

    /// The backends of this build: `mlx` and `laya`, not built yet, and `verdict`.
    static let standard = BackendRegistry(backends: [
        Backend(
            name: "mlx", modelName: ServedModels.diffusionGemmaVersion, kind: .diffusion,
            servedModels: .diffusionGemma,
            availability: .unavailable(
                "OPENJEV_BACKEND=mlx: DiffusionGemma on MLX is not in this build yet; issue #29 "
                    + "brings it (\(issue(29)))")),
        Backend(
            name: "laya", modelName: KnownEncoderModels.laya.name, kind: .encoder,
            servedModels: .encoder(KnownEncoderModels.laya),
            availability: .unavailable(
                "OPENJEV_BACKEND=laya: the Laya backend is not in this build yet; issue #58 "
                    + "brings it (\(issue(58)))")),
        Backend(
            name: "verdict", modelName: KnownEncoderModels.verdict.name, kind: .encoder,
            servedModels: .encoder(KnownEncoderModels.verdict), availability: verdict),
    ])

    /// Verdict on Core ML (D-011, D-034): the package from the store that `OPENJEV_ENCODER_MODELS`
    /// points at a folder of converted packages, or from the release downloads (D-033).
    private static var verdict: Backend.Availability {
        #if canImport(OpenJevEncoders)
            return .available { environment, willWarmUp in
                QuestionReadBackendProvider(
                    load: { _ in
                        try await VerdictBackend.load(
                            from: EncoderPackageStore(environment: environment))
                    }, willWarmUp: willWarmUp)
            }
        #else
            return .unavailable(
                "OPENJEV_BACKEND=verdict: Verdict runs on Core ML, which this platform does not "
                    + "have; serve it from a Mac with macOS 15 or later")
        #endif
    }
}
