// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/config.py`,
// `MODEL_VERSION`, `MODEL_ALIASES`, `SDK_ALIASES`, `MODELS`, `ENCODER_MODELS` and
// `served_models`, the `models_list` that `create_app` in `openjev/api.py` builds from them and
// the model routes, and the `Engine` contract (`decide`) that `openjev/api.py` holds as
// `app.state.engine`. Apache-2.0. See THIRD_PARTY.md.

/// What the server holds to answer `POST /v1/systemone` and `GET /v1/models` without knowing
/// which kind of model is loaded: upstream's `app.state.engine` together with
/// `config.served_models`.
///
/// ``DecisionEngine`` (DiffusionGemma) and ``EncoderDecisionEngine`` (Verdict, Laya, CLM and
/// JevK5) conform. Both also adopt ``ModelReleasing``, which the server calls when it stops.
public protocol SystemOneService: Sendable {
    /// The model version a response names, the names a request may use and the `/v1/models`
    /// listing.
    var servedModels: ServedModels { get }

    /// Answers a request. The errors are ``SchemaError``, ``OverloadedError``, the backend's own,
    /// among them ``BackendRefusal``, and, for an encoder engine, ``BackendContractError``.
    ///
    /// A service adds the time of each backend call to ``ModelTimeRecorder/current`` as the call
    /// ends, which the server reports as `server-timing`'s `model`, and lets cancellation reach
    /// its reads: the server cancels the decision of a client that has gone away.
    func decide(_ request: SystemOneRequest) async throws -> Decision

    /// The model's text generation, which the server serves at `POST /v1/chat/completions`, or
    /// `nil` for a model that does not generate text, whose server has no chat routes, as
    /// upstream's encoder containers have none (decision D-012).
    var textGenerator: (any TextGenerator)? { get }
}

extension SystemOneService {
    /// `nil`: a service generates no text unless it says otherwise.
    public var textGenerator: (any TextGenerator)? { nil }
}

/// Upstream's `served_models(backend)`: the model version a response names, the names a request
/// may use and the `GET /v1/models` listing.
public struct ServedModels: Sendable, Hashable {
    /// The `model` a response names: `openjev-0.1` for DiffusionGemma, the served model's name
    /// for an encoder.
    public var version: String
    /// Every `model` a request may name. Always includes ``sdkAliases``.
    public var acceptedNames: Set<String>
    /// The `GET /v1/models` listing, in upstream's order.
    public var listing: [ModelInfo]

    /// Creates a description of the served models.
    public init(version: String, acceptedNames: Set<String>, listing: [ModelInfo]) {
        self.version = version
        self.acceptedNames = acceptedNames
        self.listing = listing
    }

    /// Upstream's `SDK_ALIASES`, accepted by every backend so TypeSafe's SDKs work unchanged:
    /// their default model is `jev-latest`.
    public static let sdkAliases: Set<String> = ["jev-latest", "jev-preview"]

    /// Upstream's `MODEL_VERSION`, the wire name of the DiffusionGemma release.
    public static let diffusionGemmaVersion = "openjev-0.1"

    /// Upstream's `MODELS`: the listing of the vLLM and MLX backends, with `openjev-latest`, the
    /// release and the text-generation model, dated 2026-09-18.
    public static let diffusionGemmaListing: [ModelInfo] = [
        ModelInfo(
            name: "openjev-latest",
            description: "Alias for the newest OpenJev release. Currently openjev-0.1.",
            releaseDate: "2026-09-18"),
        ModelInfo(
            name: diffusionGemmaVersion,
            description:
                "OpenJev 0.1: DiffusionGemma 26B-A4B (NVFP4) on vLLM's structured reads.",
            releaseDate: "2026-09-18"),
        ModelInfo(
            name: "diffusiongemma-26b",
            description:
                "DiffusionGemma 26B-A4B (NVFP4) text generation at POST /v1/chat/completions.",
            releaseDate: "2026-09-18"),
    ]

    /// What the DiffusionGemma backends serve: version `openjev-0.1`, the names `openjev-latest`,
    /// `openjev-0.1` and the SDK aliases, and ``diffusionGemmaListing``.
    public static let diffusionGemma = ServedModels(
        version: diffusionGemmaVersion,
        acceptedNames: Set(["openjev-latest", diffusionGemmaVersion]).union(sdkAliases),
        listing: diffusionGemmaListing)

    /// What an encoder backend serves: its own name as the version, its name and the SDK
    /// aliases as the accepted names, and a listing of itself alone.
    public static func encoder(_ modelInfo: ModelInfo) -> ServedModels {
        ServedModels(
            version: modelInfo.name,
            acceptedNames: Set([modelInfo.name]).union(sdkAliases),
            listing: [modelInfo])
    }

    /// Whether a request may name `model`.
    public func accepts(_ model: String) -> Bool {
        acceptedNames.contains(model)
    }

    /// The `GET /v1/models` listing of a server that also forwards the models `routedNames`
    /// names to other servers (`OPENJEV_MODEL_ROUTES`), upstream's `models_list`: ``listing``,
    /// then each routed name this server does not accept, in the order given. A name an encoder
    /// backend serves gets ``KnownEncoderModels``' entry, upstream's `known`; any other gets an
    /// empty description and release date. The routed servers are never asked, so the listing is
    /// the same while one is down.
    public func listing(routedNames: some Sequence<String>) -> [ModelInfo] {
        listing
            + routedNames.filter { !accepts($0) }.map { name in
                KnownEncoderModels.named(name)
                    ?? ModelInfo(name: name, description: "", releaseDate: "")
            }
    }
}

/// Upstream's `ENCODER_MODELS`: the `/v1/models` entry of each encoder backend. The models are
/// other people's work; the descriptions credit them wherever the model list is shown, so the
/// texts are upstream's word for word.
public enum KnownEncoderModels {
    /// Laya by Nandakishor M / Convai Innovations, upstream's `laya` backend.
    public static let laya = ModelInfo(
        name: "laya-1.0",
        description:
            "Laya by Nandakishor M / Convai Innovations (github.com/NandhaKishorM/laya, "
            + "Apache-2.0): the laya-typed-decisions checkpoint, a ModernBERT-large encoder (421M) "
            + "fine-tuned on the typed-decisions workflows. Text only, 1,024 tokens.",
        releaseDate: "2026-09-22")

    /// Verdict by Heman10x, upstream's `verdict` backend.
    public static let verdict = ModelInfo(
        name: "verdict-1.4",
        description:
            "Verdict by Heman10x (github.com/Heman10x-NGU/Verdict-open-jev, Apache-2.0): "
            + "rlcd-modernbert-151m, a ModernBERT-base + GLiClass encoder (151M) calibrated with "
            + "RLCD, with the v1.4 inference engine. Text only, 512 tokens, up to 24 choices.",
        releaseDate: "2026-09-22")

    /// CLM v0.1 by Contrastive-LM, upstream's `clm` backend.
    public static let clm = ModelInfo(
        name: "clm-v0.1",
        description:
            "CLM v0.1 by Contrastive-LM (github.com/Contrastive-LM/CLM, Apache-2.0): state and "
            + "action projection heads over Qwen3-8B last-token embeddings, here with Qwen3-8B-FP8 "
            + "on vLLM. Text only, 2,048 tokens.",
        releaseDate: "2026-09-24")

    /// JevK5 v0.2 by Alibi Serikbay, upstream's `jevk5` backend.
    public static let jevk5 = ModelInfo(
        name: "jevk5-0.2",
        description:
            "JevK5 v0.2 by Alibi Serikbay (github.com/allebee/jevk5, Apache-2.0): Qwen3.5-4B with "
            + "a LoRA distilled from Qwen3.6-27B, merged, read as a softmax over the answer "
            + "letters' logits (SemIf's readout), on vLLM. Text only, 16,384 tokens.",
        releaseDate: "2026-09-25")

    /// Every entry, in upstream's order: laya, verdict, clm, jevk5.
    public static let all = [laya, verdict, clm, jevk5]

    /// The entry served under `name`, or `nil` for a name no encoder backend serves.
    public static func named(_ name: String) -> ModelInfo? {
        all.first { $0.name == name }
    }
}

extension DecisionEngine: SystemOneService {
    /// ``ServedModels/diffusionGemma``.
    public nonisolated var servedModels: ServedModels { .diffusionGemma }

    /// The backend, when it also generates text: a DiffusionGemma runtime that conforms to
    /// ``TextGenerator`` serves `POST /v1/chat/completions` beside its reads.
    public nonisolated var textGenerator: (any TextGenerator)? {
        backend as? any TextGenerator
    }

    /// ``decide(_:seed:)`` with the seed derived from the request, as the route does.
    public func decide(_ request: SystemOneRequest) async throws -> Decision {
        try await decide(request, seed: nil)
    }
}

extension EncoderDecisionEngine: SystemOneService {
    /// ``ServedModels/encoder(_:)`` for the backend's ``QuestionReadBackend/modelInfo``.
    public nonisolated var servedModels: ServedModels { .encoder(backend.modelInfo) }
}
