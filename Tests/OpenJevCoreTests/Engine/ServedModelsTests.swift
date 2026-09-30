import Foundation
import OpenJevCore
import OpenJevTestSupport
import Testing

/// Checks ``ServedModels`` and ``KnownEncoderModels`` against `Fixtures/wire/models.json`, the
/// `GET /v1/models` bodies upstream's app sends for every backend, with the version a response
/// names and the names a request may use.
@Suite(
    "Served models",
    .enabled(if: UpstreamFixtures.exists("wire/models.json"), UpstreamFixtures.missingMessage))
struct ServedModelsTests {
    /// The `ServedModels` this port builds for a recorded backend.
    static func served(backend: String) -> ServedModels? {
        switch backend {
        case "vllm", "mlx": return .diffusionGemma
        case "laya": return .encoder(KnownEncoderModels.laya)
        case "verdict": return .encoder(KnownEncoderModels.verdict)
        case "clm": return .encoder(KnownEncoderModels.clm)
        case "jevk5": return .encoder(KnownEncoderModels.jevk5)
        default: return nil
        }
    }

    @Test("Every recorded listing, version and accepted name set is reproduced")
    func listings() throws {
        let listings = try #require(
            UpstreamFixtures.load("wire/models.json")["listings"]?.arrayValue)
        var checked: Set<String> = []
        for listing in listings where listing["model_routes"] == nil {
            let backend = try #require(listing["backend"]?.stringValue)
            let served = try #require(Self.served(backend: backend), "unknown backend \(backend)")
            #expect(
                served.version == listing["model_version"]?.stringValue,
                Comment(rawValue: backend))
            let names = try #require(listing["accepted_names"]?.arrayValue).map {
                try #require($0.stringValue)
            }
            #expect(served.acceptedNames == Set(names), Comment(rawValue: backend))
            #expect(
                try WireEncoder().string(ModelsResponse(models: served.listing))
                    == listing["body_text"]?.stringValue, Comment(rawValue: backend))
            checked.insert(backend)
        }
        #expect(checked == ["vllm", "mlx", "laya", "verdict", "clm", "jevk5"])
    }

    @Test("The SDK aliases are accepted by every variant")
    func sdkAliases() {
        let variants =
            [ServedModels.diffusionGemma] + KnownEncoderModels.all.map(ServedModels.encoder)
        for served in variants {
            #expect(served.accepts("jev-latest"), Comment(rawValue: served.version))
            #expect(served.accepts("jev-preview"), Comment(rawValue: served.version))
            #expect(served.accepts(served.version), Comment(rawValue: served.version))
            #expect(!served.accepts("custom-model"), Comment(rawValue: served.version))
        }
        #expect(ServedModels.sdkAliases == ["jev-latest", "jev-preview"])
        // An encoder does not accept the DiffusionGemma names, upstream's "Unknown model".
        #expect(!ServedModels.encoder(KnownEncoderModels.laya).accepts("openjev-latest"))
        #expect(!ServedModels.diffusionGemma.accepts("laya-1.0"))
    }

    @Test("The known encoder models are found by name, in upstream's order")
    func knownEncoderModels() {
        #expect(
            KnownEncoderModels.all.map(\.name) == [
                "laya-1.0", "verdict-1.4", "clm-v0.1", "jevk5-0.2",
            ])
        #expect(KnownEncoderModels.named("verdict-1.4") == KnownEncoderModels.verdict)
        #expect(KnownEncoderModels.named("openjev-0.1") == nil)
        #expect(
            ServedModels.diffusionGemma.listing.map(\.name) == [
                "openjev-latest", "openjev-0.1", "diffusiongemma-26b",
            ])
    }
}
