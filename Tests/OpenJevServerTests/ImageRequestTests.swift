#if canImport(HummingbirdTesting)
    import Foundation
    import HummingbirdTesting
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// Upstream's test_api.py `PNG`, a 1 by 1 PNG.
    private let png =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    /// Upstream's test_api.py `EXAMPLE`, Jev's quickstart request, with `images` when given.
    private func example(images: JSONValue? = nil) throws -> JSONValue {
        let text = #"""
            {"state": "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
             "model": "jev-latest",
             "questions": {
               "department": {"type": "choice", "instructions": "Which team should handle this",
                 "criteria": {"billing": "Payment or subscription issues",
                              "technical": "Bugs or integration problems",
                              "sales": "Pricing or account questions"}},
               "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                 "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]},
               "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}}}
            """#
        guard let images, case .object(var object) = try JSONParser().parse(text) else {
            return try JSONParser().parse(text)
        }
        object["images"] = images
        return .object(object)
    }

    /// The `detail` of an error body.
    private func detail(_ response: some CheckedResponse) throws -> String {
        try #require(try JSONParser().parse(ServerHarness.text(response))["detail"]?.stringValue)
    }

    /// The `images` field over HTTP: upstream's test_api.py image tests against the DiffusionGemma
    /// engine over ``StubBackend``, and how the server answers images it or a backend refuses.
    @Suite("Images over HTTP")
    struct ImageRequestTests {
        @Test("test_images_go_ahead_of_the_state")
        func imagesGoAheadOfTheState() async throws {
            let backend = StubBackend()
            let service = try await ServerHarness.diffusionService(
                ServerSettings(), backend: backend)
            let body = try example(
                images: .array([
                    .string("data:image/png;base64,\(png)"),
                    .object(["content_type": .string("image/jpeg"), "base64": .string(png)]),
                ]))
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(client, body)
                #expect(response.status == .ok, "\(ServerHarness.text(response))")
            }
            let read = try #require(backend.reads.first)
            guard case .image(_, let state, let images) = read.prompt else {
                Issue.record("not an image prompt: \(read.prompt)")
                return
            }
            #expect(images.count == 2)
            #expect(images[0].dataURL.hasPrefix("data:image/png;base64,"))
            #expect(images[1].dataURL.hasPrefix("data:image/jpeg;base64,"))
            #expect(
                state
                    == "Hi, I've been trying to connect my Stripe account but keep getting a 403 error."
            )
        }

        @Test("test_image_validation")
        func imageValidation() async throws {
            let service = try await ServerHarness.diffusionService(ServerSettings())
            let cases: [(JSONValue, String)] = [
                (.array([.string("https://example.com/a.png")]), "data:image"),
                (
                    .array([
                        .object(["content_type": .string("image/bmp"), "base64": .string(png)])
                    ]),
                    "not supported"
                ),
                (
                    .array([
                        .object([
                            "content_type": .string("image/png"), "base64": .string("not base64!"),
                        ])
                    ]), "base64"
                ),
                (
                    .array(Array(repeating: .string("data:image/png;base64,\(png)"), count: 9)),
                    "at most 8"
                ),
            ]
            try await ServerHarness.withClient(service: service) { client in
                for (images, needle) in cases {
                    let response = try await ServerHarness.post(client, try example(images: images))
                    #expect(response.status == .badRequest, "\(needle)")
                    #expect(try detail(response).contains(needle), "\(needle)")
                }
                // URLs were never accepted; the error must not claim they are.
                let response = try await ServerHarness.post(
                    client, try example(images: .array([.string("https://example.com/a.png")])))
                #expect(try !detail(response).contains("URL"))
            }
        }

        /// The size bound is checked on the base64 text's length before decoding, so a body far
        /// past the limit costs no decoding.
        @Test("test_oversize_image_is_refused_before_decoding")
        func oversizeImageIsRefusedBeforeDecoding() async throws {
            let backend = StubBackend()
            let service = try await ServerHarness.diffusionService(
                ServerSettings(), backend: backend)
            // 8 MB of base64, decoding to 6 MB: over the 5 MB default.
            let big = String(repeating: "QUFB", count: 2 * 1024 * 1024)
            try await ServerHarness.withClient(service: service) { client in
                for image in [
                    JSONValue.object(["content_type": .string("image/png"), "base64": .string(big)]
                    ),
                    .string("data:image/png;base64,\(big)"),
                ] {
                    let response = try await ServerHarness.post(
                        client, try example(images: .array([image])))
                    #expect(response.status == .badRequest)
                    #expect(try detail(response).contains("larger than"))
                }
            }
            #expect(backend.reads.isEmpty)
        }

        /// `OPENJEV_MAX_IMAGES` and `OPENJEV_MAX_IMAGE_BYTES` reach the engine's ``ImageLimits``.
        @Test("The two image limits reach the engine")
        func imageLimitsReachTheEngine() async throws {
            let settings = try ServerSettings(maxImages: 1, maxImageBytes: 40)
            let service = try await ServerHarness.diffusionService(settings)
            let image = JSONValue.string("data:image/png;base64,\(png)")
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                let two = try await ServerHarness.post(
                    client, try example(images: .array([image, image])))
                #expect(two.status == .badRequest)
                #expect(try detail(two) == "at most 1 images per request")
                // The PNG's base64 bounds it at 67 bytes, past a 40-byte limit.
                let large = try await ServerHarness.post(
                    client, try example(images: .array([image])))
                #expect(large.status == .badRequest)
                #expect(try detail(large) == "image data is larger than the 40 byte limit")
            }
        }

        /// The DiffusionGemma runtime answers an image it cannot read with a ``SchemaError`` at
        /// `["body", "images", i]` (D-054); the server sends it as the plain-detail 400, not a 5xx.
        @Test("An image the backend cannot read is a 400 with its reason")
        func undecodableImage() async throws {
            let reason =
                "image could not be read: the image data is not a JPEG, PNG, WebP or GIF image"
            let backend = StubBackend(
                failure: SchemaError(reason, loc: ["body", "images", .index(0)]))
            let service = try await ServerHarness.diffusionService(
                ServerSettings(), backend: backend)
            try await ServerHarness.withClient(service: service) { client in
                let response = try await ServerHarness.post(
                    client, try example(images: .array([.string("data:image/jpeg;base64,\(png)")])))
                #expect(response.status == .badRequest)
                #expect(try detail(response) == reason)
            }
        }

        /// Verdict, Laya and JevK5 answer through the encoder engine, which refuses images with
        /// upstream's `"{model} does not support images"`.
        @Test("A backend without vision refuses images with upstream's message")
        func encoderRefusesImages() async throws {
            let service = try await ServerHarness.encoderService()
            try await ServerHarness.withClient(service: service) { client in
                let body = try example(
                    images: .array([.string("data:image/png;base64,\(png)")]))
                let response = try await ServerHarness.post(client, body)
                #expect(response.status == .badRequest)
                let name = KnownEncoderModels.laya.name
                #expect(try detail(response) == "\(name) does not support images")
            }
        }
    }
#endif
