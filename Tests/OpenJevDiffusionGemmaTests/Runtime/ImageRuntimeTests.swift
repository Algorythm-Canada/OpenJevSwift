import CoreGraphics
import Foundation
import ImageIO
import OpenJevCore
import OpenJevTestSupport
import Testing
import UniformTypeIdentifiers

@testable import OpenJevDiffusionGemma

/// Upstream's test_api.py `EXAMPLE`, Jev's quickstart request, with `extra` members (JSON text
/// without braces) added.
private func example(_ extra: String = "") throws -> SystemOneRequest {
    let tail = extra.isEmpty ? "" : ", \(extra)"
    return try SystemOneRequest(
        json: JSONParser().parse(
            #"""
            {"state": "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
             "model": "jev-latest",
             "questions": {
               "department": {"type": "choice", "instructions": "Which team should handle this",
                 "criteria": {"billing": "Payment or subscription issues",
                              "technical": "Bugs or integration problems",
                              "sales": "Pricing or account questions"}},
               "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                 "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]},
               "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}}
            """# + tail + "}"))
}

/// The JSON text of an `images` member holding `urls`.
private func images(_ urls: [String]) -> String {
    #""images": ["# + urls.map { "\"\($0)\"" }.joined(separator: ", ") + "]"
}

/// `width` by `height` pixels of one colour, encoded as `type` by ImageIO; upstream's live tests
/// draw their `solid_png` the same way.
private func encoded(
    _ type: UTType, width: Int = 64, height: Int = 64, rgb: (UInt8, UInt8, UInt8) = (255, 0, 0)
) throws -> Data {
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    for index in 0..<(width * height) {
        pixels[index * 4] = rgb.0
        pixels[index * 4 + 1] = rgb.1
        pixels[index * 4 + 2] = rgb.2
        pixels[index * 4 + 3] = 255
    }
    let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
    let context = try #require(
        CGContext(
            data: &pixels, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    let image = try #require(context.makeImage())
    let data = NSMutableData()
    let destination = try #require(
        CGImageDestinationCreateWithData(data, type.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil)
    try #require(CGImageDestinationFinalize(destination))
    return data as Data
}

/// A data URL of `data` declared as `type`.
private func url(_ data: Data, _ type: String = "image/png") -> String {
    "data:\(type);base64,\(data.base64EncodedString())"
}

/// A runtime over the stub model with the recorded tokenizer, and an engine over it.
private func engine(
    log: StubModelLog, configuration: DiffusionGemmaRuntime.Configuration = .default
) throws -> (DecisionEngine, DiffusionGemmaRuntime) {
    let runtime = DiffusionGemmaRuntime.stub(
        configuration: configuration, log: log, tokenizer: FixtureTokenizer.shared)
    return (try DecisionEngine(backend: runtime), runtime)
}

/// Images through ``DecisionEngine`` and ``DiffusionGemmaRuntime`` over a stub model, the cases
/// of upstream's tests/test_mlx_backend.py (lines 225 to 280) that run without weights. The stub
/// decodes and sizes each image for real and counts the prompt as upstream's stub does (the words
/// of the system and state texts plus 256 per image). Upstream's 1 by 1 PNG is an image the port
/// refuses (D-051), so the images here are 64 by 64 PNGs of one colour.
@Suite("Images through the engine and the DiffusionGemma runtime over a stub model")
struct ImageRuntimeTests {
    @Test("test_an_image_read_goes_to_the_runtime")
    func anImageReadGoesToTheRuntime() async throws {
        let log = StubModelLog()
        let (engine, _) = try engine(log: log)
        let red = url(try encoded(.png))
        let decision = try await engine.decide(example(images([red])))
        let prompt = try #require(log.imagePrompts.first)
        #expect(log.imagePrompts.count == 1)
        #expect(prompt.images.map(\.dataURL) == [red], "the runtime was given no images")
        #expect(
            prompt.stateText
                == "Hi, I've been trying to connect my Stripe account but keep getting a 403 error."
        )
        #expect(log.prefills.isEmpty, "an image read went through the text prefill")
        guard case .choice(let choice, _, _) = decision.answers["department"] else {
            Issue.record("no choice answer: \(decision.answers)")
            return
        }
        #expect(choice == "billing")
    }

    @Test("test_image_usage_counts_the_expanded_prompt")
    func imageUsageCountsTheExpandedPrompt() async throws {
        let log = StubModelLog()
        let (engine, _) = try engine(log: log)
        let decision = try await engine.decide(example(images([url(try encoded(.png))])))
        let prompt = try #require(log.imagePrompts.first)
        #expect(decision.inputTokens == prompt.promptTokens)
        #expect(decision.inputTokens >= stubImageTokens)
        #expect(log.reads.allSatisfy { $0.promptTokens == prompt.promptTokens })
    }

    @Test("test_images_do_not_share_a_prefill_entry")
    func imagesDoNotShareAPrefillEntry() async throws {
        let log = StubModelLog()
        let (engine, runtime) = try engine(log: log)
        let red = url(try encoded(.png, rgb: (255, 0, 0)))
        let blue = url(try encoded(.png, rgb: (0, 0, 255)))
        for extra in [images([red]), images([blue]), ""] {
            _ = try await engine.decide(example(extra))
        }
        let keys = await runtime.prefillCacheState.keys
        #expect(keys.count == 3 && Set(keys).count == 3, "\(keys)")
        // The image keys are upstream's ImagePrompt.key: the texts and each URL's SHA-256.
        let redKey = DiffusionGemmaRuntime.imageKey(
            systemText: log.imagePrompts[0].systemText, stateText: log.imagePrompts[0].stateText,
            images: [ImagePart(contentType: "image/png", base64: String(red.dropFirst(22)))])
        #expect(keys.first == redKey)
    }

    @Test("test_same_image_request_same_canvas: the second read reuses the prefill")
    func sameImageRequestSameCanvas() async throws {
        let log = StubModelLog()
        let (engine, runtime) = try engine(log: log)
        let request = try example(images([url(try encoded(.png))]))
        let first = try await engine.decide(request)
        let second = try await engine.decide(request)
        // The stub reads 0.75, so the engine re-reads each request (upstream's stub reads 0.99):
        // the requests' first reads are compared.
        let perRequest = log.reads.count / 2
        #expect(perRequest >= 1 && log.reads.count == 2 * perRequest)
        #expect(log.reads[0].canvas == log.reads[perRequest].canvas)
        // One decode and one prefill: every later read hits the cached prefill by its key.
        #expect(log.imagePrompts.count == 1)
        #expect(log.imagePrefills == 1)
        let statistics = await runtime.statistics()
        #expect(statistics.prefillMisses == 1)
        #expect(statistics.prefillHits == log.reads.count - 1)
        #expect(first.inputTokens == second.inputTokens)
    }

    /// Upstream answers both with `"{field} needs a text state; send images without it"`. The
    /// runtime has no `think` until milestone 5, so the engine refuses `think` first, with
    /// `"openjev-0.1 does not support think"` at the same location; both are 400s naming the
    /// field.
    @Test("test_images_with_think_or_sequential_are_still_refused")
    func imagesWithThinkOrSequentialAreStillRefused() async throws {
        let log = StubModelLog()
        let (engine, _) = try engine(log: log)
        let image = images([url(try encoded(.png))])
        for (field, value) in [("think", "64"), ("sequential", "true")] {
            let error = await #expect(throws: SchemaError.self) {
                try await engine.decide(example(image + #", "\#(field)": \#(value)"#))
            }
            #expect(error?.loc == ["body", .key(field)])
            #expect(error?.message.contains(field) == true, "\(error?.message ?? "")")
        }
        let error = await #expect(throws: SchemaError.self) {
            try await engine.decide(example(image + #", "sequential": true"#))
        }
        #expect(error?.message == "sequential needs a text state; send images without it")
        #expect(!log.touched)
    }

    @Test("The prompt cap applies to the expanded image prompt, with upstream's message")
    func capAfterExpansion() async throws {
        let log = StubModelLog()
        let (engine, _) = try engine(log: log, configuration: .init(maxPromptTokens: 200))
        let error = await #expect(throws: SchemaError.self) {
            try await engine.decide(example(images([url(try encoded(.png))])))
        }
        let tokens = try #require(log.imagePrompts.first).promptTokens
        #expect(tokens > 200)
        #expect(error?.message == "the request is \(tokens) tokens; the limit is 200")
        #expect(log.imagePrefills == 0 && log.reads.isEmpty)
    }

    /// The images upstream cannot read (D-054): each is a 400 at `["body", "images", i]` naming
    /// why, never a crash or a 5xx, and nothing reaches the model.
    @Test(
        "An image that does not decode is a 400 naming the image",
        arguments: [
            "HEIC bytes labelled image/jpeg", "TIFF bytes labelled image/png",
            "a JPEG missing only its EOI", "a JPEG cut in half", "an image 1 pixel high",
            "an image 3 pixels high", "a JPEG declaring 180,000,000 pixels",
            "a JPEG of garbage after its SOI",
        ])
    func undecodableImages(_ name: String) async throws {
        let baseline = try Data(
            contentsOf: VisionFixtures.directory.appendingPathComponent("baseline.jpg"))
        let bad: (data: Data, type: String, reason: String)
        switch name {
        case "HEIC bytes labelled image/jpeg":
            // Pillow cannot identify HEIC (UnidentifiedImageError); ImageIO could decode it.
            guard let heic = try? encoded(.heic) else {
                // No HEIC encoder on this machine: the TIFF case covers the same rule.
                return
            }
            bad = (heic, "image/jpeg", "not a JPEG, PNG, WebP or GIF")
        case "TIFF bytes labelled image/png":
            bad = (try encoded(.tiff), "image/png", "not a JPEG, PNG, WebP or GIF")
        case "a JPEG missing only its EOI":
            // Pillow decodes this one (libjpeg never reads past its end) but refuses the hot dog
            // and progressive.jpg without theirs; the port refuses every JPEG without EOI (D-054).
            bad = (baseline.dropLast(2), "image/jpeg", "truncated")
        case "a JPEG cut in half":
            bad = (baseline.prefix(baseline.count / 2), "image/jpeg", "truncated")
        case "an image 1 pixel high":
            bad = (try encoded(.png, width: 40, height: 1), "image/png", "high")
        case "an image 3 pixels high":
            bad = (try encoded(.png, width: 40, height: 3), "image/png", "high")
        case "a JPEG declaring 180,000,000 pixels":
            // SOI, a baseline frame header of 9,000 by 20,000 with one component, EOI.
            bad = (
                Data([
                    0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x23, 0x28, 0x4E, 0x20, 0x01,
                    0x01, 0x11, 0x00, 0xFF, 0xD9,
                ]), "image/jpeg", "pixels"
            )
        default:
            // Pillow cannot identify it; the port finds no EOI.
            bad = (
                Data([0xFF, 0xD8, 0xFF] + [UInt8](repeating: 0x13, count: 200)), "image/jpeg",
                "JPEG"
            )
        }
        let log = StubModelLog()
        let (engine, _) = try engine(log: log)
        // A good image first, so the bad one is image 1.
        let request = try example(images([url(try encoded(.png)), url(bad.data, bad.type)]))
        let error = await #expect(throws: SchemaError.self) { try await engine.decide(request) }
        #expect(error?.loc == ["body", "images", .index(1)], "\(name)")
        #expect(error?.message.hasPrefix("image could not be read: ") == true, "\(name)")
        #expect(error?.message.contains(bad.reason) == true, "\(name): \(error?.message ?? "")")
        #expect(log.imagePrefills == 0 && log.reads.isEmpty)
    }

    @Test("The chunked prefill policy keeps image prompts in one piece")
    func chunkedPrefillPolicy() {
        #expect(
            DiffusionGemmaModel.allowsChunkedPrefill(mmTokenTypeIDs: nil, hasPixelValues: false))
        #expect(
            DiffusionGemmaModel.allowsChunkedPrefill(
                mmTokenTypeIDs: [0, 0, 3], hasPixelValues: false))
        #expect(
            !DiffusionGemmaModel.allowsChunkedPrefill(
                mmTokenTypeIDs: [0, 1, 1], hasPixelValues: false))
        #expect(
            !DiffusionGemmaModel.allowsChunkedPrefill(
                mmTokenTypeIDs: [0, 2], hasPixelValues: false))
        #expect(
            !DiffusionGemmaModel.allowsChunkedPrefill(mmTokenTypeIDs: nil, hasPixelValues: true))
    }
}
