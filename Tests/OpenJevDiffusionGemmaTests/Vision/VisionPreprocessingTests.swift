import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import OpenJevCore
import OpenJevDiffusionGemma
import Testing

/// The Swift preprocessing against what upstream's processor recorded in
/// Fixtures/vision/preprocessing.json (issue #46, docs/spikes/vision-preprocessing.md).
///
/// Everything here needs only the committed synthetic images and runs in CI.
@Suite("Vision preprocessing parity on the synthetic images")
struct VisionPreprocessingTests {
    /// The committed images: every recorded image but upstream's hot dog.
    static func syntheticImages() throws -> [VisionFixtures.Image] {
        try VisionFixtures.preprocessing().images.filter { $0.name != "hotdog" }
    }

    /// Decodes and processes a recorded image with the given resampling.
    static func processed(
        _ image: VisionFixtures.Image, filter: PillowResample.Filter = .bicubic,
        widen: Bool = true
    ) throws -> Gemma4ImageProcessor.ProcessedImage {
        try Gemma4ImageProcessor().process(
            RGBImage(decoding: image.data()), filter: filter, widen: widen)
    }

    @Test(
        "The fixture holds eleven synthetic images, each under 5 KB, and the processor's settings")
    func fixtureShape() throws {
        let images = try Self.syntheticImages()
        #expect(
            images.map(\.name).sorted() == [
                "baseline", "frames", "gradients", "gray", "interlaced", "local", "offset",
                "pattern", "progressive", "small", "transparent",
            ])
        for image in images {
            #expect(image.bytes < 5000, "\(image.file) is \(image.bytes) bytes")
        }
        let processor = try VisionFixtures.preprocessing().processor
        let swift = Gemma4ImageProcessor()
        #expect(processor["max_soft_tokens"]?.intValue == swift.maxSoftTokens)
        #expect(processor["patch_size"]?.intValue == swift.patchSize)
        #expect(processor["pooling_kernel_size"]?.intValue == swift.poolingKernelSize)
        #expect(processor["rescale_factor"]?.doubleValue == swift.rescaleFactor)
        #expect(processor["do_normalize"]?.boolValue == false)
        #expect(processor["resample"]?.intValue == 3)
        #expect(processor["image_token_id"]?.intValue == 258_880)
        #expect(processor["boi_token_id"]?.intValue == 255_999)
        #expect(processor["eoi_token_id"]?.intValue == 258_882)
    }

    @Test("Image processor sizing parameters must be positive")
    func invalidSizingConfiguration() {
        for key in ["max_soft_tokens", "patch_size", "pooling_kernel_size"] {
            for value in ["0", "-1"] {
                let config = Data("{\"image_processor\":{\"\(key)\":\(value)}}".utf8)
                #expect(throws: VisionError.self) {
                    try Gemma4ImageProcessor(configuration: config)
                }
            }
        }
    }

    @Test("Every synthetic image decodes to the RGB bytes PIL gave")
    func decoding() throws {
        for image in try Self.syntheticImages() {
            let data = try image.data()
            let rgb = try RGBImage(decoding: data)
            #expect(
                rgb.width == image.decodedWidth && rgb.height == image.decodedHeight,
                "\(image.name)")
            #expect(
                VisionFixtures.sha256(rgb.pixels) == image.decodedSHA256,
                "\(image.name): the decoded RGB differs from PIL's")
            #expect(RGBImage.frameCount(of: data) == image.frames, "\(image.name)")
        }
    }

    @Test("The JPEG port decodes as Pillow does; how far ImageIO's decode is is recorded")
    func imageIOJPEG() throws {
        for image in try Self.syntheticImages() where image.contentType == "image/jpeg" {
            let data = try image.data()
            let port = try RGBImage(decoding: data)
            let imageIO = try RGBImage(imageIO: data)
            let largest = zip(port.pixels, imageIO.pixels).reduce(0) {
                max($0, abs(Int($1.0) - Int($1.1)))
            }
            let differing = zip(port.pixels, imageIO.pixels).filter { $0 != $1 }.count
            let line =
                "\(image.name): ImageIO is up to \(largest) levels from Pillow at "
                + "\(differing) of \(port.pixels.count) samples"
            print("vision JPEG decode: \(line)")
            SpikeReport.record("vision-jpeg-decode", line)
            // ImageIO's figure is recorded, not bounded: it is Apple's to change.
            #expect(VisionFixtures.sha256(port.pixels) == image.decodedSHA256, "\(line)")
        }
    }

    @Test("The GIF port decodes as Pillow does; how far ImageIO's first frame is is recorded")
    func imageIOGIF() throws {
        for image in try Self.syntheticImages() where image.contentType == "image/gif" {
            let data = try image.data()
            let port = try RGBImage(decoding: data)
            let imageIO = try RGBImage(imageIO: data)
            var line =
                "\(image.name): ImageIO's first frame is \(imageIO.width) by \(imageIO.height)"
            if imageIO.width == port.width && imageIO.height == port.height {
                let largest = zip(port.pixels, imageIO.pixels).reduce(0) {
                    max($0, abs(Int($1.0) - Int($1.1)))
                }
                let differing = zip(port.pixels, imageIO.pixels).filter { $0 != $1 }.count
                line +=
                    ", up to \(largest) levels from Pillow at \(differing) of \(port.pixels.count) samples"
            }
            print("vision GIF decode: \(line)")
            SpikeReport.record("vision-gif-decode", line)
            // ImageIO's figure is recorded, not bounded: it is Apple's to change.
            #expect(VisionFixtures.sha256(port.pixels) == image.decodedSHA256, "\(line)")
        }
    }

    @Test("Every small GIF case decodes as upstream's PIL did, or is refused where PIL raised")
    func gifCases() throws {
        let cases = try VisionFixtures.preprocessing().gifCases
        #expect(cases.count >= 22)
        #expect(cases.contains { $0.decoded == nil } && cases.contains { $0.decoded != nil })
        for gif in cases {
            #expect(VisionFixtures.sha256(gif.data) == gif.sha256, "\(gif.name)")
            if let decoded = gif.decoded {
                let rgb = try RGBImage(decoding: gif.data)
                #expect(rgb.width == decoded.width && rgb.height == decoded.height, "\(gif.name)")
                #expect(
                    VisionFixtures.sha256(rgb.pixels) == decoded.sha256,
                    "\(gif.name): the decoded RGB differs from PIL's")
            } else {
                #expect(throws: VisionError.self, "\(gif.name): PIL raised \(gif.error ?? "")") {
                    try RGBImage(decoding: gif.data)
                }
            }
        }
    }

    @Test(
        "Every synthetic image resizes and rescales to the oracle's shape, statistics and samples")
    func pixelValues() throws {
        for image in try Self.syntheticImages() {
            let processed = try Self.processed(image)
            #expect(processed.resized.width == image.resizedWidth, "\(image.name)")
            #expect(processed.resized.height == image.resizedHeight, "\(image.name)")
            #expect(processed.softTokens == image.softTokens, "\(image.name)")
            let comparison = VisionFixtures.compare(
                processed.values, width: processed.resized.width,
                height: processed.resized.height, with: image)
            print("vision parity: \(comparison)")
            SpikeReport.record("vision-parity", "\(comparison)")
            #expect(comparison.withinBound, "\(comparison)")
        }
    }

    @Test("The GIF's first frame is the one used, and its second frame breaks the bounds")
    func gifFirstFrame() throws {
        let image = try #require(Self.syntheticImages().first { $0.name == "frames" })
        let data = try image.data()
        #expect(RGBImage.frameCount(of: data) == 2)
        let first = try RGBImage(decoding: data)
        let second = try RGBImage(decoding: data, frame: 1)
        #expect(first.pixels != second.pixels)
        #expect(VisionFixtures.sha256(first.pixels) == image.decodedSHA256)
        let processor = Gemma4ImageProcessor()
        let secondValues = try processor.process(second)
        let comparison = VisionFixtures.compare(
            secondValues.values, width: secondValues.resized.width,
            height: secondValues.resized.height, with: image)
        #expect(!comparison.withinBound, "\(comparison)")
    }

    @Test("The resize rule gives the oracle's size and soft tokens for every size in the table")
    func budgetRule() throws {
        let processor = Gemma4ImageProcessor()
        var rows = 0
        for row in try VisionFixtures.preprocessing().budget {
            rows += 1
            let size = "\(row.width) by \(row.height)"
            if let target = row.target, row.height != 3 {
                let swift = try processor.targetSize(width: row.width, height: row.height)
                #expect(swift.width == target.width && swift.height == target.height, "\(size)")
                #expect(
                    processor.softTokens(width: swift.width, height: swift.height)
                        == target.softTokens, "\(size)")
            } else {
                // Upstream raised (1 pixel high) or misread the image as channels first
                // (3 pixels high); the port refuses both.
                #expect(throws: VisionError.self, "\(size)") {
                    try processor.targetSize(width: row.width, height: row.height)
                }
            }
        }
        #expect(rows >= 34)
    }

    @Test("Swapping the bicubic filter for bilinear breaks the bounds on every image")
    func plantedBilinear() throws {
        for image in try Self.syntheticImages() {
            let processed = try Self.processed(image, filter: .bilinear)
            let comparison = VisionFixtures.compare(
                processed.values, width: processed.resized.width,
                height: processed.resized.height, with: image)
            #expect(!comparison.withinBound, "\(comparison)")
        }
    }

    @Test("Not widening the kernel when shrinking breaks the bounds on the images that shrink")
    func plantedNoWidening() throws {
        for image in try Self.syntheticImages()
        where image.resizedWidth < image.decodedWidth || image.resizedHeight < image.decodedHeight {
            let processed = try Self.processed(image, widen: false)
            let comparison = VisionFixtures.compare(
                processed.values, width: processed.resized.width,
                height: processed.resized.height, with: image)
            print("vision planted no-widening: \(comparison)")
            #expect(!comparison.withinBound, "\(comparison)")
        }
    }

    /// A PNG of `width` by `height` grey pixels with no pixel data: a header ImageIO reads.
    static func headerOnlyPNG(width: Int, height: Int) -> Data {
        func crc32(_ bytes: [UInt8]) -> UInt32 {
            var crc: UInt32 = 0xFFFF_FFFF
            for byte in bytes {
                crc ^= UInt32(byte)
                for _ in 0..<8 { crc = crc & 1 == 1 ? (crc >> 1) ^ 0xEDB8_8320 : crc >> 1 }
            }
            return ~crc
        }
        func be32(_ value: Int) -> [UInt8] { (0..<4).map { UInt8((value >> (24 - 8 * $0)) & 255) } }
        func chunk(_ type: String, _ body: [UInt8]) -> [UInt8] {
            let typed = Array(type.utf8) + body
            return be32(body.count) + typed + be32(Int(crc32(typed)))
        }
        let header = be32(width) + be32(height) + [8, 0, 0, 0, 0]
        let bytes =
            [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] + chunk("IHDR", header)
            + chunk("IEND", [])
        return Data(bytes)
    }

    @Test("An image past Pillow's decompression bomb limit is refused before it is decoded")
    func decompressionBomb() throws {
        // 13,400 by 13,400 is 179,560,000 pixels, past the limit of 178,956,970.
        #expect(throws: VisionError.self) {
            try RGBImage(decoding: Self.headerOnlyPNG(width: 13_400, height: 13_400))
        }
        #expect(throws: VisionError.self) {
            try RGBImage(imageIO: Self.headerOnlyPNG(width: 13_400, height: 13_400))
        }
        #expect(RGBImage.maxPixels == 2 * 89_478_485)
    }

    @Test("A state that spells out more image placeholders than there are images is refused")
    func extraPlaceholder() throws {
        #expect(throws: VisionError.self) {
            try ImagePromptInputs.expanded("<|image|> and <|image|>", softTokens: [3])
        }
        #expect(
            try ImagePromptInputs.expanded("a<|image|>b", softTokens: [2])
                == "a<|image><|image|><|image|><image|>b")
    }

}

/// MLXVLM's `Gemma4Processor` with the pinned checkpoint's image settings. Its tokenizer is
/// never used by `preprocess(image:processing:)`.
enum CoreImageProcessor {
    static func make() throws -> Gemma4Processor {
        let json = """
            {"processor_class": "DiffusionGemma4Processor", "image_processor": {
              "do_normalize": false, "image_mean": [0, 0, 0], "image_std": [1, 1, 1],
              "image_seq_length": 280, "max_soft_tokens": 280, "patch_size": 16,
              "pooling_kernel_size": 3}}
            """
        let config = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: Data(json.utf8))
        return Gemma4Processor(config, tokenizer: UnusedTokenizer())
    }

    struct UnusedTokenizer: MLXLMCommon.Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }
}

/// The hot dog photo, the prompts and the full tensors, which need the pinned upstream checkout,
/// the tokenizer files or the oracle's run.
@Suite(
    "Vision parity with upstream's files",
    .enabled(if: VisionFixtures.hotdogAvailable, VisionFixtures.hotdogMessage))
struct VisionUpstreamParityTests {
    static func image(_ name: String) throws -> VisionFixtures.Image {
        try #require(VisionFixtures.preprocessing().images.first { $0.name == name })
    }

    @Test("The hot dog JPEG decodes, resizes and rescales within the bounds")
    func hotdog() throws {
        let image = try Self.image("hotdog")
        let data = try image.data()
        let rgb = try RGBImage(decoding: data)
        #expect(rgb.width == 384 && rgb.height == 188)
        let decodedExactly = VisionFixtures.sha256(rgb.pixels) == image.decodedSHA256
        print("vision hot dog: the decode equals PIL's: \(decodedExactly)")
        let processed = try Gemma4ImageProcessor().process(rgb)
        #expect(processed.softTokens == 253)
        let comparison = VisionFixtures.compare(
            processed.values, width: processed.resized.width, height: processed.resized.height,
            with: image)
        print("vision parity: \(comparison)")
        SpikeReport.record("vision-parity", "\(comparison)")
        #expect(comparison.withinBound, "\(comparison)")
    }

}

/// The expanded prompt ids and `mm_token_type_ids`, which need the tokenizer files.
@Suite(
    "Vision prompt parity with upstream's processor",
    .enabled(if: TokenizerFixtures.available, TokenizerFixtures.missingMessage),
    .enabled(if: VisionFixtures.hotdogAvailable, VisionFixtures.hotdogMessage))
struct VisionPromptParityTests {
    @Test("An image prompt's system turn ends in the space a text prompt's lacks")
    func systemSpace() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let fixture = try VisionFixtures.preprocessing()
        let prompt = try #require(fixture.prompts.first { $0.key == "hotdog" })
        let text = try tokenizer.chatPromptIDs(
            system: prompt.system, user: prompt.state, thinking: false)
        #expect(text == fixture.textPromptIDs)
        // <turn|> (106) closes the system turn; the image prompt has token 236743 before it.
        let textClose = try #require(text.firstIndex(of: 106))
        let imageClose = try #require(prompt.ids.firstIndex(of: 106))
        #expect(imageClose == textClose + 1)
        #expect(prompt.ids[imageClose - 1] == 236_743)
        #expect(Array(prompt.ids[..<textClose]) == Array(text[..<textClose]))
    }
}

extension MLXTests {
    @Suite("Vision preprocessing MLX checks")
    struct VisionPreprocessingMLXTests {
        @Test("pixel_values stack to (n, 3, H, W) only when the images share a size")
        func stacking() throws {
            MetalLibrary.configure()
            let images = try VisionPreprocessingTests.syntheticImages()
            let gray = try VisionPreprocessingTests.processed(
                #require(images.first { $0.name == "gray" }))
            let small = try VisionPreprocessingTests.processed(
                #require(images.first { $0.name == "small" }))
            let gradients = try VisionPreprocessingTests.processed(
                #require(images.first { $0.name == "gradients" }))
            let same = Gemma4ImageProcessor.pixelValues([gray, small])
            #expect(same.count == 1)
            #expect(same[0].shape == [2, 3, 624, 1008])
            let mixed = Gemma4ImageProcessor.pixelValues([gray, gradients])
            #expect(mixed.map(\.shape) == [[3, 624, 1008], [3, 576, 1008]])
        }

        @Test("MLXVLM's Gemma4Processor, on Core Image's bicubic, misses the bounds")
        func coreImage() throws {
            MetalLibrary.configure()
            let processor = try CoreImageProcessor.make()
            var largest: Float = 0
            for image in try VisionPreprocessingTests.syntheticImages() {
                let ciImage = try #require(CIImage(data: image.data()))
                let (pixels, frame) = try processor.preprocess(image: ciImage, processing: nil)
                let values = pixels.asType(.float32).reshaped(-1).asArray(Float.self)
                let comparison = VisionFixtures.compare(
                    values, width: frame.w, height: frame.h, with: image)
                print("vision Core Image: \(comparison)")
                SpikeReport.record("vision-core-image", "\(comparison)")
                if comparison.shapeMatches {
                    largest = max(largest, comparison.maxSampleDifference)
                }
            }
            #expect(largest > VisionFixtures.bound)
        }
    }

    @Suite(
        "Vision parity with upstream's MLX files",
        .enabled(if: VisionFixtures.hotdogAvailable, VisionFixtures.hotdogMessage))
    struct VisionUpstreamMLXParityTests {
        @Test(
            "Every value of every image against the oracle's full tensors",
            .enabled(if: VisionFixtures.tensorsAvailable, VisionFixtures.tensorsMessage))
        func fullTensors() throws {
            MetalLibrary.configure()
            let ciProcessor = try CoreImageProcessor.make()
            for image in try VisionFixtures.preprocessing().images {
                let data = try image.data()
                let rgb = try RGBImage(decoding: data)
                let oracleRGB = try VisionFixtures.fullRGB(image.name)
                let decodeDiff = zip(rgb.pixels, oracleRGB).reduce(0) {
                    max($0, abs(Int($1.0) - Int($1.1)))
                }
                let decodeDiffering = zip(rgb.pixels, oracleRGB).filter { $0 != $1 }.count
                let processed = try Gemma4ImageProcessor().process(rgb)
                let oracle = try VisionFixtures.fullTensor(image.name)
                let pillow = VisionFixtures.difference(processed.values, oracle)
                var line =
                    "\(image.name): decode max \(decodeDiff) levels at \(decodeDiffering) of "
                    + "\(oracleRGB.count) bytes; Pillow port max \(pillow.max) at \(pillow.differing) "
                    + "of \(oracle.count) values"
                if let ciImage = CIImage(data: data) {
                    let (pixels, frame) = try ciProcessor.preprocess(
                        image: ciImage, processing: nil)
                    let values = pixels.asType(.float32).reshaped(-1).asArray(Float.self)
                    if frame.w == processed.resized.width && frame.h == processed.resized.height {
                        let ci = VisionFixtures.difference(values, oracle)
                        line += "; Core Image max \(ci.max) at \(ci.differing) values"
                    } else {
                        line += "; Core Image size \(frame.w) by \(frame.h) differs"
                    }
                }
                print("vision full tensors: \(line)")
                SpikeReport.record("vision-full-tensors", line)
                #expect(pillow.max <= VisionFixtures.bound, "\(line)")
            }
        }

        @Test(
            "Every fixture prompt expands to the oracle's ids and mm_token_type_ids",
            .enabled(if: TokenizerFixtures.available, TokenizerFixtures.missingMessage))
        func prompts() async throws {
            let tokenizer = try await TokenizerFixtures.tokenizer()
            let fixture = try VisionFixtures.preprocessing()
            #expect(fixture.prompts.count == 13)
            for prompt in fixture.prompts {
                let parts = try prompt.images.map { name in
                    let image = try #require(fixture.images.first { $0.name == name })
                    return ImagePart(
                        contentType: image.contentType,
                        base64: try image.data().base64EncodedString())
                }
                let inputs = try ImageReadInputs(
                    system: prompt.system, state: prompt.state, parts: parts, tokenizer: tokenizer)
                #expect(inputs.prompt.softTokens == prompt.softTokens, "\(prompt.key)")
                #expect(
                    inputs.prompt.ids == prompt.ids,
                    "\(prompt.key): \(ChatTemplateParityTests.firstDifference(expected: prompt.ids, actual: inputs.prompt.ids))"
                )
                #expect(inputs.prompt.mmTokenTypeIDs == prompt.mmTokenTypeIDs, "\(prompt.key)")
                #expect(inputs.pixelValues.map(\.shape) == prompt.shapes, "\(prompt.key)")
            }
        }
    }
}
