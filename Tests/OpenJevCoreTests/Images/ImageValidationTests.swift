import Foundation
import OpenJevCore
import Testing

/// The image checks of ``ImageValidation``, against upstream's recorded responses and rule by rule.
@Suite("Image validation")
struct ImageValidationTests {
    /// A 1 by 1 PNG, the smallest real image the tests send.
    private static let pngBase64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    /// The error `parts` throws for the images, or `nil` when it accepts them.
    private func failure(
        _ images: [ImageInput], limits: ImageLimits = ImageLimits()
    ) -> SchemaError? {
        do {
            _ = try ImageValidation.parts(images, limits: limits)
            return nil
        } catch {
            return error
        }
    }

    // MARK: Recorded responses

    /// The recorded image cases whose 400 comes from the image checks. `images_with_think` and
    /// `images_with_sequential` are the engine's 400s (#17) and are checked to pass here.
    private static let imageCheckCases: Set<String> = [
        "images_url", "images_bmp", "images_invalid_base64", "images_nine",
        "images_oversize_object", "images_oversize_data_url",
    ]

    @Test(
        "Every recorded image 400 is reproduced byte for byte",
        .enabled(if: WireFixtures.exists("cases.json"), WireFixtures.missingMessage))
    func recordedImageErrors() throws {
        let cases = try #require(WireFixtures.load("cases.json")["cases"]?.arrayValue)
        var seen: Set<String> = []
        for fixture in cases {
            let name = try #require(fixture["name"]?.stringValue)
            guard name.hasPrefix("images_"), fixture["response"]?["status"]?.intValue == 400 else {
                continue
            }
            let recorded = try #require(fixture["request"])
            let bytes = try #require(try WireFixtures.bodyBytes(of: recorded))
            let request = try RequestValidator().validate(JSONParser().parse(bytes))
            let images = try #require(request.images, "\(name)")
            let expected = try #require(fixture["response"]?["body_text"]?.stringValue)
            guard Self.imageCheckCases.contains(name) else {
                #expect(failure(images) == nil, "\(name) is the engine's 400, not an image check's")
                continue
            }
            seen.insert(name)
            let error = try #require(failure(images), "\(name) was accepted")
            let body = try WireEncoder().string(WireError.semantic400(error))
            #expect(body == expected, "\(name)")
            #expect(WireError.semantic400(error).status == 400)
            if name == "images_url" {
                #expect(!error.message.contains("URL"), "the message must not mention URLs")
            }
        }
        #expect(seen == Self.imageCheckCases)
    }

    @Test(
        "The recorded valid images are accepted",
        .enabled(if: WireFixtures.exists("cases.json"), WireFixtures.missingMessage))
    func recordedValidImages() throws {
        let cases = try #require(WireFixtures.load("cases.json")["cases"]?.arrayValue)
        let fixture = try #require(cases.first { $0["name"]?.stringValue == "images_valid" })
        let recorded = try #require(fixture["request"])
        let bytes = try #require(try WireFixtures.bodyBytes(of: recorded))
        let request = try RequestValidator().validate(JSONParser().parse(bytes))
        let parts = try ImageValidation.parts(#require(request.images))
        #expect(!parts.isEmpty)
    }

    // MARK: Rules

    @Test("Both forms give the same part and data URL")
    func normalisation() throws {
        let fromURL = try ImageValidation.parts([
            .dataURL("data:image/png;base64,\(Self.pngBase64)")
        ])
        let fromObject = try ImageValidation.parts([
            .object(contentType: "image/png", base64: Self.pngBase64)
        ])
        #expect(fromURL == fromObject)
        #expect(fromURL.first?.dataURL == "data:image/png;base64,\(Self.pngBase64)")
        #expect(fromURL.first?.contentType == "image/png")
        #expect(fromURL.first?.base64 == Self.pngBase64)
    }

    @Test("Every supported type is accepted and parts keep their order")
    func supportedTypes() throws {
        let images = ImageValidation.supportedTypes.map {
            ImageInput.object(contentType: $0, base64: "QUFB")
        }
        let parts = try ImageValidation.parts(images)
        #expect(parts.map(\.contentType) == ["image/jpeg", "image/png", "image/webp", "image/gif"])
        #expect(try ImageValidation.parts([]).isEmpty)
    }

    @Test("More than the limit is refused before any image is looked at")
    func tooMany() {
        let images = Array(repeating: ImageInput.dataURL("not an image"), count: 9)
        let expected = SchemaError("at most 8 images per request", loc: ["body", "images"])
        #expect(failure(images) == expected)
        let empty = ImageInput.object(contentType: "image/gif", base64: "")
        let eight = Array(repeating: empty, count: 8)
        #expect(failure(eight) == nil)
        #expect(
            failure(images, limits: ImageLimits(maxImages: 2))?.message
                == "at most 2 images per request")
    }

    @Test(
        "A string that is not a base64 data URL is refused",
        arguments: [
            "https://example.com/cat.png", "data:image/png;base64", "image/png;base64,QUFB",
            "data:image/png,QUFB", "data:image/png;base64 ,QUFB", "Data:image/png;base64,QUFB", "",
        ])
    func notDataURL(url: String) {
        let message =
            "an image is a data:image/...;base64 string or a {content_type, base64} object"
        let images: [ImageInput] = [
            .object(contentType: "image/png", base64: "QUFB"), .dataURL(url),
        ]
        #expect(failure(images) == SchemaError(message, loc: ["body", "images", .index(1)]))
    }

    @Test("The data URL splits at the first comma")
    func firstComma() {
        // The data after the first comma holds a comma, which is not base64.
        #expect(
            failure([.dataURL("data:image/png;base64,QU,B")])?.message
                == "image data is not valid base64")
        // A comma inside the head ends the head early, so it no longer ends in ";base64".
        #expect(
            failure([.dataURL("data:image/png,x;base64,QUFB")])?.message
                == "an image is a data:image/...;base64 string or a {content_type, base64} object")
    }

    @Test(
        "An unsupported type is refused with Python's repr of it",
        arguments: [
            ("image/bmp", "'image/bmp'"),
            ("", "''"),
            ("IMAGE/PNG", "'IMAGE/PNG'"),
            ("image/png ", "'image/png '"),
            ("it's", "\"it's\""),
            ("a\"b", "'a\"b'"),
            ("a'b\"c", #"'a\'b"c'"#),
            ("x\\y", #"'x\\y'"#),
            ("\t\n\r\u{0}\u{1F}\u{7F}", #"'\t\n\r\x00\x1f\x7f'"#),
            ("\u{A0}\u{AD}", #"'\xa0\xad'"#),
            ("\u{200B}\u{2028}", #"'\u200b\u2028'"#),
            ("\u{E9}\u{301}", "'\u{E9}\u{301}'"),
            ("\u{1F600}", "'\u{1F600}'"),
            ("\u{E000}", #"'\ue000'"#),
            ("\u{E0001}", #"'\U000e0001'"#),
            ("\u{378}", #"'\u0378'"#),
            ("\u{10FFFF}", #"'\U0010ffff'"#),
        ])
    func unsupportedType(type: String, repr: String) {
        let expected = SchemaError(
            "image type \(repr) is not supported; use JPEG, PNG, WebP or GIF",
            loc: ["body", "images", .index(0)])
        #expect(failure([.object(contentType: type, base64: "QUFB")]) == expected)
    }

    @Test("The data URL head is compared by Unicode scalars, as Python compares it")
    func headComparedByScalars() {
        // A combining mark after "data:" joins the colon into one Swift Character, so a
        // Character-based prefix test would refuse the head. Python sees the prefix and reads the
        // type as "\u{301}image/png", which it refuses as a type, and so does this port.
        #expect(
            failure([.dataURL("data:\u{301}image/png;base64,QUFB")])?.message
                == "image type '\u{301}image/png' is not supported; use JPEG, PNG, WebP or GIF")
        #expect(
            failure([.dataURL("data:image/png\u{301};base64,QUFB")])?.message
                == "image type 'image/png\u{301}' is not supported; use JPEG, PNG, WebP or GIF")
        // A combining mark after the comma belongs to the data, which is then not base64.
        #expect(
            failure([.dataURL("data:image/png;base64,\u{301}QUF")])?.message
                == "image data is not valid base64")
    }

    @Test(
        "Python's strict base64 rules are reproduced",
        arguments: [
            ("", 0), ("QQ==", 1), ("QUE=", 2), ("QUFB", 3), ("QR==", 1), ("QUF=", 2),
            ("QUFBQUFB", 6), ("+/+/", 3),
        ])
    func validBase64(text: String, size: Int) throws {
        let limits = ImageLimits(maxImageBytes: size)
        #expect(failure([.object(contentType: "image/png", base64: text)], limits: limits) == nil)
    }

    @Test(
        "Base64 that Python's b64decode(validate=True) refuses is refused",
        arguments: [
            "QQ=", "QQ", "Q", "Q=", "Q==", "Q===", "QUE", "QUFB=", "QUFB==", "QUFB====", "QQ==QQ==",
            "=QQ=", "QU=B", "QUE==", "QQ= =", "QQ==\n", " QUFB", "QU FB", "QUFB\n", "-_-_",
            "QUFB=QUFB", "QQ===", "\u{E9}\u{E9}==", "QUFB\u{0}", "QQ=A", "QUF\u{E9}",
        ])
    func invalidBase64(text: String) {
        let expected = SchemaError(
            "image data is not valid base64", loc: ["body", "images", .index(0)])
        #expect(failure([.object(contentType: "image/png", base64: text)]) == expected)
    }

    @Test("The decoded size is checked against the limit after decoding")
    func decodedSize() {
        // 8 characters decode to 6 bytes. The bound 3 * (8 / 4) - 2 = 4 passes a limit of 5, and
        // the decoded size then exceeds it.
        let expected = SchemaError(
            "image is 6 bytes; the limit is 5", loc: ["body", "images", .index(0)])
        let limits = ImageLimits(maxImageBytes: 5)
        let over: [ImageInput] = [.object(contentType: "image/png", base64: "QUFBQUFB")]
        #expect(failure(over, limits: limits) == expected)
        #expect(
            failure([.object(contentType: "image/png", base64: "QUFBQUE=")], limits: limits) == nil)
    }

    @Test("The length bound refuses data before it is decoded")
    func lengthBound() {
        // 12 characters: 3 * 3 - 2 = 7 is over a limit of 6, although "QUFBQUFBQQ==" decodes to
        // exactly 7 bytes; invalid text of that length gets the size message, not the base64 one.
        let limits = ImageLimits(maxImageBytes: 6)
        let expected = SchemaError(
            "image data is larger than the 6 byte limit", loc: ["body", "images", .index(0)])
        for text in ["!!!!!!!!!!!!", "QUFBQUFBQQ=="] {
            let images: [ImageInput] = [.object(contentType: "image/png", base64: text)]
            #expect(failure(images, limits: limits) == expected, "\(text)")
        }
        // Python's len counts code points, not UTF-8 bytes: four two-byte characters stay under
        // the bound and fail as base64 instead.
        let wide: [ImageInput] = [
            .object(contentType: "image/png", base64: "\u{E9}\u{E9}\u{E9}\u{E9}")
        ]
        #expect(
            failure(wide, limits: ImageLimits(maxImageBytes: 1))?.message
                == "image data is not valid base64")
    }

    @Test("An 8 MB payload is refused by the length bound, fast, without decoding")
    func oversizedPayload() throws {
        let valid = String(repeating: "QUFB", count: 2 * 1024 * 1024)
        // The same length with a character no decoder accepts at the end: if the decode ran
        // first, this would be the base64 message.
        let invalid = String(valid.dropLast()) + "!"
        let expected = "image data is larger than the 5242880 byte limit"
        let clock = ContinuousClock()
        var errors: [SchemaError?] = []
        let elapsed = clock.measure {
            errors.append(failure([.dataURL("data:image/jpeg;base64,\(valid)")]))
            errors.append(failure([.object(contentType: "image/jpeg", base64: valid)]))
            errors.append(failure([.object(contentType: "image/jpeg", base64: invalid)]))
        }
        #expect(errors.map { $0?.message } == [expected, expected, expected])
        #expect(errors.allSatisfy { $0?.loc == ["body", "images", .index(0)] })
        #expect(elapsed < .milliseconds(500), "took \(elapsed)")
    }

    @Test("Each image is checked in order and the first failure is reported with its index")
    func firstFailureWins() {
        let images: [ImageInput] = [
            .object(contentType: "image/png", base64: "QUFB"),
            .object(contentType: "image/tiff", base64: "!"),
            .dataURL("nope"),
        ]
        #expect(failure(images)?.loc == ["body", "images", .index(1)])
        #expect(failure(images)?.message.hasPrefix("image type 'image/tiff'") == true)
    }

    @Test("A SchemaError renders as a plain-detail 400 without its loc")
    func semanticBody() throws {
        let error = SchemaError(
            "image data is not valid base64", loc: ["body", "images", .index(3)])
        let wire = WireError.semantic400(error)
        #expect(wire == WireError.semantic400("image data is not valid base64"))
        #expect(try WireEncoder().string(wire) == #"{"detail":"image data is not valid base64"}"#)
        #expect(SchemaError("x").loc == ["body"])
        #expect(error.description == "body.images.3: image data is not valid base64")
    }
}
