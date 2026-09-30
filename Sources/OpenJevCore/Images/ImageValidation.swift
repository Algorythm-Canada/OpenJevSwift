// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/api.py`, function
// `image_parts` and the set `IMAGE_TYPES`, with the limits of `openjev/config.py`. Apache-2.0.
// See THIRD_PARTY.md.

/// One accepted image, ready to go ahead of the state in a read.
public struct ImagePart: Sendable, Hashable {
    /// The media type, one of ``ImageValidation/supportedTypes``.
    public let contentType: String
    /// The base64 text as sent, already checked to be strict base64.
    public let base64: String
    /// `data:{contentType};base64,{base64}`, the same for both input forms.
    ///
    /// This exact string enters the request's seed key and the prefill cache digest.
    public let dataURL: String

    /// Creates a part and its data URL.
    public init(contentType: String, base64: String) {
        self.contentType = contentType
        self.base64 = base64
        self.dataURL = "data:\(contentType);base64,\(base64)"
    }
}

/// How many images a request may carry and how large each may be once decoded.
public struct ImageLimits: Sendable, Hashable {
    /// The most images per request (`OPENJEV_MAX_IMAGES`, default 8).
    public var maxImages: Int
    /// The most bytes one decoded image may have (`OPENJEV_MAX_IMAGE_BYTES`, default 5 MiB).
    public var maxImageBytes: Int

    /// Creates limits, by default upstream's.
    public init(maxImages: Int = 8, maxImageBytes: Int = 5 * 1024 * 1024) {
        self.maxImages = maxImages
        self.maxImageBytes = maxImageBytes
    }
}

/// The image checks upstream runs before a read, with its exact messages.
///
/// Only the envelope is checked: the content type, the base64 encoding and the decoded size. No
/// pixel data is decoded here, and whether images may be combined with `think` or `sequential`
/// is the engine's decision.
public enum ImageValidation {
    /// The accepted media types: JPEG, PNG, WebP and GIF.
    public static let supportedTypes = ["image/jpeg", "image/png", "image/webp", "image/gif"]

    /// Checks a request's images in order and returns them as parts.
    ///
    /// The checks, in upstream's order:
    /// 1. more than `maxImages` images, before any image is looked at;
    /// 2. for each image, a string that is not `data:...;base64,...`;
    /// 3. a content type outside ``supportedTypes``, compared exactly;
    /// 4. base64 text whose decoded size is already over the limit by its length alone, checked
    ///    before decoding so an oversized body costs no memory;
    /// 5. text that Python's `base64.b64decode(data, validate=True)` refuses;
    /// 6. a decoded size over the limit.
    ///
    /// Upstream skips this function for an absent or empty list; an empty list returns no parts.
    ///
    /// - Throws: The first ``SchemaError`` found. Its `loc` is `["body", "images"]` for the count
    ///   and `["body", "images", i]` for image `i`.
    public static func parts(
        _ images: [ImageInput], limits: ImageLimits = ImageLimits()
    ) throws(SchemaError) -> [ImagePart] {
        if images.count > limits.maxImages {
            throw SchemaError(
                "at most \(limits.maxImages) images per request", loc: ["body", "images"])
        }
        var parts: [ImagePart] = []
        parts.reserveCapacity(images.count)
        for (index, image) in images.enumerated() {
            let loc: [LocComponent] = ["body", "images", .index(index)]
            parts.append(try part(image, limits: limits, loc: loc))
        }
        return parts
    }

    /// Checks one image.
    private static func part(
        _ image: ImageInput, limits: ImageLimits, loc: [LocComponent]
    ) throws(SchemaError) -> ImagePart {
        let contentType: String
        let data: String
        switch image {
        case .dataURL(let url):
            guard let split = splitDataURL(url) else {
                throw SchemaError(
                    "an image is a data:image/...;base64 string or a {content_type, base64} object",
                    loc: loc)
            }
            contentType = split.contentType
            data = split.data
        case .object(let type, let base64):
            contentType = type
            data = base64
        }
        guard supportedTypes.contains(where: { $0.utf8.elementsEqual(contentType.utf8) }) else {
            throw SchemaError(
                "image type \(contentType.pythonRepr) is not supported; use JPEG, PNG, WebP or GIF",
                loc: loc)
        }
        // Three bytes per four characters, less at most two of padding. Python's `len` counts
        // code points, which is what the scalar view counts.
        if 3 * (data.unicodeScalars.count / 4) - 2 > limits.maxImageBytes {
            throw SchemaError(
                "image data is larger than the \(limits.maxImageBytes) byte limit", loc: loc)
        }
        guard let size = strictBase64DecodedCount(data) else {
            throw SchemaError("image data is not valid base64", loc: loc)
        }
        if size > limits.maxImageBytes {
            throw SchemaError(
                "image is \(size) bytes; the limit is \(limits.maxImageBytes)", loc: loc)
        }
        return ImagePart(contentType: contentType, base64: data)
    }

    /// Splits `data:{type};base64,{data}` into the type and the data, or returns `nil`.
    ///
    /// As upstream's `str.partition(",")`, the split is at the first comma. The comparisons are
    /// by Unicode scalar, as Python's are, not by Swift's canonical equivalence.
    private static func splitDataURL(_ url: String) -> (contentType: String, data: String)? {
        let scalars = url.unicodeScalars
        guard let comma = scalars.firstIndex(of: ",") else { return nil }
        let head = Array(scalars[..<comma])
        let prefix = Array("data:".unicodeScalars)
        let suffix = Array(";base64".unicodeScalars)
        guard head.starts(with: prefix), head.reversed().starts(with: suffix.reversed()) else {
            return nil
        }
        // The prefix ends in ":" and the suffix starts with ";", so they cannot overlap.
        var type = String.UnicodeScalarView()
        type.append(contentsOf: head[prefix.count..<(head.count - suffix.count)])
        let data = String(String.UnicodeScalarView(scalars[scalars.index(after: comma)...]))
        return (String(type), data)
    }

    /// The decoded length of strict base64 text, or `nil` when Python's
    /// `base64.b64decode(text, validate=True)` would refuse it.
    ///
    /// Python accepts exactly this: characters from `A-Z`, `a-z`, `0-9`, `+` and `/`, a length
    /// that is a multiple of four, and at most two `=` of padding, only at the end. Whitespace,
    /// the URL-safe alphabet and non-ASCII text are refused. Non-zero bits in the last character
    /// before the padding are accepted, as Python accepts `QR==`. The length is computed without
    /// allocating the decoded bytes.
    static func strictBase64DecodedCount(_ text: String) -> Int? {
        let bytes = text.utf8
        let count = bytes.count
        guard count % 4 == 0 else { return nil }
        var padding = 0
        for (offset, byte) in bytes.enumerated() {
            if byte == UInt8(ascii: "=") {
                guard offset >= count - 2 else { return nil }
                padding += 1
            } else {
                guard padding == 0, isBase64Digit(byte) else { return nil }
            }
        }
        return count / 4 * 3 - padding
    }

    /// True for a character of the standard base64 alphabet.
    private static func isBase64Digit(_ byte: UInt8) -> Bool {
        switch byte {
        case UInt8(ascii: "A")...UInt8(ascii: "Z"), UInt8(ascii: "a")...UInt8(ascii: "z"),
            UInt8(ascii: "0")...UInt8(ascii: "9"), UInt8(ascii: "+"), UInt8(ascii: "/"):
            return true
        default:
            return false
        }
    }
}
