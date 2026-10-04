import CoreGraphics
import Foundation
import ImageIO
import OpenJevCore

/// An image as 8-bit RGB: what `PIL.Image.open(...).convert("RGB")` gives upstream's
/// `ImagePrompt.pil` (`openjev/mlx_backend.py:66-72`).
public struct RGBImage: Sendable, Hashable {
    /// Width in pixels.
    public let width: Int
    /// Height in pixels.
    public let height: Int
    /// The pixels row by row from the top, three bytes (red, green, blue) each.
    public let pixels: [UInt8]

    /// Creates an image from its pixels, which must hold `width * height * 3` bytes.
    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3, "an RGB image needs 3 bytes per pixel")
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

extension RGBImage {
    /// Decodes an accepted image part the way upstream does: the base64 text after the comma,
    /// opened by content sniffing (the declared type plays no part, as in PIL), first frame.
    ///
    /// - Throws: A ``VisionError`` when the bytes are not an image ImageIO can decode.
    public init(decoding part: ImagePart) throws(VisionError) {
        guard let data = Data(base64Encoded: part.base64) else {
            throw VisionError("the image's base64 text does not decode")
        }
        try self.init(decoding: data)
    }

    /// Decodes JPEG, PNG, WebP or GIF bytes to 8-bit RGB without colour management.
    ///
    /// A GIF's first frame goes through `PillowGIFDecoder`, a port of Pillow's GIF reader,
    /// because ImageIO leaves the pixels of the transparent index, and the logical screen around
    /// the first frame, as (0, 0, 0, 0), where Pillow has a palette colour; it refuses every GIF
    /// on which Pillow raises. A JPEG goes through `LibjpegTurboDecoder`, a port of the
    /// libjpeg-turbo that Pillow runs, because decoders are free to differ in the inverse DCT and
    /// chroma upsampling and ImageIO's does (by up to 30 levels on upstream's hot dog photo). A
    /// JPEG it does not cover (arithmetic-coded, lossless, 12-bit, CMYK) falls back to ImageIO.
    /// Other formats are decoded by ImageIO, whose samples of 8-bit PNGs and of lossless and
    /// lossy WebPs, translucent ones included, equal PIL's (docs/spikes/vision-preprocessing.md).
    ///
    /// PIL's `convert("RGB")` copies the decoded samples: it ignores embedded colour profiles,
    /// copies a grey level into all three channels, looks palette indices up in the palette and
    /// drops alpha without compositing. This reads the samples ImageIO decoded in the same way
    /// for 8-bit RGB, grey and indexed images, with or without alpha, in either byte order. Any
    /// other layout (16-bit or floating-point samples, CMYK) is drawn into an 8-bit sRGB context
    /// instead, which Core Graphics colour-manages and composites over black, so it may differ
    /// from PIL.
    ///
    /// - Parameters:
    ///   - data: The encoded image.
    ///   - frame: The frame to decode. Upstream reads the first (PIL opens at frame 0); the
    ///     parameter exists so a test can show that the second frame of a GIF differs, and any
    ///     other frame is ImageIO's.
    /// - Throws: A ``VisionError`` when the image is one Pillow refuses, or one ImageIO cannot
    ///   decode, or has no such frame.
    public init(decoding data: Data, frame: Int = 0) throws(VisionError) {
        let bytes = [UInt8](data)
        if frame == 0, PillowGIFDecoder.isGIF(bytes) {
            // Pillow sizes the canvas from the headers and checks it against its limit itself.
            do {
                self = try PillowGIFDecoder.decode(bytes)
                return
            } catch {
                throw VisionError(error.description)
            }
        }
        try Self.checkSize(of: data)
        if frame == 0, LibjpegTurboDecoder.isJPEG(bytes) {
            do {
                self = try LibjpegTurboDecoder.decode(bytes)
                return
            } catch {
                switch error {
                case .refused(let refusal):
                    throw VisionError(refusal.description)
                case .unsupported:
                    // ImageIO may still decode it, if not as Pillow would.
                    break
                }
            }
        }
        try self.init(imageIO: data, frame: frame)
    }

    /// Decodes with ImageIO, reading the samples as ``init(decoding:frame:)`` describes.
    ///
    /// - Throws: A ``VisionError`` when ImageIO cannot decode the data or has no such frame.
    public init(imageIO data: Data, frame: Int = 0) throws(VisionError) {
        try Self.checkSize(of: data)
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, options) else {
            throw VisionError("the image data is not an image ImageIO can open")
        }
        let count = CGImageSourceGetCount(source)
        guard frame >= 0, frame < count,
            let image = CGImageSourceCreateImageAtIndex(source, frame, options)
        else {
            throw VisionError("the image has \(count) frames; frame \(frame) does not decode")
        }
        if let samples = Self.samples(of: image) {
            self = samples
        } else {
            self = try Self.drawn(image)
        }
    }

    /// The most pixels an image may have: Pillow's `2 * Image.MAX_IMAGE_PIXELS`. Above it,
    /// `Image.open` raises `DecompressionBombError` in upstream, so a small file that declares a
    /// huge size is refused before anything is allocated for it.
    public static let maxPixels = 178_956_970

    /// Refuses an image whose header declares more than ``maxPixels`` pixels, read by ImageIO
    /// without decoding. A header ImageIO cannot read is left to the decoders to refuse.
    static func checkSize(of data: Data) throws(VisionError) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
            let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
            let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue
        else { return }
        let pixels = max(1, width) * max(1, height)
        if pixels > maxPixels {
            throw VisionError(
                "the image is \(pixels) pixels; the limit is \(maxPixels) (Pillow's "
                    + "decompression bomb limit)")
        }
    }

    /// The number of frames ImageIO finds in `data`, or zero when it is not an image.
    public static func frameCount(of data: Data) -> Int {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return 0 }
        return CGImageSourceGetCount(source)
    }

    /// The decoded samples copied as PIL's `convert("RGB")` copies them, or `nil` for a layout
    /// this does not read directly.
    static func samples(of image: CGImage) -> RGBImage? {
        let info = image.bitmapInfo
        guard image.bitsPerComponent == 8, !info.contains(.floatComponents),
            let space = image.colorSpace, let data = image.dataProvider?.data
        else { return nil }
        let width = image.width
        let height = image.height
        let bytesPerPixel = image.bitsPerPixel / 8
        let bytesPerRow = image.bytesPerRow
        guard CFDataGetLength(data) >= bytesPerRow * (height - 1) + bytesPerPixel * width else {
            return nil
        }
        let alpha = image.alphaInfo
        let alphaFirst = [.first, .premultipliedFirst, .noneSkipFirst].contains(alpha)
        let hasAlpha = alpha != .none
        // A 32-bit little-endian pixel stores its components in reverse order.
        let reversed = image.byteOrderInfo == .order32Little && bytesPerPixel == 4

        // The byte offsets of red, green and blue (or of the grey level, or the palette index)
        // within a pixel.
        let offsets: [Int]
        var table: [UInt8]?
        switch space.model {
        case .rgb:
            guard bytesPerPixel == (hasAlpha ? 4 : 3) else { return nil }
            let logical = alphaFirst ? [1, 2, 3] : [0, 1, 2]
            offsets = reversed ? logical.map { 3 - $0 } : logical
        case .monochrome:
            guard bytesPerPixel == (hasAlpha ? 2 : 1), !reversed else { return nil }
            let level = alphaFirst ? 1 : 0
            offsets = [level, level, level]
        case .indexed:
            guard bytesPerPixel == 1, !hasAlpha, let colors = space.colorTable,
                space.baseColorSpace?.model == .rgb
            else { return nil }
            table = colors
            offsets = [0, 0, 0]
        default:
            return nil
        }
        let bytes = CFDataGetBytePtr(data)!
        // Premultiplied samples equal the stored ones only where alpha is 255, and PIL never
        // divides alpha back out, so an image handed over premultiplied with any translucent
        // pixel goes to the drawn path. On macOS 27.0.1 none of the images measured arrives
        // premultiplied: ImageIO hands opaque GIFs and WebPs over as `noneSkipLast`, translucent
        // PNGs and WebPs as `last` with their stored colour samples, and GIFs with a transparent
        // index as `last` with (0, 0, 0, 0) for those pixels, which is why a GIF's first frame
        // goes to `PillowGIFDecoder` instead.
        if [.premultipliedFirst, .premultipliedLast].contains(alpha) {
            let alphaOffset = (alphaFirst ? 0 : bytesPerPixel - 1)
            let alphaByte = reversed ? bytesPerPixel - 1 - alphaOffset : alphaOffset
            for y in 0..<height {
                let row = bytes + y * bytesPerRow
                for x in 0..<width where row[x * bytesPerPixel + alphaByte] != 255 {
                    return nil
                }
            }
        }

        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        pixels.withUnsafeMutableBufferPointer { out in
            for y in 0..<height {
                let row = bytes + y * bytesPerRow
                for x in 0..<width {
                    let pixel = row + x * bytesPerPixel
                    let target = (y * width + x) * 3
                    if let table {
                        let index = min(Int(pixel[0]) * 3, table.count - 3)
                        out[target] = table[index]
                        out[target + 1] = table[index + 1]
                        out[target + 2] = table[index + 2]
                    } else {
                        out[target] = pixel[offsets[0]]
                        out[target + 1] = pixel[offsets[1]]
                        out[target + 2] = pixel[offsets[2]]
                    }
                }
            }
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    /// The image drawn into an 8-bit sRGB context, for layouts ``samples(of:)`` does not read.
    static func drawn(_ image: CGImage) throws(VisionError) -> RGBImage {
        let width = image.width
        let height = image.height
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw VisionError("could not make a \(width) by \(height) context to decode into")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else {
            throw VisionError("the decoding context has no pixels")
        }
        let bytes = data.assumingMemoryBound(to: UInt8.self)
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        for index in 0..<(width * height) {
            pixels[index * 3] = bytes[index * 4]
            pixels[index * 3 + 1] = bytes[index * 4 + 1]
            pixels[index * 3 + 2] = bytes[index * 4 + 2]
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }
}
