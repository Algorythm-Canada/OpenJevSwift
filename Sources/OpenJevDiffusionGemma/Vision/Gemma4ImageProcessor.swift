// A port of mlx-vlm 0.6.15's `models/gemma4/processing_gemma4.py` (`Gemma4ImageProcessor`:
// `aspect_ratio_preserving_resize` and `preprocess`), which `DiffusionGemma4Processor` inherits.
// MIT. See THIRD_PARTY.md.

import Foundation
import MLX
import OpenJevCore

/// The image half of the DiffusionGemma processor: the size an image is resized to, its soft
/// token count, and its `pixel_values`, as mlx-vlm computes them for upstream's image reads.
///
/// The rule, from the code rather than the configuration's `size` (224 by 224, which the
/// processor never reads): the budget is `maxSoftTokens` (280) soft tokens, so
/// `maxSoftTokens * poolingKernelSize²` (2,520) patches of `patchSize` (16) pixels. The image is
/// scaled, preserving its aspect ratio, to the largest sides that are multiples of
/// `poolingKernelSize * patchSize` (48) and fit that many patches, up or down. Its soft token
/// count is then `(height / 16) * (width / 16) / 9`, at most 280 and fewer for most aspect
/// ratios (253 for upstream's hot dog photo). The resize is Pillow's 8-bit bicubic
/// (``PillowResample``), the rescale multiplies each byte by `rescaleFactor` in float32, and
/// nothing is normalised (`do_normalize` is false).
public struct Gemma4ImageProcessor: Sendable, Hashable {
    /// The soft token budget per image (`max_soft_tokens`).
    public var maxSoftTokens: Int
    /// The vision tower's patch side in pixels (`patch_size`).
    public var patchSize: Int
    /// The side of the pooling window over patches (`pooling_kernel_size`).
    public var poolingKernelSize: Int
    /// What each byte is multiplied by (`rescale_factor`, 1/255).
    public var rescaleFactor: Double

    /// The pinned checkpoint's `processor_config.json` values.
    public init(
        maxSoftTokens: Int = 280, patchSize: Int = 16, poolingKernelSize: Int = 3,
        rescaleFactor: Double = 0.00392156862745098
    ) {
        self.maxSoftTokens = maxSoftTokens
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.rescaleFactor = rescaleFactor
    }

    /// Reads the `image_processor` object of a checkpoint's `processor_config.json`, as
    /// mlx-vlm's `Gemma4Processor.from_pretrained` does, keeping this type's defaults for keys
    /// it lacks.
    ///
    /// - Throws: A ``VisionError`` when the file is not JSON, normalises (which this port does
    ///   not), or turns resizing or rescaling off.
    public init(configuration data: Data) throws(VisionError) {
        self.init()
        let root: JSONValue
        do {
            root = try JSONParser().parse(data)
        } catch {
            throw VisionError("processor_config.json is not JSON: \(error)")
        }
        guard let config = root["image_processor"] else { return }
        if let value = config["max_soft_tokens"]?.intValue { maxSoftTokens = value }
        if let value = config["patch_size"]?.intValue { patchSize = value }
        if let value = config["pooling_kernel_size"]?.intValue { poolingKernelSize = value }
        if let value = config["rescale_factor"]?.doubleValue { rescaleFactor = value }
        for key in ["do_resize", "do_rescale", "do_convert_rgb"]
        where config[key]?.boolValue == false {
            throw VisionError(
                "processor_config.json sets \(key) false, which this port does not do")
        }
        if config["do_normalize"]?.boolValue == true {
            throw VisionError(
                "processor_config.json sets do_normalize, which this port does not do")
        }
    }

    /// The side every resized dimension is a multiple of: one pooled soft token's worth of
    /// pixels (48).
    public var sideMultiple: Int { poolingKernelSize * patchSize }

    /// The most patches an image may have (2,520).
    public var maxPatches: Int { maxSoftTokens * poolingKernelSize * poolingKernelSize }

    /// The size `aspect_ratio_preserving_resize` gives an image of `width` by `height`.
    ///
    /// - Throws: A ``VisionError`` for the sizes upstream cannot process: both resized sides
    ///   zero, and heights of 1 and 3. Transformers' `infer_channel_dimension_format` reads an
    ///   (height, width, 3) array whose height is 1 or 3 as channels first, so upstream fails on
    ///   an image 1 pixel high (Pillow cannot make an image of the misread array) and resizes an
    ///   image 3 pixels high as if it were 3 wide; neither is a read this port reproduces.
    public func targetSize(width: Int, height: Int) throws(VisionError) -> (
        width: Int, height: Int
    ) {
        guard width > 0, height > 0 else {
            throw VisionError("an image of \(width) by \(height) pixels has no pixels")
        }
        if height == 1 || height == 3 {
            throw VisionError(
                "an image \(height) pixels high is read as channels first by the processor "
                    + "upstream runs, which cannot resize it")
        }
        // Python's float arithmetic, in the order processing_gemma4.py writes it.
        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / Double(height * width)).squareRoot()
        let side = sideMultiple
        var targetHeight = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var targetWidth = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        if targetHeight == 0 && targetWidth == 0 {
            throw VisionError("an image of \(width) by \(height) pixels resizes to nothing")
        }
        let maxSide = (maxPatches / (poolingKernelSize * poolingKernelSize)) * side
        if targetHeight == 0 {
            targetHeight = side
            targetWidth = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maxSide)
        } else if targetWidth == 0 {
            targetWidth = side
            targetHeight = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maxSide)
        }
        return (targetWidth, targetHeight)
    }

    /// The soft tokens the vision tower makes of an image already resized to `width` by
    /// `height`: its patches, divided by the pooling window.
    public func softTokens(width: Int, height: Int) -> Int {
        (height / patchSize) * (width / patchSize) / (poolingKernelSize * poolingKernelSize)
    }

    /// One image's processed form.
    public struct ProcessedImage: Sendable, Hashable {
        /// The image after the resize, still 8-bit.
        public let resized: RGBImage
        /// The soft tokens it becomes.
        public let softTokens: Int
        /// The rescaled values in channel, row, column order: `(3, height, width)` as float32.
        public let values: [Float]
    }

    /// Resizes (unless the size is already the target) and rescales one image.
    ///
    /// - Parameters:
    ///   - filter: The resampling filter. Bicubic is the processor's; another is only for a
    ///     planted-bug test.
    ///   - widen: Whether shrinking widens the kernel, as Pillow does; false only for a
    ///     planted-bug test.
    /// - Throws: A ``VisionError`` from ``targetSize(width:height:)``.
    public func process(
        _ image: RGBImage, filter: PillowResample.Filter = .bicubic, widen: Bool = true
    ) throws(VisionError) -> ProcessedImage {
        let target = try targetSize(width: image.width, height: image.height)
        let resized =
            target.width == image.width && target.height == image.height
            ? image
            : PillowResample.resize(
                image, width: target.width, height: target.height, filter: filter, widen: widen)
        return ProcessedImage(
            resized: resized, softTokens: softTokens(width: resized.width, height: resized.height),
            values: rescaled(resized))
    }

    /// `image.astype(np.float32) * rescale_factor` moved to channels first. NumPy 2 casts the
    /// Python float to float32 before multiplying, so this does too.
    public func rescaled(_ image: RGBImage) -> [Float] {
        let factor = Float(rescaleFactor)
        let plane = image.width * image.height
        var values = [Float](repeating: 0, count: plane * 3)
        image.pixels.withUnsafeBufferPointer { pixels in
            values.withUnsafeMutableBufferPointer { out in
                for index in 0..<plane {
                    out[index] = Float(pixels[index * 3]) * factor
                    out[plane + index] = Float(pixels[index * 3 + 1]) * factor
                    out[2 * plane + index] = Float(pixels[index * 3 + 2]) * factor
                }
            }
        }
        return values
    }

    /// The `pixel_values` mlx-vlm hands the model for these images: one `(n, 3, H, W)` float32
    /// array when every image has the same size, else one `(3, H, W)` array per image, as
    /// `Gemma4ImageProcessor.preprocess` stacks them only when their shapes agree.
    public static func pixelValues(_ images: [ProcessedImage]) -> [MLXArray] {
        let arrays = images.map {
            MLXArray($0.values, [3, $0.resized.height, $0.resized.width])
        }
        let sizes = Set(images.map { [$0.resized.width, $0.resized.height] })
        if sizes.count == 1 {
            return [stacked(arrays, axis: 0)]
        }
        return arrays
    }
}
