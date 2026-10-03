// A Swift translation of Pillow 12.3.0's `src/libImaging/Resample.c` (`precompute_coeffs`,
// `normalize_coeffs_8bpc`, the 8-bit horizontal and vertical passes and `ImagingResampleInner`),
// for the 3-band 8-bit images `Image.resize` gets from mlx-vlm.
//
// The Python Imaging Library (PIL) is Copyright (c) 1997-2011 by Secret Labs AB and Copyright (c)
// 1995-2011 by Fredrik Lundh and contributors; Pillow is Copyright (c) 2010 by Jeffrey 'Alex'
// Clark and contributors. MIT-CMU licence: see ThirdPartyLicenses/Pillow-LICENSE beside this file,
// and THIRD_PARTY.md. Translated to Swift by the OpenJevSwift contributors, 2026.

import Foundation

/// Pillow's `Image.resize(size, resample)` for 8-bit RGB with a convolution filter, to the bit.
///
/// Pillow resizes in two separable passes, horizontal then vertical, each skipped when that
/// dimension does not change. For every output pixel a pass weighs the input pixels within the
/// filter's support around the pixel's centre; when it shrinks, the support widens by the scale
/// so every input pixel contributes (an antialiasing resize, unlike Core Image's). The weights are
/// normalised to sum to one in double precision, rounded to signed 22-bit fixed point, and every
/// sum is rounded and clipped to 0...255 after each pass. The intermediate image is 8-bit, so the
/// result is too, and a float pipeline cannot reproduce it.
public enum PillowResample {
    /// A resampling kernel and its support at scale 1.
    public struct Filter: Sendable {
        /// The name Pillow gives the filter.
        public let name: String
        /// Half the width of the kernel, in input pixels, before widening.
        public let support: Double
        /// The kernel.
        public let weight: @Sendable (Double) -> Double

        /// Pillow's `BICUBIC` (resample 3): the Keys cubic convolution kernel with a = -0.5,
        /// support 2. This is what mlx-vlm's Gemma 4 processor uses.
        public static let bicubic = Filter(name: "bicubic", support: 2.0) { value in
            let a = -0.5
            let x = value < 0 ? -value : value
            if x < 1.0 {
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
            }
            if x < 2.0 {
                return (((x - 5) * x + 8) * x - 4) * a
            }
            return 0.0
        }

        /// Pillow's `BILINEAR` (resample 2): the triangle kernel, support 1. Not what the
        /// processor uses; it exists so a test can show that the fixture's bounds catch a
        /// swapped filter.
        public static let bilinear = Filter(name: "bilinear", support: 1.0) { value in
            let x = value < 0 ? -value : value
            return x < 1.0 ? 1.0 - x : 0.0
        }
    }

    /// The fixed-point precision of the 8-bit passes: 32 bits less 8 for the result and 2 for
    /// overflow, because a kernel with negative lobes can sum past 1 on the way.
    static let precisionBits = 32 - 8 - 2

    /// One pass's weights: for each output index, the first input index and how many follow,
    /// and `kernelSize` fixed-point weights (zero past the count).
    struct Coefficients {
        var kernelSize: Int
        var bounds: [(start: Int, count: Int)]
        var weights: [Int32]
    }

    /// `precompute_coeffs` then `normalize_coeffs_8bpc` for resizing `inSize` pixels to
    /// `outSize` over the whole input (`box` from 0 to `inSize`, as `Image.resize` passes it).
    ///
    /// - Parameter widen: Whether a shrink widens the support by its scale, as Pillow does.
    ///   False only for a test that plants the bug of not widening.
    static func coefficients(inSize: Int, outSize: Int, filter: Filter, widen: Bool = true)
        -> Coefficients
    {
        // The box edges are floats in C; their difference is exact for any image size.
        let scale = Double(Float(inSize) - Float(0)) / Double(outSize)
        var filterScale = scale
        if filterScale < 1.0 || !widen {
            filterScale = 1.0
        }
        let support = filter.support * filterScale
        let kernelSize = Int(ceil(support)) * 2 + 1
        let inverseScale = 1.0 / filterScale
        var bounds: [(start: Int, count: Int)] = []
        bounds.reserveCapacity(outSize)
        var weights = [Int32](repeating: 0, count: outSize * kernelSize)
        var kernel = [Double](repeating: 0, count: kernelSize)
        let one = Double(1 << precisionBits)
        for xx in 0..<outSize {
            let center = 0.0 + (Double(xx) + 0.5) * scale
            var total = 0.0
            // C's (int) cast truncates toward zero, as Swift's Int(_:) does.
            var xmin = Int(center - support + 0.5)
            if xmin < 0 {
                xmin = 0
            }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize {
                xmax = inSize
            }
            xmax -= xmin
            for x in 0..<kernelSize {
                kernel[x] = 0
            }
            for x in 0..<xmax {
                let w = filter.weight((Double(x + xmin) - center + 0.5) * inverseScale)
                kernel[x] = w
                total += w
            }
            if total != 0.0 {
                for x in 0..<xmax {
                    kernel[x] /= total
                }
            }
            for x in 0..<kernelSize {
                let k = kernel[x]
                weights[xx * kernelSize + x] =
                    k < 0 ? Int32(-0.5 + k * one) : Int32(0.5 + k * one)
            }
            bounds.append((xmin, xmax))
        }
        return Coefficients(kernelSize: kernelSize, bounds: bounds, weights: weights)
    }

    /// `clip8`: the fixed-point sum shifted down and clipped to a byte.
    @inline(__always)
    static func clip8(_ sum: Int) -> UInt8 {
        let value = sum >> precisionBits
        return value < 0 ? 0 : value > 255 ? 255 : UInt8(value)
    }

    /// `image` resized to `width` by `height` as Pillow's `Image.resize((width, height),
    /// resample)` does it. An unchanged size returns the image as it is.
    ///
    /// - Parameter widen: Whether a shrink widens the kernel (Pillow always does); false only
    ///   for a planted-bug test.
    public static func resize(
        _ image: RGBImage, width: Int, height: Int, filter: Filter = .bicubic,
        widen: Bool = true
    ) -> RGBImage {
        precondition(width > 0 && height > 0, "a resize needs a positive size")
        let needHorizontal = width != image.width
        let needVertical = height != image.height
        if !needHorizontal && !needVertical {
            return image
        }
        let half = 1 << (precisionBits - 1)
        var vertical = coefficients(
            inSize: image.height, outSize: height, filter: filter, widen: widen)
        // The rows the vertical pass reads, so the horizontal pass computes only those.
        let firstRow = vertical.bounds[0].start
        let lastRow = vertical.bounds[height - 1].start + vertical.bounds[height - 1].count

        var source = image.pixels
        var sourceWidth = image.width
        if needHorizontal {
            let horizontal = coefficients(
                inSize: image.width, outSize: width, filter: filter, widen: widen)
            let rows = lastRow - firstRow
            let k = horizontal.kernelSize
            var out = [UInt8](repeating: 0, count: width * rows * 3)
            image.pixels.withUnsafeBufferPointer { input in
                horizontal.weights.withUnsafeBufferPointer { weights in
                    out.withUnsafeMutableBufferPointer { output in
                        for yy in 0..<rows {
                            let inRow = (yy + firstRow) * image.width * 3
                            let outRow = yy * width * 3
                            for xx in 0..<width {
                                let (start, count) = horizontal.bounds[xx]
                                var s0 = half
                                var s1 = half
                                var s2 = half
                                for x in 0..<count {
                                    let w = Int(weights[xx * k + x])
                                    let p = inRow + (x + start) * 3
                                    s0 += Int(input[p]) * w
                                    s1 += Int(input[p + 1]) * w
                                    s2 += Int(input[p + 2]) * w
                                }
                                output[outRow + xx * 3] = clip8(s0)
                                output[outRow + xx * 3 + 1] = clip8(s1)
                                output[outRow + xx * 3 + 2] = clip8(s2)
                            }
                        }
                    }
                }
            }
            for i in 0..<height {
                vertical.bounds[i].start -= firstRow
            }
            source = out
            sourceWidth = width
        }
        if !needVertical {
            return RGBImage(width: sourceWidth, height: height, pixels: source)
        }
        let k = vertical.kernelSize
        var out = [UInt8](repeating: 0, count: sourceWidth * height * 3)
        source.withUnsafeBufferPointer { input in
            vertical.weights.withUnsafeBufferPointer { weights in
                out.withUnsafeMutableBufferPointer { output in
                    for yy in 0..<height {
                        let (start, count) = vertical.bounds[yy]
                        let outRow = yy * sourceWidth * 3
                        for xx in 0..<sourceWidth {
                            var s0 = half
                            var s1 = half
                            var s2 = half
                            for y in 0..<count {
                                let w = Int(weights[yy * k + y])
                                let p = ((y + start) * sourceWidth + xx) * 3
                                s0 += Int(input[p]) * w
                                s1 += Int(input[p + 1]) * w
                                s2 += Int(input[p + 2]) * w
                            }
                            output[outRow + xx * 3] = clip8(s0)
                            output[outRow + xx * 3 + 1] = clip8(s1)
                            output[outRow + xx * 3 + 2] = clip8(s2)
                        }
                    }
                }
            }
        }
        return RGBImage(width: sourceWidth, height: height, pixels: out)
    }
}
