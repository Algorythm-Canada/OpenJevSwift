// A Swift translation of libjpeg-turbo 3.1.4.1's default decompression path, the one Pillow
// 12.3.0 runs: `jdhuff.c`, `jdphuff.c`, `jidctint.c` (`jpeg_idct_islow`), `jdsample.c` (fancy
// upsampling), `jdcolor.c` (`ycc_rgb_convert`), `jdmaster.c` (`prepare_range_limit_table`) and
// `jdapimin.c` (`default_decompress_parms`).
//
// This file was derived from the Independent JPEG Group's software:
// Copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
// libjpeg-turbo Modifications: Copyright (C) 2009-2026, D. R. Commander.
// Translated to Swift, restructured to decode a whole image in memory and reduced to the 8-bit
// Huffman-coded path by the OpenJevSwift contributors, 2026. For conditions of distribution and
// use, see ThirdPartyLicenses/libjpeg-turbo-README.ijg and libjpeg-turbo-LICENSE.md beside this
// file, and THIRD_PARTY.md. This software is based in part on the work of the Independent JPEG
// Group.

/// Decodes the JPEGs Pillow decodes for upstream's image reads, sample for sample.
///
/// JPEG leaves the inverse DCT, chroma upsampling and colour conversion to the decoder, and
/// decoders differ: ImageIO's decode of upstream's hot dog photo is up to 30 levels from Pillow's
/// at a third of its samples, far outside issue #46's 1e-3. This reproduces libjpeg-turbo with
/// Pillow's settings (the library defaults): the accurate integer IDCT, "fancy" triangle-filter
/// upsampling, and its fixed-point YCbCr to RGB.
///
/// Covered: 8-bit Huffman-coded baseline, extended sequential and progressive JPEGs with one
/// (grey) or three (YCbCr or RGB) components, any sampling factors, and restart intervals.
/// Anything else (arithmetic coding, lossless or 12-bit JPEGs, CMYK and YCCK) throws
/// ``Unsupported`` so the caller can fall back to ImageIO.
enum LibjpegTurboDecoder {
    /// A JPEG this decoder does not cover, or one too damaged to decode. The caller may try
    /// another decoder.
    struct Unsupported: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    /// A JPEG refused outright: one that declares more pixels than ``RGBImage/maxPixels``, or
    /// more scans than ``maxScans``. No other decoder should be tried on it.
    struct Refused: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    /// The most scans a JPEG may have. Each scan walks the whole image, so a small file of empty
    /// scans would otherwise cost time in proportion to its size times the image's. Encoders
    /// write about 10 for a progressive JPEG and 1 for a baseline one.
    static let maxScans = 100

    /// Errors the decoder throws.
    enum Failure: Error {
        case unsupported(Unsupported)
        case refused(Refused)
    }

    /// True when `bytes` start with a JPEG's SOI marker.
    static func isJPEG(_ bytes: [UInt8]) -> Bool {
        bytes.count > 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
    }

    /// `jpeg_natural_order`: zigzag position to natural position, with libjpeg's 16 guard
    /// entries for corrupt data.
    static let naturalOrder: [Int] =
        [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
            27,
            20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44,
            51,
            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        ] + Array(repeating: 63, count: 16)

    struct HuffmanTable {
        var maxCode = [Int](repeating: -1, count: 18)
        var valueOffset = [Int](repeating: 0, count: 17)
        var values: [UInt8] = []

        init() {}

        /// `jpeg_make_d_derived_tbl`'s canonical codes from the 16 counts and the values.
        init(counts: [Int], values: [UInt8]) {
            self.values = values
            var code = 0
            var index = 0
            for length in 1...16 {
                let count = counts[length - 1]
                if count > 0 {
                    valueOffset[length] = index - code
                    code += count
                    index += count
                    maxCode[length] = code - 1
                } else {
                    maxCode[length] = -1
                }
                code <<= 1
            }
            maxCode[17] = Int.max
        }
    }

    struct Component {
        var id: Int
        var h: Int
        var v: Int
        var quantTable: Int
        /// The quantization table as it stood at the component's first scan, which is when
        /// libjpeg latches it (`latch_quant_tables`).
        var quant: [Int]?
        /// Blocks per line and per column of the coefficient grid, padded to whole MCUs.
        var blocksPerLine = 0
        var blocksPerColumn = 0
        /// Samples per line and column the component really has (`downsampled_width`, `_height`).
        var width = 0
        var height = 0
        var coefficients: [Int16] = []
        var dcTable = 0
        var acTable = 0
        var predictor = 0
    }

    /// Reads entropy-coded bits, undoing 0xFF00 stuffing and stopping at markers.
    struct BitReader {
        let bytes: [UInt8]
        var position: Int
        var buffer: UInt64 = 0
        var count = 0
        /// Set once a marker is reached; libjpeg then supplies zeros.
        var hitMarker = false

        init(bytes: [UInt8], position: Int) {
            self.bytes = bytes
            self.position = position
        }

        mutating func fill() {
            while count <= 56 {
                var byte: UInt64 = 0
                if !hitMarker, position < bytes.count {
                    let b = bytes[position]
                    if b == 0xFF {
                        let next = position + 1 < bytes.count ? bytes[position + 1] : 0xD9
                        if next == 0x00 {
                            byte = 0xFF
                            position += 2
                        } else {
                            hitMarker = true
                        }
                    } else {
                        byte = UInt64(b)
                        position += 1
                    }
                }
                buffer |= byte << UInt64(56 - count)
                count += 8
            }
        }

        mutating func bits(_ n: Int) -> Int {
            if n == 0 { return 0 }
            if count < n { fill() }
            let value = Int(buffer >> UInt64(64 - n))
            buffer <<= UInt64(n)
            count -= n
            return value
        }

        mutating func bit() -> Int { bits(1) }

        mutating func decode(_ table: HuffmanTable) throws(Unsupported) -> Int {
            var code = bit()
            var length = 1
            while code > table.maxCode[length] {
                code = (code << 1) | bit()
                length += 1
                if length > 16 {
                    throw Unsupported("a Huffman code is longer than 16 bits")
                }
            }
            let index = table.valueOffset[length] + code
            guard index >= 0, index < table.values.count else {
                throw Unsupported("a Huffman code has no value")
            }
            return Int(table.values[index])
        }

        /// `HUFF_EXTEND`: an s-bit magnitude as a signed value.
        mutating func received(_ s: Int) -> Int {
            if s == 0 { return 0 }
            let value = bits(s)
            return value < (1 << (s - 1)) ? value - (1 << s) + 1 : value
        }

        /// Drops buffered bits and moves past the restart marker that must follow.
        mutating func restart() throws(Unsupported) {
            buffer = 0
            count = 0
            hitMarker = false
            while position + 1 < bytes.count {
                if bytes[position] == 0xFF, (0xD0...0xD7).contains(bytes[position + 1]) {
                    position += 2
                    return
                }
                position += 1
            }
            throw Unsupported("a restart marker is missing")
        }
    }

    // MARK: Decoding

    static func decode(_ bytes: [UInt8]) throws(Failure) -> RGBImage {
        do {
            return try decodeChecked(bytes)
        } catch let error as Refused {
            throw .refused(error)
        } catch let error as Unsupported {
            throw .unsupported(error)
        } catch {
            throw .unsupported(Unsupported("\(error)"))
        }
    }

    static func decodeChecked(_ bytes: [UInt8]) throws -> RGBImage {
        guard isJPEG(bytes) else { throw Unsupported("not a JPEG") }
        var quantTables = [[Int]](repeating: [], count: 4)
        var dcTables = [HuffmanTable](repeating: HuffmanTable(), count: 4)
        var acTables = [HuffmanTable](repeating: HuffmanTable(), count: 4)
        var components: [Component] = []
        var width = 0
        var height = 0
        var progressive = false
        var restartInterval = 0
        var sawJFIF = false
        var adobeTransform: Int?
        var maxH = 1
        var maxV = 1
        var mcusPerLine = 0
        var mcusPerColumn = 0
        var position = 2
        var scans = 0

        func u16(_ at: Int) throws(Unsupported) -> Int {
            guard at + 1 < bytes.count else { throw Unsupported("the JPEG is cut short") }
            return Int(bytes[at]) << 8 | Int(bytes[at + 1])
        }

        markers: while true {
            // Find the next marker, skipping fill bytes.
            while position < bytes.count, bytes[position] != 0xFF { position += 1 }
            while position < bytes.count, bytes[position] == 0xFF { position += 1 }
            guard position < bytes.count else { throw Unsupported("the JPEG has no EOI") }
            let marker = bytes[position]
            position += 1
            switch marker {
            case 0xD9:
                break markers
            case 0xD0...0xD7, 0x01:
                continue
            default:
                break
            }
            let length = try u16(position)
            let start = position + 2
            let end = position + length
            guard length >= 2, end <= bytes.count else { throw Unsupported("a marker overruns") }
            switch marker {
            case 0xE0:
                // examine_app0: "JFIF\0" with at least 14 bytes of data.
                if length - 2 >= 14,
                    Array(bytes[start..<(start + 5)]) == [0x4A, 0x46, 0x49, 0x46, 0]
                {
                    sawJFIF = true
                }
            case 0xEE:
                // examine_app14: "Adobe" with at least 12 bytes; the transform is the last.
                if length - 2 >= 12,
                    Array(bytes[start..<(start + 5)]) == [0x41, 0x64, 0x6F, 0x62, 0x65]
                {
                    adobeTransform = Int(bytes[start + 11])
                }
            case 0xDB:
                var at = start
                while at < end {
                    let precision = Int(bytes[at] >> 4)
                    let id = Int(bytes[at] & 15)
                    at += 1
                    guard id < 4, precision <= 1 else {
                        throw Unsupported("a quantization table is bad")
                    }
                    guard at + 64 * (precision + 1) <= end else {
                        throw Unsupported("a quantization table overruns its segment")
                    }
                    var table = [Int](repeating: 0, count: 64)
                    for k in 0..<64 {
                        let value = precision == 0 ? Int(bytes[at]) : try u16(at)
                        at += precision == 0 ? 1 : 2
                        table[naturalOrder[k]] = value
                    }
                    quantTables[id] = table
                }
            case 0xC4:
                var at = start
                while at < end {
                    let tableClass = Int(bytes[at] >> 4)
                    let id = Int(bytes[at] & 15)
                    guard id < 4, tableClass <= 1, at + 17 <= end else {
                        throw Unsupported("a Huffman table is bad")
                    }
                    let counts = (0..<16).map { Int(bytes[at + 1 + $0]) }
                    let total = counts.reduce(0, +)
                    guard at + 17 + total <= end else {
                        throw Unsupported("a Huffman table overruns")
                    }
                    let values = Array(bytes[(at + 17)..<(at + 17 + total)])
                    // jpeg_make_d_derived_tbl: a DC symbol is a bit count of at most 15.
                    guard total <= 256, tableClass == 1 || values.allSatisfy({ $0 <= 15 }) else {
                        throw Unsupported("a Huffman table has a bad value")
                    }
                    let table = HuffmanTable(counts: counts, values: values)
                    if tableClass == 0 { dcTables[id] = table } else { acTables[id] = table }
                    at += 17 + total
                }
            case 0xDD:
                guard length == 4 else { throw Unsupported("a DRI segment is bad") }
                restartInterval = try u16(start)
            case 0xC0, 0xC1, 0xC2:
                guard components.isEmpty else { throw Unsupported("the JPEG has two frames") }
                guard length >= 8 else { throw Unsupported("a frame header is cut short") }
                progressive = marker == 0xC2
                guard bytes[start] == 8 else { throw Unsupported("only 8-bit JPEGs are covered") }
                height = try u16(start + 1)
                width = try u16(start + 3)
                let count = Int(bytes[start + 5])
                guard width > 0, height > 0 else { throw Unsupported("the JPEG has no size (DNL)") }
                guard count == 1 || count == 3 else {
                    throw Unsupported("\(count)-component JPEGs are not covered")
                }
                guard length == 8 + 3 * count else {
                    throw Unsupported("a frame header's length does not fit its components")
                }
                guard max(1, width) * max(1, height) <= RGBImage.maxPixels else {
                    throw Refused(
                        "the image is \(width * height) pixels; the limit is \(RGBImage.maxPixels)")
                }
                components = (0..<count).map { i in
                    let at = start + 6 + i * 3
                    return Component(
                        id: Int(bytes[at]), h: Int(bytes[at + 1] >> 4), v: Int(bytes[at + 1] & 15),
                        quantTable: Int(bytes[at + 2] & 3))
                }
                maxH = components.map(\.h).max() ?? 1
                maxV = components.map(\.v).max() ?? 1
                guard components.allSatisfy({ (1...4).contains($0.h) && (1...4).contains($0.v) })
                else { throw Unsupported("a sampling factor is out of range") }
                // jinit_upsampler refuses fractional ratios; jdinput refuses more than 10 blocks
                // per MCU (D_MAX_BLOCKS_IN_MCU).
                guard components.allSatisfy({ maxH % $0.h == 0 && maxV % $0.v == 0 }) else {
                    throw Unsupported("a sampling ratio is fractional")
                }
                guard count == 1 || components.reduce(0, { $0 + $1.h * $1.v }) <= 10 else {
                    throw Unsupported("an MCU has more than 10 blocks")
                }
                mcusPerLine = (width + 8 * maxH - 1) / (8 * maxH)
                mcusPerColumn = (height + 8 * maxV - 1) / (8 * maxV)
                for i in components.indices {
                    components[i].blocksPerLine = mcusPerLine * components[i].h
                    components[i].blocksPerColumn = mcusPerColumn * components[i].v
                    components[i].width = (width * components[i].h + maxH - 1) / maxH
                    components[i].height = (height * components[i].v + maxV - 1) / maxV
                    components[i].coefficients = [Int16](
                        repeating: 0,
                        count: components[i].blocksPerLine * components[i].blocksPerColumn * 64)
                }
            case 0xC3, 0xC5...0xC7, 0xC9...0xCB, 0xCD...0xCF:
                throw Unsupported(
                    "lossless, hierarchical and arithmetic-coded JPEGs are not covered")
            case 0xDA:
                guard !components.isEmpty else {
                    throw Unsupported("a scan comes before the frame")
                }
                scans += 1
                guard scans <= maxScans else {
                    throw Refused("the JPEG has more than \(maxScans) scans")
                }
                guard length >= 3 else { throw Unsupported("a scan header is cut short") }
                let count = Int(bytes[start])
                guard (1...4).contains(count), length == 6 + 2 * count else {
                    throw Unsupported("a scan header's length does not fit its components")
                }
                var scan: [Int] = []
                for i in 0..<count {
                    let at = start + 1 + i * 2
                    guard let index = components.firstIndex(where: { $0.id == Int(bytes[at]) })
                    else {
                        throw Unsupported("a scan names a component the frame lacks")
                    }
                    components[index].dcTable = Int(bytes[at + 1] >> 4) & 3
                    components[index].acTable = Int(bytes[at + 1] & 15) & 3
                    if components[index].quant == nil {
                        let table = quantTables[components[index].quantTable]
                        guard table.count == 64 else {
                            throw Unsupported("a component's quantization table is missing")
                        }
                        components[index].quant = table
                    }
                    scan.append(index)
                }
                let at = start + 1 + count * 2
                let ss = Int(bytes[at])
                let se = Int(bytes[at + 1])
                let ah = Int(bytes[at + 2] >> 4)
                let al = Int(bytes[at + 2] & 15)
                // The parameter checks of jdphuff.c's start_pass_phuff_decoder. (libjpeg only
                // warns when a sequential scan's are off, and its decoder, like this one, never
                // reads them.)
                if progressive {
                    let dcScan = ss == 0
                    guard dcScan ? se == 0 : (ss <= se && se <= 63 && count == 1),
                        ah == 0 || al == ah - 1, al <= 13
                    else { throw Unsupported("a progressive scan's parameters are bad") }
                }
                var reader = BitReader(bytes: bytes, position: end)
                try decodeScan(
                    &components, scan: scan, reader: &reader, dcTables: dcTables,
                    acTables: acTables, progressive: progressive, ss: ss, se: se, ah: ah, al: al,
                    restartInterval: restartInterval, mcusPerLine: mcusPerLine,
                    mcusPerColumn: mcusPerColumn)
                position = reader.position
                continue markers
            default:
                break
            }
            position = end
        }
        guard !components.isEmpty else { throw Unsupported("the JPEG has no frame") }

        // default_decompress_parms: the colour space libjpeg guesses for three components.
        var ycc = true
        if components.count == 3 {
            if sawJFIF {
                ycc = true
            } else if let adobeTransform {
                ycc = adobeTransform != 0
            } else {
                let ids = components.map(\.id)
                ycc = ids != [82, 71, 66]
            }
        }

        let planes = try components.map { component throws(Unsupported) in
            guard let quant = component.quant else {
                throw Unsupported("a component is in no scan")
            }
            return inverseDCT(component, quant: quant)
        }
        let full = components.indices.map { i in
            upsample(
                planes[i], component: components[i], maxH: maxH, maxV: maxV, width: width,
                height: height)
        }
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        if components.count == 1 {
            for index in 0..<(width * height) {
                let level = full[0][index]
                pixels[index * 3] = level
                pixels[index * 3 + 1] = level
                pixels[index * 3 + 2] = level
            }
        } else if ycc {
            let tables = ColorTables.shared
            for index in 0..<(width * height) {
                let y = Int(full[0][index])
                let cb = Int(full[1][index])
                let cr = Int(full[2][index])
                pixels[index * 3] = clamp(y + tables.crR[cr])
                pixels[index * 3 + 1] = clamp(y + ((tables.cbG[cb] + tables.crG[cr]) >> 16))
                pixels[index * 3 + 2] = clamp(y + tables.cbB[cb])
            }
        } else {
            for index in 0..<(width * height) {
                pixels[index * 3] = full[0][index]
                pixels[index * 3 + 1] = full[1][index]
                pixels[index * 3 + 2] = full[2][index]
            }
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    @inline(__always)
    static func clamp(_ value: Int) -> UInt8 {
        value < 0 ? 0 : value > 255 ? 255 : UInt8(value)
    }

    // MARK: Entropy decoding

    static func decodeScan(
        _ components: inout [Component], scan: [Int], reader: inout BitReader,
        dcTables: [HuffmanTable], acTables: [HuffmanTable], progressive: Bool, ss: Int, se: Int,
        ah: Int, al: Int, restartInterval: Int, mcusPerLine: Int, mcusPerColumn: Int
    ) throws(Unsupported) {
        for index in scan { components[index].predictor = 0 }
        var eobrun = 0

        // One block: decode it in place in the component's coefficients.
        func block(_ c: Int, row: Int, column: Int) throws(Unsupported) {
            let base = (row * components[c].blocksPerLine + column) * 64
            if !progressive {
                let s = try reader.decode(dcTables[components[c].dcTable])
                components[c].predictor += reader.received(s)
                components[c].coefficients[base] = Int16(
                    truncatingIfNeeded: components[c].predictor)
                let table = acTables[components[c].acTable]
                var k = 1
                while k < 64 {
                    let rs = try reader.decode(table)
                    let r = rs >> 4
                    let s = rs & 15
                    if s != 0 {
                        k += r
                        components[c].coefficients[base + naturalOrder[k]] = Int16(
                            truncatingIfNeeded: reader.received(s))
                    } else {
                        if r != 15 { break }
                        k += 15
                    }
                    k += 1
                }
                return
            }
            if ss == 0 {
                if ah == 0 {
                    let s = try reader.decode(dcTables[components[c].dcTable])
                    components[c].predictor += reader.received(s)
                    components[c].coefficients[base] = Int16(
                        truncatingIfNeeded: components[c].predictor << al)
                } else if reader.bit() != 0 {
                    components[c].coefficients[base] |= Int16(truncatingIfNeeded: 1 << al)
                }
                return
            }
            let table = acTables[components[c].acTable]
            if ah == 0 {
                // decode_mcu_AC_first
                if eobrun > 0 {
                    eobrun -= 1
                    return
                }
                var k = ss
                while k <= se {
                    let rs = try reader.decode(table)
                    let r = rs >> 4
                    let s = rs & 15
                    if s != 0 {
                        k += r
                        components[c].coefficients[base + naturalOrder[k]] = Int16(
                            truncatingIfNeeded: reader.received(s) << al)
                    } else if r == 15 {
                        k += 15
                    } else {
                        eobrun = 1 << r
                        if r != 0 { eobrun += reader.bits(r) }
                        eobrun -= 1
                        break
                    }
                    k += 1
                }
                return
            }
            // decode_mcu_AC_refine
            let p1 = 1 << al
            let m1 = -1 << al
            func refine(_ position: Int) {
                let value = Int(components[c].coefficients[position])
                if reader.bit() != 0, value & p1 == 0 {
                    components[c].coefficients[position] = Int16(
                        truncatingIfNeeded: value >= 0 ? value + p1 : value + m1)
                }
            }
            var k = ss
            if eobrun == 0 {
                while k <= se {
                    let rs = try reader.decode(table)
                    var r = rs >> 4
                    var s = rs & 15
                    if s != 0 {
                        s = reader.bit() != 0 ? p1 : m1
                    } else if r != 15 {
                        eobrun = 1 << r
                        if r != 0 { eobrun += reader.bits(r) }
                        break
                    }
                    repeat {
                        let position = base + naturalOrder[k]
                        if components[c].coefficients[position] != 0 {
                            refine(position)
                        } else {
                            r -= 1
                            if r < 0 { break }
                        }
                        k += 1
                    } while k <= se
                    if s != 0 {
                        components[c].coefficients[base + naturalOrder[k]] = Int16(
                            truncatingIfNeeded: s)
                    }
                    k += 1
                }
            }
            if eobrun > 0 {
                while k <= se {
                    let position = base + naturalOrder[k]
                    if components[c].coefficients[position] != 0 {
                        refine(position)
                    }
                    k += 1
                }
                eobrun -= 1
            }
        }

        var mcusToRestart = restartInterval
        func restartIfDue() throws(Unsupported) {
            guard restartInterval > 0 else { return }
            if mcusToRestart == 0 {
                try reader.restart()
                for index in scan { components[index].predictor = 0 }
                eobrun = 0
                mcusToRestart = restartInterval
            }
            mcusToRestart -= 1
        }

        if scan.count == 1 {
            // A non-interleaved scan covers the component's own blocks, one per MCU.
            let c = scan[0]
            let columns = (components[c].width + 7) / 8
            let rows = (components[c].height + 7) / 8
            for row in 0..<rows {
                for column in 0..<columns {
                    try restartIfDue()
                    try block(c, row: row, column: column)
                }
            }
        } else {
            for mcuRow in 0..<mcusPerColumn {
                for mcuColumn in 0..<mcusPerLine {
                    try restartIfDue()
                    for c in scan {
                        for by in 0..<components[c].v {
                            for bx in 0..<components[c].h {
                                try block(
                                    c, row: mcuRow * components[c].v + by,
                                    column: mcuColumn * components[c].h + bx)
                            }
                        }
                    }
                }
            }
        }
        // Leave the reader at the marker that ends the scan.
        reader.position = nextMarker(in: reader.bytes, from: reader.position)
    }

    /// The position of the next marker other than a restart, from `position`.
    static func nextMarker(in bytes: [UInt8], from position: Int) -> Int {
        var at = position
        while at + 1 < bytes.count {
            if bytes[at] == 0xFF {
                let next = bytes[at + 1]
                if next != 0x00 && next != 0xFF && !(0xD0...0xD7).contains(next) {
                    return at
                }
            }
            at += 1
        }
        return bytes.count
    }

    // MARK: Inverse DCT

    /// `jpeg_idct_islow` over every block of a component: its samples on the padded grid,
    /// `blocksPerLine * 8` per row.
    static func inverseDCT(_ component: Component, quant: [Int]) -> [UInt8] {
        let stride = component.blocksPerLine * 8
        var plane = [UInt8](repeating: 0, count: stride * component.blocksPerColumn * 8)
        var workspace = [Int](repeating: 0, count: 64)
        var input = [Int](repeating: 0, count: 64)
        for row in 0..<component.blocksPerColumn {
            for column in 0..<component.blocksPerLine {
                let base = (row * component.blocksPerLine + column) * 64
                for i in 0..<64 { input[i] = Int(component.coefficients[base + i]) }
                idctIslow(input, quant: quant, workspace: &workspace) { y, x, value in
                    plane[(row * 8 + y) * stride + column * 8 + x] = value
                }
            }
        }
        return plane
    }

    static let constBits = 13
    static let pass1Bits = 2

    /// `range_limit[x & RANGE_MASK]` of the post-IDCT table: the value plus 128, clamped, with
    /// the table's wraparound for the 10 low bits.
    @inline(__always)
    static func rangeLimit(_ x: Int) -> UInt8 {
        let masked = x & 1023
        let signed = masked < 512 ? masked : masked - 1024
        return clamp(signed + 128)
    }

    @inline(__always)
    static func descale(_ x: Int, _ n: Int) -> Int { (x + (1 << (n - 1))) >> n }

    static func idctIslow(
        _ input: [Int], quant: [Int], workspace: inout [Int],
        store: (Int, Int, UInt8) -> Void
    ) {
        let fix0298631336 = 2446
        let fix0390180644 = 3196
        let fix0541196100 = 4433
        let fix0765366865 = 6270
        let fix0899976223 = 7373
        let fix1175875602 = 9633
        let fix1501321110 = 12299
        let fix1847759065 = 15137
        let fix1961570560 = 16069
        let fix2053119869 = 16819
        let fix2562915447 = 20995
        let fix3072711026 = 25172
        // Pass 1: columns.
        for col in 0..<8 {
            func q(_ row: Int) -> Int { input[row * 8 + col] * quant[row * 8 + col] }
            if (1..<8).allSatisfy({ input[$0 * 8 + col] == 0 }) {
                let dc = q(0) << pass1Bits
                for row in 0..<8 { workspace[row * 8 + col] = dc }
                continue
            }
            var z2 = q(2)
            var z3 = q(6)
            var z1 = (z2 + z3) * fix0541196100
            var tmp2 = z1 + z3 * -fix1847759065
            var tmp3 = z1 + z2 * fix0765366865
            z2 = q(0)
            z3 = q(4)
            var tmp0 = (z2 + z3) << constBits
            var tmp1 = (z2 - z3) << constBits
            let tmp10 = tmp0 + tmp3
            let tmp13 = tmp0 - tmp3
            let tmp11 = tmp1 + tmp2
            let tmp12 = tmp1 - tmp2
            tmp0 = q(7)
            tmp1 = q(5)
            tmp2 = q(3)
            tmp3 = q(1)
            z1 = tmp0 + tmp3
            z2 = tmp1 + tmp2
            z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * fix1175875602
            tmp0 *= fix0298631336
            tmp1 *= fix2053119869
            tmp2 *= fix3072711026
            tmp3 *= fix1501321110
            z1 *= -fix0899976223
            z2 *= -fix2562915447
            z3 *= -fix1961570560
            z4 *= -fix0390180644
            z3 += z5
            z4 += z5
            tmp0 += z1 + z3
            tmp1 += z2 + z4
            tmp2 += z2 + z3
            tmp3 += z1 + z4
            let shift = constBits - pass1Bits
            workspace[0 * 8 + col] = descale(tmp10 + tmp3, shift)
            workspace[7 * 8 + col] = descale(tmp10 - tmp3, shift)
            workspace[1 * 8 + col] = descale(tmp11 + tmp2, shift)
            workspace[6 * 8 + col] = descale(tmp11 - tmp2, shift)
            workspace[2 * 8 + col] = descale(tmp12 + tmp1, shift)
            workspace[5 * 8 + col] = descale(tmp12 - tmp1, shift)
            workspace[3 * 8 + col] = descale(tmp13 + tmp0, shift)
            workspace[4 * 8 + col] = descale(tmp13 - tmp0, shift)
        }
        // Pass 2: rows.
        for row in 0..<8 {
            let w = row * 8
            if (1..<8).allSatisfy({ workspace[w + $0] == 0 }) {
                let dc = rangeLimit(descale(workspace[w], pass1Bits + 3))
                for x in 0..<8 { store(row, x, dc) }
                continue
            }
            var z2 = workspace[w + 2]
            var z3 = workspace[w + 6]
            var z1 = (z2 + z3) * fix0541196100
            var tmp2 = z1 + z3 * -fix1847759065
            var tmp3 = z1 + z2 * fix0765366865
            var tmp0 = (workspace[w] + workspace[w + 4]) << constBits
            var tmp1 = (workspace[w] - workspace[w + 4]) << constBits
            let tmp10 = tmp0 + tmp3
            let tmp13 = tmp0 - tmp3
            let tmp11 = tmp1 + tmp2
            let tmp12 = tmp1 - tmp2
            tmp0 = workspace[w + 7]
            tmp1 = workspace[w + 5]
            tmp2 = workspace[w + 3]
            tmp3 = workspace[w + 1]
            z1 = tmp0 + tmp3
            z2 = tmp1 + tmp2
            z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * fix1175875602
            tmp0 *= fix0298631336
            tmp1 *= fix2053119869
            tmp2 *= fix3072711026
            tmp3 *= fix1501321110
            z1 *= -fix0899976223
            z2 *= -fix2562915447
            z3 *= -fix1961570560
            z4 *= -fix0390180644
            z3 += z5
            z4 += z5
            tmp0 += z1 + z3
            tmp1 += z2 + z4
            tmp2 += z2 + z3
            tmp3 += z1 + z4
            let shift = constBits + pass1Bits + 3
            store(row, 0, rangeLimit(descale(tmp10 + tmp3, shift)))
            store(row, 7, rangeLimit(descale(tmp10 - tmp3, shift)))
            store(row, 1, rangeLimit(descale(tmp11 + tmp2, shift)))
            store(row, 6, rangeLimit(descale(tmp11 - tmp2, shift)))
            store(row, 2, rangeLimit(descale(tmp12 + tmp1, shift)))
            store(row, 5, rangeLimit(descale(tmp12 - tmp1, shift)))
            store(row, 3, rangeLimit(descale(tmp13 + tmp0, shift)))
            store(row, 4, rangeLimit(descale(tmp13 - tmp0, shift)))
        }
    }

    // MARK: Upsampling and colour

    /// One component brought to the full `width` by `height`, as `jinit_upsampler` chooses:
    /// unchanged at full size, the triangle ("fancy") filters for 2:1 horizontally (when the
    /// component is more than 2 samples wide), vertically, or both, and replication otherwise.
    /// Rows above the first and below the last are the first and last rows, as libjpeg's
    /// context rows replicate them.
    static func upsample(
        _ plane: [UInt8], component: Component, maxH: Int, maxV: Int, width: Int, height: Int
    ) -> [UInt8] {
        let stride = component.blocksPerLine * 8
        let cw = component.width
        let ch = component.height
        var out = [UInt8](repeating: 0, count: width * height)
        func sample(_ y: Int, _ x: Int) -> Int { Int(plane[y * stride + x]) }
        let h2 = component.h * 2 == maxH
        let v2 = component.v * 2 == maxV
        let hSame = component.h == maxH
        let vSame = component.v == maxV
        if hSame && vSame {
            for y in 0..<height {
                for x in 0..<width { out[y * width + x] = plane[y * stride + x] }
            }
        } else if h2 && vSame && cw > 2 {
            var row = [UInt8](repeating: 0, count: cw * 2)
            for y in 0..<height {
                row[0] = UInt8(sample(y, 0))
                row[1] = UInt8((sample(y, 0) * 3 + sample(y, 1) + 2) >> 2)
                if cw > 2 {
                    for c in 1..<(cw - 1) {
                        let value = sample(y, c) * 3
                        row[2 * c] = UInt8((value + sample(y, c - 1) + 1) >> 2)
                        row[2 * c + 1] = UInt8((value + sample(y, c + 1) + 2) >> 2)
                    }
                }
                let last = cw - 1
                row[2 * last] = UInt8((sample(y, last) * 3 + sample(y, last - 1) + 1) >> 2)
                row[2 * last + 1] = UInt8(sample(y, last))
                for x in 0..<width { out[y * width + x] = row[x] }
            }
        } else if hSame && v2 {
            for y in 0..<height {
                let r = y >> 1
                let below = y & 1 == 1
                let other = below ? min(r + 1, ch - 1) : max(r - 1, 0)
                let bias = below ? 2 : 1
                for x in 0..<width {
                    out[y * width + x] = UInt8((sample(r, x) * 3 + sample(other, x) + bias) >> 2)
                }
            }
        } else if h2 && v2 && cw > 2 {
            var sums = [Int](repeating: 0, count: cw)
            var row = [UInt8](repeating: 0, count: cw * 2)
            for y in 0..<height {
                let r = y >> 1
                let other = y & 1 == 1 ? min(r + 1, ch - 1) : max(r - 1, 0)
                for c in 0..<cw { sums[c] = sample(r, c) * 3 + sample(other, c) }
                row[0] = UInt8((sums[0] * 4 + 8) >> 4)
                row[1] = UInt8((sums[0] * 3 + sums[1] + 7) >> 4)
                for c in 1..<(cw - 1) {
                    row[2 * c] = UInt8((sums[c] * 3 + sums[c - 1] + 8) >> 4)
                    row[2 * c + 1] = UInt8((sums[c] * 3 + sums[c + 1] + 7) >> 4)
                }
                let last = cw - 1
                row[2 * last] = UInt8((sums[last] * 3 + sums[last - 1] + 8) >> 4)
                row[2 * last + 1] = UInt8((sums[last] * 4 + 7) >> 4)
                for x in 0..<width { out[y * width + x] = row[x] }
            }
        } else {
            // int_upsample (and h2v1_upsample, h2v2_upsample): replication.
            let hExpand = maxH / component.h
            let vExpand = maxV / component.v
            for y in 0..<height {
                for x in 0..<width {
                    out[y * width + x] = plane[(y / vExpand) * stride + x / hExpand]
                }
            }
        }
        return out
    }

    /// `build_ycc_rgb_table`, 16-bit fixed point.
    struct ColorTables: Sendable {
        var crR = [Int](repeating: 0, count: 256)
        var cbB = [Int](repeating: 0, count: 256)
        var crG = [Int](repeating: 0, count: 256)
        var cbG = [Int](repeating: 0, count: 256)

        static let shared = ColorTables()

        init() {
            func fix(_ x: Double) -> Int { Int(x * Double(1 << 16) + 0.5) }
            let half = 1 << 15
            for i in 0..<256 {
                let x = i - 128
                crR[i] = (fix(1.40200) * x + half) >> 16
                cbB[i] = (fix(1.77200) * x + half) >> 16
                crG[i] = -fix(0.71414) * x
                cbG[i] = -fix(0.34414) * x + half
            }
        }
    }
}
