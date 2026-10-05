// A Swift translation of how Pillow 12.3.0 opens a JPEG before libjpeg-turbo reads it:
// `src/PIL/JpegImagePlugin.py` (`JpegImageFile._open` and the marker handlers `Skip`, `APP`, `COM`,
// `SOF` and `DQT`), the checks `src/PIL/ImageFile.py` (`ImageFile.__init__`) makes of what it
// read, and `src/PIL/Image.py`'s decompression bomb check (`open`, `_decompression_bomb_check`).
//
// The Python Imaging Library (PIL) is Copyright (c) 1997-2011 by Secret Labs AB and Copyright (c)
// 1995-2011 by Fredrik Lundh and contributors; Pillow is Copyright (c) 2010 by Jeffrey 'Alex'
// Clark and contributors. JpegImagePlugin.py is Copyright (c) 1997-2003 by Secret Labs AB and
// Copyright (c) 1995-1996 by Fredrik Lundh; ImageFile.py is Copyright (c) 1997-2004 by Secret Labs
// AB and Copyright (c) 1995-2004 by Fredrik Lundh; Image.py is Copyright (c) 1997-2009 by Secret
// Labs AB and Copyright (c) 1995-2009 by Fredrik Lundh. MIT-CMU licence: see
// ThirdPartyLicenses/Pillow-LICENSE beside this file, and THIRD_PARTY.md. Translated to Swift and
// reduced to the checks that make `Image.open` raise by the OpenJevSwift contributors, 2026.

/// The JPEG headers as Pillow reads them in Python, up to the first scan, before libjpeg-turbo
/// sees the file.
///
/// `Image.open` walks the markers itself to learn the size and mode, and raises where its own
/// reading fails, whatever libjpeg-turbo would make of the file: a marker code below 0xC0 (TEM
/// and the reserved codes included), a segment that runs past the end of the file, a frame that
/// is not 8-bit or has other than 1, 3 or 4 components, a quantization table segment that ends
/// inside a table, a JFIF or Adobe segment too short for the field it reads first, an ICC profile
/// fragment ahead of the frame too short to hold its sequence number, a Photoshop resource cut
/// off before its name, no frame before the first scan, and an image past its decompression bomb
/// limit. Upstream answers each of those with an exception.
///
/// Not reproduced: Pillow's reading of EXIF (for the resolution) and of an MPF segment (for MPO
/// files), whose own parsers raise on some malformed data; this reads neither. Nor does it try
/// Pillow's other formats, as `Image.open` does when the JPEG plugin raises, which can open a
/// file built to be both (D-055).
enum PillowJPEGHeader {
    /// What `JpegImageFile._open` read from the last frame header before the first scan.
    struct Header: Equatable {
        var width: Int
        var height: Int
        /// The frame's component count, which sets Pillow's mode: 1 for L, 3 for RGB, 4 for CMYK.
        var layers: Int
    }

    /// Reads the headers as `Image.open` does.
    ///
    /// - Throws: A ``LibjpegTurboDecoder/Refused`` where `Image.open` raises.
    static func read(_ bytes: [UInt8]) throws(LibjpegTurboDecoder.Refused) -> Header {
        var reader = Reader(bytes: bytes)
        // `_accept` read the first three bytes, FF D8 FF; the walk starts at the third.
        guard LibjpegTurboDecoder.isJPEG(bytes) else { throw refusal("the data is not a JPEG") }
        reader.position = 3
        var current: UInt8? = 0xFF
        var size: (width: Int, height: Int)?
        var layers: Int?
        // ICC profile fragments seen since the last frame header.
        var icc: [ArraySlice<UInt8>] = []
        walk: while true {
            guard let byte = current else {
                throw refusal("the JPEG ends before its first scan")
            }
            guard byte == 0xFF else {
                // Junk between markers is skipped a byte at a time.
                current = reader.byte()
                continue
            }
            guard let code = reader.byte() else {
                throw refusal("the JPEG ends inside a marker before its first scan")
            }
            switch code {
            case 0x00:
                current = reader.byte()
                continue walk
            case 0xFF:
                current = 0xFF
                continue walk
            case 0x01...0xBF:
                throw refusal(
                    "marker 0x\(hex(code)) before the first scan is not one Pillow knows "
                        + "(\"no marker found\")")
            case 0xC0...0xC3, 0xC5...0xC7, 0xC9...0xCB, 0xCD...0xCF, 0xDE:
                let segment = try reader.segment()
                guard segment.count >= 5 else {
                    throw refusal("a frame header is too short to hold the image size")
                }
                size = (be16(segment, 3), be16(segment, 1))
                let bits = Int(segment[segment.startIndex])
                guard bits == 8 else {
                    throw refusal("the JPEG has \(bits)-bit samples; Pillow reads only 8-bit ones")
                }
                guard segment.count >= 6 else {
                    throw refusal("a frame header is too short to hold its component count")
                }
                let count = Int(segment[segment.startIndex + 5])
                guard [1, 3, 4].contains(count) else {
                    throw refusal("the JPEG has \(count) components; Pillow reads 1, 3 or 4")
                }
                layers = count
                if !icc.isEmpty {
                    // `icclist.sort()` and `icclist[0][13]`: the first fragment in byte order must
                    // hold the sequence count.
                    let first = icc.min { $0.lexicographicallyPrecedes($1) }!
                    guard first.count >= 14 else {
                        throw refusal("an ICC profile fragment is too short to number itself")
                    }
                    icc = []
                }
                // The component records are read three bytes at a time to the segment's end.
                guard (segment.count - 6) % 3 == 0 else {
                    throw refusal("a frame header ends inside a component record")
                }
            case 0xDB:
                var table = try reader.segment()
                while let first = table.first {
                    let length = 1 + 64 * (first >> 4 == 0 ? 1 : 2)
                    guard table.count >= length else {
                        throw refusal("a quantization table segment ends inside a table")
                    }
                    table = table.dropFirst(length)
                }
            case 0xC4, 0xCC, 0xDA, 0xDC, 0xDD, 0xDF:
                _ = try reader.segment()
                if code == 0xDA { break walk }
            case 0xE0...0xEF:
                let data = try reader.segment()
                try checkApplication(code, data, icc: &icc)
            case 0xFE:
                _ = try reader.segment()
            default:
                // SOI, EOI, RST0-7, JPG and JPG0-13: Pillow reads no length for these.
                break
            }
            current = reader.byte()
        }
        guard let size, let layers, size.width > 0, size.height > 0 else {
            throw refusal("the JPEG has no frame of a known size before its first scan")
        }
        let pixels = size.width * size.height
        guard pixels <= RGBImage.maxPixels else {
            throw refusal(
                "the image is \(pixels) pixels; the limit is \(RGBImage.maxPixels) (Pillow's "
                    + "decompression bomb limit)")
        }
        return Header(width: size.width, height: size.height, layers: layers)
    }

    /// The checks `APP` makes that raise: a JFIF or Adobe segment shorter than the 16-bit field it
    /// reads at offset 5, and a Photoshop resource block that ends before a resource's name length.
    /// ICC profile fragments are collected for the next frame header.
    static func checkApplication(
        _ code: UInt8, _ data: ArraySlice<UInt8>, icc: inout [ArraySlice<UInt8>]
    ) throws(LibjpegTurboDecoder.Refused) {
        func starts(_ prefix: String) -> Bool { data.starts(with: Array(prefix.utf8)) }
        if code == 0xE0, starts("JFIF") {
            guard data.count >= 7 else { throw refusal("a JFIF segment is too short") }
        } else if code == 0xE1, starts("Exif\0\0") || starts("http://ns.adobe.com/xap/1.0/\0") {
            return
        } else if code == 0xE2, starts("FPXR\0") {
            return
        } else if code == 0xE2, starts("ICC_PROFILE\0") {
            icc.append(data)
        } else if code == 0xED, starts("Photoshop 3.0\0") {
            // A resource is "8BIM", a 16-bit code, a name length and name, padded to even, a
            // 32-bit size and the data, padded to even. Short data stops the loop quietly
            // (`except struct.error`), except where the name length is read.
            let bytes = Array(data)
            let signature = Array("8BIM".utf8)
            var offset = 14
            while offset + 4 <= bytes.count, Array(bytes[offset..<(offset + 4)]) == signature {
                offset += 4
                guard offset + 2 <= bytes.count else { return }
                let resource = Int(bytes[offset]) << 8 | Int(bytes[offset + 1])
                offset += 2
                guard offset < bytes.count else {
                    throw refusal("a Photoshop resource block ends before a resource's name")
                }
                offset += 1 + Int(bytes[offset])
                offset += offset & 1
                guard offset + 4 <= bytes.count else { return }
                let size =
                    Int(bytes[offset]) << 24 | Int(bytes[offset + 1]) << 16
                    | Int(bytes[offset + 2]) << 8 | Int(bytes[offset + 3])
                offset += 4
                // ResolutionInfo reads 14 bytes of its data.
                if resource == 0x03ED, min(size, max(0, bytes.count - offset)) < 14 { return }
                offset += size
                offset += offset & 1
            }
        } else if code == 0xEE, starts("Adobe") {
            guard data.count >= 7 else { throw refusal("an Adobe segment is too short") }
        }
    }

    /// `BytesIO` reads over the file.
    struct Reader {
        let bytes: [UInt8]
        var position = 0

        /// `fp.read(1)`: the next byte, or nil at the end.
        mutating func byte() -> UInt8? {
            guard position < bytes.count else { return nil }
            defer { position += 1 }
            return bytes[position]
        }

        /// A handler's `n = i16(fp.read(2)) - 2` and `ImageFile._safe_read(fp, n)`: the segment's
        /// body, empty when its length is below 3.
        mutating func segment() throws(LibjpegTurboDecoder.Refused) -> ArraySlice<UInt8> {
            guard position + 2 <= bytes.count else {
                throw refusal("the JPEG ends inside a segment's length")
            }
            let length = Int(bytes[position]) << 8 | Int(bytes[position + 1])
            position += 2
            let count = length - 2
            guard count > 0 else { return [] }
            guard position + count <= bytes.count else {
                throw refusal("a segment runs past the end of the JPEG (\"Truncated File Read\")")
            }
            defer { position += count }
            return bytes[position..<(position + count)]
        }
    }

    static func be16(_ data: ArraySlice<UInt8>, _ offset: Int) -> Int {
        Int(data[data.startIndex + offset]) << 8 | Int(data[data.startIndex + offset + 1])
    }

    static func hex(_ byte: UInt8) -> String {
        let digits = Array("0123456789ABCDEF")
        return String([digits[Int(byte >> 4)], digits[Int(byte & 15)]])
    }

    static func refusal(_ message: String) -> LibjpegTurboDecoder.Refused {
        LibjpegTurboDecoder.Refused("Pillow does not open the JPEG: \(message)")
    }
}
