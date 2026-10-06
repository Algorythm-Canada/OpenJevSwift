// A Swift translation of how Pillow 12.3.0 opens a JPEG before libjpeg-turbo reads it:
// `src/PIL/JpegImagePlugin.py` (`JpegImageFile._open`, the marker handlers `Skip`, `APP`, `COM`,
// `SOF` and `DQT`, `_read_dpi_from_exif`, `_getmp` and `jpeg_factory`), the checks
// `src/PIL/ImageFile.py` (`ImageFile.__init__`) makes of what it read, `src/PIL/Image.py`'s
// decompression bomb check (`open`, `_decompression_bomb_check`) and EXIF reading (`getexif`,
// `Exif.load`, `Exif.__getitem__`), and the TIFF directories of `src/PIL/TiffImagePlugin.py`
// (`ImageFileDirectory_v2` and its value loaders) with the tag lengths of `src/PIL/TiffTags.py`.
//
// The Python Imaging Library (PIL) is Copyright (c) 1997-2011 by Secret Labs AB and Copyright (c)
// 1995-2011 by Fredrik Lundh and contributors; Pillow is Copyright (c) 2010 by Jeffrey 'Alex'
// Clark and contributors. JpegImagePlugin.py is Copyright (c) 1997-2003 by Secret Labs AB and
// Copyright (c) 1995-1996 by Fredrik Lundh; ImageFile.py is Copyright (c) 1997-2004 by Secret Labs
// AB and Copyright (c) 1995-2004 by Fredrik Lundh; Image.py is Copyright (c) 1997-2009 by Secret
// Labs AB and Copyright (c) 1995-2009 by Fredrik Lundh; TiffImagePlugin.py is Copyright (c)
// 1997-2006 by Secret Labs AB and Copyright (c) 1995-1997 by Fredrik Lundh; TiffTags.py is
// Copyright (c) 1999 by Secret Labs AB. MIT-CMU licence: see ThirdPartyLicenses/Pillow-LICENSE
// beside this file, and THIRD_PARTY.md. Translated to Swift and reduced to the checks that make
// `Image.open` raise by the OpenJevSwift contributors, 2026.

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
/// limit. It also reads the EXIF block, for the resolution, and the MPF index, for MPO files, and
/// of the errors their malformed data gives, two escape: an EXIF XResolution too short for what
/// `_read_dpi_from_exif` takes from it (``checkResolution(_:)``), and an MP Entry shorter than the
/// image count (``checkPictureIndex(_:)``). Upstream answers each of those with an exception.
///
/// Not reproduced: Pillow's other formats, which `Image.open` tries when the JPEG plugin raises,
/// and which can open a file built to be both (D-055).
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
        // The JFIF resolution, EXIF block and MPF index, read after the walk.
        var metadata = Metadata()
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
                try checkApplication(code, data, icc: &icc, metadata: &metadata)
            case 0xFE:
                _ = try reader.segment()
            default:
                // SOI, EOI, RST0-7, JPG and JPG0-13: Pillow reads no length for these.
                break
            }
            current = reader.byte()
        }
        // `_open` ends with the EXIF resolution, read where no JFIF segment gave one.
        if !metadata.dpi, let exif = metadata.exif { try checkResolution(exif) }
        guard let size, let layers, size.width > 0, size.height > 0 else {
            throw refusal("the JPEG has no frame of a known size before its first scan")
        }
        // Then `jpeg_factory` reads the MPF index.
        if let index = metadata.mp { try checkPictureIndex(index) }
        let pixels = size.width * size.height
        guard pixels <= RGBImage.maxPixels else {
            throw refusal(
                "the image is \(pixels) pixels; the limit is \(RGBImage.maxPixels) (Pillow's "
                    + "decompression bomb limit)")
        }
        return Header(width: size.width, height: size.height, layers: layers)
    }

    /// What `APP` keeps for the readings after the walk: `info["dpi"]`, `info["exif"]` and
    /// `info["mp"]`.
    struct Metadata {
        /// Whether a JFIF segment gave the resolution (unit 1 or 2), so that
        /// `_read_dpi_from_exif` leaves the EXIF block unread.
        var dpi = false
        /// The EXIF block: the first EXIF segment whole, then each later one after its six-byte
        /// header.
        var exif: [UInt8]?
        /// The last MPF segment, after its four-byte header.
        var mp: ArraySlice<UInt8>?
    }

    /// The checks `APP` makes that raise: a JFIF or Adobe segment shorter than the 16-bit field it
    /// reads at offset 5, and a Photoshop resource block that ends before a resource's name length.
    /// ICC profile fragments are collected for the next frame header, and the JFIF resolution, the
    /// EXIF block and the MPF index for the checks after the walk.
    static func checkApplication(
        _ code: UInt8, _ data: ArraySlice<UInt8>, icc: inout [ArraySlice<UInt8>],
        metadata: inout Metadata
    ) throws(LibjpegTurboDecoder.Refused) {
        func starts(_ prefix: String) -> Bool { data.starts(with: Array(prefix.utf8)) }
        if code == 0xE0, starts("JFIF") {
            guard data.count >= 7 else { throw refusal("a JFIF segment is too short") }
            // The unit and both densities, when the segment holds them.
            if data.count >= 12, [1, 2].contains(data[data.startIndex + 7]) { metadata.dpi = true }
        } else if code == 0xE1, starts("Exif\0\0") {
            if metadata.exif == nil {
                metadata.exif = Array(data)
            } else {
                metadata.exif?.append(contentsOf: data.dropFirst(6))
            }
        } else if code == 0xE1, starts("http://ns.adobe.com/xap/1.0/\0") {
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
        } else if code == 0xE2, starts("MPF\0") {
            metadata.mp = data.dropFirst(4)
        }
    }

    /// `_read_dpi_from_exif`, where no JFIF segment gave the resolution: `getexif` reads the EXIF
    /// block's first directory (``Directory``), and the resolution is `x_resolution[0] /
    /// x_resolution[1]` of its XResolution, or the value itself where that is a number, beside its
    /// ResolutionUnit. A malformed block, a missing tag and a value that is not a number are
    /// caught (Pillow then takes 72 dpi), but not the `IndexError` of an XResolution, of length 1
    /// in `TiffTags`, that decodes to less than the expression reads: one byte (BYTE or UNDEFINED,
    /// count 1), or a string (ASCII, its trailing NUL dropped) that is empty or one digit (any
    /// other character fails `float` first, with a `ValueError`). It escapes as "cannot identify
    /// image file".
    static func checkResolution(_ exif: [UInt8]) throws(LibjpegTurboDecoder.Refused) {
        // `Exif.load` drops every leading header.
        var block = exif[...]
        while block.starts(with: Array("Exif\0\0".utf8)) { block = block.dropFirst(6) }
        guard let directory = Directory(block), directory.entries[0x0128] != nil,
            let resolution = directory.entries[0x011A]
        else { return }
        let value = resolution.value
        switch resolution.type {
        case 1, 7:
            guard value.count == 1 else { return }
            throw refusal(
                "the EXIF XResolution is one byte, and Pillow divides its first by its second "
                    + "(\"IndexError: index out of range\")")
        case 2:
            let text = value.last == 0 ? value.dropLast() : value
            guard text.isEmpty || text.count == 1 && (0x30...0x39).contains(text[text.startIndex])
            else { return }
            throw refusal(
                "the EXIF XResolution is \(text.isEmpty ? "an empty string" : "one digit"), and "
                    + "Pillow divides its first character by its second (\"IndexError: string "
                    + "index out of range\")")
        default:
            return
        }
    }

    /// `_getmp`, which `jpeg_factory` calls to tell an MPO file from a JPEG: it reads the MPF
    /// index as a TIFF directory (``Directory``), then 16 bytes of MP Entry for each image that
    /// NumberOfImages counts. A malformed index leaves the file a JPEG (`jpeg_factory` catches the
    /// `SyntaxError`, `TypeError` and `IndexError` it gives), except for the `struct.error` of an
    /// MP Entry too short for the count, which escapes as "cannot identify image file". That needs
    /// NumberOfImages an integer (SHORT, LONG, SBYTE, SSHORT, SLONG, IFD or LONG8; its first
    /// value), MP Entry a BYTE or UNDEFINED value of fewer whole entries, and no entry ahead of the
    /// missing one whose image data format is other than JPEG's (`_getmp` raises a `SyntaxError`
    /// there). An index read to its end makes an MPO file, whose first image is this JPEG.
    static func checkPictureIndex(_ index: ArraySlice<UInt8>) throws(LibjpegTurboDecoder.Refused) {
        guard let directory = Directory(index), let count = directory.entries[0xB001],
            let images = directory.integer(count), let list = directory.entries[0xB002],
            list.type == 1 || list.type == 7
        else { return }
        // The entries are unpacked big-endian only after the header "MM\0*".
        let bigEndian = index.starts(with: [0x4D, 0x4D, 0x00, 0x2A])
        let whole = list.value.count / 16
        for entry in 0..<min(max(images, 0), whole) {
            // ImageDataFormat, bits 24 to 26 of the entry's first 32-bit field.
            let start = list.value.startIndex + 16 * entry
            guard list.value[bigEndian ? start : start + 3] & 7 == 0 else { return }
        }
        guard images > whole else { return }
        throw refusal(
            "the MPF index's NumberOfImages is \(images) and its MP Entry \(list.value.count) bytes, "
                + "16 per image (\"struct.error: unpack_from requires a buffer of at least "
                + "\(16 * whole + 16) bytes for unpacking 16 bytes at offset \(16 * whole) (actual "
                + "buffer size is \(list.value.count))\")")
    }

    /// A TIFF directory as `ImageFileDirectory_v2` reads one from an EXIF block or an MPF index:
    /// the header (`__init__`), and the entries `load` keeps, before any value is decoded.
    struct Directory {
        typealias Entry = (type: Int, value: ArraySlice<UInt8>)

        /// The byte order the header gives, "MM" or "II".
        var bigEndian: Bool
        /// Each tag's entry: a later entry of a tag replaces an earlier one.
        var entries: [Int: Entry] = [:]

        /// The headers `_accept` takes, but BigTIFF's "II+\0", whose first offset is past the 8
        /// bytes `Exif.load` and `_getmp` read (a `struct.error`).
        static let headers: [[UInt8]] = [
            [0x4D, 0x4D, 0x00, 0x2A], [0x49, 0x49, 0x2A, 0x00], [0x4D, 0x4D, 0x2A, 0x00],
            [0x49, 0x49, 0x00, 0x2A], [0x4D, 0x4D, 0x00, 0x2B],
        ]

        /// The size of a value of each type `load` reads (`_load_dispatch`); it skips entries of
        /// any other type.
        static let unitSizes: [Int: Int] = [
            1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8,
        ]

        /// Reads `data`, or nil where the header raises: fewer than 8 bytes, or a header Pillow
        /// does not read.
        init?(_ data: ArraySlice<UInt8>) {
            let bytes = Array(data)
            guard bytes.count >= 8, Self.headers.contains(Array(bytes[0..<4])) else { return nil }
            bigEndian = bytes[0] == 0x4D
            func number(_ at: Int, _ size: Int) -> Int {
                (0..<size).reduce(0) { $0 << 8 | Int(bytes[at + (bigEndian ? $1 : size - 1 - $1)]) }
            }
            // A read that comes up short (`OSError`) ends `load`, keeping the entries before it.
            var position = number(4, 4)
            guard position + 2 <= bytes.count else { return }
            let tagCount = number(position, 2)
            position += 2
            for _ in 0..<tagCount {
                guard position + 12 <= bytes.count else { return }
                let tag = number(position, 2)
                let type = number(position + 2, 2)
                let count = number(position + 4, 4)
                let field = position + 8
                position += 12
                guard let unit = Self.unitSizes[type] else { continue }
                let size = count * unit
                let value: ArraySlice<UInt8>
                if size > 4 {
                    let offset = number(field, 4)
                    guard offset + size <= bytes.count else { return }
                    value = bytes[offset..<(offset + size)]
                } else {
                    value = bytes[field..<(field + size)]
                }
                // No value (count 0): the entry is skipped.
                guard !value.isEmpty else { continue }
                entries[tag] = (type, value)
            }
        }

        /// The first value of an entry of a type `load` decodes to integers, as `_setitem` keeps
        /// it for a tag of length 1; nil for the other types, which `range` refuses (`TypeError`).
        func integer(_ entry: Entry) -> Int? {
            let sizes = [3: 2, 4: 4, 6: 1, 8: 2, 9: 4, 13: 4, 16: 8]
            guard let size = sizes[entry.type] else { return nil }
            let start = entry.value.startIndex
            let bits = (0..<size).reduce(UInt64(0)) {
                $0 << 8 | UInt64(entry.value[start + (bigEndian ? $1 : size - 1 - $1)])
            }
            switch entry.type {
            case 6: return Int(Int8(truncatingIfNeeded: bits))
            case 8: return Int(Int16(truncatingIfNeeded: bits))
            case 9: return Int(Int32(truncatingIfNeeded: bits))
            default: return Int(exactly: bits) ?? Int.max
            }
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
