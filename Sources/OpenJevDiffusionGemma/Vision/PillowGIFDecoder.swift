// A Swift translation of how Pillow 12.3.0 reads a GIF's first frame for
// `Image.open(...).convert("RGB")`: `src/PIL/GifImagePlugin.py` (`GifImageFile._open`,
// `_is_palette_needed`, `data`, `_seek(0)` and `load_prepare`), the frame-0 path of
// `src/PIL/ImageFile.py` (`ImageFile.load`), `src/libImaging/GifDecode.c` (`ImagingGifDecode`),
// and the palette lookup of `src/libImaging/Convert.c` (`p2rgb`, `l2rgb`) with
// `src/libImaging/Palette.c` (`ImagingPaletteNew`).
//
// The Python Imaging Library (PIL) is Copyright (c) 1997-2011 by Secret Labs AB and Copyright (c)
// 1995-2011 by Fredrik Lundh and contributors; Pillow is Copyright (c) 2010 by Jeffrey 'Alex'
// Clark and contributors. GifImagePlugin.py is Copyright (c) 1997-2004 by Secret Labs AB and
// Copyright (c) 1995-2004 by Fredrik Lundh; GifDecode.c is Copyright (c) Secret Labs AB 1997-99
// and Copyright (c) Fredrik Lundh 1995-97. MIT-CMU licence: see ThirdPartyLicenses/Pillow-LICENSE
// beside this file, and THIRD_PARTY.md. Translated to Swift and reduced to the first frame by the
// OpenJevSwift contributors, 2026.

/// Decodes a GIF's first frame to the RGB bytes `PIL.Image.open(...).convert("RGB")` gives.
///
/// ImageIO hands a GIF's first frame over as RGBA and leaves the pixels it treats as transparent,
/// and the logical screen outside the frame, as (0, 0, 0, 0). Pillow opens the frame in palette
/// mode instead, and `convert("RGB")` drops alpha by looking each index up in the palette. So
/// where ImageIO has nothing, Pillow has a colour:
///
/// - The canvas is the logical screen, grown to hold the frame if the frame reaches past it, and
///   is filled with the graphic control extension's transparent index, or with index 0 when there
///   is none, before the frame is decoded into its rectangle. Transparent pixels keep their index.
/// - The palette is the frame's local colour table, else the global one. A table that is the
///   identity grey ramp (entry *i* is (*i*, *i*, *i*)) is dropped, and the indices are then grey
///   levels, unless the frame's local table is the ramp and a global table exists: Pillow then
///   looks the indices up in the global table. Indices past the table's end are black.
/// - LZW data that breaks, or has a code size above 12, makes Pillow raise, and so does data that
///   runs out before the frame is full. An end code does not stop it while the file still has
///   bytes `ImageFile.load` has not read (it reads 65,536 at a time): it decodes on into them.
///   A header cut short, a frame of zero width or height and a canvas past
///   ``RGBImage/maxPixels`` are refused too.
enum PillowGIFDecoder {
    /// A GIF Pillow does not open or load: upstream's `ImagePrompt.pil` raises on it.
    struct Refused: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    /// True when `bytes` start with a GIF signature, Pillow's `_accept`.
    static func isGIF(_ bytes: [UInt8]) -> Bool {
        bytes.starts(with: Array("GIF87a".utf8)) || bytes.starts(with: Array("GIF89a".utf8))
    }

    /// Pillow's `ImageFile.MAXBLOCK`: how many bytes `ImageFile.load` reads at a time.
    static let readSize = 65_536

    /// `fp.read` and `GifImageFile.data` over the bytes.
    struct Reader {
        let bytes: [UInt8]
        var position = 0

        /// Up to `count` bytes, fewer at the end, as `BytesIO.read` gives them.
        mutating func read(_ count: Int) -> ArraySlice<UInt8> {
            let end = min(bytes.count, position + count)
            defer { position = end }
            return bytes[position..<end]
        }

        /// One data sub-block: nil when the size byte is 0 or missing, else up to that many bytes.
        mutating func data() -> ArraySlice<UInt8>? {
            guard let size = read(1).first, size > 0 else { return nil }
            return read(Int(size))
        }
    }

    /// A colour table as `_is_palette_needed` judges it.
    enum Table {
        /// A table that is the identity grey ramp, which Pillow drops (`palette = False`).
        case identity
        /// Any other table: its bytes, three per entry.
        case colors([UInt8])
    }

    /// The little-endian 16-bit value at `offset` of a slice, as `_binary.i16le` reads it.
    static func le16(_ bytes: ArraySlice<UInt8>, _ offset: Int) -> Int {
        let base = bytes.startIndex + offset
        return Int(bytes[base]) | Int(bytes[base + 1]) << 8
    }

    /// `_is_palette_needed`: false only for the identity grey ramp. A table cut short in the middle
    /// of an entry raises `IndexError` there, as Python's chained comparison does.
    static func classify(_ bytes: ArraySlice<UInt8>) throws(Refused) -> Table {
        let p = Array(bytes)
        var i = 0
        while i < p.count {
            if i / 3 != Int(p[i]) { return .colors(p) }
            guard i + 1 < p.count else { throw Refused("the GIF's colour table is cut short") }
            if p[i] != p[i + 1] { return .colors(p) }
            guard i + 2 < p.count else { throw Refused("the GIF's colour table is cut short") }
            if p[i + 1] != p[i + 2] { return .colors(p) }
            i += 3
        }
        return .identity
    }

    /// Refuses a canvas past ``RGBImage/maxPixels``, as `Image._decompression_bomb_check` does.
    static func checkSize(width: Int, height: Int) throws(Refused) {
        let pixels = max(1, width) * max(1, height)
        if pixels > RGBImage.maxPixels {
            throw Refused(
                "the GIF is \(pixels) pixels; the limit is \(RGBImage.maxPixels) (Pillow's "
                    + "decompression bomb limit)")
        }
    }

    /// What `GifImageFile._open` and `_seek(0)` learn from the headers.
    struct Frame {
        var canvasWidth: Int
        var canvasHeight: Int
        /// The frame's rectangle, `(x0, y0, x1, y1)`.
        var x0: Int
        var y0: Int
        var x1: Int
        var y1: Int
        var interlaced: Bool
        /// The transparent index of the frame's graphic control extension.
        var transparency: Int?
        /// The LZW minimum code size.
        var bits: Int
        /// Where the frame's image data starts.
        var dataOffset: Int
        /// The table `convert("RGB")` looks indices up in, or nil for grey levels.
        var palette: [UInt8]?
    }

    /// Reads the logical screen descriptor, the global colour table and the blocks up to the first
    /// image descriptor, as `GifImageFile._open` and `_seek(0)` do.
    static func firstFrame(in bytes: [UInt8]) throws(Refused) -> Frame {
        var reader = Reader(bytes: bytes)
        // _open: the screen, then the global palette.
        let screen = reader.read(13)
        guard isGIF(Array(screen)) else { throw Refused("not a GIF file") }
        guard screen.count >= 11 else { throw Refused("the GIF's screen descriptor is cut short") }
        var width = le16(screen, 6)
        var height = le16(screen, 8)
        let screenFlags = screen[screen.startIndex + 10]
        var global: Table?
        if screenFlags & 128 != 0 {
            guard screen.count >= 12 else {
                throw Refused("the GIF's screen descriptor is cut short")
            }
            global = try classify(reader.read(3 << (Int(screenFlags & 7) + 1)))
        }

        // _seek(0): the blocks before the first image.
        var s = reader.read(1)
        if s.isEmpty || s.first == 0x3B { throw Refused("no more images in GIF file") }
        var local: Table?
        var transparency: Int?
        var interlaced: Bool?
        var extent = (x0: 0, y0: 0, x1: 0, y1: 0)
        var bits = 0
        var dataOffset = 0
        blocks: while true {
            if s.isEmpty { s = reader.read(1) }
            guard let introducer = s.first, introducer != 0x3B else { break }
            if introducer == 0x21 {
                // An extension: its label, then its first data sub-block.
                guard let label = reader.read(1).first else {
                    throw Refused("a GIF extension is cut short")
                }
                let block = reader.data()
                if label == 0xF9, let block {
                    // The graphic control extension.
                    guard let flags = block.first else {
                        throw Refused("a graphic control extension is cut short")
                    }
                    if flags & 1 != 0 {
                        guard block.count >= 4 else {
                            throw Refused("a graphic control extension is cut short")
                        }
                        transparency = Int(block[block.startIndex + 3])
                    }
                    // The duration, i16(block, 1), is read whether or not it is used.
                    guard block.count >= 3 else {
                        throw Refused("a graphic control extension is cut short")
                    }
                } else if label == 0xFE {
                    // A comment: its sub-blocks up to the terminator, and nothing after it.
                    var comment = block
                    while let part = comment, !part.isEmpty { comment = reader.data() }
                    s = []
                    continue blocks
                } else if label == 0xFF, let block,
                    block.starts(with: Array("NETSCAPE2.0".utf8))
                {
                    // The looping extension: its second sub-block holds the loop count.
                    _ = reader.data()
                }
                // Skip the remaining sub-blocks. When the first one was already the terminator,
                // this reads the next block's bytes as sub-blocks, as Pillow does.
                while let part = reader.data(), !part.isEmpty {}
            } else if introducer == 0x2C {
                // The image descriptor.
                let descriptor = reader.read(9)
                guard descriptor.count >= 9 else {
                    throw Refused("the GIF's image descriptor is cut short")
                }
                let x0 = le16(descriptor, 0)
                let y0 = le16(descriptor, 2)
                extent = (x0, y0, x0 + le16(descriptor, 4), y0 + le16(descriptor, 6))
                if extent.x1 > width || extent.y1 > height {
                    width = max(extent.x1, width)
                    height = max(extent.y1, height)
                    try checkSize(width: width, height: height)
                }
                let flags = descriptor[descriptor.startIndex + 8]
                interlaced = flags & 64 != 0
                if flags & 128 != 0 {
                    local = try classify(reader.read(3 << (Int(flags & 7) + 1)))
                }
                guard let codeSize = reader.read(1).first else {
                    throw Refused("the GIF's image data is missing")
                }
                bits = Int(codeSize)
                dataOffset = reader.position
                break blocks
            }
            s = []
        }
        guard let interlaced else { throw Refused("image not found in GIF frame") }
        // ImageFile.__init__ and Image.open check the canvas.
        guard width > 0, height > 0 else { throw Refused("the GIF has no size") }
        try checkSize(width: width, height: height)

        // The frame's palette is its local table, else the global one; a table that is the grey
        // ramp is none, and the image is then mode L, whose indices are grey levels. With a ramp
        // for its local table and a global table, the frame is mode L but keeps a copy of the
        // global palette, which Image.load puts on the image after decoding, making it P.
        var palette: [UInt8]?
        switch (local, global) {
        case (.colors(let colors)?, _): palette = colors
        case (_, .colors(let colors)?): palette = colors
        default: palette = nil
        }
        return Frame(
            canvasWidth: width, canvasHeight: height, x0: extent.x0, y0: extent.y0,
            x1: extent.x1, y1: extent.y1, interlaced: interlaced, transparency: transparency,
            bits: bits, dataOffset: dataOffset, palette: palette)
    }

    /// Decodes the first frame as `Image.open(...).convert("RGB")` does.
    ///
    /// - Throws: ``Refused`` for every GIF on which Pillow raises before `convert` returns.
    static func decode(_ bytes: [UInt8]) throws(Refused) -> RGBImage {
        let frame = try firstFrame(in: bytes)
        // The decoder's setimage refuses an empty tile. Pillow makes the canvas first; nothing
        // else differs if this one is not made.
        guard frame.x1 > frame.x0, frame.y1 > frame.y0 else {
            throw Refused("the GIF's first frame has no pixels (tile cannot extend outside image)")
        }
        // load_prepare: the canvas, filled with the transparent index, or zeroed.
        let width = frame.canvasWidth
        var canvas = [UInt8](
            repeating: UInt8(truncatingIfNeeded: frame.transparency ?? 0),
            count: width * frame.canvasHeight)
        var decoder = LZWDecoder(
            bits: frame.bits, interlaced: frame.interlaced, xsize: frame.x1 - frame.x0,
            ysize: frame.y1 - frame.y0, xoff: frame.x0, yoff: frame.y0, stride: width)
        // ImageFile.load: read MAXBLOCK bytes at a time from the data's start, hand the decoder
        // what it has not consumed, and stop when it returns a negative count.
        var consumed = frame.dataOffset
        var read = frame.dataOffset
        while true {
            let end = min(bytes.count, read + readSize)
            guard end > read else {
                throw Refused("the GIF is cut short (image file is truncated)")
            }
            read = end
            let count = decoder.decode(bytes, from: consumed, to: read, into: &canvas)
            if count < 0 { break }
            consumed += count
        }
        if decoder.errcode < 0 {
            throw Refused(
                decoder.errcode == LZWDecoder.configError
                    ? "the GIF's LZW code size is above 12"
                    : "the GIF's image data is broken (broken data stream)")
        }

        // convert("RGB"): p2rgb through a palette whose entries past the table are black, or
        // l2rgb's grey.
        var pixels = [UInt8](repeating: 0, count: canvas.count * 3)
        if let table = frame.palette {
            var lookup = [UInt8](repeating: 0, count: 256 * 3)
            let entries = min(table.count / 3, 256)
            lookup.replaceSubrange(0..<(entries * 3), with: table[0..<(entries * 3)])
            lookup.withUnsafeBufferPointer { lookup in
                pixels.withUnsafeMutableBufferPointer { out in
                    for (index, value) in canvas.enumerated() {
                        let entry = Int(value) * 3
                        out[index * 3] = lookup[entry]
                        out[index * 3 + 1] = lookup[entry + 1]
                        out[index * 3 + 2] = lookup[entry + 2]
                    }
                }
            }
        } else {
            pixels.withUnsafeMutableBufferPointer { out in
                for (index, value) in canvas.enumerated() {
                    out[index * 3] = value
                    out[index * 3 + 1] = value
                    out[index * 3 + 2] = value
                }
            }
        }
        return RGBImage(width: width, height: frame.canvasHeight, pixels: pixels)
    }

    /// `ImagingGifDecode` and its `GIFDECODERSTATE`, writing the frame's indices into the canvas.
    /// It is suspendable as the C is: a call consumes only whole data sub-blocks and returns how
    /// many bytes it consumed, or -1 when the frame is full (``errcode`` 0) or broken.
    struct LZWDecoder {
        static let codeBits = 12
        static let tableSize = 1 << codeBits
        static let brokenError = -2
        static let overrunError = -1
        static let configError = -8

        // Configuration.
        let bits: Int
        var interlace: Int

        // The tile: its size, its offset in the canvas and the canvas's row length.
        let xsize: Int
        let ysize: Int
        let xoff: Int
        let yoff: Int
        let stride: Int

        // The decoder state, zeroed as calloc leaves it.
        var state = 0
        var x = 0
        var y = 0
        var errcode = 0
        var step = 0
        var bitbuffer = 0
        var bitcount = 0
        var blocksize = 0
        var codesize = 0
        var codemask = 0
        var clear = 0
        var end = 0
        var lastcode = 0
        var lastdata: UInt8 = 0
        var bufferindex = 0
        var buffer = [UInt8](repeating: 0, count: tableSize)
        var link = [UInt16](repeating: 0, count: tableSize)
        var data = [UInt8](repeating: 0, count: tableSize)
        var next = 0

        init(
            bits: Int, interlaced: Bool, xsize: Int, ysize: Int, xoff: Int, yoff: Int, stride: Int
        ) {
            self.bits = bits
            self.interlace = interlaced ? 1 : 0
            self.xsize = xsize
            self.ysize = ysize
            self.xoff = xoff
            self.yoff = yoff
            self.stride = stride
        }

        /// The `NEWLINE` macro: the next row, or the next interlace pass. False when the frame is
        /// full, where the C returns -1.
        mutating func newline(_ out: inout Int) -> Bool {
            x = 0
            y += step
            while y >= ysize {
                switch interlace {
                case 1:
                    y = 4
                    interlace = 2
                case 2:
                    step = 4
                    y = 2
                    interlace = 3
                case 3:
                    step = 2
                    y = 1
                    interlace = 0
                default:
                    return false
                }
            }
            out = (y + yoff) * stride + xoff
            return true
        }

        /// One call of `ImagingGifDecode` on `bytes[start..<end]`.
        mutating func decode(
            _ bytes: [UInt8], from start: Int, to end: Int, into canvas: inout [UInt8]
        ) -> Int {
            var ptr = start
            if state == 0 {
                guard bits >= 0, bits <= Self.codeBits else {
                    errcode = Self.configError
                    return -1
                }
                clear = 1 << bits
                self.end = clear + 1
                if interlace != 0 {
                    interlace = 1
                    step = 8
                } else {
                    step = 1
                }
                state = 1
            }
            var out = (y + yoff) * stride + xoff + x

            while true {
                if state == 1 {
                    next = clear + 2
                    codesize = bits + 1
                    codemask = (1 << codesize) - 1
                    bufferindex = Self.tableSize
                    state = 2
                }

                // The string to write: one byte (lastdata), or the rest of a string the last code
                // expanded to, waiting at the buffer's right end.
                var single = true
                var stringStart = 0
                if bufferindex < Self.tableSize {
                    single = false
                    stringStart = bufferindex
                    bufferindex = Self.tableSize
                } else {
                    while bitcount < codesize {
                        if blocksize > 0 {
                            bitbuffer |= Int(bytes[ptr]) << bitcount
                            ptr += 1
                            blocksize -= 1
                            bitcount += 8
                        } else {
                            // A new sub-block, started only when all of it is here.
                            guard ptr < end else { return ptr - start }
                            let size = Int(bytes[ptr])
                            guard end - ptr >= size + 1 else { return ptr - start }
                            blocksize = size
                            ptr += 1
                        }
                    }
                    var c = bitbuffer & codemask
                    bitbuffer >>= codesize
                    bitcount -= codesize

                    if c == clear {
                        if state != 2 { state = 1 }
                        continue
                    }
                    if c == self.end { break }

                    if state == 2 {
                        // The first code after a clear is used as it is.
                        guard c <= clear else {
                            errcode = Self.brokenError
                            return -1
                        }
                        lastdata = UInt8(truncatingIfNeeded: c)
                        lastcode = c
                        state = 3
                    } else {
                        let thiscode = c
                        guard c <= next else {
                            errcode = Self.brokenError
                            return -1
                        }
                        if c == next {
                            guard bufferindex > 0 else {
                                errcode = Self.brokenError
                                return -1
                            }
                            bufferindex -= 1
                            buffer[bufferindex] = lastdata
                            c = lastcode
                        }
                        while c >= clear {
                            guard bufferindex > 0, c < Self.tableSize else {
                                errcode = Self.brokenError
                                return -1
                            }
                            bufferindex -= 1
                            buffer[bufferindex] = data[c]
                            c = Int(link[c])
                        }
                        lastdata = UInt8(truncatingIfNeeded: c)
                        if next < Self.tableSize {
                            data[next] = UInt8(truncatingIfNeeded: c)
                            link[next] = UInt16(truncatingIfNeeded: lastcode)
                            if next == codemask && codesize < Self.codeBits {
                                codesize += 1
                                codemask = (1 << codesize) - 1
                            }
                            next += 1
                        }
                        lastcode = thiscode
                    }
                }

                guard y < ysize else {
                    errcode = Self.overrunError
                    return -1
                }
                // Pixel by pixel; the C's shortcuts write the same bytes. At frame 0 there is no
                // transparency to skip, so every index is written.
                let count = single ? 1 : Self.tableSize - stringStart
                for offset in 0..<count {
                    canvas[out] = single ? lastdata : buffer[stringStart + offset]
                    out += 1
                    x += 1
                    if x >= xsize, !newline(&out) { return -1 }
                }
            }
            return ptr - start
        }
    }
}
