// A Swift translation of libjpeg-turbo 3.1.4.1's default decompression path, the one Pillow
// 12.3.0 runs, with the reading Pillow's JPEG decoder does around it. Translated to Swift,
// restructured to decode a whole image in memory and reduced to the 8-bit Huffman-coded path,
// with how libjpeg-turbo reads arithmetic-coded and lossless JPEGs short of decoding them, by
// the OpenJevSwift contributors, 2026. This software is based in part on the work of the
// Independent JPEG Group. Below are the files it draws on, each with its copyright block as
// libjpeg-turbo has it. The README.ijg those blocks name is ThirdPartyLicenses/libjpeg-turbo-README.ijg
// beside this file, next to libjpeg-turbo-LICENSE.md; THIRD_PARTY.md lists the files. The inverse
// DCT, a translation of the Arm Neon code, is in LibjpegTurboNEONIDCT.swift with its own notice.
//
// jdmarker.c (the marker segments, the checks it makes of them, restart markers and
// `jpeg_resync_to_restart`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1998, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2012, 2015, 2022, 2024, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdinput.c (`initial_setup`, `per_scan_setup`, `latch_quant_tables`, `consume_markers`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1997, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2010, 2016, 2018, 2022, 2024, D. R. Commander.
//   Copyright (C) 2015, Google, Inc.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdapimin.c (`default_decompress_parms`, `jpeg_read_header`, `jpeg_finish_decompress`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1994-1998, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2016, 2022, 2024, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdapistd.c (`jpeg_start_decompress`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1994-1996, Thomas G. Lane.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2010, 2015-2020, 2022-2026, D. R. Commander.
//   Copyright (C) 2015, Google, Inc.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdhuff.c (baseline Huffman decoding, its fast and slow paths, `jpeg_make_d_derived_tbl`,
// `jpeg_fill_bit_buffer`, `jpeg_huff_decode`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1997, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2009-2011, 2016, 2018-2019, 2022, D. R. Commander.
//   Copyright (C) 2018, Matthias Räncker.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdhuff.h (the bit-reading and Huffman-decoding macros):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1997, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2010-2011, 2015-2016, 2021, D. R. Commander.
//   Copyright (C) 2018, Matthias Räncker.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jstdhuff.c (the standard Huffman tables of a sequential JPEG that defines none):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1998, Thomas G. Lane.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2013, 2022, 2024, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdlhuff.c (the DC tables a lossless scan checks before its data, and how `decode_mcus` and
// `process_restart` read the data):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1997, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2022, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdlossls.c (`start_pass_lossless`'s checks of a lossless scan):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1998, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2022, 2024, 2026, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jddiffct.c (the restart interval a lossless scan allows, its restarts by MCU row, and the
// whole-image arrays it does not have zeroed):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1994-1997, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2022, 2024, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdphuff.c (progressive Huffman decoding, `start_pass_phuff_decoder`'s checks):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1995-1997, Thomas G. Lane.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2015-2016, 2018-2022, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdarith.c (`start_pass`'s checks of an arithmetic-coded scan, and `get_byte` and
// `process_restart`, which cannot suspend, and how far a segment's first decision reads):
//   This file was part of the Independent JPEG Group's software:
//   Developed 1997-2015 by Guido Vollbeding.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2015-2020, 2022, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdcoefct.c (`decompress_onepass`, `consume_data`, block smoothing: `smoothing_ok` and
// `decompress_smooth_data`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1994-1997, Thomas G. Lane.
//   libjpeg-turbo Modifications:
//   Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
//   Copyright (C) 2010, 2015-2016, 2019-2020, 2022-2024, D. R. Commander.
//   Copyright (C) 2015, 2020, Google, Inc.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jddctmgr.c (the multiplier tables, all zero for a component no scan has reached):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1994-1996, Thomas G. Lane.
//   Modified 2002-2010 by Guido Vollbeding.
//   libjpeg-turbo Modifications:
//   Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
//   Copyright (C) 2010, 2015, 2022, 2026, D. R. Commander.
//   Copyright (C) 2013, MIPS Technologies, Inc., California.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdsample.c (fancy upsampling and `jinit_upsampler`'s checks):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1996, Thomas G. Lane.
//   libjpeg-turbo Modifications:
//   Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
//   Copyright (C) 2010, 2015-2016, 2022, 2024-2026, D. R. Commander.
//   Copyright (C) 2014, MIPS Technologies, Inc., California.
//   Copyright (C) 2015, Google, Inc.
//   Copyright (C) 2019-2020, Arm Limited.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdmainct.c (the context rows the fancy upsamplers read at the image's edges):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1994-1996, Thomas G. Lane.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2010, 2016, 2022, 2024, 2026, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdcolor.c (`build_ycc_rgb_table`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1997, Thomas G. Lane.
//   Modified 2011 by Guido Vollbeding.
//   libjpeg-turbo Modifications:
//   Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
//   Copyright (C) 2009, 2011-2012, 2014-2015, 2022, 2024, D. R. Commander.
//   Copyright (C) 2013, Linaro Limited.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdcolext.c (`ycc_rgb_convert`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1997, Thomas G. Lane.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2009, 2011, 2015, 2022-2023, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jdmaster.c (`master_selection`'s order of module set-up):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1997, Thomas G. Lane.
//   Modified 2002-2009 by Guido Vollbeding.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2009-2011, 2016, 2019, 2022-2024, 2026, D. R. Commander.
//   Copyright (C) 2013, Linaro Limited.
//   Copyright (C) 2015, Google, Inc.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jmemmgr.c (`access_virt_sarray`, which refuses to read rows of an array not zeroed that
// nothing wrote):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1997, Thomas G. Lane.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2016, 2021-2022, 2024, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jerror.h (the errors, named in the refusals):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1994-1997, Thomas G. Lane.
//   Modified 1997-2009 by Guido Vollbeding.
//   Lossless JPEG Modifications:
//   Copyright (C) 1999, Ken Murchison.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2014, 2017, 2021-2023, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// jutils.c (`jpeg_natural_order`):
//   This file was part of the Independent JPEG Group's software:
//   Copyright (C) 1991-1996, Thomas G. Lane.
//   libjpeg-turbo Modifications:
//   Copyright (C) 2022, D. R. Commander.
//   For conditions of distribution and use, see the accompanying README.ijg
//   file.
//
// It also follows Pillow 12.3.0's `src/libImaging/JpegDecode.c` (`ImagingJpegDecode` and its
// suspending source manager; Copyright (c) 1998-2000 Secret Labs AB, Copyright (c) 1996-2000
// Fredrik Lundh) and the 65,536-byte reads of `src/PIL/ImageFile.py`'s `ImageFile.load`
// (Copyright (c) 1997-2004 by Secret Labs AB, Copyright (c) 1995-2004 by Fredrik Lundh): the
// Python Imaging Library (PIL) is Copyright (c) 1997-2011 by Secret Labs AB and Copyright (c)
// 1995-2011 by Fredrik Lundh and contributors, Pillow is Copyright (c) 2010 by Jeffrey 'Alex'
// Clark and contributors, MIT-CMU licence: see ThirdPartyLicenses/Pillow-LICENSE.

/// Decodes the JPEGs Pillow decodes for upstream's image reads, sample for sample, and refuses
/// the ones on which Pillow raises.
///
/// JPEG leaves the inverse DCT, chroma upsampling and colour conversion to the decoder, and
/// decoders differ: ImageIO's decode of upstream's hot dog photo is up to 30 levels from Pillow's
/// at a third of its samples, far outside issue #46's 1e-3. This reproduces what
/// `PIL.Image.open(...).convert("RGB")` does: Pillow's own reading of the headers
/// (``PillowJPEGHeader``), then libjpeg-turbo with Pillow's settings (the library defaults) and
/// Pillow's suspending input. That is the accurate integer IDCT as the Arm Neon code computes it,
/// "fancy" triangle-filter upsampling, the fixed-point YCbCr to RGB, block smoothing for
/// progressive JPEGs whose scans stop short of full precision, the standard Huffman tables for a
/// sequential JPEG that defines none, restart-marker resynchronisation, and libjpeg-turbo's rule
/// for scan data that runs out: the MCU being decoded is finished with zero bits and the rest of
/// the segment is left as it is.
///
/// A JPEG on which Pillow or libjpeg-turbo raises is refused (``Refused``), never handed to
/// another decoder: the errors of libjpeg-turbo's `jerror.h`, Pillow's own header errors, data
/// that ends before libjpeg-turbo stops reading (Pillow's "image file is truncated"), and an
/// image past Pillow's decompression bomb limit.
///
/// Covered: 8-bit Huffman-coded baseline, extended sequential and progressive JPEGs with one
/// (grey) or three (YCbCr or RGB) components, any sampling factors, and restart intervals.
/// Pillow also decodes arithmetic-coded, lossless and 4-component (CMYK and YCCK) JPEGs; this
/// throws ``Unsupported`` for those, so the caller may fall back to ImageIO, once it has made the
/// checks it shares with them: a 4-component JPEG's scans are decoded, and an arithmetic-coded or
/// lossless one is read as libjpeg-turbo reads it short of decoding its scans' data: every scan's
/// checks, the markers between and after the scans, and the end of the file. Where Pillow's
/// answer depends on that data, which happens chiefly when an arithmetic-coded scan runs past
/// one of Pillow's 65,536-byte reads (jdarith.c cannot wait for more, so Pillow often raises),
/// ``Unsupported`` says so (D-057).
enum LibjpegTurboDecoder {
    /// A JPEG Pillow decodes but this decoder does not cover: arithmetic-coded, lossless, CMYK or
    /// YCCK; or an arithmetic-coded or lossless one on which only decoding its data would tell
    /// whether Pillow raises. The caller may try another decoder.
    struct Unsupported: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    /// A JPEG upstream's Pillow raises on, or past this decoder's work limit. No other decoder
    /// should be tried on it.
    struct Refused: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    /// Errors the decoder throws.
    enum Failure: Error {
        case unsupported(Unsupported)
        case refused(Refused)
    }

    /// How much decoding a JPEG took, for the tests' bound on work.
    struct Work: Equatable, Sendable {
        /// Scans decoded, or passed over without decoding (arithmetic-coded and lossless).
        var scans = 0
        /// Blocks whose entropy-coded data the decoder read, or finished with zero bits when the
        /// data ran out. A block left as it was (in an end-of-band run, or after the data ran
        /// out) is not counted.
        var blocks = 0
        /// Restart markers looked for.
        var restarts = 0
        /// Blocks visited one at a time: those decoded, and those of a refinement scan's
        /// end-of-band runs, checked for coefficients to refine. Blocks left as they are after
        /// the data runs out, and in a first AC scan's end-of-band runs, are passed over in bulk.
        var visits = 0
    }

    /// The most blocks the scans of one JPEG may visit one at a time: 100 passes over three
    /// components of the largest image allowed. Pillow has no such limit, but a refinement scan
    /// visits every block of an end-of-band run whether it reads a bit for it or not, so without
    /// one a small file of many such scans over a large image would cost time in proportion to
    /// its size times the image's. Well-formed JPEGs visit their blocks about 10 times.
    static let maxVisits = 100 * 3 * (RGBImage.maxPixels / 64)

    /// True when `bytes` start as a JPEG does for Pillow (`_accept`): FF D8 FF.
    static func isJPEG(_ bytes: [UInt8]) -> Bool {
        bytes.count >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
    }

    /// `jpeg_natural_order`: zigzag position to natural position, with libjpeg's 16 guard
    /// entries for corrupt data.
    static let naturalOrder: [Int] =
        [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37,
            44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        ] + Array(repeating: 63, count: 16)

    // MARK: Decoding

    static func decode(_ bytes: [UInt8]) throws(Failure) -> RGBImage {
        var work = Work()
        return try decode(bytes, work: &work)
    }

    /// Decodes `bytes`, counting the work into `work`, refusing past `visitLimit` visits.
    static func decode(
        _ bytes: [UInt8], work: inout Work, visitLimit: Int = maxVisits
    ) throws(Failure) -> RGBImage {
        // Pillow reads the headers itself first; when libjpeg-turbo then succeeds, its frame is
        // the one Pillow took its size and mode from.
        do {
            _ = try PillowJPEGHeader.read(bytes)
        } catch {
            throw .refused(error)
        }
        var decompressor = Decompressor(bytes: bytes)
        decompressor.visitLimit = visitLimit
        do {
            let image = try decompressor.run()
            work = decompressor.work
            return image
        } catch {
            work = decompressor.work
            switch error {
            case .refused(let message): throw .refused(Refused(message))
            case .unsupported(let message): throw .unsupported(Unsupported(message))
            case .endOfData:
                throw .refused(
                    Refused(
                        "the JPEG ends before libjpeg-turbo stops reading it (Pillow's \"image "
                            + "file is truncated\")"))
            }
        }
    }

    /// Why decoding stopped.
    enum Stop: Error {
        /// Pillow or libjpeg-turbo raises.
        case refused(String)
        /// Pillow decodes it and this decoder does not, or only decoding it would tell whether
        /// Pillow raises.
        case unsupported(String)
        /// A read past the last byte: Pillow's `ImageFile.load` raises "image file is truncated",
        /// unless the image is already complete. Once it is, also a read past the bytes Pillow
        /// has handed over (`finishing`).
        case endOfData
    }

    /// A refusal naming libjpeg-turbo's error.
    static func error(_ code: String, _ message: String) -> Stop {
        .refused("libjpeg-turbo stops with \(code): \(message)")
    }

    // MARK: Huffman tables

    /// `JHUFF_TBL`: the code counts per length and the symbols, as a DHT segment defines them.
    struct HuffmanTable {
        /// `bits[1...16]`; index 0 is unused.
        var counts: [Int]
        /// `huffval`, zero past the defined symbols.
        var values: [UInt8]

        init(counts: [Int], values: [UInt8]) {
            self.counts = counts
            self.values = values + [UInt8](repeating: 0, count: 256 - values.count)
        }
    }

    /// `std_huff_tables`: Annex K.3's tables, which a sequential JPEG gets in slots 0 and 1 when
    /// it defines none there by the time decompression starts.
    static let standardTables: (dc: [HuffmanTable], ac: [HuffmanTable]) = (
        [
            HuffmanTable(
                counts: [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                values: Array(0...11)),
            HuffmanTable(
                counts: [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                values: Array(0...11)),
        ],
        [
            HuffmanTable(
                counts: [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D],
                values: [
                    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13,
                    0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42,
                    0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A,
                    0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35,
                    0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
                    0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67,
                    0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84,
                    0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
                    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3,
                    0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
                    0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1,
                    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
                    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
                ]),
            HuffmanTable(
                counts: [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
                values: [
                    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51,
                    0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1,
                    0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24,
                    0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A,
                    0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
                    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66,
                    0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82,
                    0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
                    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA,
                    0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
                    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
                    0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4,
                    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
                ]),
        ]
    )

    /// `d_derived_tbl`, laid out in one array: the 256-entry lookahead table (code length << 8 |
    /// symbol, or 9 << 8 for a longer code), `maxcode[0...17]`, `valoffset[0...17]` and the 256
    /// symbols.
    enum Derived {
        static let lookup = 0
        static let maxcode = 256
        static let valoffset = 274
        static let values = 292
        static let size = 548

        /// `jpeg_make_d_derived_tbl`, with its checks, appended to `tables`. A DC table's symbols
        /// may go up to 15, or to 16 in a lossless JPEG.
        static func append(
            _ table: HuffmanTable?, isDC: Bool, slot: Int, to tables: inout [Int32],
            largestDCSymbol: Int = 15
        ) throws(Stop) {
            guard (0..<4).contains(slot), let table else {
                throw error(
                    "JERR_NO_HUFF_TABLE",
                    "a scan uses \(isDC ? "DC" : "AC") Huffman table \(slot), which is not defined")
            }
            var sizes: [Int] = []
            for length in 1...16 {
                let count = table.counts[length]
                guard sizes.count + count <= 256 else {
                    throw error("JERR_BAD_HUFF_TABLE", "a Huffman table has over 256 codes")
                }
                sizes += [Int](repeating: length, count: count)
            }
            // Figure C.2: the canonical codes; none may be all ones.
            var codes = [Int](repeating: 0, count: sizes.count)
            var code = 0
            var p = 0
            if var size = sizes.first {
                while p < sizes.count {
                    while p < sizes.count && sizes[p] == size {
                        codes[p] = code
                        p += 1
                        code += 1
                    }
                    guard code < 1 << size else {
                        throw error("JERR_BAD_HUFF_TABLE", "a Huffman table's codes do not fit")
                    }
                    code <<= 1
                    size += 1
                }
            }
            let base = tables.count
            tables += [Int32](repeating: 0, count: size)
            for i in 0..<256 { tables[base + lookup + i] = 9 << 8 }
            p = 0
            for length in 1...16 {
                if table.counts[length] > 0 {
                    tables[base + valoffset + length] = Int32(p - codes[p])
                    p += table.counts[length]
                    tables[base + maxcode + length] = Int32(codes[p - 1])
                } else {
                    tables[base + maxcode + length] = -1
                }
            }
            tables[base + valoffset + 17] = 0
            tables[base + maxcode + 17] = 0xFFFFF
            p = 0
            for length in 1...8 {
                for _ in 0..<table.counts[length] {
                    let first = codes[p] << (8 - length)
                    for fill in 0..<(1 << (8 - length)) {
                        tables[base + lookup + first + fill] =
                            Int32(length << 8 | Int(table.values[p]))
                    }
                    p += 1
                }
            }
            for i in 0..<256 { tables[base + values + i] = Int32(table.values[i]) }
            if isDC, table.values.prefix(sizes.count).contains(where: { $0 > largestDCSymbol }) {
                throw error(
                    "JERR_BAD_HUFF_TABLE",
                    "a DC Huffman table has a symbol above \(largestDCSymbol)")
            }
        }
    }

    // MARK: The decompressor

    /// The colour space libjpeg-turbo takes the components to be in (`jpeg_color_space`).
    enum ColorSpace { case grey, ycc, rgb, cmyk, ycck }

    struct Component {
        var id: Int
        var h: Int
        var v: Int
        /// The quantization table selector as the frame header has it, a whole byte.
        var quantSelector: Int
        var dcTable = 0
        var acTable = 0
        /// The quantization table latched at the component's first scan (`latch_quant_tables`),
        /// in natural order; nil until a scan reaches the component.
        var quant: [UInt16]?
        /// The component's blocks (`width_in_blocks`, `height_in_blocks`) and samples
        /// (`downsampled_width`, `downsampled_height`).
        var widthInBlocks = 0
        var heightInBlocks = 0
        var width = 0
        var height = 0
        /// The coefficient grid, padded to whole MCUs: blocks per line and per column.
        var gridWidth = 0
        var gridHeight = 0
        /// Where the component's blocks start in the shared coefficient storage, in blocks.
        var gridOffset = 0
        /// `coef_bits` and its copy from before the component's latest scan, for progressive
        /// JPEGs: the successive-approximation bit of each coefficient so far, -1 before any.
        var coefBits = [Int](repeating: -1, count: 64)
        var previousCoefBits = [Int](repeating: 0, count: 64)
    }

    /// The scan being decoded: what `get_sos` read.
    struct Scan {
        var components: [Int] = []
        var ss = 0
        var se = 0
        var ah = 0
        var al = 0
    }

    /// What stops one MCU's decode: more of Pillow's buffer is needed (libjpeg-turbo suspends and
    /// starts the MCU again with it), the data has ended, or a DC coefficient of a progressive
    /// JPEG overflows (`JERR_BAD_DCT_COEF`).
    enum Interrupt: Error { case chunk, end, overflow }

    /// Pillow's `ImageFile.MAXBLOCK`: `ImageFile.load` hands the decoder 65,536 more bytes each
    /// time it suspends, and libjpeg-turbo takes its fast path only with plenty of them at hand.
    static let readSize = 65_536

    /// libjpeg-turbo's bit buffer (`jdhuff.h`): a 64-bit register, refilled to at least 57 bits
    /// on the slow path and by 48 bits on the fast one, over Pillow's suspending source.
    struct BitReader {
        let input: UnsafeBufferPointer<UInt8>
        var position: Int
        var chunkEnd: Int
        var buffer: UInt64 = 0
        var bitsLeft = 0
        var unreadMarker: Int
        var insufficient = false

        /// The byte at `index`, or zero past the end (never read on the paths that use it).
        @inline(__always) func peek(_ index: Int) -> Int {
            index < input.count ? Int(input[index]) : 0
        }

        /// The suspension for a read at `position`.
        @inline(__always) func suspension() -> Interrupt {
            chunkEnd < input.count ? .chunk : .end
        }

        /// `jpeg_fill_bit_buffer`: loads bytes until 57 bits are held or a marker is met, undoing
        /// FF 00 stuffing and swallowing the FF fill before a marker. Past the marker that ends
        /// the segment, a request for more bits than are left gets zero bits and marks the data
        /// as exhausted (`JWRN_HIT_MARKER`).
        mutating func fill(_ nbits: Int) throws(Interrupt) {
            if unreadMarker == 0 {
                while bitsLeft < 57 {
                    guard position < chunkEnd else { throw suspension() }
                    var c = Int(input[position])
                    position += 1
                    if c == 0xFF {
                        repeat {
                            guard position < chunkEnd else { throw suspension() }
                            c = Int(input[position])
                            position += 1
                        } while c == 0xFF
                        if c == 0 {
                            c = 0xFF
                        } else {
                            unreadMarker = c
                            break
                        }
                    }
                    buffer = buffer << 8 | UInt64(c)
                    bitsLeft += 8
                }
                if unreadMarker == 0 { return }
            }
            if nbits > bitsLeft {
                insufficient = true
                buffer <<= UInt64(57 - bitsLeft)
                bitsLeft = 57
            }
        }

        /// `CHECK_BIT_BUFFER` and `GET_BITS`.
        @inline(__always) mutating func bits(_ n: Int) throws(Interrupt) -> Int {
            if bitsLeft < n { try fill(n) }
            bitsLeft -= n
            return Int(truncatingIfNeeded: buffer >> UInt64(bitsLeft)) & (1 << n - 1)
        }

        /// `HUFF_DECODE` and `jpeg_huff_decode`: a code longer than 16 bits gives symbol 0
        /// (`JWRN_HUFF_BAD_CODE`).
        @inline(__always) mutating func decode(_ t: UnsafePointer<Int32>) throws(Interrupt) -> Int {
            var length = 1
            if bitsLeft < 8 {
                try fill(0)
                if bitsLeft < 8 { return try decodeLong(t, length) }
            }
            let entry = Int(
                t[Derived.lookup + Int(truncatingIfNeeded: buffer >> UInt64(bitsLeft - 8)) & 0xFF])
            length = entry >> 8
            if length <= 8 {
                bitsLeft -= length
                return entry & 0xFF
            }
            return try decodeLong(t, length)
        }

        mutating func decodeLong(_ t: UnsafePointer<Int32>, _ minimum: Int) throws(Interrupt) -> Int
        {
            var length = minimum
            var code = try bits(length)
            while code > Int(t[Derived.maxcode + length]) {
                code = code << 1 | (try bits(1))
                length += 1
            }
            if length > 16 { return 0 }
            return Int(t[Derived.values + (code + Int(t[Derived.valoffset + length])) & 0xFF])
        }

        /// `GET_BYTE` of the fast path: a marker reads as zero bytes and is remembered, without
        /// moving past it; so is FF FF, which the slow path reads differently.
        @inline(__always) mutating func fastByte() {
            let c0 = peek(position)
            position += 1
            let c1 = peek(position)
            buffer = buffer << 8 | UInt64(c0)
            bitsLeft += 8
            if c0 == 0xFF {
                position += 1
                if c1 != 0 {
                    unreadMarker = c1
                    position -= 2
                    buffer &= ~0xFF
                }
            }
        }

        /// `FILL_BIT_BUFFER_FAST`: six bytes when 16 bits or fewer are left.
        @inline(__always) mutating func fastFill() {
            if bitsLeft <= 16 {
                fastByte()
                fastByte()
                fastByte()
                fastByte()
                fastByte()
                fastByte()
            }
        }

        /// `FILL_BIT_BUFFER_FAST` and `GET_BITS`.
        @inline(__always) mutating func fastBits(_ n: Int) -> Int {
            fastFill()
            bitsLeft -= n
            return Int(truncatingIfNeeded: buffer >> UInt64(bitsLeft)) & (1 << n - 1)
        }

        /// `HUFF_DECODE_FAST`.
        @inline(__always) mutating func fastDecode(_ t: UnsafePointer<Int32>) -> Int {
            fastFill()
            let entry = Int(
                t[Derived.lookup + Int(truncatingIfNeeded: buffer >> UInt64(bitsLeft - 8)) & 0xFF])
            var length = entry >> 8
            bitsLeft -= length
            if length <= 8 { return entry & 0xFF }
            var code = Int(truncatingIfNeeded: buffer >> UInt64(bitsLeft)) & (1 << length - 1)
            while code > Int(t[Derived.maxcode + length]) {
                bitsLeft -= 1
                code = code << 1 | Int(truncatingIfNeeded: buffer >> UInt64(bitsLeft)) & 1
                length += 1
            }
            if length > 16 { return 0 }
            return Int(t[Derived.values + (code + Int(t[Derived.valoffset + length])) & 0xFF])
        }
    }

    /// The fewest bits jdlhuff.c's `decode_mcus` reads for one sample coded with `table`: a
    /// symbol's code and the difference bits that follow it (none for 0 and 16), or the 17 bits
    /// `jpeg_huff_decode` reads of a string that is no code, which every table has (no code is
    /// all ones).
    static func fewestSampleBits(_ table: HuffmanTable) -> Int {
        var fewest = 17
        var index = 0
        for length in 1...16 {
            for _ in 0..<table.counts[length] {
                let symbol = Int(table.values[index])
                fewest = min(fewest, length + (symbol == 0 || symbol == 16 ? 0 : symbol))
                index += 1
            }
        }
        return fewest
    }

    /// `HUFF_EXTEND`: an s-bit magnitude as a signed value.
    @inline(__always) static func extend(_ x: Int, _ s: Int) -> Int {
        x < 1 << (s - 1) ? x - (1 << s) + 1 : x
    }

    /// The state libjpeg-turbo and Pillow's decoder keep while reading one JPEG.
    struct Decompressor {
        let bytes: [UInt8]
        var work = Work()

        // The source manager and the marker reader.
        var position = 0
        var chunkEnd: Int
        var unreadMarker = 0
        var sawSOI = false
        var sawSOF = false
        /// Set once every row of a single-scan JPEG is out: Pillow's decoder then stops where the
        /// bytes it has been handed end (`jpeg_finish_decompress` suspends there), and reads no
        /// more of the file.
        var finishing = false
        /// While an arithmetic-coded scan's data is read: the end of the bytes Pillow had handed
        /// over when the scan started. jdarith.c cannot suspend (`get_byte`, `process_restart`),
        /// so a read past it is `JERR_CANT_SUSPEND` rather than a request for more.
        var readLimit = Int.max
        var nextRestartNumber = 0
        var inputScanNumber = 0

        // Tables and markers that outlive a tables-only datastream.
        var quantTables: [[UInt16]?] = [nil, nil, nil, nil]
        var dcTables: [HuffmanTable?] = [nil, nil, nil, nil]
        var acTables: [HuffmanTable?] = [nil, nil, nil, nil]
        var restartInterval = 0
        var sawJFIF = false
        var sawAdobe = false
        var adobeTransform = 0

        // The frame.
        var progressive = false
        var lossless = false
        var arithmetic = false
        var precision = 0
        var width = 0
        var height = 0
        var components: [Component] = []
        var maxH = 1
        var maxV = 1
        var mcusPerLine = 0
        var totalIMCURows = 0
        var hasMultipleScans = false
        var colorSpace = ColorSpace.grey

        // The scan and the coefficients.
        var scan = Scan()
        var visitLimit = LibjpegTurboDecoder.maxVisits
        /// Every component's blocks, 64 coefficients each, in natural order: the whole image for
        /// a JPEG of several scans, one row of MCUs for a single-scan one.
        var coefficients: [Int16] = []
        /// Per block, which coefficients are nonzero (bit = natural position), for refinement
        /// scans to pass over blocks with nothing to refine.
        var nonzero: [UInt64] = []
        /// `last_good_iMCU_row`: the last MCU row the latest scan reached before its data ran out.
        var lastGoodIMCURow = 0
        /// The samples of each component after the inverse DCT, `widthInBlocks * 8` per row.
        var planes: [[UInt8]] = []

        init(bytes: [UInt8]) {
            self.bytes = bytes
            chunkEnd = min(bytes.count, LibjpegTurboDecoder.readSize)
        }

        /// What `ImagingJpegDecode` does with the whole file: read the header, start
        /// decompression (which consumes every scan of a multi-scan JPEG), read the rows, then
        /// finish, which reads the markers after a single scan up to EOI.
        mutating func run() throws(Stop) -> RGBImage {
            try readHeader()
            try startDecompress()
            if hasMultipleScans {
                try consumeScans()
                guard components.count != 4 else { throw unsupportedFeature() }
                makePlanesFromCoefficients()
            } else {
                try decodeScan(singleScan: true)
                try finishSingleScan()
                guard components.count != 4 else { throw unsupportedFeature() }
            }
            return output()
        }

        /// `jpeg_finish_decompress` after a single scan's last row: the markers up to EOI. The
        /// image is complete, so Pillow takes the end of the bytes it has handed over as the end,
        /// without reading more of the file, but a broken marker before it still raises.
        mutating func finishSingleScan() throws(Stop) {
            finishing = true
            do throws(Stop) {
                if try readMarkers() == .sos {
                    throw error("JERR_EOI_EXPECTED", "a scan follows a single-scan image")
                }
            } catch {
                if case .endOfData = error {} else { throw error }
            }
        }

        func unsupportedFeature() -> Stop {
            if arithmetic { return .unsupported("arithmetic-coded JPEGs are not covered") }
            if lossless { return .unsupported("lossless JPEGs are not covered") }
            return .unsupported("4-component (CMYK and YCCK) JPEGs are not covered")
        }

        // MARK: Reading bytes

        /// Makes `index` readable: Pillow hands the decoder another 65,536 bytes each time it
        /// asks, until the file ends or the image is complete. Inside an arithmetic-coded scan
        /// libjpeg-turbo cannot ask (``readLimit``).
        @inline(__always) mutating func need(_ index: Int) throws(Stop) {
            guard index < readLimit else {
                let end =
                    readLimit == bytes.count
                    ? "the end of the file" : "the \(readLimit) bytes Pillow has read"
                throw error(
                    "JERR_CANT_SUSPEND",
                    "an arithmetic-coded scan's data runs past \(end), and jdarith.c cannot "
                        + "suspend for more (Pillow's \"broken data stream\")")
            }
            guard index < bytes.count else { throw .endOfData }
            while index >= chunkEnd {
                if finishing { throw .endOfData }
                chunkEnd = min(bytes.count, chunkEnd + readSize)
            }
        }

        /// `INPUT_BYTE`.
        mutating func byte() throws(Stop) -> Int {
            try need(position)
            defer { position += 1 }
            return Int(bytes[position])
        }

        /// `INPUT_2BYTES`.
        mutating func u16() throws(Stop) -> Int {
            let high = try byte()
            return high << 8 | (try byte())
        }

        /// `skip_input_data`.
        mutating func skip(_ count: Int) {
            if count > 0 { position += count }
        }

        /// `next_marker`: skips anything but FF, FF fill, and FF 00.
        mutating func nextMarker() throws(Stop) {
            while true {
                var c = try byte()
                while c != 0xFF { c = try byte() }
                repeat { c = try byte() } while c == 0xFF
                if c != 0 {
                    unreadMarker = c
                    return
                }
            }
        }

        /// `first_marker`: SOI, with nothing before it.
        mutating func firstMarker() throws(Stop) {
            let c = try byte()
            let c2 = try byte()
            guard c == 0xFF, c2 == 0xD8 else {
                throw error("JERR_NO_SOI", "the datastream does not start with SOI")
            }
            unreadMarker = c2
        }

        // MARK: Markers

        enum Reached: Equatable { case sos, eoi }

        /// `read_markers`: processes markers up to SOS or EOI.
        mutating func readMarkers() throws(Stop) -> Reached {
            while true {
                if unreadMarker == 0 {
                    if !sawSOI { try firstMarker() } else { try nextMarker() }
                }
                let marker = unreadMarker
                switch marker {
                case 0xD8: try getSOI()
                case 0xC0, 0xC1: try getSOF(progressive: false, lossless: false, arithmetic: false)
                case 0xC2: try getSOF(progressive: true, lossless: false, arithmetic: false)
                case 0xC3: try getSOF(progressive: false, lossless: true, arithmetic: false)
                case 0xC9: try getSOF(progressive: false, lossless: false, arithmetic: true)
                case 0xCA: try getSOF(progressive: true, lossless: false, arithmetic: true)
                case 0xCB: try getSOF(progressive: false, lossless: true, arithmetic: true)
                case 0xC5, 0xC6, 0xC7, 0xC8, 0xCD, 0xCE, 0xCF:
                    throw error(
                        "JERR_SOF_UNSUPPORTED",
                        "SOF type 0x\(PillowJPEGHeader.hex(UInt8(marker))) is not supported")
                case 0xDA:
                    try getSOS()
                    unreadMarker = 0
                    return .sos
                case 0xD9:
                    unreadMarker = 0
                    return .eoi
                case 0xCC: try getDAC()
                case 0xC4: try getDHT()
                case 0xDB: try getDQT()
                case 0xDD: try getDRI()
                case 0xE0, 0xEE: try getInterestingAPPn(marker)
                case 0xE1...0xED, 0xEF, 0xFE, 0xDC: try skipVariable()
                case 0xD0...0xD7, 0x01: break
                default:
                    throw error(
                        "JERR_UNKNOWN_MARKER",
                        "marker 0x\(PillowJPEGHeader.hex(UInt8(marker))) is reserved or unknown")
                }
                unreadMarker = 0
            }
        }

        mutating func getSOI() throws(Stop) {
            guard !sawSOI else { throw error("JERR_SOI_DUPLICATE", "the JPEG has two SOI markers") }
            restartInterval = 0
            sawJFIF = false
            sawAdobe = false
            adobeTransform = 0
            sawSOI = true
        }

        mutating func getSOF(progressive: Bool, lossless: Bool, arithmetic: Bool) throws(Stop) {
            guard !sawSOF else { throw error("JERR_SOF_DUPLICATE", "the JPEG has two SOF markers") }
            self.progressive = progressive
            self.lossless = lossless
            self.arithmetic = arithmetic
            let length = try u16()
            precision = try byte()
            height = try u16()
            width = try u16()
            let count = try byte()
            guard height > 0, width > 0, count > 0 else {
                throw error("JERR_EMPTY_IMAGE", "the frame is empty (DNL is not supported)")
            }
            guard length - 8 == count * 3 else {
                throw error(
                    "JERR_BAD_LENGTH", "a frame header's length does not fit its components")
            }
            components = []
            for _ in 0..<count {
                let id = try byte()
                let factors = try byte()
                let table = try byte()
                components.append(
                    Component(id: id, h: factors >> 4, v: factors & 15, quantSelector: table))
            }
            sawSOF = true
        }

        mutating func getSOS() throws(Stop) {
            guard sawSOF else { throw error("JERR_SOS_NO_SOF", "a scan comes before the frame") }
            let length = try u16()
            let count = try byte()
            guard length == count * 2 + 6, (1...4).contains(count) else {
                throw error("JERR_BAD_LENGTH", "a scan header's length does not fit its components")
            }
            // `cur_comp_info`, by scan position. libjpeg-turbo looks a component up only among
            // the first four of the frame, and only while the scan slot numbered by its frame
            // position is still empty, which also refuses some scans that list their components
            // out of frame order (2 then 1, but not 3 then 2).
            var slots = [Int](repeating: -1, count: 4)
            for i in 0..<count {
                let id = try byte()
                let tables = try byte()
                guard
                    let index = (0..<min(components.count, 4)).first(where: {
                        components[$0].id == id && slots[$0] < 0
                    })
                else {
                    throw error(
                        "JERR_BAD_COMPONENT_ID",
                        "a scan names component \(id), which the frame lacks or the scan repeats")
                }
                slots[i] = index
                components[index].dcTable = tables >> 4
                components[index].acTable = tables & 15
                guard !slots[0..<i].contains(index) else {
                    throw error("JERR_BAD_COMPONENT_ID", "a scan names component \(id) twice")
                }
            }
            scan.components = Array(slots.prefix(count))
            scan.ss = try byte()
            scan.se = try byte()
            let approximation = try byte()
            scan.ah = approximation >> 4
            scan.al = approximation & 15
            nextRestartNumber = 0
            inputScanNumber += 1
        }

        mutating func getDHT() throws(Stop) {
            var length = try u16() - 2
            while length > 16 {
                let index = try byte()
                var counts = [0]
                for _ in 1...16 { counts.append(try byte()) }
                let count = counts.reduce(0, +)
                length -= 17
                guard count <= 256, count <= length else {
                    throw error("JERR_BAD_HUFF_TABLE", "a Huffman table overruns its segment")
                }
                var values: [UInt8] = []
                for _ in 0..<count { values.append(UInt8(try byte())) }
                length -= count
                let table = HuffmanTable(counts: counts, values: values)
                if index & 0x10 != 0 {
                    guard index - 0x10 < 4 else {
                        throw error("JERR_DHT_INDEX", "a Huffman table's index \(index) is bad")
                    }
                    acTables[index - 0x10] = table
                } else {
                    guard index < 4 else {
                        throw error("JERR_DHT_INDEX", "a Huffman table's index \(index) is bad")
                    }
                    dcTables[index] = table
                }
            }
            guard length == 0 else {
                throw error("JERR_BAD_LENGTH", "a DHT segment's length does not fit its tables")
            }
        }

        mutating func getDQT() throws(Stop) {
            var length = try u16() - 2
            while length > 0 {
                let n = try byte()
                let precise = n >> 4 != 0
                let index = n & 15
                guard index < 4 else {
                    throw error("JERR_DQT_INDEX", "a quantization table's index \(index) is bad")
                }
                var table = quantTables[index] ?? [UInt16](repeating: 0, count: 64)
                for k in 0..<64 {
                    table[naturalOrder[k]] = UInt16(precise ? try u16() : try byte())
                }
                quantTables[index] = table
                length -= precise ? 129 : 65
            }
            guard length == 0 else {
                throw error("JERR_BAD_LENGTH", "a DQT segment's length does not fit its tables")
            }
        }

        mutating func getDRI() throws(Stop) {
            guard try u16() == 4 else { throw error("JERR_BAD_LENGTH", "a DRI segment is bad") }
            restartInterval = try u16()
        }

        /// `get_dac`: arithmetic conditioning values, checked even in a Huffman-coded JPEG.
        mutating func getDAC() throws(Stop) {
            var length = try u16() - 2
            while length > 0 {
                let index = try byte()
                let value = try byte()
                length -= 2
                guard index < 32 else {
                    throw error("JERR_DAC_INDEX", "an arithmetic table's index \(index) is bad")
                }
                if index < 16, value & 15 > value >> 4 {
                    throw error("JERR_DAC_VALUE", "an arithmetic table's value \(value) is bad")
                }
            }
            guard length == 0 else { throw error("JERR_BAD_LENGTH", "a DAC segment is bad") }
        }

        /// `get_interesting_appn`: JFIF in APP0 and Adobe's transform in APP14.
        mutating func getInterestingAPPn(_ marker: Int) throws(Stop) {
            var length = try u16() - 2
            let count = length >= 14 ? 14 : max(length, 0)
            var data: [Int] = []
            for _ in 0..<count { data.append(try byte()) }
            length -= count
            if marker == 0xE0, count >= 14, data.prefix(5) == [0x4A, 0x46, 0x49, 0x46, 0] {
                sawJFIF = true
            } else if marker == 0xEE, count >= 12, data.prefix(5) == [0x41, 0x64, 0x6F, 0x62, 0x65]
            {
                sawAdobe = true
                adobeTransform = data[11]
            }
            skip(length)
        }

        mutating func skipVariable() throws(Stop) {
            skip(try u16() - 2)
        }

        // MARK: The header and the start of decompression

        /// `jpeg_read_header`, called by Pillow until it is not a tables-only datastream.
        mutating func readHeader() throws(Stop) {
            while true {
                // reset_input_controller and reset_marker_reader. Tables survive.
                sawSOI = false
                sawSOF = false
                components = []
                inputScanNumber = 0
                unreadMarker = 0
                if try readMarkers() == .sos { break }
                guard !sawSOF else {
                    throw error("JERR_SOF_NO_SOS", "the JPEG has a frame header but no scan")
                }
            }
            try initialSetup()
            // default_decompress_parms: the colour space libjpeg guesses.
            switch components.count {
            case 1:
                colorSpace = .grey
            case 3:
                if sawJFIF {
                    colorSpace = .ycc
                } else if sawAdobe {
                    colorSpace = adobeTransform == 0 ? .rgb : .ycc
                } else {
                    let ids = components.map(\.id)
                    colorSpace =
                        ids == [1, 2, 3]
                        ? (lossless ? .rgb : .ycc)
                        : ids == [82, 71, 66] ? .rgb : (lossless ? .rgb : .ycc)
                }
            default:
                colorSpace = sawAdobe && adobeTransform != 0 ? .ycck : .cmyk
            }
        }

        /// `initial_setup`, at the first SOS.
        mutating func initialSetup() throws(Stop) {
            guard height <= 65_500, width <= 65_500 else {
                throw error(
                    "JERR_IMAGE_TOO_BIG",
                    "the image is \(width) by \(height); sides may be up to 65,500")
            }
            guard lossless ? (2...16).contains(precision) : precision == 8 || precision == 12 else {
                throw error("JERR_BAD_PRECISION", "\(precision)-bit samples are not supported")
            }
            guard components.count <= 10 else {
                throw error("JERR_COMPONENT_COUNT", "the frame has \(components.count) components")
            }
            guard components.allSatisfy({ (1...4).contains($0.h) && (1...4).contains($0.v) }) else {
                throw error("JERR_BAD_SAMPLING", "a sampling factor is out of range")
            }
            maxH = components.map(\.h).max() ?? 1
            maxV = components.map(\.v).max() ?? 1
            let unit = lossless ? 1 : 8
            func roundUp(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }
            mcusPerLine = roundUp(width, maxH * unit)
            totalIMCURows = roundUp(height, maxV * unit)
            for i in components.indices {
                components[i].widthInBlocks = roundUp(width * components[i].h, maxH * unit)
                components[i].heightInBlocks = roundUp(height * components[i].v, maxV * unit)
                components[i].width = roundUp(width * components[i].h, maxH)
                components[i].height = roundUp(height * components[i].v, maxV)
                components[i].gridWidth = mcusPerLine * components[i].h
                components[i].gridHeight = totalIMCURows * components[i].v
            }
            hasMultipleScans = scan.components.count < components.count || progressive
        }

        /// `jpeg_start_decompress` up to the first scan's data: `master_selection`'s checks, in
        /// its order, then `start_input_pass`. An arithmetic-coded or lossless JPEG, which this
        /// decoder does not decode, then has the rest of its reading walked, and never returns
        /// (``walkUndecodedScans()``).
        mutating func startDecompress() throws(Stop) {
            // jinit_color_deconverter: Pillow asks for L, RGB or CMYK from 1, 3 or 4 components,
            // and a lossless JPEG may not be converted to any of them.
            let output: ColorSpace =
                components.count == 1 ? .grey : components.count == 3 ? .rgb : .cmyk
            if lossless, colorSpace != output {
                throw error(
                    "JERR_CONVERSION_NOTIMPL", "a lossless JPEG cannot be converted for Pillow")
            }
            // jinit_upsampler.
            for component in components {
                guard maxH % component.h == 0, maxV % component.v == 0 else {
                    throw error(
                        "JERR_FRACT_SAMPLE_NOTIMPL", "a component's sampling ratio is fractional")
                }
            }
            if lossless && arithmetic {
                throw error(
                    "JERR_ARITH_NOTIMPL", "arithmetic-coded lossless JPEGs are not supported")
            }
            if lossless {
                try checkLosslessScan()
                try walkUndecodedScans()
            }
            if arithmetic {
                // jdarith.c's start_pass checks a scan as jdphuff.c does; the arithmetic decoding
                // itself is not covered.
                try startInputPass()
                try walkUndecodedScans()
            }
            if !progressive {
                // jinit_huff_decoder: the standard tables fill slots no DHT has defined yet.
                for slot in 0..<2 {
                    if dcTables[slot] == nil { dcTables[slot] = standardTables.dc[slot] }
                    if acTables[slot] == nil { acTables[slot] = standardTables.ac[slot] }
                }
            }
            // The coefficient storage: the whole image for several scans, one MCU row for one.
            var offset = 0
            for i in components.indices {
                components[i].gridOffset = offset
                offset +=
                    components[i].gridWidth
                    * (hasMultipleScans ? components[i].gridHeight : components[i].v)
            }
            coefficients = [Int16](repeating: 0, count: offset * 64)
            if progressive { nonzero = [UInt64](repeating: 0, count: offset) }
            if !hasMultipleScans, components.count != 4 {
                // 128 is what an all-zero block comes out as, so those need no inverse DCT.
                planes = components.map {
                    [UInt8](repeating: 128, count: $0.widthInBlocks * 64 * $0.heightInBlocks)
                }
            }
            try startInputPass()
        }

        /// `per_scan_setup`'s check: an interleaved scan's MCU holds at most 10 blocks.
        func checkMCUSize() throws(Stop) {
            if scan.components.count > 1 {
                let blocks = scan.components.reduce(0) { $0 + components[$1].h * components[$1].v }
                guard blocks <= 10 else {
                    throw error(
                        "JERR_BAD_MCU_SIZE",
                        "an interleaved scan's MCU has \(blocks) blocks; the most is 10")
                }
            }
        }

        /// `start_input_pass` for a lossless JPEG's first scan, up to its data, which this decoder
        /// does not decode: per_scan_setup, no quantization tables, jdlhuff.c's DC tables (no
        /// standard ones), then jdlossls.c's checks of the predictor, Se, Ah and the point
        /// transform, and jddiffct.c's of the restart interval.
        func checkLosslessScan() throws(Stop) {
            try checkMCUSize()
            for index in scan.components {
                let slot = components[index].dcTable
                var derived: [Int32] = []
                try Derived.append(
                    slot < 4 ? dcTables[slot] : nil, isDC: true, slot: slot, to: &derived,
                    largestDCSymbol: 16)
            }
            guard (1...7).contains(scan.ss), scan.se == 0, scan.ah == 0, scan.al < precision else {
                throw error(
                    "JERR_BAD_PROGRESSION",
                    "a lossless scan's parameters are bad (Ss \(scan.ss), Se \(scan.se), "
                        + "Ah \(scan.ah), Al \(scan.al))")
            }
            let first = components[scan.components[0]]
            let mcusPerRow = scan.components.count > 1 ? mcusPerLine : first.widthInBlocks
            guard restartInterval % mcusPerRow == 0 else {
                throw error(
                    "JERR_BAD_RESTART",
                    "a lossless JPEG's restart interval, \(restartInterval) MCUs, is not a whole "
                        + "number of rows of \(mcusPerRow)")
            }
        }

        /// `start_input_pass`: per_scan_setup, latch_quant_tables and the entropy decoder's
        /// start_pass, each with its checks. jdarith.c's start_pass makes jdphuff.c's checks of a
        /// progressive scan's parameters and updates `coef_bits` the same way (a sequential
        /// scan's parameters only warn there); its one other check, a table number past 15
        /// (`JERR_NO_ARITH_TABLE`), cannot fail, as a scan header gives 4-bit table numbers. The
        /// Huffman tables of a Huffman-coded scan are derived when it is decoded (``scanTables()``).
        mutating func startInputPass() throws(Stop) {
            try checkMCUSize()
            for index in scan.components where components[index].quant == nil {
                let selector = components[index].quantSelector
                guard selector < 4, let table = quantTables[selector] else {
                    throw error(
                        "JERR_NO_QUANT_TABLE",
                        "a component uses quantization table \(selector), which is not defined")
                }
                components[index].quant = table
            }
            if progressive {
                let dc = scan.ss == 0
                let bad =
                    (dc
                        ? scan.se != 0
                        : scan.ss > scan.se || scan.se > 63
                            || scan.components.count != 1)
                    || (scan.ah != 0 && scan.al != scan.ah - 1) || scan.al > 13
                guard !bad else {
                    throw error(
                        "JERR_BAD_PROGRESSION",
                        "a progressive scan's parameters are bad (Ss \(scan.ss), Se \(scan.se), "
                            + "Ah \(scan.ah), Al \(scan.al))")
                }
                for index in scan.components {
                    for k in min(scan.ss, 1)...max(scan.se, 9) {
                        components[index].previousCoefBits[k] =
                            inputScanNumber > 1 ? components[index].coefBits[k] : 0
                    }
                    for k in scan.ss...scan.se { components[index].coefBits[k] = scan.al }
                }
            }
        }

        /// The derived Huffman tables of the scan, in `start_pass` order, and each scan
        /// component's offsets into them (DC, AC; -1 where the scan uses none).
        mutating func scanTables() throws(Stop) -> (tables: [Int32], dc: [Int], ac: [Int]) {
            var tables: [Int32] = []
            var dc: [Int] = []
            var ac: [Int] = []
            for index in scan.components {
                let component = components[index]
                if !progressive || (scan.ss == 0 && scan.ah == 0) {
                    dc.append(tables.count)
                    try Derived.append(
                        component.dcTable < 4 ? dcTables[component.dcTable] : nil, isDC: true,
                        slot: component.dcTable, to: &tables)
                } else {
                    dc.append(-1)
                }
                if !progressive || scan.ss != 0 {
                    ac.append(tables.count)
                    try Derived.append(
                        component.acTable < 4 ? acTables[component.acTable] : nil, isDC: false,
                        slot: component.acTable, to: &tables)
                } else {
                    ac.append(-1)
                }
            }
            return (tables, dc, ac)
        }

        // MARK: Scans

        /// `jpeg_start_decompress`'s loop for a JPEG of several scans: every scan, and the markers
        /// between them, up to EOI.
        mutating func consumeScans() throws(Stop) {
            while true {
                try decodeScan(singleScan: false)
                if try readMarkers() == .eoi { return }
                try startInputPass()
            }
        }

        /// One scan's entropy-coded data: `decode_mcu` of jdhuff.c or jdphuff.c for every MCU,
        /// in `consume_data`'s or `decompress_onepass`'s order, with restart markers, the MCUs
        /// after the data runs out left as they are, and Pillow's suspensions. A single-scan
        /// JPEG's MCU rows go to the inverse DCT as they are completed.
        mutating func decodeScan(singleScan: Bool) throws(Stop) {
            work.scans += 1
            let (tables, dcOffsets, acOffsets) = try scanTables()
            let interleaved = scan.components.count > 1
            let first = components[scan.components[0]]
            let mcusPerRow = interleaved ? mcusPerLine : first.widthInBlocks
            let mcuRows = interleaved ? totalIMCURows : first.heightInBlocks
            let totalMCUs = mcusPerRow * mcuRows
            /// The iMCU row an MCU belongs to (`input_iMCU_row`).
            func iMCURow(_ mcu: Int) -> Int {
                interleaved ? mcu / mcusPerRow : mcu / mcusPerRow / first.v
            }
            var members: [Member] = []
            for (slot, index) in scan.components.enumerated() {
                let component = components[index]
                let h = interleaved ? component.h : 1
                let v = interleaved ? component.v : 1
                for row in 0..<v {
                    for column in 0..<h {
                        members.append(
                            Member(
                                slot: slot, gridOffset: component.gridOffset,
                                gridWidth: component.gridWidth, rowFactor: v, columnFactor: h,
                                row: row, column: column,
                                storageRows: singleScan ? component.v : Int.max,
                                dcTable: dcOffsets[slot], acTable: acOffsets[slot]))
                    }
                }
            }
            let blocksInMCU = members.count
            let kind: ScanKind =
                !progressive
                ? .sequential
                : scan.ss == 0
                    ? (scan.ah == 0 ? .dcFirst : .dcRefine) : (scan.ah == 0 ? .acFirst : .acRefine)
            var bandMask: UInt64 = 0
            if kind == .acRefine {
                for k in scan.ss...scan.se { bandMask |= 1 << UInt64(naturalOrder[k]) }
            }
            let parameters = (ss: scan.ss, se: scan.se, al: scan.al)
            var predictors = Predictors()
            var eobrun = 0
            var restartsToGo = restartInterval
            // The iMCU row a single-scan JPEG is collecting, and whether any MCU of it was
            // decoded.
            var bufferedRow = 0
            var rowDecoded = false
            var newlyNonzero: [Int] = []
            newlyNonzero.reserveCapacity(64)
            var coefficients = self.coefficients
            self.coefficients = []
            var nonzero = self.nonzero
            self.nonzero = []
            defer {
                self.coefficients = coefficients
                self.nonzero = nonzero
            }
            let input = bytes
            try input.withUnsafeBufferPointer { (raw) throws(Stop) in
                var reader = BitReader(
                    input: raw, position: position, chunkEnd: chunkEnd, unreadMarker: unreadMarker)
                defer {
                    position = reader.position
                    chunkEnd = reader.chunkEnd
                    unreadMarker = reader.unreadMarker
                }
                try tables.withUnsafeBufferPointer { (table) throws(Stop) in
                    try naturalOrder.withUnsafeBufferPointer { (natural) throws(Stop) in
                        try coefficients.withUnsafeMutableBufferPointer { (store) throws(Stop) in
                            try nonzero.withUnsafeMutableBufferPointer { (masks) throws(Stop) in
                                var mcu = 0
                                while mcu < totalMCUs {
                                    if singleScan, iMCURow(mcu) != bufferedRow {
                                        flushRows(
                                            bufferedRow..<iMCURow(mcu), decoded: rowDecoded,
                                            store: store)
                                        bufferedRow = iMCURow(mcu)
                                        rowDecoded = false
                                    }
                                    // consume_data notes the last good row before decode_mcu
                                    // processes a restart marker, which may reset the flag.
                                    let good = !reader.insufficient
                                    if restartInterval != 0 && restartsToGo == 0 {
                                        // process_restart: the bits left are dropped, and the
                                        // next restart marker is looked for (or resynchronised
                                        // to).
                                        position = reader.position
                                        chunkEnd = reader.chunkEnd
                                        unreadMarker = reader.unreadMarker
                                        try readRestartMarker()
                                        reader.position = position
                                        reader.chunkEnd = chunkEnd
                                        reader.unreadMarker = unreadMarker
                                        reader.bitsLeft = 0
                                        if unreadMarker == 0 { reader.insufficient = false }
                                        predictors = Predictors()
                                        eobrun = 0
                                        restartsToGo = restartInterval
                                    }
                                    if reader.insufficient {
                                        // The data ran out: this segment's MCUs stay as they
                                        // are, and past a marker other than a restart marker,
                                        // every later segment is empty too (resync's action 3).
                                        let marker = reader.unreadMarker
                                        let skip: Int
                                        if restartInterval == 0
                                            || (marker >= 0xC0 && !(0xD0...0xD7).contains(marker))
                                        {
                                            skip = totalMCUs - mcu
                                        } else {
                                            skip = restartsToGo
                                            restartsToGo = 0
                                        }
                                        mcu += skip
                                        continue
                                    }
                                    if good { lastGoodIMCURow = iMCURow(mcu) }
                                    work.visits += blocksInMCU
                                    guard work.visits <= visitLimit else {
                                        throw Stop.refused(
                                            "the JPEG's scans would visit more than "
                                                + "\(visitLimit) blocks one at a time; this "
                                                + "decoder's limit on work")
                                    }
                                    let mcuRow = mcu / mcusPerRow
                                    let mcuColumn = mcu % mcusPerRow
                                    if kind == .acRefine, eobrun > 0,
                                        masks[members[0].offset(mcuRow, mcuColumn) / 64] & bandMask
                                            == 0
                                    {
                                        // A block of an end-of-band run with nothing to refine
                                        // reads no bits.
                                        eobrun -= 1
                                        if restartInterval != 0 { restartsToGo -= 1 }
                                        mcu += 1
                                        continue
                                    }
                                    work.blocks += blocksInMCU
                                    rowDecoded = true
                                    attempt: while true {
                                        let saved = (reader, predictors, eobrun)
                                        if singleScan {
                                            for member in members {
                                                (store.baseAddress!
                                                    + member.offset(mcuRow, mcuColumn))
                                                    .initialize(repeating: 0, count: 64)
                                            }
                                        }
                                        newlyNonzero.removeAll(keepingCapacity: true)
                                        do throws(Interrupt) {
                                            switch kind {
                                            case .sequential:
                                                if restartInterval == 0 && reader.unreadMarker == 0
                                                    && reader.chunkEnd - reader.position
                                                        >= 512 * blocksInMCU
                                                {
                                                    try LibjpegTurboDecoder.sequentialMCU(
                                                        members, mcuRow, mcuColumn, &reader,
                                                        &predictors, table.baseAddress!,
                                                        natural.baseAddress!, store.baseAddress!,
                                                        fast: true)
                                                    if reader.unreadMarker == 0 { break attempt }
                                                    // The slow path decodes the MCU again, over
                                                    // what the fast one wrote.
                                                    reader = saved.0
                                                    predictors = saved.1
                                                }
                                                try LibjpegTurboDecoder.sequentialMCU(
                                                    members, mcuRow, mcuColumn, &reader,
                                                    &predictors, table.baseAddress!,
                                                    natural.baseAddress!, store.baseAddress!,
                                                    fast: false)
                                            case .dcFirst:
                                                for member in members {
                                                    var s = try reader.decode(
                                                        table.baseAddress! + member.dcTable)
                                                    if s != 0 { s = extend(try reader.bits(s), s) }
                                                    let last = predictors[member.slot]
                                                    let sum = last + s
                                                    guard sum >= Int(Int32.min),
                                                        sum <= Int(Int32.max)
                                                    else { throw Interrupt.overflow }
                                                    predictors[member.slot] = sum
                                                    store[member.offset(mcuRow, mcuColumn)] =
                                                        Int16(
                                                            truncatingIfNeeded: sum << parameters.al
                                                        )
                                                }
                                            case .dcRefine:
                                                for member in members {
                                                    if try reader.bits(1) != 0 {
                                                        store[member.offset(mcuRow, mcuColumn)] |=
                                                            Int16(1 << parameters.al)
                                                    }
                                                }
                                            case .acFirst:
                                                let offset = members[0].offset(mcuRow, mcuColumn)
                                                try LibjpegTurboDecoder.acFirstBlock(
                                                    store.baseAddress! + offset,
                                                    mask: &masks[offset / 64], &reader,
                                                    table.baseAddress! + members[0].acTable,
                                                    natural.baseAddress!, parameters,
                                                    eobrun: &eobrun)
                                            case .acRefine:
                                                let offset = members[0].offset(mcuRow, mcuColumn)
                                                try LibjpegTurboDecoder.acRefineBlock(
                                                    store.baseAddress! + offset,
                                                    mask: &masks[offset / 64], &reader,
                                                    table.baseAddress! + members[0].acTable,
                                                    natural.baseAddress!, parameters,
                                                    bandMask: bandMask, eobrun: &eobrun,
                                                    newlyNonzero: &newlyNonzero)
                                            }
                                            break attempt
                                        } catch {
                                            switch error {
                                            case .chunk:
                                                // Pillow reads 65,536 more bytes, and
                                                // libjpeg-turbo starts the MCU again; a
                                                // refinement first undoes its new coefficients.
                                                if kind == .acRefine {
                                                    let offset = members[0].offset(
                                                        mcuRow, mcuColumn)
                                                    for position in newlyNonzero {
                                                        store[offset + position] = 0
                                                        masks[offset / 64] &=
                                                            ~(1 << UInt64(position))
                                                    }
                                                }
                                                let extended = min(
                                                    raw.count, reader.chunkEnd + readSize)
                                                (reader, predictors, eobrun) = saved
                                                reader.chunkEnd = extended
                                            case .end:
                                                throw Stop.endOfData
                                            case .overflow:
                                                throw LibjpegTurboDecoder.error(
                                                    "JERR_BAD_DCT_COEF",
                                                    "a DC coefficient of a progressive scan "
                                                        + "overflows 32 bits")
                                            }
                                        }
                                    }
                                    if restartInterval != 0 { restartsToGo -= 1 }
                                    mcu += 1
                                    // An end-of-band run of a first AC scan leaves the next
                                    // blocks as they are, up to the next restart marker.
                                    if kind == .acFirst, eobrun > 0, !reader.insufficient {
                                        var skip = min(eobrun, totalMCUs - mcu)
                                        if restartInterval != 0 { skip = min(skip, restartsToGo) }
                                        if skip > 0 {
                                            eobrun -= skip
                                            mcu += skip
                                            if restartInterval != 0 { restartsToGo -= skip }
                                            lastGoodIMCURow = iMCURow(mcu - 1)
                                        }
                                    }
                                }
                                if singleScan {
                                    flushRows(
                                        bufferedRow..<(iMCURow(totalMCUs - 1) + 1),
                                        decoded: rowDecoded, store: store)
                                }
                            }
                        }
                    }
                }
            }
        }

        /// `read_restart_marker`: the expected restart marker, or `jpeg_resync_to_restart`'s
        /// recovery when another marker is found.
        mutating func readRestartMarker() throws(Stop) {
            work.restarts += 1
            if unreadMarker == 0 { try nextMarker() }
            if unreadMarker == 0xD0 + nextRestartNumber {
                unreadMarker = 0
            } else {
                // jpeg_resync_to_restart: discard a restart marker too far from the expected
                // one, scan on past an earlier one or a code that is not a marker, and leave a
                // later one, or any other marker, for the segments to come.
                let desired = nextRestartNumber
                while true {
                    let marker = unreadMarker
                    if marker < 0xC0 {
                        try nextMarker()
                    } else if marker < 0xD0 || marker > 0xD7 {
                        break
                    } else if marker == 0xD0 + ((desired + 1) & 7)
                        || marker == 0xD0 + ((desired + 2) & 7)
                    {
                        break
                    } else if marker == 0xD0 + ((desired - 1) & 7)
                        || marker == 0xD0 + ((desired - 2) & 7)
                    {
                        try nextMarker()
                    } else {
                        unreadMarker = 0
                        break
                    }
                }
            }
            nextRestartNumber = (nextRestartNumber + 1) & 7
        }

        // MARK: Arithmetic-coded and lossless scans, walked without decoding

        /// How far libjpeg-turbo reads a scan's data, as far as that is known without decoding it.
        enum Reach: Equatable {
            /// No further than the marker that ends the data, which lies within the bytes Pillow
            /// has handed over.
            case settled
            /// The data runs on past the bytes Pillow has handed over, and how far into it the
            /// entropy decoder reads depends on decoding it: at most to the marker whose code byte
            /// is at `marker`, or to the end of the file when it is nil.
            case open(marker: Int?)
        }

        /// The reading `ImagingJpegDecode` makes of an arithmetic-coded or lossless JPEG, which
        /// this decoder does not decode, after the first scan's checks: every scan's data passed
        /// over as ``walkScanData()`` reads it, the markers between scans with each later scan's
        /// checks, up to EOI; or, after a single scan, the markers Pillow reads once the image is
        /// complete.
        ///
        /// Neither jdarith.c nor jdlhuff.c raises on the content of scan data: a bad code is a
        /// warning, and data that runs out is filled with zeros. What the data decides is how far
        /// libjpeg-turbo reads. jdarith.c cannot suspend, so reading past the bytes Pillow has
        /// handed over raises (``readLimit``); jdlhuff.c suspends, and a read past the end of the
        /// file is Pillow's "image file is truncated"; and after a single scan's last row Pillow
        /// reads no further than it has. Where that depends on decoding the data, this throws
        /// ``Stop/unsupported(_:)`` saying so, unless something later raises whatever the data
        /// holds. It throws a refusal where libjpeg-turbo or Pillow raises, and
        /// ``Stop/unsupported(_:)`` where Pillow decodes the JPEG.
        mutating func walkUndecodedScans() throws(Stop) -> Never {
            if !hasMultipleScans {
                switch try walkScanData() {
                case .settled:
                    try finishSingleScan()
                case .open(let marker?) where lossless:
                    // jdlhuff.c reads at most to the marker, so when the last row is out Pillow has
                    // read at most the 65,536 bytes that hold it, and at least what it has now. A
                    // fault after the scan raises only if Pillow has read that far.
                    let fewest = chunkEnd
                    chunkEnd = min(bytes.count, (marker / readSize + 1) * readSize)
                    do throws(Stop) {
                        try finishSingleScan()
                    } catch {
                        guard case .refused(let fault) = error else { throw error }
                        throw .unsupported(
                            "a lossless JPEG's scan ends past the \(fewest) bytes Pillow is sure "
                                + "to have read by its last row, and a fault follows (\(fault)); "
                                + "whether Pillow has read that far when the last row is out depends "
                                + "on decoding the scan")
                    }
                case .open where lossless:
                    throw .unsupported(
                        "a lossless JPEG's scan runs to the end of the file without a marker; "
                            + "whether libjpeg-turbo reads past the end before the last row is out "
                            + "(Pillow's \"image file is truncated\") depends on decoding the scan")
                case .open:
                    throw .unsupported(openArithmeticScan)
                }
                throw unsupportedFeature()
            }
            // Every scan is read in jpeg_start_decompress, up to EOI.
            var undecided: String?
            var reached = Set(scan.components)
            do throws(Stop) {
                while true {
                    if try walkScanData() != .settled, arithmetic, undecided == nil {
                        undecided = openArithmeticScan
                    }
                    if try readMarkers() == .eoi { break }
                    if lossless { try checkLosslessScan() } else { try startInputPass() }
                    reached.formUnion(scan.components)
                }
            } catch {
                if case .endOfData = error, undecided != nil {
                    throw .refused(
                        "the JPEG ends before its EOI (Pillow's \"image file is truncated\", or "
                            + "\"broken data stream\" where the arithmetic decoder runs out first)")
                }
                throw error
            }
            // jddiffct.c keeps a lossless JPEG's samples in arrays it does not ask jmemmgr.c to
            // zero (jdcoefct.c does), so the first output row reads a component no scan wrote:
            // JERR_BAD_VIRTUAL_ACCESS.
            if lossless, let missing = components.indices.first(where: { !reached.contains($0) }) {
                throw error(
                    "JERR_BAD_VIRTUAL_ACCESS",
                    "no scan reaches component \(components[missing].id) of a lossless JPEG, whose "
                        + "samples jddiffct.c then reads undefined")
            }
            throw undecided.map { .unsupported($0) } ?? unsupportedFeature()
        }

        /// Why an arithmetic-coded JPEG whose scan may read past Pillow's buffer goes to ImageIO.
        var openArithmeticScan: String {
            "an arithmetic-coded scan's data runs on past the bytes Pillow had read when the scan "
                + "started, or to the end of the file without a marker; whether jdarith.c reads past "
                + "them, which Pillow answers with \"broken data stream\" (JERR_CANT_SUSPEND), "
                + "depends on decoding the scan"
        }

        /// One scan's entropy-coded data as libjpeg-turbo reads it, without decoding it: the
        /// restart markers, which `read_restart_marker` and `jpeg_resync_to_restart` look for
        /// every `restart_interval` MCUs (counted in MCU rows in a lossless scan, by jddiffct.c,
        /// to the same effect), and the bytes the entropy decoder is sure to read of the last
        /// segment. jdarith.c's first decision of a segment loads two bytes. jdlhuff.c's first
        /// fill loads 57 bits, eight bytes, and its samples take at least so many bits each: their
        /// Huffman code and the difference bits after it, or the 17 bits of a string that is no
        /// code. Each stops early at a marker. The other segments are read to the restart marker
        /// that ends them, whatever their data holds. In an arithmetic-coded scan, none of it may
        /// lie past the bytes Pillow handed over before the scan (``readLimit``).
        ///
        /// A marker other than RST that `jpeg_resync_to_restart` leaves stays for every later
        /// segment, which then reads nothing, so the work is in proportion to the bytes read.
        mutating func walkScanData() throws(Stop) -> Reach {
            work.scans += 1
            let interleaved = scan.components.count > 1
            let first = components[scan.components[0]]
            let mcusPerRow = interleaved ? mcusPerLine : first.widthInBlocks
            let totalMCUs = mcusPerRow * (interleaved ? totalIMCURows : first.heightInBlocks)
            let segments = restartInterval > 0 ? (totalMCUs - 1) / restartInterval + 1 : 1
            if arithmetic { readLimit = chunkEnd }
            defer { readLimit = .max }
            var segment = 1
            while segment < segments {
                segment += 1
                try readRestartMarker()
                if unreadMarker >= 0xC0 && !(0xD0...0xD7).contains(unreadMarker) { break }
            }
            if unreadMarker == 0 {
                // The last segment, which is not empty.
                if arithmetic {
                    try readEntropyBytes(2)
                } else {
                    // checkLosslessScan made sure every table is there; a bit a sample is the
                    // least any table can give.
                    var bits = 0
                    for index in scan.components {
                        let component = components[index]
                        let samples = interleaved ? component.h * component.v : 1
                        let table = component.dcTable < 4 ? dcTables[component.dcTable] : nil
                        bits += samples * (table.map(LibjpegTurboDecoder.fewestSampleBits) ?? 1)
                    }
                    let mcus = totalMCUs - (segments - 1) * restartInterval
                    try readEntropyBytes(max(8, (mcus * bits + 7) / 8))
                }
            }
            if unreadMarker != 0 { return .settled }
            let marker = markerAhead()
            if let marker, marker < chunkEnd { return .settled }
            return .open(marker: marker)
        }

        /// Up to `count` bytes of entropy-coded data, read as the entropy decoders fetch them (FF
        /// 00 is one byte, and FF fill before a marker is swallowed), stopping at a marker, which
        /// is left in `unreadMarker`.
        mutating func readEntropyBytes(_ count: Int) throws(Stop) {
            var left = count
            while left > 0 {
                var c = try byte()
                if c == 0xFF {
                    repeat { c = try byte() } while c == 0xFF
                    if c != 0 {
                        unreadMarker = c
                        return
                    }
                }
                left -= 1
            }
        }

        /// The index of the code byte of the marker `next_marker` would find from `position`,
        /// without reading anything; nil when the file ends first.
        func markerAhead() -> Int? {
            var i = position
            while i < bytes.count {
                guard bytes[i] == 0xFF else {
                    i += 1
                    continue
                }
                var j = i + 1
                while j < bytes.count && bytes[j] == 0xFF { j += 1 }
                guard j < bytes.count else { return nil }
                if bytes[j] != 0 { return j }
                i = j + 1
            }
            return nil
        }

        /// The inverse DCT of a single-scan JPEG's completed MCU rows, as `decompress_onepass`
        /// runs it: the blocks inside the image, dummy blocks skipped. A row no MCU of which was
        /// decoded holds only zero blocks, which come out as 128.
        mutating func flushRows(
            _ rows: Range<Int>, decoded: Bool, store: UnsafeMutableBufferPointer<Int16>
        ) {
            guard !rows.isEmpty, components.count != 4 else { return }
            var multipliers = [Int16](repeating: 0, count: 64)
            var workspace = [Int16](repeating: 0, count: 64)
            for c in components.indices {
                let component = components[c]
                let stride = component.widthInBlocks * 8
                for (i, value) in (component.quant ?? []).enumerated() {
                    multipliers[i] = Int16(truncatingIfNeeded: value)
                }
                // Only the first row can hold decoded MCUs; the others stay 128.
                guard decoded else { continue }
                let first = rows.lowerBound * component.v
                let blockRows = first..<min(first + component.v, component.heightInBlocks)
                planes[c].withUnsafeMutableBufferPointer { plane in
                    multipliers.withUnsafeBufferPointer { m in
                        workspace.withUnsafeMutableBufferPointer { w in
                            for blockRow in blockRows {
                                for column in 0..<component.widthInBlocks {
                                    let block =
                                        store.baseAddress!
                                        + (component.gridOffset
                                            + (blockRow % component.v) * component.gridWidth
                                            + column) * 64
                                    if LibjpegTurboDecoder.isZero(block) { continue }
                                    LibjpegTurboDecoder.inverseDCT(
                                        block, m.baseAddress!, w.baseAddress!,
                                        plane.baseAddress! + blockRow * 8 * stride + column * 8,
                                        stride: stride)
                                }
                            }
                        }
                    }
                }
            }
            if decoded {
                store.baseAddress!.initialize(repeating: 0, count: store.count)
            }
        }

        // MARK: The inverse DCT of a multi-scan JPEG and block smoothing

        /// `decompress_data`, or `decompress_smooth_data` when `smoothing_ok`: every component's
        /// blocks to samples. A component no scan reached has an all-zero multiplier table
        /// (jddctmgr.c) and comes out as 128.
        mutating func makePlanesFromCoefficients() {
            let latches = smoothingLatches()
            let storage = coefficients
            coefficients = []
            nonzero = []
            planes = []
            var workspace = [Int16](repeating: 0, count: 64)
            var smoothed = [Int16](repeating: 0, count: 64)
            var registers = [Int](repeating: 0, count: 26)
            for c in components.indices {
                let component = components[c]
                let stride = component.widthInBlocks * 8
                var plane = [UInt8](repeating: 128, count: stride * component.heightInBlocks * 8)
                guard let quant = component.quant else {
                    planes.append(plane)
                    continue
                }
                let multipliers = quant.map { Int16(truncatingIfNeeded: $0) }
                // The rows past the last good one take coef_bits from before the last scan.
                let smoothings = latches.map {
                    (
                        current: Smoothing(
                            component: component, quant: quant, coefBits: $0.current[c]),
                        previous: Smoothing(
                            component: component, quant: quant, coefBits: $0.previous[c])
                    )
                }
                storage.withUnsafeBufferPointer { store in
                    plane.withUnsafeMutableBufferPointer { out in
                        multipliers.withUnsafeBufferPointer { m in
                            workspace.withUnsafeMutableBufferPointer { w in
                                smoothed.withUnsafeMutableBufferPointer { s in
                                    registers.withUnsafeMutableBufferPointer { d in
                                        for blockRow in 0..<component.heightInBlocks {
                                            let iMCURow = blockRow / component.v
                                            let smoothing = smoothings.map {
                                                iMCURow > lastGoodIMCURow ? $0.previous : $0.current
                                            }
                                            let neighbours =
                                                smoothing == nil
                                                ? nil
                                                : Self.neighbourRows(
                                                    blockRow, component: component,
                                                    totalIMCURows: totalIMCURows)
                                            let rowBase =
                                                component.gridOffset + blockRow
                                                * component.gridWidth
                                            let rowOut = out.baseAddress! + blockRow * 8 * stride
                                            for column in 0..<component.widthInBlocks {
                                                var block =
                                                    store.baseAddress! + (rowBase + column) * 64
                                                if smoothing == nil,
                                                    LibjpegTurboDecoder.isZero(block)
                                                {
                                                    continue
                                                }
                                                if let smoothing, let neighbours {
                                                    smoothing.estimate(
                                                        block, column: column, rows: neighbours,
                                                        store: store.baseAddress!,
                                                        into: s.baseAddress!,
                                                        registers: d.baseAddress!)
                                                    block = UnsafePointer(s.baseAddress!)
                                                }
                                                LibjpegTurboDecoder.inverseDCT(
                                                    block, m.baseAddress!, w.baseAddress!,
                                                    rowOut + column * 8, stride: stride)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                planes.append(plane)
            }
        }

        /// `smoothing_ok`: whether block smoothing applies, and the latched `coef_bits` of each
        /// component for coefficients 0 to 9, as of the last scan and before it.
        func smoothingLatches() -> (current: [[Int]], previous: [[Int]])? {
            guard progressive else { return nil }
            var useful = false
            var current: [[Int]] = []
            var previous: [[Int]] = []
            for component in components {
                guard let quant = component.quant,
                    [0, 1, 8, 16, 9, 2, 3, 10, 17, 24].allSatisfy({ quant[$0] != 0 }),
                    component.coefBits[0] >= 0
                else { return nil }
                var latch = [Int](repeating: 0, count: 10)
                var previousLatch = [Int](repeating: 0, count: 10)
                latch[0] = component.coefBits[0]
                for k in 1..<10 {
                    previousLatch[k] = inputScanNumber > 1 ? component.previousCoefBits[k] : -1
                    latch[k] = component.coefBits[k]
                    if component.coefBits[k] != 0 { useful = true }
                }
                current.append(latch)
                previous.append(previousLatch)
            }
            return useful ? (current, previous) : nil
        }

        /// The block rows `decompress_smooth_data` takes DC values from for a block row: two
        /// above, two below, repeating the edge rows, with libjpeg-turbo's count of the image's
        /// block rows (which, in the last iMCU row, uses that row's height for every row).
        static func neighbourRows(
            _ blockRow: Int, component: Component, totalIMCURows: Int
        ) -> (Int, Int, Int, Int, Int) {
            let v = component.v
            let iMCURow = blockRow / v
            let rowInIMCU = blockRow % v
            var blockRows = v
            if iMCURow == totalIMCURows - 1 {
                blockRows = component.heightInBlocks % v
                if blockRows == 0 { blockRows = v }
            }
            let imageBlockRow = iMCURow * blockRows + rowInIMCU
            let imageBlockRows = blockRows * totalIMCURows
            let previous = imageBlockRow > 0 ? blockRow - 1 : blockRow
            let previousPrevious = imageBlockRow > 1 ? blockRow - 2 : previous
            let next = imageBlockRow < imageBlockRows - 1 ? blockRow + 1 : blockRow
            let nextNext = imageBlockRow < imageBlockRows - 2 ? blockRow + 2 : next
            return (previousPrevious, previous, blockRow, next, nextNext)
        }

        // MARK: Output

        /// Upsampling and colour conversion, a row at a time, into the image.
        func output() -> RGBImage {
            var pixels = [UInt8](repeating: 0, count: width * height * 3)
            let methods = components.map { upsampling(for: $0) }
            let count = components.count
            var rowStorage = [UInt8](repeating: 0, count: width * count)
            var scratch = [Int](repeating: 0, count: width / 2 + 3)
            let tables = ColorTables.shared
            let width = width
            let height = height
            let colorSpace = colorSpace
            let components = components
            withPlanes { planes in
                pixels.withUnsafeMutableBufferPointer { out in
                    rowStorage.withUnsafeMutableBufferPointer { rowBuffer in
                        scratch.withUnsafeMutableBufferPointer { scratch in
                            tables.crR.withUnsafeBufferPointer { crR in
                                tables.cbB.withUnsafeBufferPointer { cbB in
                                    tables.crG.withUnsafeBufferPointer { crG in
                                        tables.cbG.withUnsafeBufferPointer { cbG in
                                            var rows = [UnsafePointer<UInt8>](
                                                repeating: UnsafePointer(rowBuffer.baseAddress!),
                                                count: count)
                                            for y in 0..<height {
                                                for c in 0..<count {
                                                    let stride = components[c].widthInBlocks * 8
                                                    if methods[c] == .full {
                                                        rows[c] =
                                                            planes[c].baseAddress! + y * stride
                                                    } else {
                                                        let target =
                                                            rowBuffer.baseAddress! + c * width
                                                        LibjpegTurboDecoder.upsampleRow(
                                                            y, methods[c], components[c],
                                                            plane: planes[c].baseAddress!,
                                                            width: width, into: target,
                                                            scratch: scratch.baseAddress!)
                                                        rows[c] = UnsafePointer(target)
                                                    }
                                                }
                                                let o = out.baseAddress! + y * width * 3
                                                switch colorSpace {
                                                case .grey:
                                                    let g = rows[0]
                                                    for x in 0..<width {
                                                        o[x * 3] = g[x]
                                                        o[x * 3 + 1] = g[x]
                                                        o[x * 3 + 2] = g[x]
                                                    }
                                                case .rgb:
                                                    let (r, g, b) = (rows[0], rows[1], rows[2])
                                                    for x in 0..<width {
                                                        o[x * 3] = r[x]
                                                        o[x * 3 + 1] = g[x]
                                                        o[x * 3 + 2] = b[x]
                                                    }
                                                default:
                                                    let (ys, cbs, crs) = (rows[0], rows[1], rows[2])
                                                    for x in 0..<width {
                                                        let luma = Int(ys[x])
                                                        let cb = Int(cbs[x])
                                                        let cr = Int(crs[x])
                                                        o[x * 3] = clamp(luma + crR[cr])
                                                        o[x * 3 + 1] = clamp(
                                                            luma + ((cbG[cb] + crG[cr]) >> 16))
                                                        o[x * 3 + 2] = clamp(luma + cbB[cb])
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return RGBImage(width: width, height: height, pixels: pixels)
        }

        /// Calls `body` with a pointer to every component's samples.
        func withPlanes(
            _ index: Int = 0, _ gathered: [UnsafeBufferPointer<UInt8>] = [],
            _ body: ([UnsafeBufferPointer<UInt8>]) -> Void
        ) {
            if index == planes.count { return body(gathered) }
            planes[index].withUnsafeBufferPointer { withPlanes(index + 1, gathered + [$0], body) }
        }

        /// `jinit_upsampler`'s choice of method for a component.
        func upsampling(for component: Component) -> Upsampling {
            let hSame = component.h == maxH
            let vSame = component.v == maxV
            if hSame && vSame { return .full }
            if component.h * 2 == maxH && vSame {
                return component.width > 2 ? .h2v1Fancy : .replicate(2, 1)
            }
            if hSame && component.v * 2 == maxV { return .h1v2Fancy }
            if component.h * 2 == maxH && component.v * 2 == maxV {
                return component.width > 2 ? .h2v2Fancy : .replicate(2, 2)
            }
            return .replicate(maxH / component.h, maxV / component.v)
        }
    }

    /// The kinds of scan, by their decoding routine.
    enum ScanKind { case sequential, dcFirst, dcRefine, acFirst, acRefine }

    /// The last DC values of the scan's components (`last_dc_val`).
    struct Predictors {
        var values: (Int, Int, Int, Int) = (0, 0, 0, 0)

        subscript(_ index: Int) -> Int {
            get {
                switch index {
                case 0: values.0
                case 1: values.1
                case 2: values.2
                default: values.3
                }
            }
            set {
                switch index {
                case 0: values.0 = newValue
                case 1: values.1 = newValue
                case 2: values.2 = newValue
                default: values.3 = newValue
                }
            }
        }
    }

    /// One block of an MCU: its component's place in the scan, the storage and the tables.
    struct Member {
        var slot: Int
        var gridOffset: Int
        var gridWidth: Int
        /// Blocks per MCU down and across (the sampling factors in an interleaved scan, else 1).
        var rowFactor: Int
        var columnFactor: Int
        var row: Int
        var column: Int
        /// Block rows the storage holds: one MCU row for a single-scan JPEG, else all.
        var storageRows: Int
        var dcTable: Int
        var acTable: Int

        /// The block's first coefficient in the storage for MCU (`mcuRow`, `mcuColumn`).
        @inline(__always) func offset(_ mcuRow: Int, _ mcuColumn: Int) -> Int {
            let blockRow = (mcuRow * rowFactor + row) % storageRows
            return (gridOffset + blockRow * gridWidth + mcuColumn * columnFactor + column) * 64
        }
    }

    /// `decode_mcu_slow` and `decode_mcu_fast` of jdhuff.c: the same decoding, with the slow
    /// path's or the fast path's refills.
    @inline(__always)
    static func sequentialMCU(
        _ members: [Member], _ mcuRow: Int, _ mcuColumn: Int, _ reader: inout BitReader,
        _ predictors: inout Predictors, _ tables: UnsafePointer<Int32>,
        _ natural: UnsafePointer<Int>, _ store: UnsafeMutablePointer<Int16>, fast: Bool
    ) throws(Interrupt) {
        for member in members {
            let block = store + member.offset(mcuRow, mcuColumn)
            let dc = tables + member.dcTable
            let ac = tables + member.acTable
            var s = fast ? reader.fastDecode(dc) : try reader.decode(dc)
            if s != 0 { s = extend(fast ? reader.fastBits(s) : try reader.bits(s), s) }
            predictors[member.slot] += s
            block[0] = Int16(truncatingIfNeeded: predictors[member.slot])
            var k = 1
            while k < 64 {
                let rs = fast ? reader.fastDecode(ac) : try reader.decode(ac)
                let r = rs >> 4
                let size = rs & 15
                if size != 0 {
                    k += r
                    let bits = fast ? reader.fastBits(size) : try reader.bits(size)
                    block[natural[k]] = Int16(truncatingIfNeeded: extend(bits, size))
                } else {
                    if r != 15 { break }
                    k += 15
                }
                k += 1
            }
        }
    }

    /// `decode_mcu_AC_first`, for one block.
    @inline(__always)
    static func acFirstBlock(
        _ block: UnsafeMutablePointer<Int16>, mask: inout UInt64, _ reader: inout BitReader,
        _ table: UnsafePointer<Int32>, _ natural: UnsafePointer<Int>,
        _ parameters: (ss: Int, se: Int, al: Int), eobrun: inout Int
    ) throws(Interrupt) {
        if eobrun > 0 {
            eobrun -= 1
            return
        }
        var k = parameters.ss
        while k <= parameters.se {
            let rs = try reader.decode(table)
            let r = rs >> 4
            let s = rs & 15
            if s != 0 {
                k += r
                let value = Int16(
                    truncatingIfNeeded: extend(try reader.bits(s), s) << parameters.al)
                let position = natural[k]
                block[position] = value
                if value != 0 {
                    mask |= 1 << UInt64(position)
                } else {
                    mask &= ~(1 << UInt64(position))
                }
            } else if r == 15 {
                k += 15
            } else {
                eobrun = 1 << r
                if r != 0 { eobrun += try reader.bits(r) }
                eobrun -= 1
                break
            }
            k += 1
        }
    }

    /// `decode_mcu_AC_refine`, for one block. A block of an end-of-band run with no nonzero
    /// coefficient in the band reads nothing, which `mask` shows without looking.
    @inline(__always)
    static func acRefineBlock(
        _ block: UnsafeMutablePointer<Int16>, mask: inout UInt64, _ reader: inout BitReader,
        _ table: UnsafePointer<Int32>, _ natural: UnsafePointer<Int>,
        _ parameters: (ss: Int, se: Int, al: Int), bandMask: UInt64, eobrun: inout Int,
        newlyNonzero: inout [Int]
    ) throws(Interrupt) {
        let p1 = 1 << parameters.al
        let m1 = -1 << parameters.al
        let se = parameters.se
        var k = parameters.ss
        /// A correction bit for an already nonzero coefficient.
        @inline(__always) func refine(_ position: Int) throws(Interrupt) {
            let value = Int(block[position])
            if try reader.bits(1) != 0, value & p1 == 0 {
                block[position] = Int16(truncatingIfNeeded: value >= 0 ? value + p1 : value + m1)
            }
        }
        if eobrun == 0 {
            while k <= se {
                let rs = try reader.decode(table)
                var r = rs >> 4
                var s = rs & 15
                if s != 0 {
                    // A newly nonzero coefficient is always of size 1 (JWRN_HUFF_BAD_CODE).
                    s = try reader.bits(1) != 0 ? p1 : m1
                } else if r != 15 {
                    eobrun = 1 << r
                    if r != 0 { eobrun += try reader.bits(r) }
                    break
                }
                repeat {
                    let position = natural[k]
                    if block[position] != 0 {
                        try refine(position)
                    } else {
                        r -= 1
                        if r < 0 { break }
                    }
                    k += 1
                } while k <= se
                if s != 0 {
                    let position = natural[k]
                    block[position] = Int16(truncatingIfNeeded: s)
                    mask |= 1 << UInt64(position)
                    newlyNonzero.append(position)
                }
                k += 1
            }
        }
        if eobrun > 0 {
            if mask & bandMask != 0 {
                while k <= se {
                    let position = natural[k]
                    if block[position] != 0 { try refine(position) }
                    k += 1
                }
            }
            eobrun -= 1
        }
    }

    // MARK: Block smoothing

    /// `decompress_smooth_data` for one component: estimates of the first nine AC coefficients
    /// (and, while no AC data has arrived, the DC) from the DC values of the 5 by 5 blocks
    /// around each block.
    struct Smoothing {
        var widthInBlocks: Int
        var gridOffset: Int
        var gridWidth: Int
        /// `coef_bits` for coefficients 0 to 9 (zigzag order).
        var coefBits: [Int]
        var changeDC: Bool
        var q: [Int]

        init(component: Component, quant: [UInt16], coefBits: [Int]) {
            widthInBlocks = component.widthInBlocks
            gridOffset = component.gridOffset
            gridWidth = component.gridWidth
            self.coefBits = coefBits
            changeDC = (1...9).allSatisfy { coefBits[$0] == -1 }
            // Q00, Q01, Q10, Q20, Q11, Q02, Q03, Q12, Q21, Q30 at their natural positions.
            q = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24].map { Int(quant[$0]) }
        }

        /// The block at `column` of the current row with its estimates, into `workspace`.
        func estimate(
            _ block: UnsafePointer<Int16>, column: Int, rows: (Int, Int, Int, Int, Int),
            store: UnsafePointer<Int16>, into workspace: UnsafeMutablePointer<Int16>,
            registers d: UnsafeMutablePointer<Int>
        ) {
            workspace.update(from: block, count: 64)
            let last = widthInBlocks - 1
            /// The DC value of the block `dx` columns from `column` in neighbour row `row`, held
            /// at the row's ends as libjpeg-turbo's sliding registers hold it.
            func dc(_ row: Int, _ dx: Int) -> Int {
                var x = column + dx
                if dx < 0 { x = max(x, 0) }
                if dx > 0 { x = min(x, last) }
                return Int(store[(gridOffset + row * gridWidth + x) * 64])
            }
            // DC01...DC25 in d[1...25]: rows from two above to two below, columns from two left
            // to two right.
            func load(_ i: Int, _ row: Int) {
                for j in 0..<5 { d[i * 5 + j + 1] = dc(row, j - 2) }
            }
            load(0, rows.0)
            load(1, rows.1)
            load(2, rows.2)
            load(3, rows.3)
            load(4, rows.4)
            let q00 = q[0]
            /// One estimate: `num` over `Q << 8`, rounded half away from zero, limited to the bits
            /// not yet received, as libjpeg-turbo computes it in `JLONG` and stores it as `JCOEF`.
            func predict(_ num: Int, _ qk: Int, _ al: Int, limit: Bool) -> Int16 {
                var pred: Int32
                if num >= 0 {
                    pred = Int32(truncatingIfNeeded: ((qk << 7) + num) / (qk << 8))
                    if limit, al > 0, pred >= Int32(1 << al) { pred = Int32(1 << al) - 1 }
                } else {
                    pred = Int32(truncatingIfNeeded: ((qk << 7) - num) / (qk << 8))
                    if limit, al > 0, pred >= Int32(1 << al) { pred = Int32(1 << al) - 1 }
                    pred = 0 &- pred
                }
                return Int16(truncatingIfNeeded: pred)
            }
            let changeDC = changeDC
            // AC01
            if coefBits[1] != 0 && workspace[1] == 0 {
                let num =
                    q00
                    * (changeDC
                        ? (-d[1] - d[2] + d[4] + d[5] - 3 * d[6] + 13 * d[7] - 13 * d[9] + 3 * d[10]
                            - 3 * d[11] + 38 * d[12] - 38 * d[14] + 3 * d[15] - 3 * d[16]
                            + 13 * d[17] - 13 * d[19] + 3 * d[20] - d[21] - d[22] + d[24] + d[25])
                        : (-7 * d[11] + 50 * d[12] - 50 * d[14] + 7 * d[15]))
                workspace[1] = predict(num, q[1], coefBits[1], limit: true)
            }
            // AC10
            if coefBits[2] != 0 && workspace[8] == 0 {
                let num =
                    q00
                    * (changeDC
                        ? (-d[1] - 3 * d[2] - 3 * d[3] - 3 * d[4] - d[5] - d[6] + 13 * d[7]
                            + 38 * d[8] + 13 * d[9] - d[10] + d[16] - 13 * d[17] - 38 * d[18]
                            - 13 * d[19] + d[20] + d[21] + 3 * d[22] + 3 * d[23] + 3 * d[24]
                            + d[25])
                        : (-7 * d[3] + 50 * d[8] - 50 * d[18] + 7 * d[23]))
                workspace[8] = predict(num, q[2], coefBits[2], limit: true)
            }
            // AC20
            if coefBits[3] != 0 && workspace[16] == 0 {
                let num =
                    q00
                    * (changeDC
                        ? (d[3] + 2 * d[7] + 7 * d[8] + 2 * d[9] - 5 * d[12] - 14 * d[13]
                            - 5 * d[14] + 2 * d[17] + 7 * d[18] + 2 * d[19] + d[23])
                        : (-d[3] + 13 * d[8] - 24 * d[13] + 13 * d[18] - d[23]))
                workspace[16] = predict(num, q[3], coefBits[3], limit: true)
            }
            // AC11
            if coefBits[4] != 0 && workspace[9] == 0 {
                let num =
                    q00
                    * (changeDC
                        ? (-d[1] + d[5] + 9 * d[7] - 9 * d[9] - 9 * d[17] + 9 * d[19] + d[21]
                            - d[25])
                        : (d[10] + d[16] - 10 * d[17] + 10 * d[19] - d[2] - d[20] + d[22] - d[24]
                            + d[4] - d[6] + 10 * d[7] - 10 * d[9]))
                workspace[9] = predict(num, q[4], coefBits[4], limit: true)
            }
            // AC02
            if coefBits[5] != 0 && workspace[2] == 0 {
                let num =
                    q00
                    * (changeDC
                        ? (2 * d[7] - 5 * d[8] + 2 * d[9] + d[11] + 7 * d[12] - 14 * d[13]
                            + 7 * d[14] + d[15] + 2 * d[17] - 5 * d[18] + 2 * d[19])
                        : (-d[11] + 13 * d[12] - 24 * d[13] + 13 * d[14] - d[15]))
                workspace[2] = predict(num, q[5], coefBits[5], limit: true)
            }
            if changeDC {
                // AC03
                if coefBits[6] != 0 && workspace[3] == 0 {
                    let num = q00 * (d[7] - d[9] + 2 * d[12] - 2 * d[14] + d[17] - d[19])
                    workspace[3] = predict(num, q[6], coefBits[6], limit: true)
                }
                // AC12
                if coefBits[7] != 0 && workspace[10] == 0 {
                    let num = q00 * (d[7] - 3 * d[8] + d[9] - d[17] + 3 * d[18] - d[19])
                    workspace[10] = predict(num, q[7], coefBits[7], limit: true)
                }
                // AC21
                if coefBits[8] != 0 && workspace[17] == 0 {
                    let num = q00 * (d[7] - d[9] - 3 * d[12] + 3 * d[14] + d[17] - d[19])
                    workspace[17] = predict(num, q[8], coefBits[8], limit: true)
                }
                // AC30
                if coefBits[9] != 0 && workspace[24] == 0 {
                    let num = q00 * (d[7] + 2 * d[8] + d[9] - d[17] - 2 * d[18] - d[19])
                    workspace[24] = predict(num, q[9], coefBits[9], limit: true)
                }
                // DC
                let num =
                    q00
                    * (-2 * d[1] - 6 * d[2] - 8 * d[3] - 6 * d[4] - 2 * d[5] - 6 * d[6] + 6 * d[7]
                        + 42 * d[8] + 6 * d[9] - 6 * d[10] - 8 * d[11] + 42 * d[12] + 152 * d[13]
                        + 42 * d[14] - 8 * d[15] - 6 * d[16] + 6 * d[17] + 42 * d[18] + 6 * d[19]
                        - 6 * d[20] - 2 * d[21] - 6 * d[22] - 8 * d[23] - 6 * d[24] - 2 * d[25])
                workspace[0] = predict(num, q00, 0, limit: false)
            }
        }
    }

    // MARK: Upsampling and colour

    /// A component's upsampling method (jdsample.c).
    enum Upsampling: Equatable {
        case full, h2v1Fancy, h1v2Fancy, h2v2Fancy
        /// `int_upsample`, `h2v1_upsample` and `h2v2_upsample`: replication.
        case replicate(Int, Int)
    }

    /// Output row `y` of a component at the full width: the triangle ("fancy") filters for 2:1
    /// horizontally (when the component is more than 2 samples wide), vertically, or both, and
    /// replication otherwise. Rows above the first and below the last are the first and last
    /// rows, as libjpeg's context rows (jdmainct.c) repeat them.
    static func upsampleRow(
        _ y: Int, _ method: Upsampling, _ component: Component, plane: UnsafePointer<UInt8>,
        width: Int, into out: UnsafeMutablePointer<UInt8>, scratch: UnsafeMutablePointer<Int>
    ) {
        let stride = component.widthInBlocks * 8
        let cw = component.width
        let ch = component.height
        switch method {
        case .full:
            out.update(from: plane + y * stride, count: width)
        case .h2v1Fancy:
            let row = plane + y * stride
            // Two samples per input sample, written to a full pair and trimmed to `width`.
            for c in 0..<cw {
                let value = Int(row[c]) * 3
                let left = c == 0 ? Int(row[0]) : (value + Int(row[c - 1]) + 1) >> 2
                let right = c == cw - 1 ? Int(row[c]) : (value + Int(row[c + 1]) + 2) >> 2
                if 2 * c < width { out[2 * c] = UInt8(left) }
                if 2 * c + 1 < width { out[2 * c + 1] = UInt8(right) }
            }
        case .h1v2Fancy:
            let r = y >> 1
            let below = y & 1 == 1
            let near = plane + r * stride
            let far = plane + (below ? min(r + 1, ch - 1) : max(r - 1, 0)) * stride
            let bias = below ? 2 : 1
            for x in 0..<width {
                out[x] = UInt8((Int(near[x]) * 3 + Int(far[x]) + bias) >> 2)
            }
        case .h2v2Fancy:
            let r = y >> 1
            let near = plane + r * stride
            let far = plane + (y & 1 == 1 ? min(r + 1, ch - 1) : max(r - 1, 0)) * stride
            for c in 0..<cw { scratch[c] = Int(near[c]) * 3 + Int(far[c]) }
            for c in 0..<cw {
                let sum = scratch[c]
                let left = c == 0 ? (sum * 4 + 8) >> 4 : (sum * 3 + scratch[c - 1] + 8) >> 4
                let right = c == cw - 1 ? (sum * 4 + 7) >> 4 : (sum * 3 + scratch[c + 1] + 7) >> 4
                if 2 * c < width { out[2 * c] = UInt8(left) }
                if 2 * c + 1 < width { out[2 * c + 1] = UInt8(right) }
            }
        case .replicate(let hExpand, let vExpand):
            let row = plane + (y / vExpand) * stride
            for x in 0..<width { out[x] = row[x / hExpand] }
        }
    }

    /// Whether a block's 64 coefficients are all zero.
    @inline(__always)
    static func isZero(_ block: UnsafePointer<Int16>) -> Bool {
        let words = UnsafeRawPointer(block).assumingMemoryBound(to: UInt64.self)
        var any: UInt64 = 0
        for i in 0..<16 { any |= words[i] }
        return any == 0
    }

    @inline(__always)
    static func clamp(_ value: Int) -> UInt8 {
        value < 0 ? 0 : value > 255 ? 255 : UInt8(value)
    }

    /// `build_ycc_rgb_table`, 16-bit fixed point. The Neon conversion libjpeg-turbo runs on
    /// Apple silicon gives the same bytes for every input.
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
