import Foundation
import Testing

@testable import OpenJevDiffusionGemma

/// The JPEG port against what upstream's Pillow made of the regression cases in
/// Fixtures/vision/jpeg_cases.json (Tools/fixtures/jpeg_cases.py): the same pixels where Pillow
/// decodes, a refusal where it raises, and decoding work in proportion to the input.
///
/// The cases cover what PR #121's fuzzing review found: scan data that runs out, the refusals of
/// libjpeg-turbo and of Pillow's header reading, the standard Huffman tables, codes longer than 16
/// bits, restart markers out of sequence or missing, blocks per MCU counted per scan, block
/// smoothing, the Neon inverse DCT, libjpeg-turbo's fast Huffman path and Pillow's 65,536-byte
/// reads. They also cover the end of a single-scan JPEG, where Pillow reads no further than the
/// 65,536-byte reads it has made, the checks of a lossless JPEG's first scan, and the EXIF
/// resolution and MPF index Pillow reads with the headers. Four of them declare 13,376 by 13,376
/// pixels from a few hundred bytes; the CI run decodes their entropy-coded data but not their
/// pixels, which take minutes in a debug build; set `OPENJEV_TEST_JPEG_MUTATIONS=1` to decode them
/// fully.
@Suite("JPEG decoder parity with Pillow on the regression cases")
struct JPEGParityTests {
    typealias Decoder = LibjpegTurboDecoder

    /// Whether to decode the large cases' pixels too (the variable of JPEGRobustnessTests).
    static var everything: Bool {
        ProcessInfo.processInfo.environment[JPEGRobustnessTests.allMutationsVariable] == "1"
    }

    /// Cases of more than this many pixels are left out of the CI run's pixel comparison.
    static let largeCase = 4_000_000

    @Test("The cases are the recorded bytes, and cover decodes, refusals and departures")
    func cases() throws {
        let cases = try JPEGCases.all()
        #expect(cases.count >= 204)
        for jpeg in cases {
            #expect(jpeg.bytes.count == jpeg.count, "\(jpeg.name)")
            #expect(VisionFixtures.sha256(jpeg.bytes) == jpeg.sha256, "\(jpeg.name)")
        }
        #expect(cases.filter { $0.decoded != nil }.count >= 125)
        #expect(cases.filter { $0.error != nil }.count >= 79)
        #expect(cases.filter { $0.port == "unsupported" }.count == 5)
    }

    @Test("Each case decodes to Pillow's bytes, or is refused where Pillow raised")
    func parity() throws {
        var compared = 0
        for jpeg in try JPEGCases.all() {
            if let decoded = jpeg.decoded, decoded.width * decoded.height > Self.largeCase,
                !Self.everything
            {
                continue
            }
            var work = Decoder.Work()
            do {
                let image = try Decoder.decode(jpeg.bytes, work: &work)
                guard let decoded = jpeg.decoded, jpeg.port == nil else {
                    Issue.record("\(jpeg.name): decoded where Pillow raised \(jpeg.error ?? "")")
                    continue
                }
                #expect(
                    image.width == decoded.width && image.height == decoded.height, "\(jpeg.name)")
                #expect(
                    VisionFixtures.sha256(image.pixels) == decoded.sha256,
                    "\(jpeg.name): the decoded RGB differs from Pillow's")
            } catch {
                switch error {
                case .refused(let refusal):
                    #expect(
                        jpeg.error != nil,
                        "\(jpeg.name): refused (\(refusal)) where Pillow decoded")
                case .unsupported(let reason):
                    let pillow = jpeg.error == nil ? "decoded" : "raised"
                    #expect(
                        jpeg.port == "unsupported",
                        "\(jpeg.name): handed to ImageIO (\(reason)) where Pillow \(pillow)")
                }
            }
            compared += 1
        }
        #expect(compared >= (Self.everything ? 204 : 200))
    }

    /// The bound on decoding work: each block read costs at least a bit, and a block finished
    /// with zero bits (at most an MCU of 10 per scan or restart segment) follows a scan header of
    /// at least 10 bytes or a restart marker of 2.
    static func withinBound(_ work: Decoder.Work, bytes: Int) -> Bool {
        work.blocks <= 14 * bytes && work.restarts <= 2 * bytes + 2 * work.scans
    }

    @Test("Decoding work stays in proportion to the input on every case")
    func workBound() throws {
        for jpeg in try JPEGCases.all() {
            var work = Decoder.Work()
            if let decoded = jpeg.decoded, decoded.width * decoded.height > Self.largeCase,
                !Self.everything
            {
                // The entropy-coded data alone: the scans run to their end without the pixels.
                work = try Self.entropyWork(jpeg.bytes)
            } else {
                _ = try? Decoder.decode(jpeg.bytes, work: &work)
            }
            #expect(
                Self.withinBound(work, bytes: jpeg.bytes.count),
                "\(jpeg.name): \(work) for \(jpeg.bytes.count) bytes")
        }
    }

    /// Runs a JPEG's scans without the inverse DCT and colour stages, as `Decoder.decode` runs
    /// them, and returns the work.
    static func entropyWork(_ bytes: [UInt8]) throws -> Decoder.Work {
        _ = try PillowJPEGHeader.read(bytes)
        var decompressor = Decoder.Decompressor(bytes: bytes)
        try decompressor.readHeader()
        try decompressor.startDecompress()
        if decompressor.hasMultipleScans {
            try decompressor.consumeScans()
        } else {
            try decompressor.decodeScan(singleScan: true)
        }
        return decompressor.work
    }

    @Test("The review's slow cases decode in a handful of blocks however large the frame")
    func slowCases() throws {
        let cases = try JPEGCases.all().filter { $0.name.hasPrefix("dos_") }
        #expect(cases.count == 4)
        for jpeg in cases {
            let work = try Self.entropyWork(jpeg.bytes)
            // Before the port followed libjpeg-turbo, 99 empty AC scans over 13,376 by 13,376
            // decoded every block of each: 277 million blocks in 147 s.
            #expect(
                work.blocks <= 3 * work.scans && work.visits <= 3 * work.scans,
                "\(jpeg.name): \(work)")
        }
    }

    /// A progressive grey JPEG of `side` by `side` pixels: a DC scan, then `scans` AC refinement
    /// scans that each pass over every block in end-of-band runs of 2^14 blocks, reading no
    /// correction bits because no AC coefficient is nonzero.
    static func refinementScans(side: Int, scans: Int) -> [UInt8] {
        func be16(_ v: Int) -> [UInt8] { [UInt8(v >> 8 & 255), UInt8(v & 255)] }
        func segment(_ marker: UInt8, _ body: [UInt8]) -> [UInt8] {
            [0xFF, marker] + be16(body.count + 2) + body
        }
        let blocks = (side / 8) * (side / 8)
        var out: [UInt8] = [0xFF, 0xD8]
        out += segment(0xDB, [0] + Array(repeating: 1, count: 64))
        out += segment(0xC2, [8] + be16(side) + be16(side) + [1, 1, 0x11, 0])
        // DC: one 1-bit code, 0, for a zero difference. AC: one 1-bit code, 0, for EOB14.
        out += segment(0xC4, [0x00, 1] + Array(repeating: 0, count: 15) + [0])
        out += segment(0xC4, [0x10, 1] + Array(repeating: 0, count: 15) + [0xE0])
        // The DC scan: a zero bit per block.
        out += segment(0xDA, [1, 1, 0x00, 0, 0, 0x00])
        out += Array(repeating: 0, count: (blocks + 7) / 8)
        for _ in 0..<scans {
            // Ss 1, Se 63, Ah 1, Al 0: each run is the code 0 and 14 zero bits, 15 bits; the
            // runs are padded to a byte with ones.
            out += segment(0xDA, [1, 1, 0x00, 1, 63, 0x10])
            let runs = (blocks + 16_383) / 16_384
            var bits = Array(repeating: false, count: runs * 15)
            while bits.count % 8 != 0 { bits.append(true) }
            for i in stride(from: 0, to: bits.count, by: 8) {
                let byte = bits[i..<(i + 8)].reduce(0) { $0 << 1 | ($1 ? 1 : 0) }
                out.append(UInt8(byte))
                if byte == 0xFF { out.append(0) }
            }
        }
        return out + [0xFF, 0xD9]
    }

    @Test("Refinement scans that would visit too many blocks are refused")
    func visitLimit() throws {
        let jpeg = Self.refinementScans(side: 256, scans: 40)
        var work = Decoder.Work()
        // The default limit allows it.
        _ = try Decoder.decode(jpeg, work: &work)
        let visits = work.visits
        #expect(visits == 41 * 1024, "\(work)")
        // A limit below the visits refuses it, before the pixels.
        var limited = Decoder.Work()
        #expect(throws: Decoder.Failure.self) {
            _ = try Decoder.decode(jpeg, work: &limited, visitLimit: visits / 2)
        }
        #expect(limited.visits <= visits / 2 + 1024)
    }
}
