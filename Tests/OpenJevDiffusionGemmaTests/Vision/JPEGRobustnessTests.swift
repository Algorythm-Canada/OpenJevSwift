import Foundation
import Testing

@testable import OpenJevDiffusionGemma

/// The JPEG decoder on malformed input: a server decodes whatever a request sends, so every
/// malformed JPEG must throw, never trap. A trap would end the test process.
@Suite("JPEG decoder robustness")
struct JPEGRobustnessTests {
    typealias Decoder = LibjpegTurboDecoder

    /// Big-endian 16-bit.
    static func be16(_ value: Int) -> [UInt8] { [UInt8(value >> 8 & 255), UInt8(value & 255)] }

    /// A marker segment: FF, the marker, the length and the body.
    static func segment(_ marker: UInt8, _ body: [UInt8]) -> [UInt8] {
        [0xFF, marker] + be16(body.count + 2) + body
    }

    /// SOF of a `width` by `height` frame with components (id, h, v), all on table 0.
    static func frame(
        _ marker: UInt8 = 0xC0, width: Int = 8, height: Int = 8,
        components: [(Int, Int, Int)] = [(1, 1, 1)]
    ) -> [UInt8] {
        segment(
            marker,
            [8] + be16(height) + be16(width) + [UInt8(components.count)]
                + components.flatMap { [UInt8($0.0), UInt8($0.1 << 4 | $0.2), 0] })
    }

    static let quant = segment(0xDB, [0] + Array(repeating: 1, count: 64))
    static let soi: [UInt8] = [0xFF, 0xD8]
    static let eoi: [UInt8] = [0xFF, 0xD9]

    /// What decoding gives: decoded, unsupported or refused.
    enum Outcome: Equatable { case decoded, unsupported, refused }

    static func outcome(_ bytes: [UInt8]) -> Outcome {
        do {
            _ = try Decoder.decode(bytes)
            return .decoded
        } catch {
            switch error {
            case .unsupported: return .unsupported
            case .refused: return .refused
            }
        }
    }

    @Test("Headers that end early or lie about their length are refused, not read past")
    func shortSegments() {
        // A DQT whose 64 entries are not there.
        #expect(Self.outcome([0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x03, 0x00]) == .unsupported)
        // A SOF that declares 3 components in a 6-byte body.
        #expect(
            Self.outcome([0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x08, 0x08, 0x00, 0x01, 0x00, 0x01, 0x03])
                == .unsupported)
        // A SOS at the end of the file.
        #expect(Self.outcome([0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02]) == .unsupported)
        // A SOS with no components.
        let empty = Self.soi + Self.frame() + Self.quant + Self.segment(0xDA, [0, 0, 63, 0])
        #expect(Self.outcome(empty + Self.eoi) == .unsupported)
    }

    @Test("A DC Huffman table with a symbol above 15 is refused")
    func dcSymbol() {
        let table = Self.segment(0xC4, [0x00, 1] + Array(repeating: 0, count: 15) + [0x50])
        let scan = Self.segment(0xDA, [1, 1, 0x00, 0, 63, 0])
        let bytes = Self.soi + Self.frame() + Self.quant + table + scan + [0, 0] + Self.eoi
        #expect(Self.outcome(bytes) == .unsupported)
    }

    @Test("A progressive scan whose spectral range runs past 63 is refused")
    func spectralRange() {
        let table = Self.segment(0xC4, [0x10, 1] + Array(repeating: 0, count: 15) + [0x01])
        let scan = Self.segment(0xDA, [1, 1, 0x00, 200, 255, 0])
        let bytes =
            Self.soi + Self.frame(0xC2) + Self.quant + table + scan + [0, 0] + Self.eoi
        #expect(Self.outcome(bytes) == .unsupported)
    }

    @Test("Fractional sampling ratios and a second frame are refused")
    func frames() {
        let fractional = Self.frame(
            width: 32, height: 32, components: [(1, 4, 4), (2, 3, 3), (3, 1, 1)])
        #expect(Self.outcome(Self.soi + fractional + Self.quant + Self.eoi) == .unsupported)
        let two = Self.frame() + Self.frame(width: 65_535, height: 65_535)
        #expect(Self.outcome(Self.soi + two + Self.eoi) == .unsupported)
    }

    @Test("A frame past the pixel limit, or more than 100 scans, is refused outright")
    func limits() {
        let huge = Self.frame(width: 65_535, height: 65_535)
        #expect(Self.outcome(Self.soi + huge + Self.eoi) == .refused)
        let table = Self.segment(0xC4, [0x00, 1] + Array(repeating: 0, count: 15) + [0x00])
        let dcScan = Self.segment(0xDA, [1, 1, 0x00, 0, 0, 0]) + [0x00]
        let many =
            Self.soi + Self.frame(0xC2) + Self.quant + table
            + Array([[UInt8]](repeating: dcScan, count: Decoder.maxScans + 1).joined()) + Self.eoi
        #expect(Self.outcome(many) == .refused)
    }

    @Test("A header-only large frame is refused without allocating its coefficient grid")
    func headerOnlyLargeFrame() {
        let frame = Self.frame(
            width: 13_376, height: 13_376, components: [(1, 1, 1), (2, 1, 1), (3, 1, 1)])
        #expect(Self.outcome(Self.soi + frame + Self.eoi) == .unsupported)
    }

    @Test("Oversubscribed and all-ones Huffman code trees are refused")
    func invalidHuffmanCodeTrees() {
        let allOnes = Self.segment(
            0xC4, [0x00, 2] + Array(repeating: 0, count: 15) + [0x00, 0x01])
        let oversubscribed = Self.segment(
            0xC4, [0x00, 3] + Array(repeating: 0, count: 15) + [0x00, 0x01, 0x02])
        #expect(Self.outcome(Self.soi + allOnes + Self.eoi) == .unsupported)
        #expect(Self.outcome(Self.soi + oversubscribed + Self.eoi) == .unsupported)
    }

    @Test("Every truncation and every corrupted header byte of the fixture JPEGs throws or decodes")
    func mutations() throws {
        var outcomes: [Outcome: Int] = [:]
        for name in ["baseline.jpg", "progressive.jpg"] {
            let original = [UInt8](
                try Data(contentsOf: VisionFixtures.directory.appendingPathComponent(name)))
            #expect(Self.outcome(original) == .decoded)
            for length in stride(from: 0, to: original.count, by: 5) {
                outcomes[Self.outcome(Array(original.prefix(length))), default: 0] += 1
            }
            // The headers come before the first scan's data; corrupt each of their bytes.
            let headerEnd = min(original.count, 700)
            for index in 0..<headerEnd {
                for value: UInt8 in [0x00, 0xFF, 0x7F, original[index] ^ 0x5A] {
                    var bytes = original
                    bytes[index] = value
                    outcomes[Self.outcome(bytes), default: 0] += 1
                }
            }
        }
        print("JPEG mutations: \(outcomes)")
        #expect((outcomes[.unsupported] ?? 0) > 0)
    }
}
