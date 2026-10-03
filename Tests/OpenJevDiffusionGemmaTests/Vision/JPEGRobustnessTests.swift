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

    /// Set to `1` to run every mutation (6,799 cases, about 3 minutes in a debug build on an M3
    /// Max and 6 on CI) rather than the subset CI runs.
    static let allMutationsVariable = "OPENJEV_TEST_JPEG_MUTATIONS"

    /// The bytes the mutations corrupt: the first 700, which hold the headers before the first
    /// scan's data.
    static let headerEnd = 700

    /// The four values each corrupted byte takes.
    static func corruptions(of byte: UInt8) -> [UInt8] { [0x00, 0xFF, 0x7F, byte ^ 0x5A] }

    /// The marker segments of a JPEG after SOI, as `(start, end, marker)`, walking past each
    /// scan's entropy-coded data to the next marker.
    static func segments(_ bytes: [UInt8]) -> [(start: Int, end: Int, marker: UInt8)] {
        var out: [(start: Int, end: Int, marker: UInt8)] = []
        var position = 2
        while position + 3 < bytes.count {
            let marker = bytes[position + 1]
            guard bytes[position] == 0xFF, marker != 0x00, marker != 0xFF,
                !(0xD0...0xD7).contains(marker)
            else {
                position += 1
                continue
            }
            if marker == 0xD9 { break }
            let end = position + 2 + (Int(bytes[position + 2]) << 8 | Int(bytes[position + 3]))
            out.append((position, end, marker))
            position = end
        }
        return out
    }

    /// The mutations of one JPEG: the lengths it is cut to and the `(index, value)` corruptions.
    ///
    /// All of them: every fifth length, and each of the first 700 bytes set to each of four
    /// values. The subset keeps every kind (cuts, and each value) on every segment's structure:
    /// cuts at each segment's start, inside its length and one byte short of its end, every
    /// 211th length and the last two; corruptions of each header segment's marker code, the low
    /// byte of its length, its first and last bytes, and the first data byte after a scan header.
    static func mutations(of original: [UInt8], all: Bool) -> (
        lengths: [Int], corruptions: [(index: Int, value: UInt8)]
    ) {
        let headerEnd = min(original.count, Self.headerEnd)
        if all {
            return (
                Array(stride(from: 0, to: original.count, by: 5)),
                (0..<headerEnd).flatMap { index in
                    corruptions(of: original[index]).map { (index, $0) }
                }
            )
        }
        let segments = segments(original)
        var lengths = Set(stride(from: 0, to: original.count, by: 211))
        lengths.formUnion([original.count - 2, original.count - 1])
        var indices: Set<Int> = [1]
        for segment in segments {
            lengths.formUnion([segment.start, segment.start + 3, segment.end - 1])
            indices.formUnion([
                segment.start + 1, segment.start + 3, segment.start + 4, segment.end - 1,
            ])
            if segment.marker == 0xDA { indices.insert(segment.end) }
        }
        return (
            lengths.filter { $0 < original.count }.sorted(),
            indices.filter { $0 < headerEnd }.sorted().flatMap { index in
                corruptions(of: original[index]).map { (index, $0) }
            }
        )
    }

    @Test("Truncations and corrupted header bytes of the fixture JPEGs throw or decode, never trap")
    func mutations() throws {
        let all = ProcessInfo.processInfo.environment[Self.allMutationsVariable] == "1"
        var outcomes: [Outcome: Int] = [:]
        var cases = 0
        for name in ["baseline.jpg", "progressive.jpg"] {
            let original = [UInt8](
                try Data(contentsOf: VisionFixtures.directory.appendingPathComponent(name)))
            #expect(Self.outcome(original) == .decoded)
            let mutations = Self.mutations(of: original, all: all)
            for length in mutations.lengths {
                outcomes[Self.outcome(Array(original.prefix(length))), default: 0] += 1
            }
            for corruption in mutations.corruptions {
                var bytes = original
                bytes[corruption.index] = corruption.value
                outcomes[Self.outcome(bytes), default: 0] += 1
            }
            cases += mutations.lengths.count + mutations.corruptions.count
        }
        print(
            "JPEG mutations (\(all ? "all" : "the subset; \(Self.allMutationsVariable)=1 runs all")): "
                + "\(cases) cases, \(outcomes)")
        if all { #expect(cases == 6_799) }
        #expect((outcomes[.decoded] ?? 0) > 0 && (outcomes[.unsupported] ?? 0) > 0)
    }

    @Test("The mutation subset keeps every kind of mutation on every header segment")
    func mutationSubset() throws {
        for name in ["baseline.jpg", "progressive.jpg"] {
            let original = [UInt8](
                try Data(contentsOf: VisionFixtures.directory.appendingPathComponent(name)))
            let all = Self.mutations(of: original, all: true)
            let subset = Self.mutations(of: original, all: false)
            #expect(Set(subset.lengths).isSubset(of: Set(0..<original.count)))
            #expect(subset.lengths.count + subset.corruptions.count < 300, "\(name)")
            // Each corrupted byte takes all four values, as in the full run.
            let byIndex = Dictionary(grouping: subset.corruptions, by: \.index)
            #expect(byIndex.values.allSatisfy { $0.count == 4 }, "\(name)")
            let allPairs = Set(all.corruptions.map { [$0.index, Int($0.value)] })
            #expect(subset.corruptions.allSatisfy { allPairs.contains([$0.index, Int($0.value)]) })
            // Every header segment's marker code and length are corrupted, and every segment is cut.
            for segment in Self.segments(original) {
                if segment.start + 3 < Self.headerEnd {
                    #expect(byIndex[segment.start + 1] != nil && byIndex[segment.start + 3] != nil)
                }
                #expect(subset.lengths.contains(segment.start), "\(name) at \(segment.start)")
            }
        }
    }
}
