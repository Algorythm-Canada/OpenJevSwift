import Foundation
import Testing

@testable import OpenJevDiffusionGemma

/// The JPEG decoder on malformed input: a server decodes whatever a request sends, so every
/// malformed JPEG must decode as Pillow does or be refused, never trap, and never cost work out of
/// proportion to its size. A trap would end the test process.
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
    /// DC and AC tables 0 of one 1-bit code each: a zero difference, and end of block.
    static let tables =
        segment(0xC4, [0x00, 1] + Array(repeating: 0, count: 15) + [0x00])
        + segment(0xC4, [0x10, 1] + Array(repeating: 0, count: 15) + [0x00])

    /// A scan of the given component ids on tables 0.
    static func scan(_ ids: [Int], ss: Int = 0, se: Int = 63, ahal: Int = 0) -> [UInt8] {
        segment(
            0xDA,
            [UInt8(ids.count)] + ids.flatMap { [UInt8($0), 0] } + [
                UInt8(ss), UInt8(se), UInt8(ahal),
            ])
    }

    /// What decoding gives: decoded, unsupported or refused.
    enum Outcome: Equatable { case decoded, unsupported, refused }

    static func outcome(_ bytes: [UInt8]) -> Outcome {
        var work = Decoder.Work()
        return outcome(bytes, work: &work)
    }

    static func outcome(_ bytes: [UInt8], work: inout Decoder.Work) -> Outcome {
        do {
            _ = try Decoder.decode(bytes, work: &work)
            return .decoded
        } catch {
            switch error {
            case .unsupported: return .unsupported
            case .refused: return .refused
            }
        }
    }

    @Test("A well-formed small JPEG decodes, so the cases below fail for their one fault")
    func control() {
        let bytes =
            Self.soi + Self.quant + Self.frame() + Self.tables + Self.scan([1]) + [0, 0] + Self.eoi
        #expect(Self.outcome(bytes) == .decoded)
    }

    @Test("Headers that end early or lie about their length are refused, not read past")
    func shortSegments() {
        // A DQT whose 64 entries are not there.
        #expect(Self.outcome([0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x03, 0x00]) == .refused)
        // A SOF that declares 3 components in a 6-byte body.
        #expect(
            Self.outcome([0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x08, 0x08, 0x00, 0x01, 0x00, 0x01, 0x03])
                == .refused)
        // A SOS at the end of the file.
        #expect(Self.outcome([0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02]) == .refused)
        // A SOS with no components (JERR_BAD_LENGTH).
        let empty = Self.soi + Self.frame() + Self.quant + Self.segment(0xDA, [0, 0, 63, 0])
        #expect(Self.outcome(empty + Self.eoi) == .refused)
    }

    @Test("A DC Huffman table with a symbol above 15 is refused when a scan uses it, not before")
    func dcSymbol() {
        let bad = Self.segment(0xC4, [0x01, 1] + Array(repeating: 0, count: 15) + [0x50])
        let base = Self.soi + Self.quant + Self.frame() + Self.tables + bad
        #expect(Self.outcome(base + Self.scan([1]) + [0, 0] + Self.eoi) == .decoded)
        let uses = Self.segment(0xDA, [1, 1, 0x10, 0, 63, 0])
        #expect(Self.outcome(base + uses + [0, 0] + Self.eoi) == .refused)
    }

    @Test("A progressive scan whose spectral range runs past 63 is refused")
    func spectralRange() {
        let bytes =
            Self.soi + Self.frame(0xC2) + Self.quant + Self.tables
            + Self.scan([1], ss: 200, se: 255) + [0, 0] + Self.eoi
        #expect(Self.outcome(bytes) == .refused)
    }

    @Test("Fractional sampling ratios, a second frame and a second SOI are refused")
    func frames() {
        let fractional = Self.frame(
            width: 32, height: 32, components: [(1, 4, 4), (2, 3, 3), (3, 1, 1)])
        #expect(
            Self.outcome(
                Self.soi + fractional + Self.quant + Self.tables + Self.scan([1]) + Self.eoi)
                == .refused)
        let two = Self.frame() + Self.frame(width: 16, height: 16)
        #expect(
            Self.outcome(Self.soi + two + Self.quant + Self.tables + Self.scan([1]) + Self.eoi)
                == .refused)
        let soi = Self.soi + Self.quant + Self.soi + Self.frame() + Self.tables + Self.scan([1])
        #expect(Self.outcome(soi + [0, 0] + Self.eoi) == .refused)
    }

    @Test("A frame past the pixel limit, or a side past 65,500, is refused outright")
    func limits() {
        let huge = Self.frame(width: 65_000, height: 65_000)
        #expect(
            Self.outcome(Self.soi + huge + Self.quant + Self.tables + Self.scan([1]) + Self.eoi)
                == .refused)
        let wide = Self.frame(width: 65_501, height: 8)
        #expect(
            Self.outcome(Self.soi + wide + Self.quant + Self.tables + Self.scan([1]) + Self.eoi)
                == .refused)
    }

    @Test("A header-only large frame is refused without allocating its coefficient grid")
    func headerOnlyLargeFrame() {
        let frame = Self.frame(
            width: 13_376, height: 13_376, components: [(1, 1, 1), (2, 1, 1), (3, 1, 1)])
        #expect(Self.outcome(Self.soi + frame + Self.eoi) == .refused)
    }

    @Test("Oversubscribed and all-ones Huffman code trees are refused when a scan uses them")
    func invalidHuffmanCodeTrees() {
        let allOnes = Self.segment(
            0xC4, [0x00, 2] + Array(repeating: 0, count: 15) + [0x00, 0x01])
        let oversubscribed = Self.segment(
            0xC4, [0x00, 3] + Array(repeating: 0, count: 15) + [0x00, 0x01, 0x02])
        for table in [allOnes, oversubscribed] {
            let bytes =
                Self.soi + Self.quant + Self.frame() + Self.tables + table + Self.scan([1])
                + [0, 0] + Self.eoi
            #expect(Self.outcome(bytes) == .refused)
        }
    }

    /// Set to `1` to run every mutation (7,650 cases) rather than the subset CI runs, and to
    /// decode the large parity cases in full: about 100 seconds in a debug build on an M3 Max.
    static let allMutationsVariable = "OPENJEV_TEST_JPEG_MUTATIONS"

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

    /// One mutation: the JPEG cut to a length (with FF D9 appended when `eoi`), or one byte set.
    enum Mutation: Hashable {
        case cut(Int, eoi: Bool)
        case corrupt(index: Int, value: UInt8)

        func apply(to original: [UInt8]) -> [UInt8] {
            switch self {
            case .cut(let length, let eoi):
                return Array(original.prefix(length)) + (eoi ? [0xFF, 0xD9] : [])
            case .corrupt(let index, let value):
                var bytes = original
                bytes[index] = value
                return bytes
            }
        }
    }

    /// The mutations of one JPEG.
    ///
    /// All of them: every fifth length, cut plainly (the data ends, as a truncated download does)
    /// and with FF D9 appended (the scan's data runs out before a marker, libjpeg-turbo's
    /// zero-fill path), and every byte of every marker segment, the SOI marker's and the first
    /// data byte after each scan header set to each of four values, the headers of later scans
    /// and the tables between them included. The subset keeps every kind on every segment: cuts,
    /// both ways, at each segment's start, inside its length and one byte short of its end, every
    /// 211th length and the last two; corruptions of each segment's marker code, the low byte of
    /// its length, its first and last bytes, and the first data byte after a scan header.
    static func mutations(of original: [UInt8], all: Bool) -> [Mutation] {
        let segments = segments(original)
        var lengths: Set<Int>
        var indices: Set<Int>
        if all {
            lengths = Set(stride(from: 0, to: original.count, by: 5))
            indices = Set(segments.flatMap { $0.start..<min($0.end, original.count) })
            indices.formUnion(segments.filter { $0.marker == 0xDA }.map(\.end))
            indices.insert(1)
        } else {
            lengths = Set(stride(from: 0, to: original.count, by: 211))
            lengths.formUnion([original.count - 2, original.count - 1])
            indices = [1]
            for segment in segments {
                lengths.formUnion([segment.start, segment.start + 3, segment.end - 1])
                indices.formUnion([
                    segment.start + 1, segment.start + 3, segment.start + 4, segment.end - 1,
                ])
                if segment.marker == 0xDA { indices.insert(segment.end) }
            }
        }
        let cuts = lengths.filter { $0 < original.count }.sorted().flatMap {
            [Mutation.cut($0, eoi: false), .cut($0, eoi: true)]
        }
        let corruptions = indices.filter { $0 < original.count }.sorted().flatMap { index in
            corruptions(of: original[index]).map { Mutation.corrupt(index: index, value: $0) }
        }
        return cuts + corruptions
    }

    @Test("Truncations and corruptions of the fixture JPEGs decode or are refused, never trap")
    func mutations() throws {
        let all = ProcessInfo.processInfo.environment[Self.allMutationsVariable] == "1"
        var outcomes: [Outcome: Int] = [:]
        var cases = 0
        var outside: [String] = []
        for name in ["baseline.jpg", "progressive.jpg"] {
            let original = [UInt8](
                try Data(contentsOf: VisionFixtures.directory.appendingPathComponent(name)))
            #expect(Self.outcome(original) == .decoded)
            for mutation in Self.mutations(of: original, all: all) {
                let bytes = mutation.apply(to: original)
                var work = Decoder.Work()
                outcomes[Self.outcome(bytes, work: &work), default: 0] += 1
                if !JPEGParityTests.withinBound(work, bytes: bytes.count) {
                    outside.append("\(name) \(mutation): \(work)")
                }
                cases += 1
            }
        }
        print(
            "JPEG mutations (\(all ? "all" : "the subset; \(Self.allMutationsVariable)=1 runs all")): "
                + "\(cases) cases, \(outcomes)")
        if all { #expect(cases > 7_000) }
        #expect((outcomes[.decoded] ?? 0) > 0 && (outcomes[.refused] ?? 0) > 0)
        #expect(outside.isEmpty, "work out of proportion to the input: \(outside.prefix(5))")
    }

    @Test("The mutation subset keeps every kind of mutation on every segment")
    func mutationSubset() throws {
        for name in ["baseline.jpg", "progressive.jpg"] {
            let original = [UInt8](
                try Data(contentsOf: VisionFixtures.directory.appendingPathComponent(name)))
            let all = Set(Self.mutations(of: original, all: true))
            let subset = Self.mutations(of: original, all: false)
            #expect(subset.count < 1_200, "\(name): \(subset.count)")
            // Every subset corruption is one of the full run's, and each corrupted byte takes all
            // four values.
            var values: [Int: Int] = [:]
            for mutation in subset {
                if case .corrupt(let index, _) = mutation {
                    #expect(all.contains(mutation), "\(name): \(mutation)")
                    values[index, default: 0] += 1
                }
            }
            #expect(values.values.allSatisfy { $0 == 4 }, "\(name)")
            // Every segment, the later scan headers included, has its marker code and length
            // corrupted, and is cut both ways at its start.
            for segment in Self.segments(original) {
                #expect(values[segment.start + 1] != nil && values[segment.start + 3] != nil)
                #expect(
                    subset.contains(.cut(segment.start, eoi: false)), "\(name) at \(segment.start)")
                #expect(
                    subset.contains(.cut(segment.start, eoi: true)), "\(name) at \(segment.start)")
            }
        }
    }
}
