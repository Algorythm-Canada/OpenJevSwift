import Foundation
import OpenJevCore
import Testing

/// Fixtures/vision/jpeg_cases.json: JPEGs built by edits from the committed fixture JPEGs, or from
/// small bytes, and what upstream's `ImagePrompt.pil` (Pillow 12.3.0 with its libjpeg-turbo
/// 3.1.4.1) made of each (Tools/fixtures/jpeg_cases.py).
enum JPEGCases {
    /// One case.
    struct Case {
        var name: String
        var bytes: [UInt8]
        /// The recorded byte count and SHA-256 of the built JPEG.
        var count: Int
        var sha256: String
        /// The decoded size and the SHA-256 of the RGB bytes, or nil when upstream raised.
        var decoded: (width: Int, height: Int, sha256: String)?
        /// The exception upstream raised, when it did.
        var error: String?
        /// "unsupported" when the port knowingly hands the JPEG to ImageIO instead.
        var port: String?
    }

    private static let loaded = Result { try load() }

    /// The cases, built once.
    static func all() throws -> [Case] { try loaded.get() }

    private static func load() throws -> [Case] {
        let root = try JSONParser().parse(
            Data(contentsOf: VisionFixtures.directory.appendingPathComponent("jpeg_cases.json")))
        var sources: [String: [UInt8]] = [:]
        for name in ["baseline.jpg", "progressive.jpg"] {
            sources[name] = [UInt8](
                try Data(contentsOf: VisionFixtures.directory.appendingPathComponent(name)))
        }
        var cases: [Case] = []
        for (name, row) in try #require(root["cases"]?.objectValue) {
            let from = try #require(row["from"]?.stringValue)
            var bytes: [UInt8]
            if from == "bytes" {
                let text = try #require(row["base64"]?.stringValue)
                bytes = [UInt8](try #require(Data(base64Encoded: text)))
            } else {
                bytes = try #require(sources[from])
            }
            for edit in try #require(row["edits"]?.arrayValue) {
                try apply(try #require(edit.arrayValue), to: &bytes)
            }
            let decoded: (Int, Int, String)? =
                row["decoded"] == nil
                ? nil
                : (
                    try #require(row["decoded"]?["width"]?.intValue),
                    try #require(row["decoded"]?["height"]?.intValue),
                    try #require(row["decoded"]?["sha256"]?.stringValue)
                )
            cases.append(
                Case(
                    name: name, bytes: bytes, count: try #require(row["bytes"]?.intValue),
                    sha256: try #require(row["sha256"]?.stringValue), decoded: decoded,
                    error: row["error"]?.stringValue, port: row["port"]?.stringValue))
        }
        return cases.sorted { $0.name < $1.name }
    }

    /// One edit, as jpeg_cases.py's `apply` makes it.
    static func apply(_ edit: [JSONValue], to bytes: inout [UInt8]) throws {
        func int(_ i: Int) throws -> Int { try #require(edit[i].intValue) }
        func hex(_ i: Int) throws -> [UInt8] { try bytesFromHex(try #require(edit[i].stringValue)) }
        switch try #require(edit.first?.stringValue) {
        case "cut":
            bytes.removeSubrange(min(try int(1), bytes.count)...)
        case "set":
            let value = try hex(2)
            let at = try int(1)
            bytes.replaceSubrange(at..<min(at + value.count, bytes.count), with: value)
        case "insert":
            bytes.insert(contentsOf: try hex(2), at: try int(1))
        case "delete":
            let at = try int(1)
            bytes.removeSubrange(at..<min(at + (try int(2)), bytes.count))
        case "append":
            bytes += try hex(1)
        case "copy":
            bytes.insert(contentsOf: Array(bytes[(try int(2))..<(try int(3))]), at: try int(1))
        case "repeat":
            let value = try hex(1)
            for _ in 0..<(try int(2)) { bytes += value }
        case "entropy":
            bytes += entropy(seed: UInt64(try int(1)), count: try int(2))
        case let op:
            Issue.record("unknown edit \(op)")
        }
    }

    /// `n` bytes from splitmix64, each 0xFF followed by a stuffed 0x00.
    static func entropy(seed: UInt64, count: Int) -> [UInt8] {
        var state = seed
        var out: [UInt8] = []
        out.reserveCapacity(count + count / 128)
        for _ in 0..<count {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z ^= z >> 31
            let byte = UInt8(truncatingIfNeeded: z)
            out.append(byte)
            if byte == 0xFF { out.append(0) }
        }
        return out
    }

    static func bytesFromHex(_ text: String) throws -> [UInt8] {
        let digits = Array(text.utf8)
        try #require(digits.count.isMultiple(of: 2), "odd hex \(text)")
        func nibble(_ c: UInt8) throws -> UInt8 {
            switch c {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): return c - UInt8(ascii: "0")
            case UInt8(ascii: "a")...UInt8(ascii: "f"): return c - UInt8(ascii: "a") + 10
            default:
                Issue.record("bad hex digit in \(text)")
                return 0
            }
        }
        return try stride(from: 0, to: digits.count, by: 2).map {
            try nibble(digits[$0]) << 4 | nibble(digits[$0 + 1])
        }
    }
}
