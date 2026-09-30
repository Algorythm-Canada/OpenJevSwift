import OpenJevCore
import Testing

/// SplitMix64, a small deterministic generator, so that the random corpus is the same on every
/// run and every platform.
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Builds random `JSONValue` trees for round-trip tests.
struct RandomTreeBuilder {
    var generator: SeededGenerator

    /// Characters that exercise every escaping rule, including canonically equivalent forms.
    static let alphabet: [String] = [
        "a", "Z", "0", " ", "\"", "\\", "/", "\n", "\t", "\u{0}", "\u{1F}", "\u{7F}", "\u{E9}",
        "e\u{301}", "\u{2028}", "\u{4E2D}", "\u{1F600}", "\u{10FFFF}", "\u{FEFF}",
    ]

    mutating func string() -> String {
        let length = Int.random(in: 0...6, using: &generator)
        return (0..<length).map { _ in Self.alphabet.randomElement(using: &generator)! }.joined()
    }

    mutating func value(depth: Int) -> JSONValue {
        let kinds = depth < 5 ? 8 : 6
        switch Int.random(in: 0..<kinds, using: &generator) {
        case 0:
            return .null
        case 1:
            return .bool(Bool.random(using: &generator))
        case 2:
            // Integers beyond 64 bits as well as small ones.
            let digits = Int.random(in: 1...30, using: &generator)
            var text = String(Int.random(in: 1...9, using: &generator))
            for _ in 1..<digits {
                text += String(Int.random(in: 0...9, using: &generator))
            }
            return .integer(Bool.random(using: &generator) ? "-" + text : text)
        case 3:
            // Any finite bit pattern, which covers every exponent and subnormals.
            while true {
                let number = Double(bitPattern: generator.next())
                if number.isFinite { return .float(number) }
            }
        case 4, 5:
            return .string(string())
        case 6:
            let count = Int.random(in: 0...4, using: &generator)
            return .array((0..<count).map { _ in value(depth: depth + 1) })
        default:
            var object = JSONObject()
            for _ in 0..<Int.random(in: 0...4, using: &generator) {
                object[string()] = value(depth: depth + 1)
            }
            return .object(object)
        }
    }
}

@Suite("JSON round trips")
struct JSONRoundTripTests {
    @Test(
        "Parsing a CPython output and writing it again gives the same bytes",
        .enabled(if: PythonFixtures.exists("documents.json"), PythonFixtures.missingMessage))
    func fixtureRoundTrip() throws {
        let columns: [(String, PythonJSONWriter.Options)] = [
            ("dumps", .pythonDefault),
            ("dumps_sorted", .init(sortKeys: true)),
            ("dumps_compact", .compact),
            ("dumps_unicode", .init(ensureASCII: false)),
        ]
        for row in try PythonFixtures.rows("documents.json") {
            for (column, options) in columns {
                let text = try #require(row[column]?.stringValue)
                let rewritten = try PythonJSONWriter(options: options).string(
                    try JSONParser().parse(text))
                #expect(rewritten == text, "\(row["name"] ?? .null) \(column)")
            }
        }
    }

    @Test("parse(write(x)) == x for 400 random trees under every option set")
    func propertyRoundTrip() throws {
        var builder = RandomTreeBuilder(generator: SeededGenerator(state: 20_260_929))
        let optionSets: [PythonJSONWriter.Options] = [
            .pythonDefault, .compact, .init(ensureASCII: false),
        ]
        for _ in 0..<400 {
            let tree = builder.value(depth: 0)
            for options in optionSets {
                let writer = PythonJSONWriter(options: options)
                let bytes = try writer.bytes(tree)
                let parsed = try JSONParser().parse(bytes)
                #expect(parsed == tree)
                // Byte equality also catches differences that == ignores, such as the sign of zero.
                #expect(try writer.bytes(parsed) == bytes)
            }
            // Sorting changes order, so check that it is stable instead.
            let sorted = try PythonJSONWriter.canonicalSeedBytes(tree)
            #expect(try PythonJSONWriter.canonicalSeedBytes(JSONParser().parse(sorted)) == sorted)
        }
    }
}
