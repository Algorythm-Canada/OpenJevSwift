import Foundation
import OpenJevCore
import Testing

/// ``PythonJSONLoads`` against CPython's own `json.loads`, which
/// Tools/fixtures/python_json_tables.py recorded in Fixtures/python-json/decode_errors.json.
@Suite(
    "CPython json.loads outcomes",
    .enabled(if: PythonFixtures.exists("decode_errors.json"), PythonFixtures.missingMessage))
struct PythonJSONLoadsTests {
    /// One recorded document and what CPython did with it.
    struct Row {
        var name: String
        var bytes: [UInt8]
        var expected: PythonJSONLoads.Outcome
    }

    static func rows() throws -> [Row] {
        try PythonFixtures.rows("decode_errors.json").map { row in
            let name = try #require(row["name"]?.stringValue)
            let encoded = try #require(row["bytes"]?.stringValue, "\(name): bytes")
            let bytes = try #require(Data(base64Encoded: encoded), "\(name): base64")
            let expected: PythonJSONLoads.Outcome
            switch row["outcome"]?.stringValue {
            case "accepted":
                expected = .accepted
            case "UnicodeDecodeError":
                expected = .notUTF8
            case "ValueError":
                expected = .integerTooLong
            case "JSONDecodeError":
                expected = .decodeError(
                    message: try #require(row["msg"]?.stringValue, "\(name): msg"),
                    position: try #require(row["pos"]?.intValue, "\(name): pos"))
            case let other:
                throw FixtureValueError("\(name): unknown outcome \(String(describing: other))")
            }
            return Row(name: name, bytes: [UInt8](bytes), expected: expected)
        }
    }

    @Test("Every recorded document ends as CPython's json.loads ends")
    func recordedOutcomes() throws {
        let rows = try Self.rows()
        #expect(rows.count >= 700)
        var mismatches: [String] = []
        for row in rows {
            let outcome = PythonJSONLoads.outcome(of: row.bytes)
            if outcome != row.expected {
                mismatches.append("\(row.name): CPython \(row.expected), here \(outcome)")
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count) mismatches: \(mismatches.prefix(10))")
    }

    @Test("The table covers every message CPython's scanner raises")
    func everyMessage() throws {
        var messages: Set<String> = []
        for row in try Self.rows() {
            if case .decodeError(let message, _) = row.expected {
                messages.insert(message)
            }
        }
        #expect(
            messages == [
                "Expecting value", "Expecting property name enclosed in double quotes",
                "Expecting ':' delimiter", "Expecting ',' delimiter", "Extra data",
                "Illegal trailing comma before end of object",
                "Illegal trailing comma before end of array", "Unterminated string starting at",
                "Invalid control character at", "Invalid \\escape", "Invalid \\uXXXX escape",
            ])
    }

    /// The server answers a body ``JSONParser`` refuses by asking ``PythonJSONLoads`` first. When
    /// CPython accepts it anyway, the server picks the closest error from the parser's, which
    /// assumes the parser only refuses what decision D-016 lists, and accepts everything else
    /// CPython accepts.
    @Test("The parser refuses only what D-016 lists among what CPython accepts")
    func parserAgreesWithCPython() throws {
        var problems: [String] = []
        for row in try Self.rows() {
            let document = PythonJSONLoads.document(row.bytes)
            do {
                let value = try JSONParser().parse(document)
                switch row.expected {
                case .accepted:
                    if PythonJSONLoads.hasIntegerBeyondDigitLimit(value) {
                        problems.append("\(row.name): a long integer CPython accepted")
                    }
                case .integerTooLong:
                    if !PythonJSONLoads.hasIntegerBeyondDigitLimit(value) {
                        problems.append("\(row.name): no long integer found")
                    }
                default:
                    problems.append("\(row.name): parsed, CPython \(row.expected)")
                }
            } catch {
                guard row.expected == .accepted else { continue }
                let offset = error.offset
                let byte = offset < document.count ? document[document.startIndex + offset] : 0
                let before = offset > 0 ? document[document.startIndex + offset - 1] : 0
                let constant = byte == UInt8(ascii: "N") || byte == UInt8(ascii: "I")
                switch error.kind {
                case .unexpectedCharacter where constant:
                    break
                case .invalidNumber where byte == UInt8(ascii: "I") && before == UInt8(ascii: "-"):
                    break
                case .numberOutOfRange, .loneSurrogate, .depthExceeded:
                    break
                default:
                    problems.append("\(row.name): \(error), which CPython accepts")
                }
            }
        }
        #expect(problems.isEmpty, "\(problems.count) problems: \(problems.prefix(10))")
    }

    @Test("A UTF-8 byte order mark is dropped, once, and positions start after it")
    func byteOrderMark() {
        let mark = PythonJSONLoads.byteOrderMark
        #expect(PythonJSONLoads.outcome(of: mark + Array("{}".utf8)) == .accepted)
        #expect(
            PythonJSONLoads.outcome(of: mark + Array("{,}".utf8))
                == .decodeError(
                    message: "Expecting property name enclosed in double quotes", position: 1))
        #expect(
            PythonJSONLoads.outcome(of: mark + mark + Array("{}".utf8))
                == .decodeError(message: "Expecting value", position: 0))
        #expect(Array(PythonJSONLoads.document(mark + [1, 2])) == [1, 2])
        #expect(Array(PythonJSONLoads.document([1, 2])) == [1, 2])
    }

    @Test("Nesting deeper than any parser allows is scanned without recursion")
    func deepNesting() {
        let depth = 1_000_000
        let open = [UInt8](repeating: UInt8(ascii: "["), count: depth)
        #expect(
            PythonJSONLoads.outcome(of: open)
                == .decodeError(message: "Expecting value", position: depth))
        let closed = open + [UInt8](repeating: UInt8(ascii: "]"), count: depth)
        #expect(PythonJSONLoads.outcome(of: closed) == .accepted)
    }

    @Test("Positions count characters, not bytes")
    func characterPositions() {
        #expect(PythonJSONLoads.characterCount(of: Array("aé中😀".utf8)) == 4)
        #expect(
            PythonJSONLoads.outcome(of: Array(#"{"😀": 1 x}"#.utf8))
                == .decodeError(message: "Expecting ',' delimiter", position: 8))
    }

    @Test("Integers are limited to 4,300 digits, fractions and exponents are not")
    func integerDigitLimit() throws {
        let digits = String(repeating: "7", count: 4301)
        #expect(PythonJSONLoads.hasIntegerBeyondDigitLimit(try JSONParser().parse("[\(digits)]")))
        #expect(
            !PythonJSONLoads.hasIntegerBeyondDigitLimit(
                try JSONParser().parse("[-\(digits.dropLast())]")))
        let longFraction = "0." + String(repeating: "0", count: 5000) + "1"
        let fraction = try JSONParser().parse("[\(longFraction)]")
        #expect(!PythonJSONLoads.hasIntegerBeyondDigitLimit(fraction))
        #expect(
            PythonJSONLoads.hasIntegerBeyondDigitLimit(
                try JSONParser().parse(#"{"a": [{"b": \#(digits)}]}"#)))
    }
}

/// A fixture value the table does not allow.
struct FixtureValueError: Error, CustomStringConvertible {
    var description: String

    init(_ description: String) {
        self.description = description
    }
}
