import Foundation
import OpenJevCore
import Testing

@Suite("PythonJSONWriter")
struct PythonJSONWriterTests {
    @Test(
        "Floats match CPython repr and json.dumps",
        .enabled(if: PythonFixtures.exists("float_repr.json"), PythonFixtures.missingMessage))
    func floatTable() throws {
        let rows = try PythonFixtures.rows("float_repr.json")
        #expect(rows.count >= 5000)
        var mismatches: [String] = []
        for row in rows {
            let hex = try #require(row["hex"]?.stringValue)
            let bits = try #require(UInt64(hex, radix: 16))
            let written = try PythonJSONWriter().string(.float(Double(bitPattern: bits)))
            if written != row["repr"]?.stringValue || written != row["dumps"]?.stringValue {
                mismatches.append("\(hex): wrote \(written), Python \(row["repr"] ?? .null)")
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count) mismatches: \(mismatches.prefix(10))")
    }

    @Test(
        "Strings match CPython json.dumps with and without ensure_ascii",
        .enabled(if: PythonFixtures.exists("strings.json"), PythonFixtures.missingMessage))
    func stringTable() throws {
        let rows = try PythonFixtures.rows("strings.json")
        #expect(!rows.isEmpty)
        for row in rows {
            let codes = try #require(row["scalars"]?.arrayValue)
            var scalars = String.UnicodeScalarView()
            for code in codes {
                let value = try #require(code.intValue.flatMap { UInt32(exactly: $0) })
                scalars.append(try #require(Unicode.Scalar(value)))
            }
            let value = JSONValue.string(String(scalars))
            #expect(try PythonJSONWriter().string(value) == row["dumps"]?.stringValue)
            #expect(try PythonJSONWriter.modelText(value) == row["dumps_unicode"]?.stringValue)
        }
    }

    @Test(
        "Documents match CPython json.dumps for every option set",
        .enabled(if: PythonFixtures.exists("documents.json"), PythonFixtures.missingMessage))
    func documentTable() throws {
        let rows = try PythonFixtures.rows("documents.json")
        #expect(!rows.isEmpty)
        for row in rows {
            let name = row["name"]?.stringValue ?? "?"
            let value = try JSONParser().parse(try #require(row["text"]?.stringValue))
            let sorted = PythonJSONWriter(options: .init(sortKeys: true))
            #expect(try PythonJSONWriter().string(value) == row["dumps"]?.stringValue, "\(name)")
            #expect(try sorted.string(value) == row["dumps_sorted"]?.stringValue, "\(name)")
            #expect(
                try PythonJSONWriter(options: .compact).string(value)
                    == row["dumps_compact"]?.stringValue, "\(name)")
            #expect(
                try PythonJSONWriter.modelText(value) == row["dumps_unicode"]?.stringValue,
                "\(name)")
            let seedBytes = try PythonJSONWriter.canonicalSeedBytes(value)
            #expect(String(decoding: seedBytes, as: UTF8.self) == row["dumps_sorted"]?.stringValue)
        }
    }

    @Test("Lays floats out as CPython repr does")
    func floatLayout() throws {
        let cases: [(Double, String)] = [
            (0.0, "0.0"), (-0.0, "-0.0"), (100.0, "100.0"), (0.1 + 0.2, "0.30000000000000004"),
            (1e15, "1000000000000000.0"), (1e16, "1e+16"), (1e-4, "0.0001"), (1e-5, "1e-05"),
            (1.5e300, "1.5e+300"), (5e-324, "5e-324"), (-2.5e-7, "-2.5e-07"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (9_007_199_254_740_992.0, "9007199254740992.0"),
        ]
        for (value, expected) in cases {
            #expect(try PythonJSONWriter().string(.float(value)) == expected)
        }
    }

    @Test("Rejects non-finite floats and malformed integer text")
    func writeErrors() {
        for value in [Double.infinity, -Double.infinity] {
            #expect(throws: JSONWriteError.nonFiniteNumber(value)) {
                try PythonJSONWriter().bytes(.float(value))
            }
        }
        #expect(throws: JSONWriteError.nonFiniteNumber(.nan)) {
            try PythonJSONWriter().bytes([1, .float(.nan)])
        }
        for text in ["", "-", "+1", "01", "-0", "1.0", "1e5", " 1", "abc"] {
            #expect(throws: JSONWriteError.invalidInteger(text)) {
                try PythonJSONWriter().bytes(.integer(text))
            }
        }
    }

    @Test("Escapes strings as CPython does")
    func stringEscapes() throws {
        let text: JSONValue = "a/\"\\\u{7F}\u{1F}\u{E9}\u{1F600}"
        #expect(try PythonJSONWriter().string(text) == #""a/\"\\\u007f\u001f\u00e9\ud83d\ude00""#)
        #expect(
            try PythonJSONWriter.modelText(text) == "\"a/\\\"\\\\\u{7F}\\u001f\u{E9}\u{1F600}\"")
    }

    @Test("Writes separators, empty containers and literals as Python does")
    func layout() throws {
        let value: JSONValue = ["b": [1, 2], "a": [:], "c": [], "d": [true, false, nil]]
        #expect(
            try PythonJSONWriter().string(value)
                == #"{"b": [1, 2], "a": {}, "c": [], "d": [true, false, null]}"#)
        #expect(
            try PythonJSONWriter(options: .compact).string(value)
                == #"{"b":[1,2],"a":{},"c":[],"d":[true,false,null]}"#)
        #expect(
            String(decoding: try PythonJSONWriter.canonicalSeedBytes(value), as: UTF8.self)
                == #"{"a": {}, "b": [1, 2], "c": [], "d": [true, false, null]}"#)
    }

    @Test("Writes 1,500 levels of nesting without recursion")
    func deepWrite() throws {
        var value: JSONValue = 1
        for _ in 0..<1500 {
            value = [value]
        }
        let text = try PythonJSONWriter(options: .compact).string(value)
        #expect(
            text == String(repeating: "[", count: 1500) + "1" + String(repeating: "]", count: 1500))
    }
}
