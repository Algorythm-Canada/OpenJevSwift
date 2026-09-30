import Foundation
import OpenJevCore
import Testing

/// An input the parser must reject, with the error kind and byte offset it must report.
struct Rejection: CustomTestStringConvertible, Sendable {
    let name: String
    let bytes: [UInt8]
    let kind: JSONParseError.Kind
    let offset: Int

    init(_ name: String, _ text: String, _ kind: JSONParseError.Kind, at offset: Int) {
        self.init(name, Array(text.utf8), kind, at: offset)
    }

    init(_ name: String, _ bytes: [UInt8], _ kind: JSONParseError.Kind, at offset: Int) {
        self.name = name
        self.bytes = bytes
        self.kind = kind
        self.offset = offset
    }

    var testDescription: String { name }
}

@Suite("JSONParser")
struct JSONParserTests {
    static let rejections: [Rejection] = [
        Rejection("empty input", "", .unexpectedEndOfInput, at: 0),
        Rejection("only whitespace", "  \n", .unexpectedEndOfInput, at: 3),
        Rejection("line comment", "// c\n1", .unexpectedCharacter, at: 0),
        Rejection("block comment", "/* c */ 1", .unexpectedCharacter, at: 0),
        Rejection("comment after value", "1 // c", .trailingContent, at: 2),
        Rejection("trailing comma in array", "[1,]", .unexpectedCharacter, at: 3),
        Rejection("trailing comma in object", #"{"a":1,}"#, .unexpectedCharacter, at: 7),
        Rejection("leading comma", "[,1]", .unexpectedCharacter, at: 1),
        Rejection("missing comma", "[1 2]", .unexpectedCharacter, at: 3),
        Rejection("single-quoted string", "'a'", .unexpectedCharacter, at: 0),
        Rejection("single-quoted key", "{'a':1}", .unexpectedCharacter, at: 1),
        Rejection("unquoted key", "{a:1}", .unexpectedCharacter, at: 1),
        Rejection("number key", "{1:2}", .unexpectedCharacter, at: 1),
        Rejection("missing colon", #"{"a" 1}"#, .unexpectedCharacter, at: 5),
        Rejection("mismatched bracket", "[1}", .unexpectedCharacter, at: 2),
        Rejection("leading zero", "01", .invalidNumber, at: 1),
        Rejection("negative leading zero", "-01", .invalidNumber, at: 2),
        Rejection("leading zero in array", "[00]", .invalidNumber, at: 2),
        Rejection("bare minus", "-", .invalidNumber, at: 1),
        Rejection("plus sign", "+1", .unexpectedCharacter, at: 0),
        Rejection("missing fraction digits", "1.", .invalidNumber, at: 2),
        Rejection("missing integer digits", ".5", .unexpectedCharacter, at: 0),
        Rejection("missing exponent digits", "1e", .invalidNumber, at: 2),
        Rejection("signed exponent without digits", "1e+", .invalidNumber, at: 3),
        Rejection("hex number", "0x10", .trailingContent, at: 1),
        Rejection("NaN", "NaN", .unexpectedCharacter, at: 0),
        Rejection("Infinity", "Infinity", .unexpectedCharacter, at: 0),
        Rejection("negative Infinity", "-Infinity", .invalidNumber, at: 1),
        Rejection("float overflow", "1e400", .numberOutOfRange, at: 0),
        Rejection("truncated literal", "tru", .unexpectedEndOfInput, at: 3),
        Rejection("misspelled literal", "trux", .unexpectedCharacter, at: 3),
        Rejection("capitalized literal", "True", .unexpectedCharacter, at: 0),
        Rejection("unterminated string", #""abc"#, .unexpectedEndOfInput, at: 4),
        Rejection("raw control character", "\"a\u{01}b\"", .unescapedControlCharacter, at: 2),
        Rejection("raw tab", "\"a\tb\"", .unescapedControlCharacter, at: 2),
        Rejection("raw newline", "\"a\nb\"", .unescapedControlCharacter, at: 2),
        Rejection("unknown escape", #""\x""#, .invalidEscape, at: 1),
        Rejection("escaped single quote", #""\'""#, .invalidEscape, at: 1),
        Rejection("short unicode escape", #""\u12""#, .invalidUnicodeEscape, at: 1),
        Rejection("non-hex unicode escape", #""\u12g4""#, .invalidUnicodeEscape, at: 1),
        Rejection("lone high surrogate", #""\ud83d""#, .loneSurrogate, at: 1),
        Rejection("lone low surrogate", #""\ude00""#, .loneSurrogate, at: 1),
        Rejection("high surrogate then text", #""\ud83dx""#, .loneSurrogate, at: 1),
        Rejection("high surrogate then non-surrogate", #""\ud83dA""#, .loneSurrogate, at: 1),
        Rejection("two high surrogates", #""\ud83d\ud83d""#, .loneSurrogate, at: 1),
        Rejection("invalid continuation byte", [0x22, 0xC3, 0x28, 0x22], .invalidUTF8, at: 1),
        Rejection("overlong encoding", [0x22, 0xC0, 0xAF, 0x22], .invalidUTF8, at: 1),
        Rejection(
            "overlong three-byte encoding", [0x22, 0xE0, 0x80, 0xAF, 0x22], .invalidUTF8, at: 1),
        Rejection("encoded surrogate", [0x22, 0xED, 0xA0, 0x80, 0x22], .invalidUTF8, at: 1),
        Rejection("beyond U+10FFFF", [0x22, 0xF4, 0x90, 0x80, 0x80, 0x22], .invalidUTF8, at: 1),
        Rejection("stray continuation byte", [0x22, 0x61, 0x80, 0x22], .invalidUTF8, at: 2),
        Rejection("truncated sequence", [0x22, 0xE2, 0x82], .invalidUTF8, at: 1),
        Rejection("byte order mark", [0xEF, 0xBB, 0xBF, 0x31], .unexpectedCharacter, at: 0),
        Rejection(
            "non-ASCII outside a string", [0x5B, 0xC3, 0xA9, 0x5D], .unexpectedCharacter, at: 1),
        Rejection("two values", "1 2", .trailingContent, at: 2),
        Rejection("unclosed array", "[1", .unexpectedEndOfInput, at: 2),
        Rejection("unclosed object", #"{"a":1"#, .unexpectedEndOfInput, at: 6),
        Rejection("object without value", #"{"a":}"#, .unexpectedCharacter, at: 5),
    ]

    @Test("Rejects input that RFC 8259 does not allow", arguments: rejections)
    func rejects(_ rejection: Rejection) {
        #expect {
            try JSONParser().parse(rejection.bytes)
        } throws: { error in
            guard let error = error as? JSONParseError else { return false }
            return error.kind == rejection.kind && error.offset == rejection.offset
        }
    }

    @Test("Reports the line and column of an error")
    func errorPosition() {
        let text = "[\n  1,\n  x\n]"
        #expect(throws: JSONParseError(kind: .unexpectedCharacter, offset: 9, line: 3, column: 3)) {
            try JSONParser().parse(text)
        }
        #expect(throws: JSONParseError(kind: .unexpectedCharacter, offset: 0, line: 1, column: 1)) {
            try JSONParser().parse("x")
        }
    }

    @Test("Parses every scalar type")
    func scalars() throws {
        let value = try JSONParser().parse(
            #" [null, true, false, "s", 0, -0, 42, -7, 1.5, -0.0, 1E2, 1e-400] "#)
        let expected: JSONValue = [
            nil, true, false, "s", .integer("0"), .integer("0"), 42, -7, 1.5, -0.0, 100.0, 0.0,
        ]
        #expect(value == expected)
        #expect(value[9]?.doubleValue?.sign == .minus)
    }

    @Test("Keeps integers of any length as their digits")
    func bigIntegers() throws {
        let digits = "-" + String(repeating: "9876543210", count: 10)
        #expect(try JSONParser().parse(digits) == .integer(digits))
        #expect(try JSONParser().parse("12345678901234567890").intValue == nil)
    }

    @Test("Decodes every escape")
    func escapes() throws {
        let value = try JSONParser().parse(#""\"\\\/\b\f\n\r\t\u0041\u00e9\u4E2D""#)
        #expect(value == .string("\"\\/\u{8}\u{C}\n\r\tA\u{E9}\u{4E2D}"))
    }

    @Test("Joins a surrogate pair into one scalar")
    func surrogatePairs() throws {
        for text in [#""\ud83d\ude00""#, #""\uD83D\uDE00""#] {
            let value = try JSONParser().parse(text)
            #expect(value.stringValue?.unicodeScalars.map(\.value) == [0x1F600])
        }
    }

    @Test("Accepts raw multibyte UTF-8 and DEL in strings")
    func rawUTF8() throws {
        let text = "\"caf\u{E9} \u{4E2D} \u{1F600} \u{7F} \u{10FFFF}\""
        #expect(
            try JSONParser().parse(text)
                == .string("caf\u{E9} \u{4E2D} \u{1F600} \u{7F} \u{10FFFF}"))
    }

    @Test("A repeated key keeps its first position and its last value")
    func duplicateKeys() throws {
        let value = try JSONParser().parse(#"{"a": 1, "b": 2, "a": 3, "c": {"x": 1, "x": [2]}}"#)
        let object = try #require(value.objectValue)
        #expect(object.keys == ["a", "b", "c"])
        #expect(object["a"] == 3)
        #expect(value["c"] == ["x": [2]])
    }

    @Test("Keeps object keys in document order")
    func orderPreservation() throws {
        let value = try JSONParser().parse(#"{"z": 1, "a": 2, "m": {"q2": 0, "q1": 0, "q10": 0}}"#)
        #expect(value.objectValue?.keys == ["z", "a", "m"])
        #expect(value["m"]?.objectValue?.keys == ["q2", "q1", "q10"])
    }

    @Test("Parses 1,000 levels of nesting when the limit allows it")
    func deepNesting() throws {
        let depth = 1000
        let text = String(repeating: "[", count: depth) + String(repeating: "]", count: depth)
        let parser = JSONParser(options: .init(maximumDepth: 2048))
        var value = try parser.parse(text)
        var levels = 0
        while case .array(let elements) = value {
            levels += 1
            value = elements.first ?? .null
        }
        #expect(levels == depth)
    }

    @Test("Accepts exactly the maximum depth and rejects one level more")
    func depthLimit() throws {
        let parser = JSONParser()
        #expect(parser.options.maximumDepth == 1024)
        let allowed = String(repeating: "[", count: 1024) + String(repeating: "]", count: 1024)
        _ = try parser.parse(allowed)
        let tooDeep = String(repeating: "[", count: 1025) + String(repeating: "]", count: 1025)
        #expect(throws: JSONParseError(kind: .depthExceeded, offset: 1024, line: 1, column: 1025)) {
            try parser.parse(tooDeep)
        }
        let objects =
            String(repeating: #"{"a":"#, count: 1025) + "1" + String(repeating: "}", count: 1025)
        #expect {
            try parser.parse(objects)
        } throws: { error in
            (error as? JSONParseError)?.kind == .depthExceeded
        }
    }

    @Test("Rejects input over the size limit before parsing")
    func sizeLimit() throws {
        #expect(JSONParser().options.maximumBytes == 64 * 1024 * 1024)
        let parser = JSONParser(options: .init(maximumBytes: 4))
        #expect(try parser.parse("1234") == 1234)
        // The input is not valid JSON, so only the size check can produce this error.
        #expect(throws: JSONParseError(kind: .tooLarge, offset: 0, line: 1, column: 1)) {
            try parser.parse("[[[[[")
        }
    }

    @Test("Parses Data, String and non-contiguous collections alike")
    func inputTypes() throws {
        let text = #"{"k": [1, "é"]}"#
        let expected = try JSONParser().parse(text)
        #expect(try JSONParser().parse(Data(text.utf8)) == expected)
        #expect(try JSONParser().parse(Array(text.utf8)) == expected)
        #expect(try JSONParser().parse(Array(text.utf8).lazy.map { $0 }) == expected)
    }
}
