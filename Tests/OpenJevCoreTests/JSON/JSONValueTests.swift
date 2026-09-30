import Foundation
import OpenJevCore
import Testing

@Suite("JSONValue")
struct JSONValueTests {
    // The expectations below come from upstream's `trim` in openjev/api.py at dcd2094, run on the
    // same inputs with CPython.

    @Test("trimmed shortens long strings to 500 scalars and adds ...")
    func trimStrings() {
        let exact = String(repeating: "x", count: 500)
        #expect(JSONValue.string(exact).trimmed() == .string(exact))
        #expect(JSONValue.string(exact + "y").trimmed() == .string(exact + "..."))
        // 300 decomposed characters are 600 scalars; Python counts code points.
        let decomposed = String(repeating: "e\u{301}", count: 300)
        let trimmed = JSONValue.string(decomposed).trimmed().stringValue ?? ""
        #expect(trimmed.unicodeScalars.count == 503)
        #expect(trimmed.hasSuffix("..."))
    }

    @Test("trimmed replaces containers at depth 4 with ...")
    func trimDepth() {
        let arrays: JSONValue = [[[[[1]]]]]
        #expect(arrays.trimmed() == [[[["..."]]]])
        let objects: JSONValue = ["a": ["b": ["c": ["d": ["e": 1]]]]]
        #expect(objects.trimmed() == ["a": ["b": ["c": ["d": "..."]]]])
        // Strings are shortened at any depth but never replaced by depth.
        let deepString: JSONValue = [[[[.string(String(repeating: "x", count: 501))]]]]
        #expect(deepString.trimmed() == [[[[.string(String(repeating: "x", count: 500) + "...")]]]])
    }

    @Test("trimmed keeps 20 array elements and appends ...")
    func trimArrays() {
        let twenty = JSONValue.array((0..<20).map { JSONValue($0) })
        #expect(twenty.trimmed() == twenty)
        let twentyOne = JSONValue.array((0..<21).map { JSONValue($0) })
        #expect(twentyOne.trimmed() == .array((0..<20).map { JSONValue($0) } + ["..."]))
    }

    @Test("trimmed keeps 20 object entries and adds the N more entry")
    func trimObjects() throws {
        let object = JSONObject(uniqueKeysWithValues: (0..<25).map { ("k\($0)", JSONValue($0)) })
        let trimmed = try #require(JSONValue.object(object).trimmed().objectValue)
        #expect(trimmed.keys == (0..<20).map { "k\($0)" } + ["..."])
        #expect(trimmed["..."] == "5 more")

        // A kept "..." key takes the count in place, as a Python dict assignment does.
        var withEllipsis = JSONObject()
        withEllipsis["..."] = "orig"
        for index in 0..<21 {
            withEllipsis["k\(index)"] = JSONValue(index)
        }
        let replaced = try #require(JSONValue.object(withEllipsis).trimmed().objectValue)
        #expect(replaced.keys == ["..."] + (0..<19).map { "k\($0)" })
        #expect(replaced["..."] == "2 more")
    }

    @Test("trimmed passes numbers, Booleans and null through and honors custom limits")
    func trimScalarsAndLimits() {
        let scalars: JSONValue = [1.5, true, nil, .integer("1000000000000000000000000000000")]
        #expect(scalars.trimmed() == scalars)
        #expect(JSONValue.string("abcdef").trimmed(characters: 3) == "abc...")
        #expect(JSONValue.array([1, 2, 3]).trimmed(items: 2) == [1, 2, "..."])
        #expect(JSONValue.array([[1]]).trimmed(depth: 1) == ["..."])
    }

    @Test("Accessors and subscripts read the matching case only")
    func accessors() {
        let value: JSONValue = [
            "n": nil, "b": true, "i": 7, "f": 2.5, "s": "t", "a": [1], "o": [:],
        ]
        #expect(value["n"]?.isNull == true)
        #expect(value["b"]?.boolValue == true)
        #expect(value["i"]?.intValue == 7)
        #expect(value["i"]?.integerText == "7")
        #expect(value["i"]?.doubleValue == 7.0)
        #expect(value["f"]?.doubleValue == 2.5)
        #expect(value["f"]?.intValue == nil)
        #expect(value["s"]?.stringValue == "t")
        #expect(value["a"]?[0] == 1)
        #expect(value["a"]?[1] == nil)
        #expect(value["o"]?.objectValue?.isEmpty == true)
        #expect(value["missing"] == nil)
        #expect(value[0] == nil)
        #expect(JSONValue(Int64.min) == .integer("-9223372036854775808"))
        #expect(JSONValue(1) != JSONValue(1.0))
    }

    @Test("Codable decoding and encoding go through Foundation")
    func codable() throws {
        let decoded = try JSONDecoder().decode(
            JSONValue.self, from: Data(#"[null, true, 1, 2.5, "s", {"k": [1]}]"#.utf8))
        #expect(decoded == [nil, true, 1, 2.5, "s", ["k": [1]]])

        let original: JSONValue = ["k": [1, 2.5, "é", nil, false]]
        let encoded = try JSONEncoder().encode(original)
        #expect(try JSONParser().parse(encoded) == original)
    }

    @Test("decode(_:using:) decodes a Decodable type from the value")
    func decodeTyped() throws {
        struct Question: Decodable, Equatable {
            var type: String
            var criteria: [String]
        }
        let value: JSONValue = ["type": "choice", "criteria": ["a", "b"]]
        #expect(try value.decode(Question.self) == Question(type: "choice", criteria: ["a", "b"]))
        #expect(throws: DecodingError.self) {
            try JSONValue.string("x").decode(Question.self)
        }
    }

    @Test("description is compact JSON without ASCII escaping")
    func description() {
        let value: JSONValue = ["é": [1, 1e16]]
        #expect(value.description == #"{"é":[1,1e+16]}"#)
    }
}
