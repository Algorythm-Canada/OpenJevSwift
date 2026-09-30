import OpenJevCore
import Testing

@Suite("JSONObject")
struct JSONObjectTests {
    @Test("Order survives parsing, mutation and writing")
    func orderThroughMutation() throws {
        var object = try #require(try JSONParser().parse(#"{"b": 1, "a": 2, "c": 3}"#).objectValue)
        object["a"] = 20
        object["d"] = 4
        #expect(object.removeValue(forKey: "b") == 1)
        object["b"] = 5
        object["c"] = nil
        #expect(object.keys == ["a", "d", "b"])
        #expect(object.index(forKey: "b") == 2)
        #expect(
            try PythonJSONWriter(options: .compact).string(.object(object))
                == #"{"a":20,"d":4,"b":5}"#)
    }

    @Test("Keys are identified by Unicode scalars, not canonical equivalence")
    func scalarKeyIdentity() throws {
        let composed = "\u{E9}"
        let decomposed = "e\u{301}"
        // Swift itself considers these equal, which is why JSONObject must not use String ==.
        #expect(composed == decomposed)

        var object = JSONObject()
        object[composed] = 1
        object[decomposed] = 2
        #expect(object.count == 2)
        #expect(object[composed] == 1)
        #expect(object[decomposed] == 2)

        let parsed = try JSONParser().parse(#"{"é": 1, "é": 2}"#)
        #expect(parsed.objectValue?.count == 2)
        #expect(JSONValue.string(composed) != JSONValue.string(decomposed))
        #expect(Set<JSONValue>([.string(composed), .string(decomposed)]).count == 2)
    }

    @Test(
        "sortKeys orders non-ASCII keys as CPython does",
        .enabled(if: PythonFixtures.exists("documents.json"), PythonFixtures.missingMessage))
    func sortKeysMatchesPython() throws {
        let rows = try PythonFixtures.rows("documents.json")
        for name in ["non_ascii_keys", "equivalent_keys"] {
            let row = try #require(rows.first { $0["name"]?.stringValue == name })
            let value = try JSONParser().parse(try #require(row["text"]?.stringValue))
            let sorted = try PythonJSONWriter(options: .init(sortKeys: true)).string(value)
            #expect(sorted == row["dumps_sorted"]?.stringValue, "\(name)")
        }
    }

    @Test("Sorting is by scalar value at every level")
    func sortKeysRecursive() throws {
        let value: JSONValue = ["\u{E9}": ["z": 1, "a": 2], "e\u{301}": 0, "Z": 0, "a": 0]
        #expect(
            try PythonJSONWriter(options: .init(ensureASCII: false, sortKeys: true)).string(value)
                == "{\"Z\": 0, \"a\": 0, \"e\u{301}\": 0, \"\u{E9}\": {\"a\": 2, \"z\": 1}}")
    }

    @Test("Literals, unique-key initialization and equality follow order")
    func constructionAndEquality() {
        let literal: JSONObject = ["a": 1, "b": 2, "a": 3]
        #expect(literal.keys == ["a", "b"])
        #expect(literal["a"] == 3)
        let built = JSONObject(uniqueKeysWithValues: [("a", 3), ("b", 2)])
        #expect(built == literal)
        #expect(built.hashValue == literal.hashValue)
        let reordered = JSONObject(uniqueKeysWithValues: [("b", 2), ("a", 3)])
        #expect(reordered != literal)
        #expect(literal.map(\.key) == ["a", "b"])
        #expect(literal.values == [3, 2])
        #expect(literal[1].key == "b")
    }
}
