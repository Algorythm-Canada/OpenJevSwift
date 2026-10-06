import Foundation
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// The chat template's `dictsort`, which is jinja2's, and its objects, which keep a request's
/// keys apart as Python does or refuse them (D-058), on templates of the tests' own. No checkpoint
/// file is needed, so these run in CI. The expected values are what jinja2 3.1.6 and CPython 3.14
/// give; Fixtures/chat-completions/prompts.json has upstream's rendering of the same keys
/// through the shipped template.
@Suite("The chat template sorts a dict as jinja2 does and keeps its keys apart")
struct ChatTemplateDictsortTests {
    /// A dict of `keys` in that order, each with the value `true`.
    static func dict(_ keys: [String]) -> JSONValue {
        var object = JSONObject()
        for key in keys {
            object.updateValue(true, forKey: key)
        }
        return .object(object)
    }

    /// The keys of `keys` as a dict, in the order `dictsort` with `arguments` gives them.
    static func sorted(_ keys: [String], _ arguments: String = "") throws -> [String] {
        let text = try SwiftTransformersTokenizer.renderTemplate(
            "{% for k, v in d | dictsort" + arguments + " %}{{ k }}\u{1F}{% endfor %}",
            context: ["d": try SwiftTransformersTokenizer.templateValue(dict(keys))])
        return text.unicodeScalars.split(separator: "\u{1F}", omittingEmptySubsequences: false)
            .dropLast().map { String(String.UnicodeScalarView($0)) }
    }

    /// The scalars of each string: Swift's `==` on strings is canonical equivalence, and these
    /// tests tell a decomposed `é` from a composed one.
    static func scalars(_ strings: [String]) -> [[Unicode.Scalar]] {
        strings.map { Array($0.unicodeScalars) }
    }

    @Test("Keys sort as jinja2 sorts them: lowercased as Python does, then by code point")
    func pythonOrder() throws {
        let keys = [
            "b", "\u{E4}", "a", "f", "Z", "_x", "B", "\u{DF}", "z", "\u{C4}", "ea", "ez",
            "\u{65E5}\u{672C}", "\u{4E2D}", "e\u{301}", "A", "a1", "a_", "\u{C9}b", "1", "~",
        ]
        // swift-jinja's own dictsort puts the sharp s between f and Z, as ss, and the
        // decomposed e acute with a composed one, after the tilde.
        let python = [
            "1", "_x", "a", "A", "a1", "a_", "b", "B", "ea", "ez", "e\u{301}", "f", "Z", "z",
            "~", "\u{DF}", "\u{E4}", "\u{C4}", "\u{C9}b", "\u{4E2D}", "\u{65E5}\u{672C}",
        ]
        #expect(Self.scalars(try Self.sorted(keys)) == Self.scalars(python))
        #expect(
            Self.scalars(try Self.sorted(["j", "\u{130}", "i"]))
                == Self.scalars(["i", "\u{130}", "j"]))
    }

    @Test("Equal keys keep the dict's order, reversed or not; case_sensitive compares as is")
    func arguments() throws {
        #expect(try Self.sorted(["B", "a", "b", "A"]) == ["a", "A", "B", "b"])
        #expect(try Self.sorted(["B", "a", "b", "A"], "(reverse=true)") == ["B", "b", "a", "A"])
        #expect(
            try Self.sorted(["B", "a", "b", "A"], "(false, 'key', true)") == ["B", "b", "a", "A"])
        #expect(
            try Self.sorted(["b", "B", "a", "A"], "(case_sensitive=true)") == ["A", "B", "a", "b"])
        // Sorting by value is swift-jinja's, which agrees with Python on these.
        let values = try SwiftTransformersTokenizer.renderTemplate(
            "{% for k, v in d | dictsort(by='value') %}{{ k }}{% endfor %}",
            context: ["d": try SwiftTransformersTokenizer.templateValue(["x": 3, "y": 1, "z": 2])])
        #expect(values == "yzx")
    }

    @Test("Arguments Python refuses throw, where jinja2 raises TypeError")
    func badArguments() {
        for arguments in ["(foo=1)", "(true, case_sensitive=true)", "(true, 'key', false, 1)"] {
            #expect(throws: (any Error).self, "\(arguments)") {
                _ = try Self.sorted(["a"], arguments)
            }
        }
    }

    @Test("Keys lowercase as Python's str.lower() lowercases them, final sigma included")
    func pythonLowercase() throws {
        let cases: [(String, String)] = [
            ("\u{3A3}", "\u{3C3}"),
            ("\u{391}\u{3A3}", "\u{3B1}\u{3C2}"),
            ("\u{391}\u{3A3}\u{391}", "\u{3B1}\u{3C3}\u{3B1}"),
            ("\u{391}.\u{3A3}", "\u{3B1}.\u{3C2}"),
            ("\u{391}\u{3A3}'\u{391}", "\u{3B1}\u{3C3}'\u{3B1}"),
            ("\u{391}\u{3A3}1", "\u{3B1}\u{3C2}1"),
            ("\u{3A3}\u{3A3}", "\u{3C3}\u{3C2}"),
            ("\u{391}\u{301}\u{3A3}", "\u{3B1}\u{301}\u{3C2}"),
            ("\u{391}\u{3A3}\u{301}", "\u{3B1}\u{3C2}\u{301}"),
            ("\u{130}", "i\u{307}"),
            ("\u{1E9E}", "\u{DF}"),
            ("\u{1C5}", "\u{1C6}"),
        ]
        for (text, lowered) in cases {
            #expect(
                Array(SwiftTransformersTokenizer.pythonLowercased(text).unicodeScalars)
                    == Array(lowered.unicodeScalars), "\(text)")
        }
        // A final sigma sorts before a medial one, so the capital key sorts first.
        let sigma = ["\u{3B1}\u{3C3}", "\u{391}\u{3A3}", "\u{3B1}\u{3C2}"]
        #expect(
            Self.scalars(try Self.sorted(sigma))
                == Self.scalars(["\u{391}\u{3A3}", "\u{3B1}\u{3C2}", "\u{3B1}\u{3C3}"]))
    }

    @Test("An object whose keys differ only in Unicode normalization is refused (D-058)")
    func canonicallyEquivalentKeys() throws {
        let pairs = [("\u{E9}", "e\u{301}"), ("K", "\u{212A}"), ("\u{C5}", "\u{212B}")]
        for (first, second) in pairs {
            let object = Self.dict([first, second])
            #expect(object.objectValue?.count == 2, "JSON keeps both, as Python does")
            #expect(throws: TokenizerError.self, "\(first)") {
                _ = try SwiftTransformersTokenizer.templateValue(object)
            }
        }
        let message =
            "an object in the messages has two keys that differ only in Unicode normalization "
            + "('e\u{301}'), which the template cannot tell apart"
        let call: JSONValue = [
            [
                "role": "assistant", "content": "",
                "tool_calls": [
                    [
                        "id": "c", "type": "function",
                        "function": ["name": "f", "arguments": Self.dict(["\u{E9}", "e\u{301}"])],
                    ]
                ],
            ]
        ]
        #expect(throws: TokenizerError(message)) {
            _ = try SwiftTransformersTokenizer.templateValue(call)
        }
        // Keys alike only under compatibility mapping are different keys in Swift too.
        let ligature = try SwiftTransformersTokenizer.templateValue(Self.dict(["\u{FB01}", "fi"]))
        #expect(
            try SwiftTransformersTokenizer.renderTemplate(
                "{{ d | length }}", context: ["d": ligature])
                == "2")
    }
}
