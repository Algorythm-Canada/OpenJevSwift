import Foundation
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// The `trim` of the chat template's environment, which is jinja2's, Python's `str.strip()`
/// (decision D-056), on templates of the tests' own. No checkpoint file is needed, so these run
/// in CI; ``ChatTemplateParityTests`` checks the same states through the shipped template when
/// the tokenizer files are present.
@Suite("The chat template's trim is jinja2's, Python's str.strip()")
struct ChatTemplateTrimTests {
    static func render(_ source: String, _ context: [String: any Sendable]) throws -> String {
        try SwiftTransformersTokenizer.renderTemplate(source, context: context)
    }

    /// `U+001C` for a message.
    static func name(_ scalar: Unicode.Scalar) -> String {
        String(format: "U+%04X", scalar.value)
    }

    /// Every scalar CPython's `str.isspace()` accepts, in code point order, as the fixture
    /// generator wrote them: the state of the `trim_whitespace_only` row of
    /// Fixtures/chat-prompts/prompts.json.
    static func pythonWhitespace() throws -> [Unicode.Scalar] {
        let rows = try TokenizerFixtures.cases("chat-prompts/prompts.json")
        let row = try #require(rows.first { $0["name"]?.stringValue == "trim_whitespace_only" })
        let state = try #require(row["messages"]?[1]?["content"]?.stringValue)
        return Array(state.unicodeScalars)
    }

    @Test("Without chars it strips every scalar str.isspace() accepts, U+001C to U+001F among them")
    func pythonWhitespace() throws {
        let whitespace = try Self.pythonWhitespace()
        #expect(whitespace.count == 29)
        #expect(whitespace.allSatisfy(TextOf.isPythonWhitespace))
        let all = String(String.UnicodeScalarView(whitespace))
        #expect(try Self.render("{{ s | trim }}", ["s": all + "x y" + all]) == "x y")
        #expect(try Self.render("{{ s | trim }}", ["s": all]) == "")
        for scalar in whitespace {
            let text = "\(scalar)\(scalar)Look at the photo.\(scalar)"
            #expect(
                try Self.render("{{ s | trim }}", ["s": text]) == "Look at the photo.",
                "\(Self.name(scalar))")
        }
    }

    @Test("It keeps what Python keeps: U+200B, other format characters and inner whitespace")
    func keeps() throws {
        for text in [
            "Look at the photo.\u{200B}", "\u{200B}x\u{200B}", "\u{FEFF}x\u{2060}", "x \u{1C} y",
        ] {
            #expect(
                try Self.render("{{ s | trim }}", ["s": text]) == text, "\(text.debugDescription)")
        }
        // A space and a combining accent make one grapheme; Python strips the space, a code point.
        #expect(try Self.render("{{ s | trim }}", ["s": " \u{301}x"]) == "\u{301}x")
    }

    @Test("With chars it strips the scalars given, and an empty string strips nothing")
    func chars() throws {
        let context: [String: any Sendable] = ["s": "\u{1C}xax\u{1C}", "c": "x\u{1C}"]
        #expect(try Self.render("{{ s | trim(c) }}", context) == "a")
        #expect(try Self.render("{{ s | trim(chars=c) }}", context) == "a")
        #expect(try Self.render("{{ s | trim('ab') }}", ["s": "abxba"]) == "x")
        #expect(try Self.render("{{ s | trim('') }}", ["s": "  x  "]) == "  x  ")
        #expect(try Self.render("{{ s | trim(none) }}", ["s": " \u{1C}x\u{1F} "]) == "x")
    }

    @Test("Other arguments throw where Python raises TypeError")
    func badArguments() {
        for source in [
            "{{ s | trim(5) }}", "{{ s | trim(no_such_name) }}", "{{ s | trim('a', 'b') }}",
            "{{ s | trim('a', chars='b') }}", "{{ s | trim(other='a') }}",
        ] {
            #expect(throws: (any Error).self, "\(source)") {
                try Self.render(source, ["s": " x "])
            }
        }
    }

    @Test("Macros, set blocks, filter blocks and loops reach the same trim, as the template does")
    func scopes() throws {
        let text = "\u{1C}y\u{1F}"
        #expect(
            try Self.render("{% macro m(x) %}{{ x | trim }}{% endmacro %}{{ m(s) }}", ["s": text])
                == "y")
        #expect(
            try Self.render(
                "{% set c %}{{ s }}{% endset %}{{ c | trim | length }}", ["s": "\u{1C}\u{1D}"])
                == "0")
        #expect(try Self.render("{% filter trim %}{{ s }}{% endfilter %}", ["s": text]) == "y")
        #expect(try Self.render("{% for x in [s] %}{{ x | trim }}{% endfor %}", ["s": text]) == "y")
        // Gemma 4's system text parts: the filter binds before `+`.
        #expect(try Self.render("{{ s | trim + ' ' }}", ["s": text]) == "y ")
    }

    /// A JSON value as the template reads a chat request's.
    static func value(_ json: JSONValue) -> any Sendable {
        SwiftTransformersTokenizer.templateValue(json)
    }

    @Test("A value that is not a string is trimmed as Python's str() writes it (D-058)")
    func nonStrings() throws {
        let cases: [(JSONValue, String)] = [
            (nil, "None"), (5, "5"), (true, "True"), (false, "False"), (1.5, "1.5"),
            (0.1, "0.1"), (1e16, "1e+16"), ([1, "a", nil, ["it's"]], "[1, 'a', None, [\"it's\"]]"),
            (["k": "v", "n": [true]], "{'k': 'v', 'n': [True]}"),
        ]
        for (json, expected) in cases {
            #expect(
                try Self.render("{{ x | trim }}", ["x": Self.value(json)]) == expected, "\(json)")
        }
        // An undefined value is jinja2's Undefined, whose text is empty.
        #expect(try Self.render("[{{ missing | trim }}]", [:]) == "[]")
    }

    @Test("A dict is a sequence, as jinja2's test says of anything with len() and [] (D-058)")
    func sequences() throws {
        let source = "{% if x is sequence %}yes{% else %}no{% endif %}"
        let cases: [(JSONValue, String)] = [
            (["a"], "yes"), ("text", "yes"), (["k": 1], "yes"), (.object(JSONObject()), "yes"),
            (nil, "no"), (7, "no"), (true, "no"),
        ]
        for (json, expected) in cases {
            #expect(try Self.render(source, ["x": Self.value(json)]) == expected, "\(json)")
        }
        // Iterating a dict gives its keys, as Gemma 4's template does with a dict content.
        #expect(
            try Self.render(
                "{% for k in x %}{{ k }},{% endfor %}", ["x": Self.value(["b": 1, "a": 2])])
                == "b,a,")
    }
}
