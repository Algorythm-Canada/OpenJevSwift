// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, function
// `text_of`. Apache-2.0. See THIRD_PARTY.md.

/// Renders a Jev description or instruction as the text the model reads.
public enum TextOf {
    /// The text of a description or instruction, as upstream's `text_of` gives it.
    ///
    /// - absent or `null`: an empty string;
    /// - a string: the string without the leading and trailing scalars Python's `str.strip()`
    ///   removes (see ``isPythonWhitespace(_:)``);
    /// - anything else: `json.dumps(value, ensure_ascii=False)`, with the default separators.
    ///
    /// The wire types allow only strings, objects and arrays here, but a number or a Boolean is
    /// written as `json.dumps` would write it.
    ///
    /// - Precondition: The value holds no infinite or NaN float and no malformed integer text.
    ///   ``JSONParser`` never produces either (decision D-016).
    public static func render(_ value: Described) -> String {
        switch value {
        case .none, .some(.null):
            return ""
        case .some(.string(let text)):
            return pythonStripped(text)
        case .some(let other):
            return modelText(other)
        }
    }

    /// True for the scalars CPython's `str.isspace()` accepts, which `str.strip()` removes:
    /// U+0009 to U+000D, U+001C to U+0020, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028,
    /// U+2029, U+202F, U+205F and U+3000.
    ///
    /// This is not Swift's `whitespacesAndNewlines`: U+001C to U+001F are included, and U+200B
    /// (zero width space) is not.
    public static func isPythonWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029,
            0x202F, 0x205F, 0x3000:
            return true
        default:
            return false
        }
    }

    /// The text without leading and trailing Python whitespace, compared scalar by scalar.
    static func pythonStripped(_ text: String) -> String {
        let scalars = text.unicodeScalars
        guard let start = scalars.firstIndex(where: { !isPythonWhitespace($0) }),
            let end = scalars.lastIndex(where: { !isPythonWhitespace($0) })
        else {
            return ""
        }
        return String(scalars[start...end])
    }

    /// The text without trailing Python whitespace, as `str.rstrip()` leaves it.
    static func pythonRightStripped(_ text: String) -> String {
        let scalars = text.unicodeScalars
        guard let end = scalars.lastIndex(where: { !isPythonWhitespace($0) }) else {
            return ""
        }
        return String(scalars[...end])
    }

    /// `json.dumps(value, ensure_ascii=False)` for a value that came through the wire types.
    ///
    /// - Precondition: The value can be written. Only a hand-built value with an infinite or
    ///   NaN float, or with integer text that is not normalized digits, cannot.
    static func modelText(_ value: JSONValue) -> String {
        do {
            return try PythonJSONWriter.modelText(value)
        } catch {
            preconditionFailure("a model text value cannot be written as JSON: \(error)")
        }
    }
}
