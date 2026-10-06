// A port of upstream OpenJev (razorback16/openjev at dcd2094), `extract_json` in `openjev/chat.py`,
// and of what it calls in CPython 3.14: `re.sub` with its pattern, `json.JSONDecoder.raw_decode`
// (`Modules/_json.c`: `scan_once_unicode`, `scanstring_unicode`, `_match_number_unicode`) and
// `json.dumps(obj, ensure_ascii=False)`. Apache-2.0 and PSF-2.0. See THIRD_PARTY.md.

/// Upstream's `extract_json`: the first JSON object or array in a reply, written back as
/// `json.dumps(obj, ensure_ascii=False)` writes it, or the reply unchanged when none decodes.
///
/// JSON mode asks the model for one JSON object and nothing else; this keeps the object when the
/// model wraps it in prose or a code fence anyway. The reply is stripped of Python's whitespace,
/// one opening fence (three backticks and an optional `json`, then whitespace) and one closing
/// fence (whitespace, then three backticks at the end) are dropped, and from each `{` or `[` in
/// turn a value is decoded as CPython's `raw_decode` decodes it: text after it is ignored, `NaN`,
/// `Infinity` and `-Infinity` are numbers, a float too large for a double is infinite, a later
/// duplicate key replaces the earlier value in the earlier place, and an integer of more than
/// 4,300 digits fails. The first value that decodes is written with `, ` and `: ` between items
/// and its non-ASCII characters as they are.
///
/// Two departures (D-058): a `\u` escape of a lone surrogate decodes to U+FFFD, where CPython
/// keeps the surrogate and then cannot encode the answer, a 500; and a value nested deeper than
/// ``maximumNesting`` levels gives the reply unchanged, where CPython decodes it up to where its
/// stack runs out (about 58,000 levels) and raises `RecursionError` past that, a 500.
public enum ExtractJSON {
    /// The deepest nesting decoded: ``JSONParser``'s default, 1,024 levels. A decoded value is
    /// released level by level, so a deeper one could overflow a task's stack.
    public static let maximumNesting = JSONParser.Options().maximumDepth

    /// The first JSON object or array in `text`, or `text` itself.
    public static func extract(_ text: String) -> String {
        let scalars = withoutFences(Array(TextOf.pythonStripped(text).unicodeScalars))
        for start in scalars.indices where scalars[start] == "{" || scalars[start] == "[" {
            var decoder = RawDecoder(scalars: scalars)
            switch decoder.decode(at: start) {
            case .value(let value):
                let options = PythonJSONWriter.Options(ensureASCII: false, allowNaN: true)
                // A decoded value holds only normalized integers, so it can always be written.
                return (try? PythonJSONWriter(options: options).string(value)) ?? text
            case .failed:
                continue
            case .tooDeep:
                return text
            }
        }
        return text
    }

    /// `re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)` for a stripped text, which has no
    /// trailing newline for `$` to match before: the opening fence at the start, then the closing
    /// fence at the end if it starts after the opening one ends. `\s` is CPython's `str.isspace()`.
    static func withoutFences(_ scalars: [Unicode.Scalar]) -> [Unicode.Scalar] {
        let fence: [Unicode.Scalar] = ["`", "`", "`"]
        var start = 0
        if scalars.starts(with: fence) {
            start = 3
            if scalars[start...].starts(with: ["j", "s", "o", "n"]) {
                start += 4
            }
            while start < scalars.count, TextOf.isPythonWhitespace(scalars[start]) {
                start += 1
            }
        }
        var end = scalars.count
        if end - 3 >= start, scalars[(end - 3)...].elementsEqual(fence) {
            end -= 3
            while end > start, TextOf.isPythonWhitespace(scalars[end - 1]) {
                end -= 1
            }
        }
        return Array(scalars[start..<end])
    }

    /// How a decode from one position ends.
    enum Outcome {
        case value(JSONValue)
        case failed
        case tooDeep
    }

    /// CPython's `scan_once` from one position: a value and nothing after it, with an explicit
    /// stack of the containers still open.
    struct RawDecoder {
        let scalars: [Unicode.Scalar]
        var position = 0

        init(scalars: [Unicode.Scalar]) {
            self.scalars = scalars
        }

        /// A container still being read.
        private enum Frame {
            case array([JSONValue])
            case object(JSONObject, key: String)
        }

        /// What a value position started: a complete value, or a container to fill.
        private enum Start: Equatable {
            case value(JSONValue)
            case array
            case object(key: String)
            case emptyArray
            case emptyObject
        }

        private struct Failure: Error {}

        mutating func decode(at start: Int) -> Outcome {
            position = start
            var stack: [Frame] = []
            do {
                values: while true {
                    var value: JSONValue
                    let start = try startValue()
                    switch start {
                    case .value(let scalar):
                        value = scalar
                    case .emptyArray, .emptyObject:
                        // An empty container is a level too.
                        if stack.count + 1 > ExtractJSON.maximumNesting { return .tooDeep }
                        value = start == .emptyArray ? .array([]) : .object(JSONObject())
                    case .array:
                        stack.append(.array([]))
                        if stack.count > ExtractJSON.maximumNesting { return .tooDeep }
                        continue values
                    case .object(let key):
                        stack.append(.object(JSONObject(), key: key))
                        if stack.count > ExtractJSON.maximumNesting { return .tooDeep }
                        continue values
                    }
                    // The value goes into the innermost open container, which either closes, its
                    // own value going into the next one out, or waits for another value.
                    closing: while true {
                        guard let frame = stack.popLast() else { return .value(value) }
                        switch frame {
                        case .array(var elements):
                            elements.append(value)
                            skipWhitespace()
                            if take("]") {
                                value = .array(elements)
                                continue closing
                            }
                            guard take(",") else { throw Failure() }
                            skipWhitespace()
                            stack.append(.array(elements))
                        case .object(var object, let key):
                            object.updateValue(value, forKey: key)
                            skipWhitespace()
                            if take("}") {
                                value = .object(object)
                                continue closing
                            }
                            guard take(",") else { throw Failure() }
                            skipWhitespace()
                            let next = try string()
                            skipWhitespace()
                            guard take(":") else { throw Failure() }
                            skipWhitespace()
                            stack.append(.object(object, key: next))
                        }
                        continue values
                    }
                }
            } catch {
                return .failed
            }
        }

        /// The value at the current position: a scalar, or the opening of a container with, for
        /// an object, its first key read up to the value.
        private mutating func startValue() throws -> Start {
            guard position < scalars.count else { throw Failure() }
            switch scalars[position] {
            case "\"":
                return .value(.string(try string()))
            case "{":
                position += 1
                skipWhitespace()
                if take("}") {
                    return .emptyObject
                }
                let key = try string()
                skipWhitespace()
                guard take(":") else { throw Failure() }
                skipWhitespace()
                return .object(key: key)
            case "[":
                position += 1
                skipWhitespace()
                if take("]") {
                    return .emptyArray
                }
                return .array
            case "n" where take("null"):
                return .value(.null)
            case "t" where take("true"):
                return .value(.bool(true))
            case "f" where take("false"):
                return .value(.bool(false))
            case "N" where take("NaN"):
                return .value(.float(.nan))
            case "I" where take("Infinity"):
                return .value(.float(.infinity))
            case "-" where take("-Infinity"):
                return .value(.float(-.infinity))
            default:
                return .value(try number())
            }
        }

        /// CPython's `_match_number_unicode`: `-?(0|[1-9][0-9]*)`, then a fraction only when a
        /// digit follows the point, and an exponent only when a digit follows `e` and its sign.
        private mutating func number() throws -> JSONValue {
            let start = position
            var index = position
            if index < scalars.count, scalars[index] == "-" {
                index += 1
            }
            guard index < scalars.count, Self.isDigit(scalars[index]) else { throw Failure() }
            if scalars[index] == "0" {
                index += 1
            } else {
                while index < scalars.count, Self.isDigit(scalars[index]) {
                    index += 1
                }
            }
            var isFloat = false
            if index + 1 < scalars.count, scalars[index] == ".", Self.isDigit(scalars[index + 1]) {
                index += 2
                while index < scalars.count, Self.isDigit(scalars[index]) {
                    index += 1
                }
                isFloat = true
            }
            if index < scalars.count, scalars[index] == "e" || scalars[index] == "E" {
                var exponent = index + 1
                if exponent < scalars.count, scalars[exponent] == "-" || scalars[exponent] == "+" {
                    exponent += 1
                }
                if exponent < scalars.count, Self.isDigit(scalars[exponent]) {
                    index = exponent
                    while index < scalars.count, Self.isDigit(scalars[index]) {
                        index += 1
                    }
                    isFloat = true
                }
            }
            position = index
            let text = String(String.UnicodeScalarView(scalars[start..<index]))
            if isFloat {
                // float() rounds correctly and gives infinity past the largest double. The text
                // is a valid number, so a failed parse can only be out of range.
                return .float(Double(text) ?? (text.hasPrefix("-") ? -.infinity : .infinity))
            }
            let digits = text.hasPrefix("-") ? text.utf8.count - 1 : text.utf8.count
            guard digits <= PythonJSONLoads.maximumIntegerDigits else { throw Failure() }
            return .integer(text == "-0" ? "0" : text)
        }

        /// CPython's strict `scanstring`: no raw control characters, the JSON escapes, and a
        /// `\u` surrogate pair joined. A lone surrogate becomes U+FFFD.
        private mutating func string() throws -> String {
            guard take("\"") else { throw Failure() }
            var out = String.UnicodeScalarView()
            while true {
                guard position < scalars.count else { throw Failure() }
                let scalar = scalars[position]
                position += 1
                switch scalar {
                case "\"":
                    return String(out)
                case "\\":
                    guard position < scalars.count else { throw Failure() }
                    let escape = scalars[position]
                    position += 1
                    switch escape {
                    case "\"", "\\", "/":
                        out.append(escape)
                    case "b":
                        out.append("\u{08}")
                    case "f":
                        out.append("\u{0C}")
                    case "n":
                        out.append("\n")
                    case "r":
                        out.append("\r")
                    case "t":
                        out.append("\t")
                    case "u":
                        out.append(try unicodeEscape())
                    default:
                        throw Failure()
                    }
                default:
                    guard scalar.value >= 0x20 else { throw Failure() }
                    out.append(scalar)
                }
            }
        }

        /// The scalar of a `\u` escape whose `\u` has been read: a high surrogate followed by
        /// `\u` and a low surrogate is one scalar; any other surrogate is U+FFFD. As in CPython,
        /// four characters after a high surrogate's following `\u` that are not hexadecimal fail,
        /// and a following escape that is not a low surrogate is read again on its own.
        private mutating func unicodeEscape() throws -> Unicode.Scalar {
            let unit = try hexUnit()
            if (0xD800...0xDBFF).contains(unit), position + 1 < scalars.count,
                scalars[position] == "\\", scalars[position + 1] == "u"
            {
                let resume = position
                position += 2
                let low = try hexUnit()
                if (0xDC00...0xDFFF).contains(low) {
                    let value = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                    return Unicode.Scalar(value) ?? "\u{FFFD}"
                }
                position = resume
            }
            return Unicode.Scalar(unit) ?? "\u{FFFD}"
        }

        /// Four hexadecimal digits, either case.
        private mutating func hexUnit() throws -> UInt32 {
            guard position + 4 <= scalars.count else { throw Failure() }
            var value: UInt32 = 0
            for scalar in scalars[position..<(position + 4)] {
                guard let digit = Self.hexValue(scalar) else { throw Failure() }
                value = value << 4 | digit
            }
            position += 4
            return value
        }

        private static func hexValue(_ scalar: Unicode.Scalar) -> UInt32? {
            switch scalar {
            case "0"..."9": return scalar.value - 0x30
            case "a"..."f": return scalar.value - 0x61 + 10
            case "A"..."F": return scalar.value - 0x41 + 10
            default: return nil
            }
        }

        private static func isDigit(_ scalar: Unicode.Scalar) -> Bool {
            ("0"..."9").contains(scalar)
        }

        /// CPython's JSON whitespace: space, tab, line feed and carriage return.
        private mutating func skipWhitespace() {
            while position < scalars.count {
                switch scalars[position] {
                case " ", "\t", "\n", "\r":
                    position += 1
                default:
                    return
                }
            }
        }

        /// Consumes `text` when it comes next.
        private mutating func take(_ text: String) -> Bool {
            let expected = Array(text.unicodeScalars)
            guard scalars[position...].starts(with: expected) else { return false }
            position += expected.count
            return true
        }
    }
}
