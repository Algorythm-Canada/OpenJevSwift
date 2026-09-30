// A port of CPython 3.14's `json.loads` error behaviour: `Modules/_json.c`
// (`scanstring_unicode`, `_parse_object_unicode`, `_parse_array_unicode`, `scan_once_unicode`,
// `_match_number_unicode`) and `Lib/json/decoder.py` and `Lib/json/__init__.py` (`decode`,
// `raw_decode`, `detect_encoding`), with the `ValueError` of `int()` past 4,300 digits. PSF-2.0.
// See THIRD_PARTY.md. Checked against CPython itself: Fixtures/python-json/decode_errors.json
// records what `json.loads` does with each of its documents.

/// What CPython's `json.loads(body)` does with the bytes of a request body, which is how
/// Starlette's `Request.json()` reads one.
///
/// ``JSONParser`` is stricter than `json.loads` (decision D-016) and reports its own errors. When
/// it refuses a body, ``outcome(of:)`` says whether CPython would refuse it too, and with which
/// error, so that a server can answer as upstream does.
///
/// Only UTF-8 is read. `json.loads` also recognizes UTF-16 and UTF-32 by a byte order mark or by
/// NUL bytes among the first four, and decodes UTF-8 with `surrogatepass`, which lets an encoded
/// surrogate through; here those bytes are simply not UTF-8.
public enum PythonJSONLoads {
    /// The most digits `int()` converts, CPython's default `sys.int_max_str_digits`.
    public static let maximumIntegerDigits = 4300

    /// How `json.loads` ends on a document.
    public enum Outcome: Sendable, Hashable {
        /// It returns a value. That includes documents ``JSONParser`` refuses: `NaN`, `Infinity`,
        /// `-Infinity`, a float beyond `Double`, a lone surrogate escape and nesting of any depth
        /// (CPython stops only when its C stack runs out).
        case accepted
        /// The bytes are not UTF-8: a `UnicodeDecodeError`.
        case notUTF8
        /// A `JSONDecodeError`: its `msg`, CPython's message without the position suffix, and
        /// its `pos`, a count of characters (Unicode scalars) from the start of the document
        /// after any byte order mark.
        case decodeError(message: String, position: Int)
        /// An integer with more than ``maximumIntegerDigits`` digits, before any other error:
        /// the `ValueError` `int()` raises.
        case integerTooLong
    }

    /// The UTF-8 byte order mark, which `json.loads` drops.
    public static let byteOrderMark: [UInt8] = [0xEF, 0xBB, 0xBF]

    /// The bytes `json.loads` parses: the document after a leading UTF-8 byte order mark.
    public static func document<Bytes: Collection<UInt8>>(_ bytes: Bytes) -> Bytes.SubSequence {
        bytes.starts(with: byteOrderMark)
            ? bytes.dropFirst(byteOrderMark.count) : bytes[...]
    }

    /// How `json.loads` ends on `bytes`: the byte order mark is dropped, the rest must be UTF-8,
    /// and then CPython's scanner runs, which stops at the first error.
    ///
    /// It keeps its own stack, so deep nesting cannot overflow the thread's stack.
    public static func outcome(of bytes: some Collection<UInt8>) -> Outcome {
        let text = Array(Self.document(bytes))
        guard isUTF8(text) else {
            return .notUTF8
        }
        return text.withUnsafeBufferPointer { buffer in
            var scanner = Scanner(bytes: buffer)
            switch scanner.scan() {
            case .accepted:
                return .accepted
            case .integerTooLong:
                return .integerTooLong
            case .error(let message, let byteOffset):
                return .decodeError(
                    message: message, position: characterCount(of: buffer.prefix(byteOffset)))
            }
        }
    }

    /// Whether a parsed value holds an integer with more than ``maximumIntegerDigits`` digits,
    /// which `json.loads` refuses and ``JSONParser`` accepts.
    public static func hasIntegerBeyondDigitLimit(_ value: JSONValue) -> Bool {
        var pending = [value]
        while let next = pending.popLast() {
            switch next {
            case .integer(let text):
                let digits = text.utf8.count - (text.hasPrefix("-") ? 1 : 0)
                if digits > maximumIntegerDigits {
                    return true
                }
            case .array(let elements):
                pending.append(contentsOf: elements)
            case .object(let object):
                pending.append(contentsOf: object.values)
            case .null, .bool, .float, .string:
                break
            }
        }
        return false
    }

    /// The characters (Unicode scalars) in UTF-8 bytes, which is how CPython counts positions:
    /// every byte that does not continue a multibyte sequence starts one.
    public static func characterCount(of bytes: some Sequence<UInt8>) -> Int {
        bytes.reduce(0) { count, byte in byte & 0xC0 == 0x80 ? count : count + 1 }
    }

    /// Whether the bytes are well-formed UTF-8: no overlong forms, no encoded surrogates, nothing
    /// above U+10FFFF and no truncated sequence.
    static func isUTF8(_ bytes: [UInt8]) -> Bool {
        let failed = transcode(
            bytes.makeIterator(), from: UTF8.self, to: UTF32.self, stoppingOnError: true,
            into: { _ in })
        return !failed
    }
}

/// CPython's scanner over one UTF-8 document, reporting only how it ends.
///
/// Every decision CPython makes compares a character with an ASCII one, and every position it
/// reports is at a character boundary, so scanning bytes gives the same result as scanning
/// characters; the caller turns the byte offset into a character count.
private struct Scanner {
    /// How a scan ends.
    enum End {
        case accepted
        case integerTooLong
        case error(String, Int)
    }

    /// What the scanner expects next.
    private enum Expectation {
        /// A value, as `scan_once_unicode` reads one.
        case value
        /// An object key and its `:`, after `{` or `,`.
        case key
        /// A `,` or the close of the innermost container, or the end of the document.
        case afterValue
    }

    let bytes: UnsafeBufferPointer<UInt8>
    /// The open containers, innermost last: `true` for an object, `false` for an array.
    private var containers: [Bool] = []

    init(bytes: UnsafeBufferPointer<UInt8>) {
        self.bytes = bytes
    }

    /// `JSONDecoder.decode`: whitespace, one value, whitespace, then nothing.
    mutating func scan() -> End {
        let count = bytes.count
        var index = skipWhitespace(from: 0)
        var expectation = Expectation.value
        while true {
            switch expectation {
            case .value:
                // `scan_once_unicode`; a `StopIteration(idx)` becomes "Expecting value" at idx.
                guard index < count else {
                    return .error("Expecting value", index)
                }
                switch bytes[index] {
                case UInt8(ascii: "\""):
                    switch scanString(from: index + 1) {
                    case .success(let end): index = end
                    case .failure(let error): return error.end
                    }
                    expectation = .afterValue
                case UInt8(ascii: "{"):
                    containers.append(true)
                    index = skipWhitespace(from: index + 1)
                    if index < count, bytes[index] == UInt8(ascii: "}") {
                        containers.removeLast()
                        index += 1
                        expectation = .afterValue
                    } else {
                        expectation = .key
                    }
                case UInt8(ascii: "["):
                    containers.append(false)
                    index = skipWhitespace(from: index + 1)
                    if index < count, bytes[index] == UInt8(ascii: "]") {
                        containers.removeLast()
                        index += 1
                        expectation = .afterValue
                    }
                default:
                    if let end = constantEnd(at: index) {
                        index = end
                    } else {
                        switch scanNumber(from: index) {
                        case .success(let end): index = end
                        case .failure(let error): return error.end
                        }
                    }
                    expectation = .afterValue
                }
            case .key:
                guard index < count, bytes[index] == UInt8(ascii: "\"") else {
                    return .error("Expecting property name enclosed in double quotes", index)
                }
                switch scanString(from: index + 1) {
                case .success(let end): index = skipWhitespace(from: end)
                case .failure(let error): return error.end
                }
                guard index < count, bytes[index] == UInt8(ascii: ":") else {
                    return .error("Expecting ':' delimiter", index)
                }
                index = skipWhitespace(from: index + 1)
                expectation = .value
            case .afterValue:
                guard let isObject = containers.last else {
                    index = skipWhitespace(from: index)
                    return index == count ? .accepted : .error("Extra data", index)
                }
                let closer = isObject ? UInt8(ascii: "}") : UInt8(ascii: "]")
                index = skipWhitespace(from: index)
                if index < count, bytes[index] == closer {
                    containers.removeLast()
                    index += 1
                    continue
                }
                guard index < count, bytes[index] == UInt8(ascii: ",") else {
                    return .error("Expecting ',' delimiter", index)
                }
                let comma = index
                index = skipWhitespace(from: index + 1)
                if index < count, bytes[index] == closer {
                    let kind = isObject ? "object" : "array"
                    return .error("Illegal trailing comma before end of \(kind)", comma)
                }
                expectation = isObject ? .key : .value
            }
        }
    }

    /// A scan that stopped: how the whole document ends.
    private struct Stop: Error {
        let end: End
    }

    /// The words `scan_once_unicode` recognizes: the first character and the rest.
    private static let constants: [(first: UInt8, rest: StaticString)] = [
        (UInt8(ascii: "n"), "ull"), (UInt8(ascii: "t"), "rue"), (UInt8(ascii: "f"), "alse"),
        (UInt8(ascii: "N"), "aN"), (UInt8(ascii: "I"), "nfinity"), (UInt8(ascii: "-"), "Infinity"),
    ]

    /// `null`, `true`, `false`, `NaN`, `Infinity` or `-Infinity` at `index`, when one starts
    /// there with a character after it: CPython requires one (`idx + 3 < length` for `null`),
    /// so a document that ends inside or right after the word falls through to the number
    /// matcher, which refuses it.
    private func constantEnd(at index: Int) -> Int? {
        for word in Self.constants where bytes[index] == word.first {
            let rest = UnsafeBufferPointer(
                start: word.rest.utf8Start, count: word.rest.utf8CodeUnitCount)
            guard index + rest.count < bytes.count else { return nil }
            for (offset, byte) in rest.enumerated() where bytes[index + 1 + offset] != byte {
                return nil
            }
            return index + 1 + rest.count
        }
        return nil
    }

    /// `scanstring_unicode` from just after the opening quote at `end - 1`: the index after the
    /// closing quote. Surrogate pairs need no handling here: pairing two escapes, or not, never
    /// changes which error is raised or where.
    private func scanString(from start: Int) -> Result<Int, Stop> {
        let count = bytes.count
        let begin = start - 1
        var end = start
        while true {
            var next = end
            while next < count {
                let byte = bytes[next]
                if byte == UInt8(ascii: "\"") || byte == UInt8(ascii: "\\") {
                    break
                }
                if byte <= 0x1F {
                    return .failure(Stop(end: .error("Invalid control character at", next)))
                }
                next += 1
            }
            guard next < count else {
                return .failure(Stop(end: .error("Unterminated string starting at", begin)))
            }
            if bytes[next] == UInt8(ascii: "\"") {
                return .success(next + 1)
            }
            next += 1
            guard next < count else {
                return .failure(Stop(end: .error("Unterminated string starting at", begin)))
            }
            if bytes[next] != UInt8(ascii: "u") {
                end = next + 1
                switch bytes[next] {
                case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"), UInt8(ascii: "b"),
                    UInt8(ascii: "f"), UInt8(ascii: "n"), UInt8(ascii: "r"), UInt8(ascii: "t"):
                    break
                default:
                    return .failure(Stop(end: .error("Invalid \\escape", end - 2)))
                }
            } else {
                next += 1
                end = next + 4
                guard end <= count else {
                    return .failure(Stop(end: .error("Invalid \\uXXXX escape", next - 1)))
                }
                for position in next..<end where !isHexDigit(bytes[position]) {
                    return .failure(Stop(end: .error("Invalid \\uXXXX escape", end - 5)))
                }
            }
        }
    }

    /// `_match_number_unicode` at `start`: the index after the number. A `-` or a character that
    /// cannot start one is "Expecting value" at `start`; a fraction without digits or an
    /// exponent without digits is left for the caller to trip over, as CPython backtracks.
    private func scanNumber(from start: Int) -> Result<Int, Stop> {
        let last = bytes.count - 1
        var index = start
        let negative = bytes[index] == UInt8(ascii: "-")
        if negative {
            index += 1
            guard index <= last else {
                return .failure(Stop(end: .error("Expecting value", start)))
            }
        }
        if bytes[index] >= UInt8(ascii: "1") && bytes[index] <= UInt8(ascii: "9") {
            index += 1
            while index <= last, isDigit(bytes[index]) {
                index += 1
            }
        } else if bytes[index] == UInt8(ascii: "0") {
            index += 1
        } else {
            return .failure(Stop(end: .error("Expecting value", start)))
        }
        let integerDigits = index - start - (negative ? 1 : 0)
        var isFloat = false
        if index < last, bytes[index] == UInt8(ascii: "."), isDigit(bytes[index + 1]) {
            isFloat = true
            index += 2
            while index <= last, isDigit(bytes[index]) {
                index += 1
            }
        }
        if index < last, bytes[index] | 0x20 == UInt8(ascii: "e") {
            let exponentStart = index
            index += 1
            if index < last, bytes[index] == UInt8(ascii: "-") || bytes[index] == UInt8(ascii: "+")
            {
                index += 1
            }
            while index <= last, isDigit(bytes[index]) {
                index += 1
            }
            if isDigit(bytes[index - 1]) {
                isFloat = true
            } else {
                index = exponentStart
            }
        }
        if !isFloat && integerDigits > PythonJSONLoads.maximumIntegerDigits {
            return .failure(Stop(end: .integerTooLong))
        }
        return .success(index)
    }

    /// The index of the first byte at or after `index` that is not JSON whitespace.
    private func skipWhitespace(from index: Int) -> Int {
        var index = index
        while index < bytes.count {
            switch bytes[index] {
            case 0x20, 0x09, 0x0A, 0x0D: index += 1
            default: return index
            }
        }
        return index
    }

    private func isDigit(_ byte: UInt8) -> Bool {
        byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9")
    }

    private func isHexDigit(_ byte: UInt8) -> Bool {
        isDigit(byte) || (byte | 0x20 >= UInt8(ascii: "a") && byte | 0x20 <= UInt8(ascii: "f"))
    }
}
