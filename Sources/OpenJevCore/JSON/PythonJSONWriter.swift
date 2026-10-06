/// Writes JSON bytes identical to CPython's `json.dumps` for the same value and options.
///
/// Upstream OpenJev renders objects and arrays for the model with
/// `json.dumps(value, ensure_ascii=False)` and derives each request's seed from
/// `json.dumps(key, sort_keys=True)`. Use ``modelText(_:)`` and ``canonicalSeedBytes(_:)`` for
/// those two cases rather than choosing options by hand.
///
/// Output matches Python in these details:
///
/// - Strings escape `"`, `\`, `\n`, `\r`, `\t`, `\b` and `\f` by name and every other code
///   point below U+0020 as `\u00XX` in lowercase hex. `/` is never escaped.
/// - With ``Options/ensureASCII``, every scalar outside U+0020 to U+007E is escaped as `\uXXXX`
///   in lowercase hex, astral scalars as a UTF-16 surrogate pair. That includes U+007F, which
///   CPython escapes too. Without it, those scalars are written as UTF-8 and U+007F is left as is.
/// - Integers are written as their stored digits. Floats are written as Python's `repr(float)`.
/// - There is no trailing newline and no indentation.
public struct PythonJSONWriter: Sendable {
    /// The `json.dumps` arguments this writer reproduces.
    public struct Options: Sendable, Equatable {
        /// Escape every non-ASCII scalar, as `ensure_ascii` does. Python's default is true.
        public var ensureASCII: Bool
        /// Sort object keys by Unicode scalar value at every level, as `sort_keys` does.
        public var sortKeys: Bool
        /// The text between array elements and between object entries.
        public var itemSeparator: String
        /// The text between an object key and its value.
        public var keySeparator: String
        /// Write an infinite or NaN float as `Infinity`, `-Infinity` or `NaN`, as `allow_nan` does,
        /// instead of refusing it. Python's default is true; here it is false, so a value that is
        /// not JSON is never written unless a caller asks for upstream's output.
        public var allowNaN: Bool

        /// Creates options. The defaults are Python's `json.dumps` defaults, except `allowNaN`.
        public init(
            ensureASCII: Bool = true,
            sortKeys: Bool = false,
            itemSeparator: String = ", ",
            keySeparator: String = ": ",
            allowNaN: Bool = false
        ) {
            self.ensureASCII = ensureASCII
            self.sortKeys = sortKeys
            self.itemSeparator = itemSeparator
            self.keySeparator = keySeparator
            self.allowNaN = allowNaN
        }

        /// Python's `json.dumps(value)`.
        public static let pythonDefault = Options()

        /// Python's `json.dumps(value, separators=(",", ":"))`.
        public static let compact = Options(itemSeparator: ",", keySeparator: ":")
    }

    /// The options this writer applies.
    public var options: Options

    /// Creates a writer with the given options.
    public init(options: Options = .pythonDefault) {
        self.options = options
    }

    /// Writes a value as UTF-8 bytes.
    ///
    /// - Throws: ``JSONWriteError/nonFiniteNumber(_:)`` for an infinite or NaN float unless
    ///   ``Options/allowNaN`` is set, and ``JSONWriteError/invalidInteger(_:)`` for integer text
    ///   that is not normalized digits.
    public func bytes(_ value: JSONValue) throws(JSONWriteError) -> [UInt8] {
        var output: [UInt8] = []
        try write(value, into: &output)
        return output
    }

    /// Writes a value as a string.
    public func string(_ value: JSONValue) throws(JSONWriteError) -> String {
        String(decoding: try bytes(value), as: UTF8.self)
    }

    /// The bytes of `json.dumps(value, sort_keys=True)`, which upstream hashes to seed a request.
    public static func canonicalSeedBytes(_ value: JSONValue) throws(JSONWriteError) -> [UInt8] {
        try PythonJSONWriter(options: Options(sortKeys: true)).bytes(value)
    }

    /// The text of `json.dumps(value, ensure_ascii=False)`, which upstream puts in front of the
    /// model for object and array values.
    public static func modelText(_ value: JSONValue) throws(JSONWriteError) -> String {
        try PythonJSONWriter(options: Options(ensureASCII: false)).string(value)
    }

    /// An array or object whose children are still being written.
    private struct Frame {
        var keys: [String]?
        var values: [JSONValue]
        var next = 0
    }

    /// Writes a value with an explicit stack, so that deep nesting cannot overflow the thread's
    /// stack.
    private func write(_ root: JSONValue, into output: inout [UInt8]) throws(JSONWriteError) {
        let itemSeparator = Array(options.itemSeparator.utf8)
        let keySeparator = Array(options.keySeparator.utf8)
        var stack: [Frame] = []
        if let frame = try open(root, into: &output) {
            stack.append(frame)
        }
        while !stack.isEmpty {
            let top = stack.count - 1
            let index = stack[top].next
            if index == stack[top].values.count {
                output.append(stack[top].keys == nil ? UInt8(ascii: "]") : UInt8(ascii: "}"))
                stack.removeLast()
                continue
            }
            stack[top].next += 1
            if index > 0 {
                output.append(contentsOf: itemSeparator)
            }
            if let keys = stack[top].keys {
                writeString(keys[index], into: &output)
                output.append(contentsOf: keySeparator)
            }
            if let frame = try open(stack[top].values[index], into: &output) {
                stack.append(frame)
            }
        }
    }

    /// Writes a scalar or an empty container completely, or writes the opening bracket of a
    /// container that has children and returns its frame.
    private func open(_ value: JSONValue, into output: inout [UInt8]) throws(JSONWriteError)
        -> Frame?
    {
        switch value {
        case .null:
            output.append(contentsOf: "null".utf8)
        case .bool(let flag):
            output.append(contentsOf: (flag ? "true" : "false").utf8)
        case .integer(let digits):
            guard Self.isNormalizedInteger(digits) else {
                throw .invalidInteger(digits)
            }
            output.append(contentsOf: digits.utf8)
        case .float(let number):
            guard number.isFinite else {
                guard options.allowNaN else {
                    throw .nonFiniteNumber(number)
                }
                let text = number.isNaN ? "NaN" : number < 0 ? "-Infinity" : "Infinity"
                output.append(contentsOf: text.utf8)
                return nil
            }
            output.append(contentsOf: pythonFloatRepr(number).utf8)
        case .string(let text):
            writeString(text, into: &output)
        case .array(let elements):
            if elements.isEmpty {
                output.append(contentsOf: "[]".utf8)
                return nil
            }
            output.append(UInt8(ascii: "["))
            return Frame(keys: nil, values: elements)
        case .object(let object):
            if object.isEmpty {
                output.append(contentsOf: "{}".utf8)
                return nil
            }
            output.append(UInt8(ascii: "{"))
            if options.sortKeys {
                let keys = object.keys
                let values = object.values
                let order = keys.indices.sorted { scalarsPrecede(keys[$0], keys[$1]) }
                return Frame(keys: order.map { keys[$0] }, values: order.map { values[$0] })
            }
            return Frame(keys: object.keys, values: object.values)
        }
        return nil
    }

    /// Writes a quoted, escaped string the way CPython's encoder does.
    private func writeString(_ text: String, into output: inout [UInt8]) {
        output.append(UInt8(ascii: "\""))
        for scalar in text.unicodeScalars {
            let value = scalar.value
            switch value {
            case 0x22: output.append(contentsOf: #"\""#.utf8)
            case 0x5C: output.append(contentsOf: #"\\"#.utf8)
            case 0x0A: output.append(contentsOf: #"\n"#.utf8)
            case 0x0D: output.append(contentsOf: #"\r"#.utf8)
            case 0x09: output.append(contentsOf: #"\t"#.utf8)
            case 0x08: output.append(contentsOf: #"\b"#.utf8)
            case 0x0C: output.append(contentsOf: #"\f"#.utf8)
            case 0..<0x20:
                appendUnicodeEscape(UInt16(value), into: &output)
            case 0x20..<0x7F:
                output.append(UInt8(value))
            default:
                if options.ensureASCII {
                    for unit in UTF16.encode(scalar)! {
                        appendUnicodeEscape(unit, into: &output)
                    }
                } else {
                    output.append(contentsOf: UTF8.encode(scalar)!)
                }
            }
        }
        output.append(UInt8(ascii: "\""))
    }

    /// Appends `\u` and four lowercase hexadecimal digits.
    private func appendUnicodeEscape(_ unit: UInt16, into output: inout [UInt8]) {
        output.append(contentsOf: #"\u"#.utf8)
        for shift in stride(from: 12, through: 0, by: -4) {
            output.append(hexDigits[Int((unit >> UInt16(shift)) & 0xF)])
        }
    }

    /// True for an optional `-` followed by digits without leading zeros, and not `-0`.
    static func isNormalizedInteger(_ text: String) -> Bool {
        var digits = Substring(text).utf8
        if digits.first == UInt8(ascii: "-") {
            digits = digits.dropFirst()
        }
        guard let first = digits.first,
            digits.allSatisfy({ $0 >= UInt8(ascii: "0") && $0 <= UInt8(ascii: "9") })
        else {
            return false
        }
        if first == UInt8(ascii: "0") {
            return text == "0"
        }
        return true
    }
}

/// Lowercase hexadecimal digits, as CPython writes them in `\u` escapes.
private let hexDigits: [UInt8] = Array("0123456789abcdef".utf8)

/// An error that ``PythonJSONWriter`` throws.
public enum JSONWriteError: Error, Sendable, Equatable {
    /// An infinite or NaN float. Python would write `Infinity` or `NaN`, which is not JSON.
    case nonFiniteNumber(Double)
    /// Integer text that is not an optional `-` followed by digits without leading zeros.
    case invalidInteger(String)

    /// Returns true when both errors have the same case and payload. NaN payloads compare equal.
    public static func == (lhs: JSONWriteError, rhs: JSONWriteError) -> Bool {
        switch (lhs, rhs) {
        case (.nonFiniteNumber(let a), .nonFiniteNumber(let b)):
            return a == b || (a.isNaN && b.isNaN)
        case (.invalidInteger(let a), .invalidInteger(let b)):
            return a == b
        default:
            return false
        }
    }
}

/// Formats a finite double exactly as CPython's `repr(float)` does.
///
/// Swift's `description` already gives the shortest digits that round-trip, which is what CPython
/// computes too. Only the layout differs, so this function takes the digits and decimal exponent
/// out of `description` and lays them out again by CPython's rules: fixed notation when the
/// decimal exponent is between -4 and 15, exponential otherwise, `.0` after an integral fixed
/// value, and at least two exponent digits with an explicit sign.
func pythonFloatRepr(_ value: Double) -> String {
    let isNegative = value.sign == .minus
    let sign = isNegative ? "-" : ""
    if value == 0 {
        return sign + "0.0"
    }

    // Split the description into its digits, the count of digits before the point, and the
    // exponent. For example "1.25e-07" gives digits 125, one integer digit and exponent -7.
    var digits: [UInt8] = []
    var integerDigitCount = 0
    var seenPoint = false
    var exponent = 0
    var text = Substring(value.magnitude.description).utf8
    while let byte = text.first {
        text = text.dropFirst()
        if byte == UInt8(ascii: ".") {
            seenPoint = true
        } else if byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E") {
            exponent = Int(String(decoding: text, as: UTF8.self))!
            break
        } else {
            digits.append(byte)
            if !seenPoint {
                integerDigitCount += 1
            }
        }
    }

    // Drop leading zeros ("0.001" has digits 0001), each of which moves the point left, and
    // trailing zeros, which do not change the value.
    while digits.first == UInt8(ascii: "0") {
        digits.removeFirst()
        integerDigitCount -= 1
    }
    while digits.last == UInt8(ascii: "0") {
        digits.removeLast()
    }

    // The value is 0.DIGITS times ten to the power pointPosition.
    let pointPosition = integerDigitCount + exponent
    let decimalExponent = pointPosition - 1
    let digitText = String(decoding: digits, as: UTF8.self)

    if decimalExponent < -4 || decimalExponent > 15 {
        var mantissa = String(digitText.prefix(1))
        if digits.count > 1 {
            mantissa += "." + digitText.dropFirst()
        }
        let exponentSign = decimalExponent < 0 ? "-" : "+"
        let exponentDigits = String(abs(decimalExponent))
        let paddedExponent = exponentDigits.count < 2 ? "0" + exponentDigits : exponentDigits
        return sign + mantissa + "e" + exponentSign + paddedExponent
    }
    if pointPosition <= 0 {
        return sign + "0." + String(repeating: "0", count: -pointPosition) + digitText
    }
    if pointPosition >= digits.count {
        return sign + digitText + String(repeating: "0", count: pointPosition - digits.count) + ".0"
    }
    let integerPart = digitText.prefix(pointPosition)
    let fractionPart = digitText.dropFirst(pointPosition)
    return sign + integerPart + "." + fractionPart
}
