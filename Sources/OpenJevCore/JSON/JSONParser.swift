import Foundation

/// A strict RFC 8259 parser that keeps object order.
///
/// Foundation's `JSONDecoder` and `JSONSerialization` lose the order of object keys, which Jev
/// depends on. This parser keeps it, and it keeps integers and floats apart the way Python's
/// `json.loads` does (see ``JSONValue``).
///
/// It accepts only what RFC 8259 allows: no comments, trailing commas, single quotes, leading
/// zeros, `NaN` or `Infinity`, no unescaped control characters in strings, no byte order mark,
/// and only well-formed UTF-8. A `\u` escape for a high surrogate must be followed by one for a
/// low surrogate, and the pair becomes one scalar; a lone surrogate is an error. A float whose
/// value overflows `Double` is an error, where Python would produce infinity.
///
/// A repeated object key keeps the position of its first occurrence and takes the value of its
/// last, which is what Python's `dict` does during `json.loads`.
///
/// The parser keeps its own stack, so deep nesting cannot overflow the thread's stack. Nesting
/// beyond ``Options/maximumDepth`` throws ``JSONParseError/Kind/depthExceeded``.
public struct JSONParser: Sendable {
    /// Limits that protect the parser from hostile input.
    public struct Options: Sendable, Equatable {
        /// The deepest nesting of arrays and objects accepted. The default is 1024.
        public var maximumDepth: Int
        /// The largest input accepted, in bytes. The default is 64 MiB.
        public var maximumBytes: Int

        /// Creates options with the given limits.
        public init(maximumDepth: Int = 1024, maximumBytes: Int = 64 * 1024 * 1024) {
            self.maximumDepth = maximumDepth
            self.maximumBytes = maximumBytes
        }
    }

    /// The limits this parser applies.
    public var options: Options

    /// Creates a parser with the given limits.
    public init(options: Options = Options()) {
        self.options = options
    }

    /// Parses one JSON value from UTF-8 bytes.
    ///
    /// Whitespace may surround the value; anything else after it is an error.
    public func parse(_ bytes: some Collection<UInt8>) throws(JSONParseError) -> JSONValue {
        try parseCollection(bytes)
    }

    /// Parses one JSON value from UTF-8 data.
    public func parse(_ data: Data) throws(JSONParseError) -> JSONValue {
        try parseCollection(data)
    }

    /// Parses one JSON value from a string.
    public func parse(_ text: String) throws(JSONParseError) -> JSONValue {
        try parse(text.utf8)
    }

    private func parseCollection(_ bytes: some Collection<UInt8>) throws(JSONParseError)
        -> JSONValue
    {
        if bytes.count > options.maximumBytes {
            throw JSONParseError(kind: .tooLarge, offset: 0, line: 1, column: 1)
        }
        let contiguous = bytes.withContiguousStorageIfAvailable { buffer in
            Result { () throws(JSONParseError) in try parseBuffer(buffer) }
        }
        if let contiguous {
            return try contiguous.get()
        }
        return try Array(bytes).withUnsafeBufferPointer { buffer throws(JSONParseError) in
            try parseBuffer(buffer)
        }
    }

    private func parseBuffer(_ buffer: UnsafeBufferPointer<UInt8>) throws(JSONParseError)
        -> JSONValue
    {
        var scanner = Scanner(bytes: buffer, maximumDepth: options.maximumDepth)
        return try scanner.parseDocument()
    }
}

/// An error that ``JSONParser`` throws, with the position where parsing stopped.
public struct JSONParseError: Error, Sendable, Equatable, CustomStringConvertible {
    /// What went wrong.
    public enum Kind: Sendable, Equatable {
        /// The input ended before the value was complete.
        case unexpectedEndOfInput
        /// A byte that cannot start or continue a value here.
        case unexpectedCharacter
        /// A number that does not follow the JSON grammar, such as `01`, `1.` or `-`.
        case invalidNumber
        /// A float whose magnitude is too large for `Double`.
        case numberOutOfRange
        /// A control character below U+0020 that was not escaped inside a string.
        case unescapedControlCharacter
        /// A backslash followed by a character that is not a JSON escape.
        case invalidEscape
        /// A `\u` escape that is not followed by four hexadecimal digits.
        case invalidUnicodeEscape
        /// A `\u` escape for a surrogate that is not part of a high and low pair.
        case loneSurrogate
        /// Bytes inside a string that are not well-formed UTF-8.
        case invalidUTF8
        /// Arrays and objects nested deeper than the parser's maximum depth.
        case depthExceeded
        /// Input larger than the parser's maximum size. Parsing did not start.
        case tooLarge
        /// Something other than whitespace after the value.
        case trailingContent
    }

    /// What went wrong.
    public var kind: Kind
    /// The zero-based byte offset where the problem was found.
    public var offset: Int
    /// The one-based line of ``offset``. Lines end at line feeds.
    public var line: Int
    /// The one-based column of ``offset``, counted in bytes from the start of the line.
    public var column: Int

    /// Creates an error at a position.
    public init(kind: Kind, offset: Int, line: Int, column: Int) {
        self.kind = kind
        self.offset = offset
        self.line = line
        self.column = column
    }

    /// The kind and position, for logs.
    public var description: String {
        "\(kind) at line \(line), column \(column) (byte \(offset))"
    }
}

/// The parser's working state over one input buffer.
private struct Scanner {
    /// An array or object that is still open, with what has been read of it so far.
    struct Container {
        var isObject: Bool
        var elements: [JSONValue] = []
        var object = JSONObject()
        var pendingKey = ""
    }

    let bytes: UnsafeBufferPointer<UInt8>
    let maximumDepth: Int
    var position = 0
    var stack: [Container] = []
    /// Reused between strings to collect decoded bytes.
    var stringBuffer: [UInt8] = []

    init(bytes: UnsafeBufferPointer<UInt8>, maximumDepth: Int) {
        self.bytes = bytes
        self.maximumDepth = maximumDepth
    }

    mutating func parseDocument() throws(JSONParseError) -> JSONValue {
        let value = try parseValueTree()
        skipWhitespace()
        if position < bytes.count {
            throw error(.trailingContent, at: position)
        }
        return value
    }

    /// Reads one complete value, opening and closing containers on the explicit stack.
    mutating func parseValueTree() throws(JSONParseError) -> JSONValue {
        while true {
            // Read the start of a value. Containers that are not empty loop back here for their
            // first element; everything else produces a finished value.
            skipWhitespace()
            guard position < bytes.count else {
                throw error(.unexpectedEndOfInput, at: position)
            }
            var value: JSONValue
            switch bytes[position] {
            case UInt8(ascii: "["), UInt8(ascii: "{"):
                let isObject = bytes[position] == UInt8(ascii: "{")
                if stack.count >= maximumDepth {
                    throw error(.depthExceeded, at: position)
                }
                position += 1
                skipWhitespace()
                let closer = isObject ? UInt8(ascii: "}") : UInt8(ascii: "]")
                if position < bytes.count, bytes[position] == closer {
                    position += 1
                    value = isObject ? .object(JSONObject()) : .array([])
                } else {
                    var container = Container(isObject: isObject)
                    if isObject {
                        container.pendingKey = try parseKeyAndColon()
                    }
                    stack.append(container)
                    continue
                }
            default:
                value = try parseScalar()
            }

            // Hand the finished value to the innermost open container, then close as many
            // containers as the input closes here.
            while true {
                guard !stack.isEmpty else { return value }
                let top = stack.count - 1
                if stack[top].isObject {
                    let key = stack[top].pendingKey
                    stack[top].object.updateValue(value, forKey: key)
                } else {
                    stack[top].elements.append(value)
                }
                skipWhitespace()
                guard position < bytes.count else {
                    throw error(.unexpectedEndOfInput, at: position)
                }
                let byte = bytes[position]
                if byte == UInt8(ascii: ",") {
                    position += 1
                    if stack[top].isObject {
                        skipWhitespace()
                        stack[top].pendingKey = try parseKeyAndColon()
                    }
                    break
                }
                let closer = stack[top].isObject ? UInt8(ascii: "}") : UInt8(ascii: "]")
                guard byte == closer else {
                    throw error(.unexpectedCharacter, at: position)
                }
                position += 1
                let finished = stack.removeLast()
                value = finished.isObject ? .object(finished.object) : .array(finished.elements)
            }
        }
    }

    /// Reads an object key and the colon after it. The position is at the key's opening quote.
    mutating func parseKeyAndColon() throws(JSONParseError) -> String {
        guard position < bytes.count else {
            throw error(.unexpectedEndOfInput, at: position)
        }
        guard bytes[position] == UInt8(ascii: "\"") else {
            throw error(.unexpectedCharacter, at: position)
        }
        let key = try parseString()
        skipWhitespace()
        guard position < bytes.count else {
            throw error(.unexpectedEndOfInput, at: position)
        }
        guard bytes[position] == UInt8(ascii: ":") else {
            throw error(.unexpectedCharacter, at: position)
        }
        position += 1
        return key
    }

    mutating func parseScalar() throws(JSONParseError) -> JSONValue {
        switch bytes[position] {
        case UInt8(ascii: "\""):
            return .string(try parseString())
        case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"):
            return try parseNumber()
        case UInt8(ascii: "t"):
            try expectLiteral("true")
            return .bool(true)
        case UInt8(ascii: "f"):
            try expectLiteral("false")
            return .bool(false)
        case UInt8(ascii: "n"):
            try expectLiteral("null")
            return .null
        default:
            throw error(.unexpectedCharacter, at: position)
        }
    }

    mutating func expectLiteral(_ literal: StaticString) throws(JSONParseError) {
        let expected = UnsafeBufferPointer(
            start: literal.utf8Start, count: literal.utf8CodeUnitCount)
        for byte in expected {
            guard position < bytes.count else {
                throw error(.unexpectedEndOfInput, at: position)
            }
            guard bytes[position] == byte else {
                throw error(.unexpectedCharacter, at: position)
            }
            position += 1
        }
    }

    // MARK: Numbers

    mutating func parseNumber() throws(JSONParseError) -> JSONValue {
        let start = position
        var isFloat = false
        if bytes[position] == UInt8(ascii: "-") {
            position += 1
        }
        guard position < bytes.count else {
            throw error(.invalidNumber, at: position)
        }
        if bytes[position] == UInt8(ascii: "0") {
            position += 1
            if position < bytes.count, isDigit(bytes[position]) {
                throw error(.invalidNumber, at: position)
            }
        } else if isDigit(bytes[position]) {
            skipDigits()
        } else {
            throw error(.invalidNumber, at: position)
        }
        if position < bytes.count, bytes[position] == UInt8(ascii: ".") {
            isFloat = true
            position += 1
            guard position < bytes.count, isDigit(bytes[position]) else {
                throw error(.invalidNumber, at: position)
            }
            skipDigits()
        }
        if position < bytes.count, bytes[position] | 0x20 == UInt8(ascii: "e") {
            isFloat = true
            position += 1
            if position < bytes.count,
                bytes[position] == UInt8(ascii: "+") || bytes[position] == UInt8(ascii: "-")
            {
                position += 1
            }
            guard position < bytes.count, isDigit(bytes[position]) else {
                throw error(.invalidNumber, at: position)
            }
            skipDigits()
        }

        // The lexeme is pure ASCII, checked above.
        let lexeme = String(
            decoding: UnsafeBufferPointer(rebasing: bytes[start..<position]), as: UTF8.self)
        if isFloat {
            guard let value = Double(lexeme), value.isFinite else {
                throw error(.numberOutOfRange, at: start)
            }
            return .float(value)
        }
        return .integer(lexeme == "-0" ? "0" : lexeme)
    }

    mutating func skipDigits() {
        while position < bytes.count, isDigit(bytes[position]) {
            position += 1
        }
    }

    func isDigit(_ byte: UInt8) -> Bool {
        byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9")
    }

    // MARK: Strings

    /// Reads a string. The position is at the opening quote.
    mutating func parseString() throws(JSONParseError) -> String {
        position += 1
        stringBuffer.removeAll(keepingCapacity: true)
        while true {
            guard position < bytes.count else {
                throw error(.unexpectedEndOfInput, at: position)
            }
            let byte = bytes[position]
            switch byte {
            case UInt8(ascii: "\""):
                position += 1
                return String(decoding: stringBuffer, as: UTF8.self)
            case UInt8(ascii: "\\"):
                try parseEscape()
            case 0..<0x20:
                throw error(.unescapedControlCharacter, at: position)
            case 0x20..<0x80:
                stringBuffer.append(byte)
                position += 1
            default:
                try copyUTF8Sequence()
            }
        }
    }

    /// Copies one multibyte UTF-8 sequence after checking that it is well formed: no overlong
    /// forms, no encoded surrogates, nothing above U+10FFFF.
    mutating func copyUTF8Sequence() throws(JSONParseError) {
        let start = position
        let lead = bytes[position]
        let length: Int
        var low: UInt8 = 0x80
        var high: UInt8 = 0xBF
        switch lead {
        case 0xC2...0xDF: length = 2
        case 0xE0:
            length = 3
            low = 0xA0
        case 0xE1...0xEC, 0xEE...0xEF: length = 3
        case 0xED:
            length = 3
            high = 0x9F
        case 0xF0:
            length = 4
            low = 0x90
        case 0xF1...0xF3: length = 4
        case 0xF4:
            length = 4
            high = 0x8F
        default:
            throw error(.invalidUTF8, at: start)
        }
        for offset in 1..<length {
            let index = start + offset
            guard index < bytes.count else {
                throw error(.invalidUTF8, at: start)
            }
            let byte = bytes[index]
            let valid = offset == 1 ? (byte >= low && byte <= high) : (byte & 0xC0 == 0x80)
            guard valid else {
                throw error(.invalidUTF8, at: start)
            }
        }
        stringBuffer.append(
            contentsOf: UnsafeBufferPointer(rebasing: bytes[start..<start + length]))
        position = start + length
    }

    /// Decodes one escape sequence. The position is at the backslash.
    mutating func parseEscape() throws(JSONParseError) {
        let start = position
        position += 1
        guard position < bytes.count else {
            throw error(.unexpectedEndOfInput, at: position)
        }
        let byte = bytes[position]
        position += 1
        switch byte {
        case UInt8(ascii: "\""): stringBuffer.append(0x22)
        case UInt8(ascii: "\\"): stringBuffer.append(0x5C)
        case UInt8(ascii: "/"): stringBuffer.append(0x2F)
        case UInt8(ascii: "b"): stringBuffer.append(0x08)
        case UInt8(ascii: "f"): stringBuffer.append(0x0C)
        case UInt8(ascii: "n"): stringBuffer.append(0x0A)
        case UInt8(ascii: "r"): stringBuffer.append(0x0D)
        case UInt8(ascii: "t"): stringBuffer.append(0x09)
        case UInt8(ascii: "u"):
            var scalar = try parseHex4(escapeStart: start)
            if (0xDC00...0xDFFF).contains(scalar) {
                throw error(.loneSurrogate, at: start)
            }
            if (0xD800...0xDBFF).contains(scalar) {
                // A high surrogate must be followed by an escaped low surrogate.
                guard position + 1 < bytes.count, bytes[position] == UInt8(ascii: "\\"),
                    bytes[position + 1] == UInt8(ascii: "u")
                else {
                    throw error(.loneSurrogate, at: start)
                }
                let lowStart = position
                position += 2
                let low = try parseHex4(escapeStart: lowStart)
                guard (0xDC00...0xDFFF).contains(low) else {
                    throw error(.loneSurrogate, at: start)
                }
                scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
            }
            // Every value left is a valid scalar: surrogates were rejected or combined above.
            let unicodeScalar = Unicode.Scalar(scalar)!
            stringBuffer.append(contentsOf: UTF8.encode(unicodeScalar)!)
        default:
            throw error(.invalidEscape, at: start)
        }
    }

    /// Reads the four hexadecimal digits after `\u`.
    mutating func parseHex4(escapeStart: Int) throws(JSONParseError) -> UInt32 {
        var value: UInt32 = 0
        for _ in 0..<4 {
            guard position < bytes.count else {
                throw error(.unexpectedEndOfInput, at: position)
            }
            let byte = bytes[position]
            let digit: UInt8
            switch byte {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): digit = byte - UInt8(ascii: "0")
            case UInt8(ascii: "a")...UInt8(ascii: "f"): digit = byte - UInt8(ascii: "a") + 10
            case UInt8(ascii: "A")...UInt8(ascii: "F"): digit = byte - UInt8(ascii: "A") + 10
            default: throw error(.invalidUnicodeEscape, at: escapeStart)
            }
            value = value << 4 | UInt32(digit)
            position += 1
        }
        return value
    }

    // MARK: Whitespace and errors

    mutating func skipWhitespace() {
        while position < bytes.count {
            switch bytes[position] {
            case 0x20, 0x09, 0x0A, 0x0D: position += 1
            default: return
            }
        }
    }

    /// Builds an error at a byte offset, working out its line and column.
    func error(_ kind: JSONParseError.Kind, at offset: Int) -> JSONParseError {
        var line = 1
        var lineStart = 0
        for index in 0..<min(offset, bytes.count) where bytes[index] == 0x0A {
            line += 1
            lineStart = index + 1
        }
        return JSONParseError(
            kind: kind, offset: offset, line: line, column: offset - lineStart + 1)
    }
}
