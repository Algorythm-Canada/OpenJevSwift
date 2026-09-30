/// A JSON value that keeps object order and Python's distinction between integers and floats.
///
/// Parse one with ``JSONParser`` and write one with ``PythonJSONWriter``. Objects are
/// ``JSONObject`` values, which keep insertion order.
///
/// Numbers follow Python's `json` module. A number written without a fraction or exponent is an
/// ``integer(_:)`` of arbitrary length, stored as its decimal digits so that it prints verbatim. A
/// number with a fraction or exponent is a ``float(_:)``, so `1e2` and `100.0` both become
/// `100.0`, as in Python.
///
/// Equality is structural. Strings and object keys are equal only when their Unicode scalars are
/// identical, which is Python's rule; Swift's canonical equivalence does not apply. An integer
/// never equals a float, even when Python would say `1 == 1.0`. Floats compare as `Double` does.
public enum JSONValue: Sendable {
    /// The JSON `null` literal.
    case null
    /// The JSON `true` or `false` literal.
    case bool(Bool)
    /// An integer, stored as normalized decimal digits.
    ///
    /// The text has an optional leading `-`, no leading `+`, no leading zeros, and zero is always
    /// `0`, never `-0`. ``JSONParser`` produces this form. ``PythonJSONWriter`` writes the text
    /// verbatim and throws ``JSONWriteError/invalidInteger(_:)`` for any other text.
    case integer(String)
    /// A floating-point number, parsed from a lexeme that had a fraction or an exponent.
    case float(Double)
    /// A string.
    case string(String)
    /// An array.
    case array([JSONValue])
    /// An object whose keys keep their insertion order.
    case object(JSONObject)
}

extension JSONValue {
    /// Creates an integer value from any binary integer.
    public init(_ value: some BinaryInteger) {
        self = .integer(String(value))
    }

    /// True when the value is `null`.
    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }

    /// The Boolean, when the value is `true` or `false`.
    public var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    /// The decimal digits, when the value is an integer.
    public var integerText: String? {
        if case .integer(let digits) = self { return digits }
        return nil
    }

    /// The integer as an `Int`, when the value is an integer that fits.
    public var intValue: Int? {
        if case .integer(let digits) = self { return Int(digits) }
        return nil
    }

    /// The number as a `Double`, when the value is a float or an integer.
    ///
    /// An integer converts with correct rounding, as Python's `float(int)` does, and may lose
    /// precision beyond 2^53.
    public var doubleValue: Double? {
        switch self {
        case .float(let value): return value
        case .integer(let digits): return Double(digits)
        default: return nil
        }
    }

    /// The string, when the value is a string.
    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    /// The elements, when the value is an array.
    public var arrayValue: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }

    /// The object, when the value is an object.
    public var objectValue: JSONObject? {
        if case .object(let value) = self { return value }
        return nil
    }

    /// The element at a position, when the value is an array and the position is in range.
    public subscript(index: Int) -> JSONValue? {
        guard case .array(let elements) = self, elements.indices.contains(index) else {
            return nil
        }
        return elements[index]
    }

    /// The value for a key, when the value is an object that has the key.
    public subscript(key: String) -> JSONValue? {
        guard case .object(let object) = self else { return nil }
        return object[key]
    }
}

extension JSONValue: Hashable {
    /// Returns true when both values have the same structure and equal contents.
    public static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null): return true
        case (.bool(let a), .bool(let b)): return a == b
        case (.integer(let a), .integer(let b)): return a == b
        case (.float(let a), .float(let b)): return a == b
        case (.string(let a), .string(let b)): return a.utf8.elementsEqual(b.utf8)
        case (.array(let a), .array(let b)): return a == b
        case (.object(let a), .object(let b)): return a == b
        default: return false
        }
    }

    /// Hashes the value consistently with `==`.
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .null:
            hasher.combine(0)
        case .bool(let value):
            hasher.combine(1)
            hasher.combine(value)
        case .integer(let digits):
            hasher.combine(2)
            hasher.combine(digits)
        case .float(let value):
            hasher.combine(3)
            hasher.combine(value)
        case .string(let value):
            hasher.combine(4)
            hashScalars(of: value, into: &hasher)
        case .array(let elements):
            hasher.combine(5)
            hasher.combine(elements)
        case .object(let object):
            hasher.combine(6)
            hasher.combine(object)
        }
    }
}

extension JSONValue: ExpressibleByNilLiteral {
    /// Creates `null`.
    public init(nilLiteral: ()) {
        self = .null
    }
}

extension JSONValue: ExpressibleByBooleanLiteral {
    /// Creates `true` or `false`.
    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }
}

extension JSONValue: ExpressibleByIntegerLiteral {
    /// Creates an integer.
    public init(integerLiteral value: Int) {
        self = .integer(String(value))
    }
}

extension JSONValue: ExpressibleByFloatLiteral {
    /// Creates a float.
    public init(floatLiteral value: Double) {
        self = .float(value)
    }
}

extension JSONValue: ExpressibleByStringLiteral {
    /// Creates a string.
    public init(stringLiteral value: String) {
        self = .string(value)
    }
}

extension JSONValue: ExpressibleByArrayLiteral {
    /// Creates an array.
    public init(arrayLiteral elements: JSONValue...) {
        self = .array(elements)
    }
}

extension JSONValue: ExpressibleByDictionaryLiteral {
    /// Creates an object, keeping the order written.
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        var object = JSONObject()
        for (key, value) in elements {
            object.updateValue(value, forKey: key)
        }
        self = .object(object)
    }
}

extension JSONValue: CustomStringConvertible {
    /// The value written as compact JSON without ASCII escaping, or a note when it cannot be
    /// written (a non-finite float or an invalid integer).
    public var description: String {
        do {
            let options = PythonJSONWriter.Options(
                ensureASCII: false, sortKeys: false, itemSeparator: ",", keySeparator: ":")
            return try PythonJSONWriter(options: options).string(self)
        } catch {
            return "<unwritable JSON: \(error)>"
        }
    }
}
