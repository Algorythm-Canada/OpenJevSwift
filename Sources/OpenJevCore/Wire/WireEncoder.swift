/// A wire type that renders as a ``JSONValue``.
///
/// Wire types do not use `Codable`: Foundation's `JSONEncoder` loses object order and writes the
/// float `1.0` as `1`, and both matter on the wire. ``WireEncoder`` writes the value instead.
public protocol WireEncodable {
    /// The value as it goes on the wire, with its keys in wire order.
    var json: JSONValue { get }
}

/// Renders wire types to the bytes upstream's FastAPI server sends.
///
/// FastAPI's `JSONResponse` renders a body with
/// `json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":"))`.
/// This encoder applies the same settings through ``PythonJSONWriter``: compact separators,
/// non-ASCII text written as UTF-8, and an error for an infinite or NaN float.
public struct WireEncoder: Sendable {
    /// FastAPI's `JSONResponse` settings.
    public static let options = PythonJSONWriter.Options(
        ensureASCII: false, sortKeys: false, itemSeparator: ",", keySeparator: ":")

    /// Creates an encoder.
    public init() {}

    /// The UTF-8 bytes of a wire value.
    ///
    /// - Throws: ``JSONWriteError/nonFiniteNumber(_:)`` when a probability, score or confidence
    ///   is infinite or NaN, which FastAPI refuses as well.
    public func bytes(_ value: some WireEncodable) throws(JSONWriteError) -> [UInt8] {
        try bytes(json: value.json)
    }

    /// The UTF-8 bytes of a JSON value, rendered as FastAPI renders a response body.
    public func bytes(json: JSONValue) throws(JSONWriteError) -> [UInt8] {
        try PythonJSONWriter(options: Self.options).bytes(json)
    }

    /// The text of a wire value.
    public func string(_ value: some WireEncodable) throws(JSONWriteError) -> String {
        try PythonJSONWriter(options: Self.options).string(value.json)
    }
}

/// A response-side wire value that does not have the expected shape.
///
/// Request bodies are checked by ``RequestValidator``, which reports errors the way upstream
/// does. This error is for the other direction: decoding a response, a listing or an answer that
/// this project or upstream wrote, as the fixture tests and clients do.
public struct WireDecodingError: Error, Sendable, Hashable, CustomStringConvertible {
    /// Where in the value the problem is, from its root.
    public var path: [LocComponent]
    /// What is wrong.
    public var reason: String

    /// Creates an error.
    public init(path: [LocComponent], reason: String) {
        self.path = path
        self.reason = reason
    }

    /// The path and the reason, for logs and test failures.
    public var description: String {
        let location = path.map(\.description).joined(separator: ".")
        return location.isEmpty ? reason : "\(location): \(reason)"
    }
}

/// Reading helpers for the response-side decoders.
extension JSONValue {
    /// The object, or an error naming the path.
    func requireObject(at path: [LocComponent]) throws(WireDecodingError) -> JSONObject {
        guard let object = objectValue else {
            throw WireDecodingError(path: path, reason: "expected an object")
        }
        return object
    }

    /// The array, or an error naming the path.
    func requireArray(at path: [LocComponent]) throws(WireDecodingError) -> [JSONValue] {
        guard let array = arrayValue else {
            throw WireDecodingError(path: path, reason: "expected an array")
        }
        return array
    }

    /// The string, or an error naming the path.
    func requireString(at path: [LocComponent]) throws(WireDecodingError) -> String {
        guard let string = stringValue else {
            throw WireDecodingError(path: path, reason: "expected a string")
        }
        return string
    }

    /// The number as a `Double`, or an error naming the path.
    func requireDouble(at path: [LocComponent]) throws(WireDecodingError) -> Double {
        guard let number = doubleValue else {
            throw WireDecodingError(path: path, reason: "expected a number")
        }
        return number
    }

    /// The integer as an `Int`, or an error naming the path.
    func requireInt(at path: [LocComponent]) throws(WireDecodingError) -> Int {
        guard let number = intValue else {
            throw WireDecodingError(path: path, reason: "expected an integer")
        }
        return number
    }
}

extension JSONObject {
    /// The value for a key, or an error naming the path.
    func require(_ key: String, at path: [LocComponent]) throws(WireDecodingError) -> JSONValue {
        guard let value = self[key] else {
            throw WireDecodingError(path: path + [.key(key)], reason: "missing")
        }
        return value
    }

    /// Throws unless the object has exactly these keys, in any order.
    func requireKeys(_ expected: Set<String>, at path: [LocComponent]) throws(WireDecodingError) {
        let actual = Set(keys)
        guard actual == expected else {
            let extra = actual.subtracting(expected).sorted()
            let missing = expected.subtracting(actual).sorted()
            throw WireDecodingError(
                path: path, reason: "unexpected keys \(extra), missing keys \(missing)")
        }
    }
}
