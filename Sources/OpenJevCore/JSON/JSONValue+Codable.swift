import Foundation

/// Bridges ``JSONValue`` to `Codable` for places where object order does not matter.
///
/// Decoding goes through the decoder's keyed containers, and Foundation's `JSONDecoder` does not
/// report keys in document order, so objects decoded this way lose their order. Use
/// ``JSONParser`` for input whose order matters, such as a Jev `questions` object or a choice's
/// `criteria`.
///
/// Numbers take the first type the decoder accepts: `Int64`, then `UInt64`, then `Double`. A
/// number with a fraction may therefore decode as an integer when the decoder allows it, and an
/// integer too large for 64 bits decodes as a float. Encoding writes integers that fit 64 bits
/// exactly and larger ones as `Double`. ``PythonJSONWriter`` writes every integer exactly.
extension JSONValue: Codable {
    /// Decodes any JSON value. Object order is not preserved.
    public init(from decoder: any Decoder) throws {
        if let keyed = try? decoder.container(keyedBy: AnyCodingKey.self) {
            var object = JSONObject()
            for key in keyed.allKeys {
                object.updateValue(
                    try keyed.decode(JSONValue.self, forKey: key), forKey: key.stringValue)
            }
            self = .object(object)
            return
        }
        if var unkeyed = try? decoder.unkeyedContainer() {
            var elements: [JSONValue] = []
            while !unkeyed.isAtEnd {
                elements.append(try unkeyed.decode(JSONValue.self))
            }
            self = .array(elements)
            return
        }
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let flag = try? container.decode(Bool.self) {
            self = .bool(flag)
        } else if let integer = try? container.decode(Int64.self) {
            self = .integer(String(integer))
        } else if let integer = try? container.decode(UInt64.self) {
            self = .integer(String(integer))
        } else if let number = try? container.decode(Double.self) {
            self = .float(number)
        } else if let text = try? container.decode(String.self) {
            self = .string(text)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "The value is not a JSON value.")
        }
    }

    /// Encodes the value. Objects are encoded in order, which the encoder may not keep.
    public func encode(to encoder: any Encoder) throws {
        switch self {
        case .null:
            var container = encoder.singleValueContainer()
            try container.encodeNil()
        case .bool(let flag):
            var container = encoder.singleValueContainer()
            try container.encode(flag)
        case .integer(let digits):
            var container = encoder.singleValueContainer()
            if let integer = Int64(digits) {
                try container.encode(integer)
            } else if let integer = UInt64(digits) {
                try container.encode(integer)
            } else if let number = Double(digits) {
                try container.encode(number)
            } else {
                throw EncodingError.invalidValue(
                    self,
                    EncodingError.Context(
                        codingPath: encoder.codingPath,
                        debugDescription: "The integer text is not a decimal number."))
            }
        case .float(let number):
            var container = encoder.singleValueContainer()
            try container.encode(number)
        case .string(let text):
            var container = encoder.singleValueContainer()
            try container.encode(text)
        case .array(let elements):
            var container = encoder.unkeyedContainer()
            for element in elements {
                try container.encode(element)
            }
        case .object(let object):
            var container = encoder.container(keyedBy: AnyCodingKey.self)
            for (key, value) in object {
                try container.encode(value, forKey: AnyCodingKey(stringValue: key))
            }
        }
    }

    /// Decodes a `Decodable` type from this value.
    ///
    /// The value is written as compact JSON and handed to `decoder`, so the result is the same as
    /// decoding the original document with that decoder.
    public func decode<T: Decodable>(_ type: T.Type, using decoder: JSONDecoder = JSONDecoder())
        throws -> T
    {
        let writer = PythonJSONWriter(
            options: .init(
                ensureASCII: false, sortKeys: false, itemSeparator: ",", keySeparator: ":"))
        return try decoder.decode(type, from: Data(try writer.bytes(self)))
    }
}

/// A coding key for any string, used to walk objects of unknown shape.
private struct AnyCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}
