/// A JSON object that keeps its keys in insertion order.
///
/// Jev gives meaning to object order: the order of `questions` numbers them q1 to qN, and the
/// order of a choice's `criteria` picks the label letters. Python's `dict` keeps insertion order,
/// so this type does too.
///
/// Keys are compared by their Unicode scalars, not by Swift `String` equality. Swift treats
/// canonically equivalent strings such as `"\u{E9}"` and `"e\u{301}"` as equal; Python does not,
/// and neither does this type, so those are two different keys.
///
/// Two objects are equal when they hold the same keys with equal values in the same order.
/// Lookup by key takes constant time on average. Removing a key takes time proportional to the
/// number of keys after it.
public struct JSONObject: Sendable, Hashable {
    /// A key and its value.
    public typealias Element = (key: String, value: JSONValue)

    private var orderedKeys: [String]
    private var orderedValues: [JSONValue]
    private var positions: [ScalarKey: Int]

    /// Creates an empty object.
    public init() {
        orderedKeys = []
        orderedValues = []
        positions = [:]
    }

    /// Creates an object from key and value pairs, in the order given.
    ///
    /// - Precondition: No two keys have the same Unicode scalars.
    public init<S: Sequence>(uniqueKeysWithValues entries: S)
    where S.Element == (String, JSONValue) {
        self.init()
        for (key, value) in entries {
            let previous = updateValue(value, forKey: key)
            precondition(previous == nil, "Duplicate key in JSONObject(uniqueKeysWithValues:)")
        }
    }

    /// The keys, in order.
    public var keys: [String] { orderedKeys }

    /// The values, in the order of their keys.
    public var values: [JSONValue] { orderedValues }

    /// The value stored for a key, or `nil` when the key is absent.
    ///
    /// Setting a value for a new key appends it. Setting a value for an existing key replaces the
    /// value and keeps the key's position. Setting `nil` removes the key.
    public subscript(key: String) -> JSONValue? {
        get {
            positions[ScalarKey(key)].map { orderedValues[$0] }
        }
        set {
            if let newValue {
                updateValue(newValue, forKey: key)
            } else {
                removeValue(forKey: key)
            }
        }
    }

    /// The position of a key in the object, or `nil` when the key is absent.
    public func index(forKey key: String) -> Int? {
        positions[ScalarKey(key)]
    }

    /// Stores a value for a key and returns the value it replaced.
    ///
    /// An existing key keeps its position and gets the new value, which is what Python's `dict`
    /// does. A new key goes at the end.
    @discardableResult
    public mutating func updateValue(_ value: JSONValue, forKey key: String) -> JSONValue? {
        let scalarKey = ScalarKey(key)
        if let position = positions[scalarKey] {
            let old = orderedValues[position]
            orderedValues[position] = value
            return old
        }
        positions[scalarKey] = orderedKeys.count
        orderedKeys.append(key)
        orderedValues.append(value)
        return nil
    }

    /// Removes a key and returns its value, or returns `nil` when the key is absent.
    ///
    /// The keys after it keep their relative order.
    @discardableResult
    public mutating func removeValue(forKey key: String) -> JSONValue? {
        guard let position = positions.removeValue(forKey: ScalarKey(key)) else {
            return nil
        }
        orderedKeys.remove(at: position)
        let old = orderedValues.remove(at: position)
        for index in position..<orderedKeys.count {
            positions[ScalarKey(orderedKeys[index])] = index
        }
        return old
    }

    /// Returns true when both objects hold the same keys with equal values in the same order.
    public static func == (lhs: JSONObject, rhs: JSONObject) -> Bool {
        guard lhs.orderedKeys.count == rhs.orderedKeys.count else { return false }
        for index in lhs.orderedKeys.indices {
            guard ScalarKey(lhs.orderedKeys[index]) == ScalarKey(rhs.orderedKeys[index]),
                lhs.orderedValues[index] == rhs.orderedValues[index]
            else {
                return false
            }
        }
        return true
    }

    /// Hashes the keys and values in order, consistent with `==`.
    public func hash(into hasher: inout Hasher) {
        hasher.combine(orderedKeys.count)
        for index in orderedKeys.indices {
            hasher.combine(ScalarKey(orderedKeys[index]))
            hasher.combine(orderedValues[index])
        }
    }
}

extension JSONObject: RandomAccessCollection {
    /// The position of the first entry.
    public var startIndex: Int { 0 }

    /// The position one past the last entry.
    public var endIndex: Int { orderedKeys.count }

    /// The key and value at a position.
    public subscript(position: Int) -> Element {
        (orderedKeys[position], orderedValues[position])
    }
}

extension JSONObject: ExpressibleByDictionaryLiteral {
    /// Creates an object from a dictionary literal, keeping the order written.
    ///
    /// A repeated key keeps its first position and takes its last value, as a Python dict display
    /// does.
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self.init()
        for (key, value) in elements {
            updateValue(value, forKey: key)
        }
    }
}

extension JSONObject: CustomStringConvertible {
    /// The object written as compact JSON without ASCII escaping.
    public var description: String {
        JSONValue.object(self).description
    }
}

/// A string key that is equal to another only when their Unicode scalars are identical.
///
/// UTF-8 bytes map one to one onto Unicode scalars, so comparing and hashing the UTF-8 view gives
/// scalar identity without Swift's canonical equivalence.
struct ScalarKey: Hashable {
    let string: String

    init(_ string: String) {
        self.string = string
    }

    static func == (lhs: ScalarKey, rhs: ScalarKey) -> Bool {
        lhs.string.utf8.elementsEqual(rhs.string.utf8)
    }

    func hash(into hasher: inout Hasher) {
        hashScalars(of: string, into: &hasher)
    }
}

/// Feeds the UTF-8 bytes of a string to a hasher, followed by their count.
func hashScalars(of string: String, into hasher: inout Hasher) {
    var string = string
    string.withUTF8 { bytes in
        hasher.combine(bytes: UnsafeRawBufferPointer(bytes))
        hasher.combine(bytes.count)
    }
}

/// Returns true when the first string's scalars sort before the second's.
///
/// UTF-8 byte order is the same as Unicode scalar order, which is how Python compares `str`.
func scalarsPrecede(_ lhs: String, _ rhs: String) -> Bool {
    lhs.utf8.lexicographicallyPrecedes(rhs.utf8)
}
