/// String keys mapped to typed values, in insertion order.
///
/// The wire types use this where ``JSONObject`` would hold untyped values: a request's
/// questions, a response's answers and a choice answer's probabilities. Order is part of the
/// contract in all three. Keys compare by Unicode scalars, as in ``JSONObject`` and Python, so
/// `"\u{E9}"` and `"e\u{301}"` are two different question ids.
public struct OrderedMap<Value: Sendable & Hashable>: Sendable, Hashable {
    /// A key and its value.
    public typealias Element = (key: String, value: Value)

    private var orderedKeys: [String]
    private var orderedValues: [Value]
    private var positions: [ScalarKey: Int]

    /// Creates an empty map.
    public init() {
        orderedKeys = []
        orderedValues = []
        positions = [:]
    }

    /// Creates a map from key and value pairs, in the order given.
    ///
    /// - Precondition: No two keys have the same Unicode scalars.
    public init<S: Sequence>(uniqueKeysWithValues entries: S) where S.Element == (String, Value) {
        self.init()
        for (key, value) in entries {
            let previous = updateValue(value, forKey: key)
            precondition(previous == nil, "Duplicate key in OrderedMap(uniqueKeysWithValues:)")
        }
    }

    /// The keys, in order.
    public var keys: [String] { orderedKeys }

    /// The values, in the order of their keys.
    public var values: [Value] { orderedValues }

    /// The value stored for a key, or `nil` when the key is absent.
    public subscript(key: String) -> Value? {
        positions[ScalarKey(key)].map { orderedValues[$0] }
    }

    /// The position of a key, or `nil` when the key is absent.
    public func index(forKey key: String) -> Int? {
        positions[ScalarKey(key)]
    }

    /// Stores a value for a key and returns the value it replaced.
    ///
    /// An existing key keeps its position and gets the new value. A new key goes at the end.
    @discardableResult
    public mutating func updateValue(_ value: Value, forKey key: String) -> Value? {
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

    /// Returns true when both maps hold the same keys with equal values in the same order.
    public static func == (lhs: OrderedMap, rhs: OrderedMap) -> Bool {
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

extension OrderedMap: RandomAccessCollection {
    /// The position of the first entry.
    public var startIndex: Int { 0 }

    /// The position one past the last entry.
    public var endIndex: Int { orderedKeys.count }

    /// The key and value at a position.
    public subscript(position: Int) -> Element {
        (orderedKeys[position], orderedValues[position])
    }
}

extension OrderedMap: ExpressibleByDictionaryLiteral {
    /// Creates a map from a dictionary literal, keeping the order written.
    ///
    /// A repeated key keeps its first position and takes its last value, as a Python dict
    /// display does.
    public init(dictionaryLiteral elements: (String, Value)...) {
        self.init()
        for (key, value) in elements {
            updateValue(value, forKey: key)
        }
    }
}
