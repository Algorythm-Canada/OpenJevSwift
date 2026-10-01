/// A small cache that keeps the entries used most recently, for the package functions an
/// encoder keeps loaded (each loaded function holds its own copy of the weights once it has run).
struct LeastRecentlyUsed<Key: Hashable, Value> {
    /// The entries, least recently used first.
    private var entries: [(key: Key, value: Value)] = []

    /// The keys, least recently used first.
    var keys: [Key] { entries.map(\.key) }

    /// The value for a key, which becomes the most recently used; `nil` when absent.
    mutating func value(forKey key: Key) -> Value? {
        guard let index = entries.firstIndex(where: { $0.key == key }) else {
            return nil
        }
        let entry = entries.remove(at: index)
        entries.append(entry)
        return entry.value
    }

    /// Removes the least recently used entries until at most `count` remain, so that a new one
    /// can be loaded before the cache holds more than it may.
    mutating func trim(to count: Int) {
        entries.removeFirst(max(0, entries.count - count))
    }

    /// Adds an entry as the most recently used, replacing one with the same key.
    mutating func insert(_ value: Value, forKey key: Key) {
        entries.removeAll { $0.key == key }
        entries.append((key, value))
    }
}

extension LeastRecentlyUsed: Sendable where Key: Sendable, Value: Sendable {}
