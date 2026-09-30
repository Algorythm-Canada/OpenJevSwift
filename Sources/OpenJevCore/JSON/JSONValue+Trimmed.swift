// Ported from upstream OpenJev (razorback16/openjev at dcd2094), `openjev/api.py`, function
// `trim` and the constants TRIM_DEPTH, TRIM_ITEMS and TRIM_CHARS. Apache-2.0. See THIRD_PARTY.md.

extension JSONValue {
    /// A copy that is safe to echo back in an error: deep or long parts become placeholders.
    ///
    /// This is upstream's `trim`, which shortens the `input` it echoes in validation errors:
    ///
    /// - A string longer than `characters` Unicode scalars keeps its first `characters` scalars
    ///   followed by `...`.
    /// - An array or object nested `depth` or more levels below this value becomes the string
    ///   `...`. This value itself is at level 0.
    /// - An array keeps its first `items` elements and gains a final `...` element when some
    ///   were dropped.
    /// - An object keeps its first `items` entries and gains the entry `"...": "N more"` when N
    ///   entries were dropped. As in Python, a kept key that is already `...` takes that value in
    ///   place.
    /// - Numbers, Booleans and null are unchanged.
    ///
    /// Lengths count Unicode scalars because Python's `len` and slicing count code points.
    public func trimmed(depth: Int = 4, items: Int = 20, characters: Int = 500) -> JSONValue {
        trimmed(level: 0, depth: depth, items: items, characters: characters)
    }

    private func trimmed(level: Int, depth: Int, items: Int, characters: Int) -> JSONValue {
        switch self {
        case .string(let text):
            let scalars = text.unicodeScalars
            guard scalars.count > characters else { return self }
            return .string(String(String.UnicodeScalarView(scalars.prefix(characters))) + "...")
        case .array(let elements):
            if level >= depth {
                return .string("...")
            }
            var kept = elements.prefix(items).map {
                $0.trimmed(level: level + 1, depth: depth, items: items, characters: characters)
            }
            if elements.count > items {
                kept.append(.string("..."))
            }
            return .array(kept)
        case .object(let object):
            if level >= depth {
                return .string("...")
            }
            var kept = JSONObject()
            for (key, value) in object.prefix(items) {
                let child = value.trimmed(
                    level: level + 1, depth: depth, items: items, characters: characters)
                kept.updateValue(child, forKey: key)
            }
            if object.count > items {
                kept.updateValue(.string("\(object.count - items) more"), forKey: "...")
            }
            return .object(kept)
        case .null, .bool, .integer, .float:
            return self
        }
    }
}
