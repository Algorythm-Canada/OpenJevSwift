// The prefill cache of upstream OpenJev's MlxRuntime (razorback16/openjev at dcd2094,
// openjev/mlx_backend.py lines 27 to 50, 116 to 128 and 150 to 174, Apache-2.0, see
// THIRD_PARTY.md): an ordered map bounded by entries and tokens, with no exempt entry.

import Foundation

/// The key a prefill is cached under.
///
/// A text prompt is keyed by its token ids, as upstream keys it by `tuple(prompt)`; an image
/// prompt by upstream's `ImagePrompt.key`, the system text, the state text and the SHA-256 of
/// each image's data URL, so it never shares an entry with the same text with another image or
/// with none.
public enum PrefillKey: Hashable, Sendable {
    /// A prompt of token ids.
    case tokens([Int])
    /// An image prompt: its system and state texts and each image's data URL digest, in order.
    case image(systemText: String, stateText: String, digests: [Data])
}

/// The prefill cache's defaults, upstream's module constants.
public enum PrefillCacheDefaults {
    /// Upstream's `PROMPT_CACHE_TOKENS`: the most prompt tokens the cached prefills may hold.
    public static let tokens = 16_384
    /// Upstream's `DEFAULT_PROMPT_CACHE_ENTRIES`: the most prefills kept when nothing configures
    /// the runtime.
    public static let entries = 12
}

/// Upstream's prefill cache, generic over the cached value so the eviction rule is tested
/// without MLX. The runtime instantiates it with ``PromptCache``.
///
/// Two budgets bound it, because the two ways to fill it cost memory differently: the token
/// budget guards a few long prompts, and the entry budget the many short ones whose cost is the
/// cache itself rather than their length (upstream's comment at `mlx_backend.py` lines 27 to
/// 43). An insertion is kept first and then the oldest entries are evicted while either budget is
/// exceeded, so no entry is exempt: one prompt above the token budget is inserted and evicted at
/// once, and the caller keeps the value it was handed. A hit moves to the end. Zero entries turn
/// the cache off.
///
/// Not thread-safe; the runtime actor owns it.
public struct PrefillCache<Value> {
    private struct Entry {
        var value: Value
        var tokens: Int
    }

    /// The most entries kept, upstream's `prompt_cache_entries`. 0 keeps none.
    public var entryBudget: Int
    /// The most prompt tokens kept, upstream's `PROMPT_CACHE_TOKENS`.
    public var tokenBudget: Int
    /// The running total of the cached prompts' tokens, upstream's `prefill_tokens`, maintained
    /// on insert and eviction rather than re-summed.
    public private(set) var tokenCount = 0
    /// Lookups that found their key, for diagnostics.
    public private(set) var hits = 0
    /// Lookups that did not.
    public private(set) var misses = 0

    /// The keys, oldest first.
    public private(set) var keys: [PrefillKey] = []
    private var entries: [PrefillKey: Entry] = [:]

    /// Creates an empty cache with upstream's defaults unless told otherwise.
    public init(
        entryBudget: Int = PrefillCacheDefaults.entries,
        tokenBudget: Int = PrefillCacheDefaults.tokens
    ) {
        self.entryBudget = entryBudget
        self.tokenBudget = tokenBudget
    }

    /// The number of cached prefills.
    public var count: Int { keys.count }

    /// The cached value for `key` without counting a lookup or moving it.
    public func peek(_ key: PrefillKey) -> Value? {
        entries[key]?.value
    }

    /// The token count recorded for each cached key, oldest first.
    public var tokenCounts: [Int] {
        keys.map { entries[$0]?.tokens ?? 0 }
    }

    /// Upstream's `_prefill`: the cached value for `key`, moved to the end, or the value
    /// `make` returns, inserted and then subject to eviction.
    ///
    /// - Parameters:
    ///   - key: the prompt's key.
    ///   - tokens: the prompt's token count, its cost against the token budget.
    ///   - make: the prefill, run only on a miss. What it throws is rethrown and nothing is
    ///     inserted.
    /// - Returns: the value and whether it was a hit.
    public mutating func value(
        for key: PrefillKey, tokens: Int, make: () throws -> Value
    ) rethrows -> (value: Value, hit: Bool) {
        if let entry = entries[key] {
            hits += 1
            if let index = keys.firstIndex(of: key) {
                keys.remove(at: index)
            }
            keys.append(key)
            return (entry.value, true)
        }
        misses += 1
        let value = try make()
        insert(value, for: key, tokens: tokens)
        return (value, false)
    }

    /// Removes every entry; the hit and miss counts stay.
    public mutating func removeAll() {
        keys.removeAll()
        entries.removeAll()
        tokenCount = 0
    }

    private mutating func insert(_ value: Value, for key: PrefillKey, tokens: Int) {
        entries[key] = Entry(value: value, tokens: tokens)
        keys.append(key)
        tokenCount += tokens
        // No entry is exempt: evicting the entry just inserted is safe, the caller holds it.
        while !keys.isEmpty && (keys.count > entryBudget || tokenCount > tokenBudget) {
            let oldest = keys.removeFirst()
            if let evicted = entries.removeValue(forKey: oldest) {
                tokenCount -= evicted.tokens
            }
        }
    }
}
