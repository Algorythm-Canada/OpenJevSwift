import Foundation
import Testing

@testable import OpenJevDiffusionGemma

/// Upstream's prefill cache and settings tests (razorback16/openjev at dcd2094,
/// tests/test_mlx_backend.py lines 584 to 727), over ``PrefillCache`` and a stub runtime. No
/// weights and no MLX.
@Suite("Prefill cache and runtime settings")
struct PrefillCacheTests {
    /// A cached value whose identity the tests compare, as upstream's `object()`.
    final class Prefill {}

    /// `rt._prefill(_FakePrompt(key, tokens), max_tokens=32768)`: the key is one id per
    /// character, so distinct names are distinct prompts.
    @discardableResult
    private func prefill(
        _ cache: inout PrefillCache<Prefill>, _ name: String, tokens: Int
    ) -> Prefill {
        cache.value(for: key(name), tokens: tokens) { Prefill() }.value
    }

    private func key(_ name: String) -> PrefillKey {
        .tokens(name.unicodeScalars.map { Int($0.value) })
    }

    @Test("test_many_short_prompts_cannot_pin_the_cache: 400 short prompts keep at most 12")
    func manyShortPromptsCannotPinTheCache() {
        var cache = PrefillCache<Prefill>(entryBudget: 12)
        for i in 0..<400 {
            prefill(&cache, "k\(i)", tokens: 40)
        }
        #expect(cache.count <= 12)
    }

    @Test("test_the_newest_prompts_are_the_ones_kept")
    func theNewestPromptsAreTheOnesKept() {
        var cache = PrefillCache<Prefill>(entryBudget: 3)
        for i in 0..<10 {
            prefill(&cache, "k\(i)", tokens: 40)
        }
        #expect(cache.keys == ["k7", "k8", "k9"].map(key))
    }

    @Test("test_a_repeated_prompt_is_a_hit_and_is_not_re_evicted")
    func aRepeatedPromptIsAHitAndIsNotReEvicted() throws {
        var cache = PrefillCache<Prefill>(entryBudget: 3)
        for i in 0..<3 {
            prefill(&cache, "k\(i)", tokens: 40)
        }
        let first = try #require(cache.peek(key("k0")))
        let again = cache.value(for: key("k0"), tokens: 40) { Prefill() }
        #expect(again.hit)
        #expect(again.value === first)
        prefill(&cache, "k9", tokens: 40)  // evicts the least recently used, now k1
        #expect(cache.peek(key("k0")) != nil)
        #expect(cache.peek(key("k1")) == nil)
    }

    @Test(
        "test_long_prompts_are_still_bounded_by_tokens: 16,384 tokens hold two 8,192-token entries")
    func longPromptsAreStillBoundedByTokens() {
        var cache = PrefillCache<Prefill>(entryBudget: 12)
        for i in 0..<6 {
            prefill(&cache, "big\(i)", tokens: 8192)
        }
        #expect(cache.tokenCount <= PrefillCacheDefaults.tokens)
        #expect(cache.count == 2)
    }

    @Test("test_no_entry_is_exempt_from_eviction: one 32,768-token prompt is inserted and evicted")
    func noEntryIsExemptFromEviction() {
        var cache = PrefillCache<Prefill>(entryBudget: 12)
        let handed = prefill(&cache, "huge", tokens: 32_768)
        #expect(cache.tokenCount <= PrefillCacheDefaults.tokens)
        #expect(cache.count == 0)
        #expect(cache.keys.isEmpty)
        // The caller still has the prefill it asked for.
        _ = handed
    }

    @Test("test_zero_entries_turns_the_cache_off")
    func zeroEntriesTurnsTheCacheOff() {
        var cache = PrefillCache<Prefill>(entryBudget: 0)
        let result = cache.value(for: key("k0"), tokens: 40) { Prefill() }
        #expect(cache.count == 0 && cache.tokenCount == 0)
        #expect(!result.hit, "the caller still gets the prefill it asked for")
    }

    @Test("test_the_running_token_total_tracks_the_cache")
    func theRunningTokenTotalTracksTheCache() {
        var cache = PrefillCache<Prefill>(entryBudget: 3)
        for i in 0..<10 {
            prefill(&cache, "k\(i)", tokens: 40)
        }
        #expect(cache.tokenCount == cache.tokenCounts.reduce(0, +))
        #expect(cache.tokenCount == 3 * 40)
    }

    @Test("test_the_default_entry_count_comes_from_the_module")
    func theDefaultEntryCountComesFromTheModule() async {
        #expect(PrefillCache<Prefill>().entryBudget == PrefillCacheDefaults.entries)
        #expect(PrefillCache<Prefill>().tokenBudget == PrefillCacheDefaults.tokens)
        #expect(DiffusionGemmaRuntime.Configuration().promptCacheEntries == 12)
        #expect(DiffusionGemmaRuntime.Configuration().promptCacheTokens == 16_384)
        let state = await DiffusionGemmaRuntime.stub().prefillCacheState
        #expect(state.entryBudget == PrefillCacheDefaults.entries)
        #expect(state.tokenBudget == PrefillCacheDefaults.tokens)
    }

    @Test("test_mlx_engine_applies_both_settings_to_the_runtime")
    func bothSettingsReachTheRuntime() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(
            configuration: .init(promptCacheEntries: 5, cacheLimitGB: 2.5), log: log)
        #expect(runtime.configuration.promptCacheEntries == 5)
        #expect(runtime.configuration.cacheLimitGB == 2.5)
        #expect(await runtime.prefillCacheState.entryBudget == 5)
        try await runtime.applyConfiguredCacheLimit()
        #expect(log.limits == [Int(2.5 * 1024 * 1024 * 1024)])
    }

    @Test("test_an_unset_cache_limit_still_reaches_the_runtime_as_none: MLX is left alone")
    func anUnsetCacheLimitReachesTheRuntimeAsNil() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(
            configuration: .init(cacheLimitGB: nil), log: log)
        #expect(runtime.configuration.cacheLimitGB == nil)
        #expect(try runtime.configuration.cacheLimitBytes() == nil)
        try await runtime.applyConfiguredCacheLimit()
        #expect(log.limits.isEmpty)
    }

    @Test("A cache limit of nan, inf or below 0 is refused before MLX is touched")
    func invalidCacheLimits() async {
        for gb in [Double.nan, .infinity, -1, 1e300] {
            let log = StubModelLog()
            let runtime = DiffusionGemmaRuntime.stub(
                configuration: .init(cacheLimitGB: gb), log: log)
            #expect(throws: DiffusionGemmaRuntimeError.self) {
                try runtime.configuration.cacheLimitBytes()
            }
            await #expect(throws: DiffusionGemmaRuntimeError.self) {
                try await runtime.applyConfiguredCacheLimit()
            }
            #expect(log.limits.isEmpty, "\(gb)")
        }
        #expect(
            DiffusionGemmaRuntimeError.invalidCacheLimit(.nan).description.contains(
                "OPENJEV_MLX_CACHE_LIMIT_GB"))
    }

    @Test("test_a_zero_cache_limit_is_not_the_same_as_unset: 0 disables MLX's pool")
    func aZeroCacheLimitIsNotUnset() async throws {
        let log = StubModelLog()
        let runtime = DiffusionGemmaRuntime.stub(configuration: .init(cacheLimitGB: 0), log: log)
        #expect(try runtime.configuration.cacheLimitBytes() == 0)
        try await runtime.applyConfiguredCacheLimit()
        #expect(log.limits == [0])
    }
}
