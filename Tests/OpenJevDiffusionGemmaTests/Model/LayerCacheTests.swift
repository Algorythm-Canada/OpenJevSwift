import MLX
import Testing

@testable import OpenJevDiffusionGemma

/// `[1, 1, values.count, 1]` float32 keys or values holding `values`, one per position.
private func positions(_ values: [Float]) -> MLXArray {
    MLXArray(values, [1, 1, values.count, 1])
}

/// The values a cache holds, position by position.
private func held(_ array: MLXArray?) -> [Float] {
    array?.asArray(Float.self) ?? []
}

private func range(_ start: Int, _ count: Int) -> [Float] {
    (start..<(start + count)).map(Float.init)
}

extension MLXTests {
    /// ``LayerCache`` across updates, on synthetic tensors whose every position is distinct, so
    /// what each cache keeps is visible: the sliding window's trim, the full layer's buffer, and
    /// copies (``LayerCache/extended()``) that never change each other.
    @Suite("Layer caches across generated blocks")
    struct LayerCacheTests {
        init() {
            MetalLibrary.configure()
        }

        @Test(
            "A sliding cache keeps the last window − 1 positions and appends, update after update")
        func slidingTrim() {
            let prefill = LayerCache(isFullAttention: false, slidingWindow: 4)
            _ = prefill.update(keys: positions(range(0, 6)), values: positions(range(50, 6)))
            // A prefill keeps every position, as RotatingKVCache does on an empty cache.
            #expect(held(prefill.keys) == range(0, 6) && prefill.offset == 6)

            let cache = prefill.extended()
            _ = cache.update(keys: positions([100, 101]), values: positions([150, 151]))
            #expect(held(cache.keys) == [3, 4, 5, 100, 101])
            #expect(held(cache.values) == [53, 54, 55, 150, 151])
            #expect(cache.offset == 8)
            _ = cache.update(keys: positions([200, 201]), values: positions([250, 251]))
            #expect(held(cache.keys) == [5, 100, 101, 200, 201])
            #expect(held(cache.values) == [55, 150, 151, 250, 251])
            #expect(cache.offset == 10)

            // A cache shorter than the window keeps everything.
            let short = LayerCache(isFullAttention: false, slidingWindow: 8)
            _ = short.update(keys: positions(range(0, 3)), values: positions(range(0, 3)))
            _ = short.update(keys: positions([9, 9]), values: positions([9, 9]))
            #expect(held(short.keys) == range(0, 3) + [9, 9])

            // The prefill is as it was.
            #expect(held(prefill.keys) == range(0, 6) && held(prefill.values) == range(50, 6))
            #expect(prefill.offset == 6)
        }

        @Test("A full cache writes into its buffer, grows by 256, and keeps the filled part")
        func fullBuffer() {
            let cache = LayerCache(isFullAttention: true)
            _ = cache.update(keys: positions(range(0, 26)), values: positions(range(0, 26)))
            _ = cache.update(keys: positions(range(100, 64)), values: positions(range(100, 64)))
            #expect(held(cache.keys) == range(0, 26) + range(100, 64))
            #expect(cache.offset == 90)
            // Past 256 positions the buffer grows; the filled part is unchanged.
            _ = cache.update(keys: positions(range(1000, 200)), values: positions(range(1000, 200)))
            #expect(held(cache.keys) == range(0, 26) + range(100, 64) + range(1000, 200))
            #expect(cache.offset == 290)
        }

        @Test("Two continuations of one prefill never see each other's blocks")
        func forkedContinuations() {
            for full in [true, false] {
                let prefill = LayerCache(isFullAttention: full, slidingWindow: full ? nil : 1024)
                _ = prefill.update(keys: positions(range(0, 26)), values: positions(range(0, 26)))
                let first = prefill.extended()
                let second = prefill.extended()
                _ = first.update(keys: positions(range(100, 64)), values: positions(range(100, 64)))
                _ = second.update(
                    keys: positions(range(500, 64)), values: positions(range(500, 64)))
                _ = first.update(keys: positions(range(200, 64)), values: positions(range(200, 64)))
                #expect(
                    held(first.keys) == range(0, 26) + range(100, 64) + range(200, 64),
                    "full \(full)")
                #expect(
                    held(first.values) == range(0, 26) + range(100, 64) + range(200, 64),
                    "full \(full)")
                #expect(held(second.keys) == range(0, 26) + range(500, 64), "full \(full)")
                // A copy of a copy is independent too.
                let third = first.extended()
                _ = third.update(keys: positions([7, 7]), values: positions([7, 7]))
                _ = first.update(keys: positions([8, 8]), values: positions([8, 8]))
                #expect(held(third.keys).last == 7 && held(first.keys).last == 8, "full \(full)")
                #expect(held(prefill.keys) == range(0, 26) && prefill.offset == 26, "full \(full)")
            }
        }
    }
}
