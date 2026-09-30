// A port of CPython 3.14's `random.Random.getrandbits` (`Modules/_randommodule.c`,
// `_random_Random_getrandbits_impl`) and `Random._randbelow_with_getrandbits` (`Lib/random.py`).
// PSF-2.0. See THIRD_PARTY.md.

/// The parts of Python's `random.Random` that upstream's engine uses, with identical output.
///
/// `build_canvas` in upstream's `engine.py` fills every answer slot with
/// `random.Random(seed).randrange(VOCAB)`. The same seed gives the same draws here, so the same
/// request gives the same canvas (decision D-006).
public struct PythonRandom: Sendable {
    private var generator: MT19937

    /// A generator seeded as `random.Random(seed)` is.
    public init(seed: UInt64) {
        generator = MT19937(seed: seed)
    }

    /// `getrandbits(k)`: a value of `k` random bits.
    ///
    /// For `k` up to 32 this is one output shifted right by `32 - k`. For larger `k`, 32-bit
    /// outputs fill the result from the least significant word up, and the last one is shifted
    /// right by `32 - k % 32` when `k` is not a multiple of 32, as CPython does.
    ///
    /// - Precondition: `1 <= k <= 64`.
    public mutating func getrandbits(_ k: Int) -> UInt64 {
        precondition((1...64).contains(k), "getrandbits supports 1 to 64 bits here")
        if k <= 32 {
            return UInt64(generator.nextUInt32() >> UInt32(32 - k))
        }
        var result: UInt64 = 0
        var remaining = k
        var shift: UInt64 = 0
        while remaining > 0 {
            var word = generator.nextUInt32()
            if remaining < 32 {
                word >>= UInt32(32 - remaining)
            }
            result |= UInt64(word) << shift
            shift += 32
            remaining -= 32
        }
        return result
    }

    /// `randrange(n)`: a value in `0..<n`, drawn by rejection as `_randbelow_with_getrandbits`
    /// does.
    ///
    /// With `k` the bit length of `n`, it draws `getrandbits(k)` until the value is below `n`, so
    /// one call can consume more than one output. For upstream's `randrange(262144)`, `k` is 19.
    ///
    /// - Precondition: `n > 0`. Python raises `ValueError` for an empty range.
    public mutating func randrange(_ n: Int) -> Int {
        precondition(n > 0, "empty range for randrange")
        let k = Int.bitWidth - n.leadingZeroBitCount
        var value = getrandbits(k)
        while value >= UInt64(n) {
            value = getrandbits(k)
        }
        return Int(value)
    }
}
