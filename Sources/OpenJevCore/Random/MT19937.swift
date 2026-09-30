// A port of CPython 3.14's `Modules/_randommodule.c` (`init_genrand`, `init_by_array`,
// `genrand_uint32` and the integer branch of `random_seed`). PSF-2.0. See THIRD_PARTY.md.
//
// That file is itself based on the MT19937 reference code, which carries this notice:
//
//   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
//   All rights reserved.
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions
//   are met:
//
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/// The 32-bit Mersenne Twister, seeded as CPython's `random.Random(seed)` seeds it for an integer.
///
/// Upstream draws its canvas noise with `random.Random(seed).randrange(VOCAB)`, so reproducing
/// its canvases needs the same generator and the same seeding (decision D-006). The state is 624
/// words; the output is `genrand_uint32` with CPython's tempering.
public struct MT19937: Sendable {
    private static let n = 624
    private static let m = 397
    private static let matrixA: UInt32 = 0x9908_b0df
    private static let upperMask: UInt32 = 0x8000_0000
    private static let lowerMask: UInt32 = 0x7fff_ffff

    private var state: [UInt32]
    private var index: Int

    /// Seeds the generator as `random.Random(seed)` does for a non-negative integer.
    ///
    /// CPython splits the seed's absolute value into 32-bit words, least significant first, and
    /// passes them to `init_by_array`. A seed below 2^32 is a one-word key, and 0 is the key
    /// `[0]`. Larger seeds, such as upstream's group seeds `seed + 104729·k`, are two-word keys.
    public init(seed: UInt64) {
        let low = UInt32(truncatingIfNeeded: seed)
        let high = UInt32(truncatingIfNeeded: seed >> 32)
        self.init(key: high == 0 ? [low] : [low, high])
    }

    /// Seeds the generator with CPython's `init_genrand(19650218)` followed by
    /// `init_by_array(key)`.
    ///
    /// - Precondition: `key` is not empty. CPython always passes at least one word.
    public init(key: [UInt32]) {
        precondition(!key.isEmpty, "init_by_array needs at least one key word")
        let n = Self.n
        state = [UInt32](repeating: 0, count: n)
        index = n

        // init_genrand(19650218)
        state[0] = 19_650_218
        for i in 1..<n {
            let previous = state[i - 1]
            state[i] = 1_812_433_253 &* (previous ^ (previous >> 30)) &+ UInt32(i)
        }

        // init_by_array(key)
        var i = 1
        var j = 0
        for _ in 0..<max(n, key.count) {
            let previous = state[i - 1]
            state[i] =
                (state[i] ^ ((previous ^ (previous >> 30)) &* 1_664_525)) &+ key[j]
                &+ UInt32(truncatingIfNeeded: j)
            i += 1
            j += 1
            if i >= n {
                state[0] = state[n - 1]
                i = 1
            }
            if j >= key.count {
                j = 0
            }
        }
        for _ in 0..<(n - 1) {
            let previous = state[i - 1]
            state[i] =
                (state[i] ^ ((previous ^ (previous >> 30)) &* 1_566_083_941))
                &- UInt32(truncatingIfNeeded: i)
            i += 1
            if i >= n {
                state[0] = state[n - 1]
                i = 1
            }
        }
        state[0] = 0x8000_0000
    }

    /// The next 32-bit output, CPython's `genrand_uint32`.
    public mutating func nextUInt32() -> UInt32 {
        if index >= Self.n {
            regenerate()
        }
        var y = state[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c_5680
        y ^= (y << 15) & 0xefc6_0000
        y ^= y >> 18
        return y
    }

    /// Generates the next 624 words of state.
    private mutating func regenerate() {
        let n = Self.n
        let m = Self.m
        for k in 0..<n {
            let y = (state[k] & Self.upperMask) | (state[(k + 1) % n] & Self.lowerMask)
            let mag: UInt32 = (y & 1) == 0 ? 0 : Self.matrixA
            state[k] = state[(k + m) % n] ^ (y >> 1) ^ mag
        }
        index = 0
    }
}
