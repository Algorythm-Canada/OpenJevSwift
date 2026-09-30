import OpenJevCore
import Testing

/// Compares ``MT19937`` and ``PythonRandom`` with CPython's `random.Random`, using the tables in
/// Fixtures/seeds.json and values taken from `python3` (CPython 3.14.7).
@Suite("Python-compatible random numbers")
struct PythonRandomTests {
    // MARK: Fixtures

    @Test(
        "First getrandbits(32) and randrange(262144) for 10,000 seeds",
        .enabled(if: SeedFixtures.exists, SeedFixtures.missingMessage))
    func mt19937Seeds() throws {
        let columns = try SeedFixtures.array("mt19937_columns").compactMap(\.stringValue)
        #expect(columns == ["seed", "getrandbits(32)", "randrange(262144)"])
        let rows = try SeedFixtures.array("mt19937_seeds")
        #expect(rows.count == 10_000)
        var failures: [String] = []
        for row in rows {
            let seed = try SeedFixtures.unsigned(row[0])
            let bits = try SeedFixtures.unsigned(row[1])
            let draw = try #require(row[2]?.intValue)
            var first = PythonRandom(seed: seed)
            var second = PythonRandom(seed: seed)
            let gotBits = first.getrandbits(32)
            let gotDraw = second.randrange(262_144)
            if gotBits != bits || gotDraw != draw {
                failures.append("seed \(seed): \(gotBits), \(gotDraw); want \(bits), \(draw)")
            }
        }
        #expect(failures.isEmpty, "\(failures.count) mismatches: \(failures.prefix(5))")
    }

    @Test(
        "Long streams, past state regenerations and from a two-word seed",
        .enabled(if: SeedFixtures.exists, SeedFixtures.missingMessage))
    func mt19937Streams() throws {
        let streams = try SeedFixtures.array("mt19937_streams")
        var seen: [String] = []
        for stream in streams {
            let seed = try SeedFixtures.unsigned(stream["seed"])
            let call = try #require(stream["call"]?.stringValue)
            let values = try SeedFixtures.integers(stream["values"])
            var random = PythonRandom(seed: seed)
            let got: [Int]
            switch call {
            case "getrandbits(32)":
                got = values.map { _ in Int(random.getrandbits(32)) }
            case "randrange(262144)":
                got = values.map { _ in random.randrange(262_144) }
            default:
                Issue.record("unknown call \(call)")
                continue
            }
            #expect(got == values, "seed \(seed), \(call)")
            seen.append("\(seed) \(call) \(values.count)")
        }
        #expect(
            seen == [
                "0 getrandbits(32) 1300",
                "4295072025 getrandbits(32) 700",
                "20260929 randrange(262144) 1000",
            ])
    }

    // MARK: Hand-written

    /// `python3 -c "import random; r=random.Random(5); print(r.getrandbits(64))"` and the same for
    /// seed 4295072025 (2**32 + 104729, a two-word key).
    @Test(
        "getrandbits(64)",
        arguments: [
            (UInt64(5), UInt64(4_712_128_852_136_459_333)),
            (4_295_072_025, 15_106_292_795_604_252_561),
        ])
    func getrandbits64(seed: UInt64, expected: UInt64) {
        var random = PythonRandom(seed: seed)
        #expect(random.getrandbits(64) == expected)
    }

    /// `python3 -c "import random; r=random.Random(5); print(r.getrandbits(40))"` and the same for
    /// seed 4295072025. The second 32-bit output is shifted right by 24 bits.
    @Test(
        "getrandbits(40)",
        arguments: [
            (UInt64(5), UInt64(281_848_216_645)),
            (4_295_072_025, 901_188_728_721),
        ])
    func getrandbits40(seed: UInt64, expected: UInt64) {
        var random = PythonRandom(seed: seed)
        #expect(random.getrandbits(40) == expected)
    }

    /// `python3 -c "import random; r=random.Random(5); print([r.getrandbits(19) for _ in
    /// range(3)])"` and the same for seed 4295072025.
    @Test(
        "getrandbits(19)",
        arguments: [
            (5, [326_579, 133_926, 388_910]),
            (4_295_072_025, [432_197, 429_346, 211_472]),
        ] as [(UInt64, [UInt64])])
    func getrandbits19(seed: UInt64, expected: [UInt64]) {
        var random = PythonRandom(seed: seed)
        #expect(expected.map { _ in random.getrandbits(19) } == expected)
    }

    /// From seed 0 the first three outputs are 3626764237, 1654615998 and 3255389356
    /// (`python3 -c "import random; r=random.Random(0); print([r.getrandbits(32) for _ in
    /// range(3)])"`). The first gives 3626764237 >> 13 = 442720, which is not below 262144, so
    /// `randrange(262144)` draws again and returns 1654615998 >> 13 = 201979; the next output is
    /// then the third (`python3 -c "import random; r=random.Random(0); print(r.randrange(262144),
    /// r.getrandbits(32))"` prints `201979 3255389356`).
    @Test("randrange rejection consumes an extra output")
    func randrangeRejection() {
        var random = PythonRandom(seed: 0)
        #expect(random.randrange(262_144) == 201_979)
        #expect(random.getrandbits(32) == 3_255_389_356)

        var generator = MT19937(seed: 0)
        #expect(generator.nextUInt32() == 3_626_764_237)
        #expect(generator.nextUInt32() == 1_654_615_998)
    }

    @Test("Seed 0 is the key [0] and a two-word seed is its low word first")
    func seedKeys() {
        var fromSeed = MT19937(seed: 0)
        var fromKey = MT19937(key: [0])
        #expect(fromSeed.nextUInt32() == fromKey.nextUInt32())

        let twoWord: UInt64 = (1 << 32) + 104_729
        var fromLargeSeed = MT19937(seed: twoWord)
        var fromWords = MT19937(key: [104_729, 1])
        #expect(fromLargeSeed.nextUInt32() == fromWords.nextUInt32())
    }
}
