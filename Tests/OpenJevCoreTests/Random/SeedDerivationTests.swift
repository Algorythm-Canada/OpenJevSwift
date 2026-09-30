import OpenJevCore
import Testing

/// Compares ``SeedDerivation`` with the seeds upstream's route computed for the bodies in
/// Fixtures/seeds.json.
@Suite("Seed derivation")
struct SeedDerivationTests {
    /// Every row of `cases`. The rows of `upstream_only` are skipped by design: their bodies hold
    /// `NaN`, `Infinity`, `1e400` or a lone surrogate, which only Python's `json.loads` accepts
    /// and `JSONParser` refuses (decision D-016).
    @Test(
        "Every recorded key, digest, seed and first 64 draws are reproduced",
        .enabled(if: SeedFixtures.exists, SeedFixtures.missingMessage))
    func recordedSeeds() throws {
        let cases = try SeedFixtures.array("cases")
        #expect(cases.count > 0)
        for row in cases {
            let name = row["name"]?.stringValue ?? "?"
            let body = try JSONParser().parse(try #require(row["body_text"]?.stringValue))
            let request = try RequestValidator().validate(body)
            let images = try ImageValidation.parts(request.images ?? [])
            let key = SeedDerivation.seedKey(
                state: request.state, questions: request.questions, images: images)

            let bytes = try SeedDerivation.seedBytes(for: key)
            let keyText = try #require(row["key_text"]?.stringValue)
            #expect(bytes == Array(keyText.utf8), "\(name): key bytes")
            #expect(
                SeedFixtures.hex(OpenJevCore.SHA256.digest(bytes)) == row["sha256"]?.stringValue,
                "\(name): sha256")

            let seed = try SeedDerivation.seed(for: key)
            #expect(seed == (try SeedFixtures.unsigned(row["seed"])), "\(name): seed")

            let draws = try SeedFixtures.integers(row["randrange_262144"])
            #expect(draws.count == 64, "\(name): draw count")
            var random = PythonRandom(seed: seed)
            #expect(draws.map { _ in random.randrange(262_144) } == draws, "\(name): draws")
        }
    }

    @Test("An empty image list adds nothing to the key")
    func emptyImages() {
        let questions: OrderedMap<Question> = ["q": .noul(instructions: nil, criteria: nil)]
        let key = SeedDerivation.seedKey(state: "s", questions: questions, images: [])
        #expect(key.arrayValue?.count == 2)
    }

    @Test("Every declared question field is present, unset ones as null")
    func questionDump() throws {
        let questions: OrderedMap<Question> = [
            "a": .noul(instructions: nil, criteria: nil),
            "b": .noul(instructions: "i", criteria: NoulCriteria(whenTrue: "t")),
            "c": .choice(instructions: nil, criteria: ["x": .null, "y": "why"]),
            "d": .score(instructions: nil, criteria: ["low", "high"]),
        ]
        let key = SeedDerivation.seedKey(state: "s", questions: questions, images: [])
        let text = String(decoding: try SeedDerivation.seedBytes(for: key), as: UTF8.self)
        let expected =
            #"["s", {"a": {"criteria": null, "instructions": null, "type": "noul"}, "#
            + #""b": {"criteria": {"false": null, "true": "t"}, "instructions": "i", "#
            + #""type": "noul"}, "c": {"criteria": {"x": null, "y": "why"}, "#
            + #""instructions": null, "type": "choice"}, "d": {"criteria": ["low", "high"], "#
            + #""instructions": null, "type": "score"}}]"#
        #expect(text == expected)
    }

    @Test("Derived seeds for groups and samples")
    func derivedSeeds() {
        let base: UInt64 = 4_294_967_295
        #expect(SeedDerivation.groupSeed(base, 0) == base)
        #expect(SeedDerivation.groupSeed(base, 3) == 4_295_281_482)
        #expect(SeedDerivation.sampleSeed(base, 0) == base)
        #expect(SeedDerivation.sampleSeed(base, 5) == 4_295_006_890)
    }
}
