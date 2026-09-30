import OpenJevCore
import OpenJevTestSupport
import Testing

/// Compares ``LabelDiscovery`` and ``EngineTokens``, driven by ``FixtureTokenizer``, with
/// Fixtures/labels.json and the `engine` table of Fixtures/tokenizer/special_tokens.json.
@Suite(
    "Label discovery and engine tokens",
    .enabled(if: FixtureTokenizer.exists, FixtureTokenizer.missingMessage))
struct LabelDiscoveryTests {
    private let tokenizer = FixtureTokenizer.shared

    private static func strings(_ value: JSONValue?) throws -> [String] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.stringValue) }
    }

    private static func ints(_ value: JSONValue?) throws -> [Int] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.intValue) }
    }

    @Test(
        "Discovery reproduces the recorded labels and ids",
        .enabled(if: UpstreamFixtures.exists("labels.json"), UpstreamFixtures.missingMessage))
    func recordedLabels() throws {
        let recorded = try UpstreamFixtures.load("labels.json")
        let discovered = try LabelDiscovery.choiceLabels(using: tokenizer)

        #expect(recorded["prefix"]?.stringValue == LabelDiscovery.prefix)
        #expect(
            try tokenizer.encode("q1: A", addSpecialTokens: false)
                == Self.ints(recorded["base_ids"]))
        #expect(discovered.labels == (try Self.strings(recorded["labels"])))
        #expect(discovered.labelIDs == (try Self.ints(recorded["label_ids"])))

        // Upstream's own test: 255 labels starting A, B, C, all distinct.
        #expect(discovered.labels.count == 255)
        #expect(Array(discovered.labels.prefix(3)) == ["A", "B", "C"])
        #expect(Set(discovered.labels).count == 255)
        #expect(Set(discovered.labelIDs).count == 255)

        #expect(LabelDiscovery.candidates.count == recorded["candidate_count"]?.intValue)
        let lastLabel = try #require(discovered.labels.last)
        let examined = try #require(LabelDiscovery.candidates.firstIndex(of: lastLabel)) + 1
        #expect(examined == recorded["candidates_examined"]?.intValue)

        #expect(LabelDiscovery.noulLabels == (try Self.strings(recorded["noul_labels"])))
        #expect(LabelDiscovery.scoreLabels == (try Self.strings(recorded["score_labels"])))
    }

    @Test(
        "Rejected candidates are skipped for the recorded reason",
        .enabled(if: UpstreamFixtures.exists("labels.json"), UpstreamFixtures.missingMessage))
    func rejectedCandidates() throws {
        let recorded = try UpstreamFixtures.load("labels.json")
        let rejected = try #require(recorded["rejected"]?.arrayValue)
        let discovered = try LabelDiscovery.choiceLabels(using: tokenizer)
        let base = try tokenizer.encode("q1: A", addSpecialTokens: false)

        let names = try rejected.map { try #require($0["candidate"]?.stringValue) }
        #expect(names == ["BQ", "BZ", "FQ", "FZ", "GZ", "HZ"])
        for row in rejected {
            let candidate = try #require(row["candidate"]?.stringValue)
            let ids = try tokenizer.encode("q1: " + candidate, addSpecialTokens: false)
            #expect(ids == (try Self.ints(row["ids"])), "\(candidate)")
            #expect(!discovered.labels.contains(candidate), "\(candidate)")
            switch row["reason"]?.stringValue {
            case "not a single token after the prefix":
                #expect(
                    ids.count != base.count || ids.dropLast() != base.dropLast(), "\(candidate)")
            case let reason:
                Issue.record("\(candidate): unknown reason \(String(describing: reason))")
            }
        }
    }

    @Test(
        "Engine tokens equal the recorded engine table",
        .enabled(
            if: UpstreamFixtures.exists("tokenizer/special_tokens.json"),
            UpstreamFixtures.missingMessage))
    func engineTokens() throws {
        let special = try UpstreamFixtures.load("tokenizer/special_tokens.json")
        let engine = try #require(special["engine"])
        let tokens = try EngineTokens(tokenizer: tokenizer)

        #expect(tokens.scaffold == [100, 45518, 107, 101])
        #expect(tokens.scaffold == (try Self.ints(engine["scaffold"])))
        #expect(tokens.thoughtOpen == (try Self.ints(engine["thought_open"])))
        #expect(tokens.thoughtClose == (try Self.ints(engine["thought_close"])))
        #expect(EngineTokens.scaffoldText == engine["SCAFFOLD_TEXT"]?.stringValue)

        #expect(EngineTokens.vocabularySize == engine["VOCAB"]?.intValue)
        #expect(EngineTokens.vocabularySize == special["vocabulary_size"]?.intValue)
        #expect(EngineTokens.turnClose == engine["TURN_CLOSE"]?["id"]?.intValue)
        #expect(engine["TURN_CLOSE"]?["token"]?.stringValue == "<turn|>")
        #expect(EngineTokens.pad == engine["PAD"]?["id"]?.intValue)
        #expect(engine["PAD"]?["token"]?.stringValue == "<pad>")
        #expect(LabelDiscovery.maxChoices == engine["MAX_CHOICES"]?.intValue)
    }
}
