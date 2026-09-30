import Foundation
import OpenJevCore
import Testing

/// Replays every case of Fixtures/policies/ through ``DecisionEngine`` and ``StubBackend`` and
/// compares the group-level calls, the thoughts, the reads and the response with the recording.
///
/// Parallel groups and the reads inside a group run concurrently, so the recording's order across
/// groups is asyncio's and the stub's order is Swift's. Reads are matched by seed, which is
/// unique within a case, and each group's seeds are checked to be `groupSeed + 7919·k` in `k`
/// order.
@Suite(
    "Decision engine policies",
    .enabled(if: PolicyFixtures.exists, PolicyFixtures.missingMessage))
struct DecisionEnginePolicyTests {
    @Test("Every case of policies.json reproduces the recorded calls and body")
    func policies() async throws {
        try await Self.check(file: PolicyFixtures.policies, entropy: 0.05)
    }

    @Test("Every case of auto_rereads.json reproduces the recorded re-reads and body")
    func autoRereads() async throws {
        try await Self.check(file: PolicyFixtures.autoRereads, entropy: 0.5)
    }

    private static func check(file: String, entropy: Double) async throws {
        let cases = try UpstreamFixtures.cases(file)
        #expect(cases.count > 0)
        for row in cases {
            try await checkCase(row, entropy: entropy)
        }
    }

    private static func checkCase(_ row: JSONValue, entropy: Double) async throws {
        let name = row["name"]?.stringValue ?? "?"
        let tokenizer = FixtureTokenizer.shared
        let stub = StubBackend(tokenizer: tokenizer, entropy: entropy)
        let engine = try DecisionEngine(
            backend: stub, configuration: try PolicyFixtures.configuration(row["settings"]))
        let request = try RequestValidator().validate(row["request"])
        let decision = try await engine.decide(request)

        #expect(row["response"]?["status"]?.intValue == 200, "\(name): status")
        #expect(
            try PolicyFixtures.body(of: decision) == row["response"]?["body_text"]?.stringValue,
            "\(name): body")

        let seed = UInt64(try #require(row["seed"]?.intValue))
        let images = try ImageValidation.parts(request.images ?? [])
        #expect(
            try SeedDerivation.seed(
                for: SeedDerivation.seedKey(
                    state: request.state, questions: request.questions, images: images)) == seed,
            "\(name): seed")
        let schema = try engine.schemaBuilder.build(request.questions)
        let stateText = StateText.render(request.state)
        let options = ReadOptions(request)

        let reads = stub.reads
        var readsBySeed: [UInt64: CanvasRead] = [:]
        for read in reads {
            #expect(
                readsBySeed.updateValue(read, forKey: read.seed) == nil,
                "\(name): seed \(read.seed) was read twice")
        }
        let recordedReads = try #require(row["reads"]?.arrayValue)
        #expect(reads.count == recordedReads.count, "\(name): read count")

        let recordedGroups = try #require(row["groups"]?.arrayValue)
        let sequentialPath = options.sequential && recordedGroups.count > 1
        for (g, group) in recordedGroups.enumerated() {
            let label = "\(name) group \(g)"
            let questions = try UpstreamFixtures.questions(group["questions"], in: schema)
            #expect(
                questions.map(\.key) == (try PolicyFixtures.strings(group["keys"])),
                "\(label): keys")
            let groupSeed = UInt64(try #require(group["seed"]?.intValue))
            #expect(groupSeed == SeedDerivation.groupSeed(seed, g), "\(label): seed")
            let lead = try #require(group["lead"]?.stringValue)
            let groupPrefix = try PolicyFixtures.optionalInts(group["prefix"])
            let systemText = try #require(group["sys_text"]?.stringValue)
            let recorded = try #require(group["options"])
            #expect(recorded["steps"]?.intValue == options.steps, "\(label): steps option")
            #expect(recorded["samples"]?.intValue == options.samples, "\(label): samples option")
            #expect(
                recorded["sequential"]?.boolValue == options.sequential,
                "\(label): sequential option")
            let groupThink = sequentialPath ? 0 : options.think
            #expect(recorded["think"]?.intValue == groupThink, "\(label): think option")

            // The template the group's reads carry ties the head and the lead to the recording.
            let head: [Int]? = groupPrefix == nil && groupThink == 0 ? nil : []
            let expected = try engine.resolver.resolve(
                questions, format: schema.format, head: head, lead: lead)

            let groupReads = recordedReads.filter { $0["group"]?.intValue == g }
            #expect(!groupReads.isEmpty, "\(label): no reads")
            var order: [Int] = []
            for (k, record) in groupReads.enumerated() {
                let readLabel = "\(label) read \(k)"
                let recordedSeed = UInt64(try #require(record["seed"]?.intValue))
                #expect(
                    recordedSeed == SeedDerivation.sampleSeed(groupSeed, k),
                    "\(readLabel): seed")
                guard let read = readsBySeed[recordedSeed] else {
                    Issue.record("\(readLabel): no read at seed \(recordedSeed)")
                    continue
                }
                if let index = reads.firstIndex(where: { $0.seed == recordedSeed }) {
                    order.append(index)
                }
                #expect(read.steps == record["steps"]?.intValue, "\(readLabel): steps")
                #expect(read.systemText == systemText, "\(readLabel): group system text")
                #expect(
                    read.systemText == record["sys_text"]?.stringValue,
                    "\(readLabel): system text")
                #expect(
                    read.template == (try PolicyFixtures.ints(record["template"])),
                    "\(readLabel): template")
                #expect(read.template == expected.template, "\(readLabel): resolved template")
                #expect(
                    read.slots == (try PolicyFixtures.slots(record["slots"])),
                    "\(readLabel): slots")
                #expect(read.slots == expected.slots, "\(readLabel): resolved slots")
                #expect(
                    read.labelIDs == (try PolicyFixtures.ints(record["label_ids"])),
                    "\(readLabel): label ids")
                #expect(
                    read.canvas.width == record["canvas_width"]?.intValue,
                    "\(readLabel): canvas width")
                #expect(
                    read.canvas.tokens == (try PolicyFixtures.ints(record["canvas"])),
                    "\(readLabel): canvas")
                #expect(read.stateText == stateText, "\(readLabel): state text")
                if let content = record["content"]?.stringValue {
                    #expect(read.stateText == content, "\(readLabel): content")
                } else {
                    let parts = try #require(record["content"]?.arrayValue)
                    let urls = parts.dropLast().map { $0["image_url"]?["url"]?.stringValue }
                    #expect(urls == images.map(\.dataURL), "\(readLabel): image parts")
                    #expect(
                        parts.last?["text"]?.stringValue == read.stateText,
                        "\(readLabel): content text")
                }
                let recordedPrefix = try PolicyFixtures.optionalInts(record["prefix"])
                if let groupPrefix {
                    #expect(recordedPrefix == groupPrefix, "\(readLabel): group prefix")
                }
                let kind = try #require(record["mlx_prompt"]?.stringValue)
                switch kind {
                case "prefix":
                    let prefix = try #require(recordedPrefix, "\(readLabel): prefix")
                    #expect(read.prompt == .tokens(prefix), "\(readLabel): prefix prompt")
                case "chat_prompt_ids":
                    #expect(recordedPrefix == nil, "\(readLabel): unexpected prefix")
                    let ids = try tokenizer.chatPromptIDs(
                        system: systemText, user: stateText, thinking: false)
                    #expect(read.prompt == .tokens(ids), "\(readLabel): chat prompt")
                case "image_prompt":
                    #expect(
                        read.prompt
                            == .image(
                                systemText: systemText, stateText: stateText, images: images),
                        "\(readLabel): image prompt")
                default:
                    Issue.record("\(readLabel): unknown mlx_prompt \(kind)")
                }
            }
            // Under the automatic policy the first read finishes before any re-read starts.
            if options.samples == nil, let first = order.first {
                #expect(order.dropFirst().allSatisfy { $0 > first }, "\(label): first read order")
            }
        }

        var thinks = stub.thinks
        let recordedThinks = try #require(row["thinks"]?.arrayValue)
        #expect(thinks.count == recordedThinks.count, "\(name): think count")
        for record in recordedThinks {
            let systemText = try #require(record["sys_text"]?.stringValue)
            let recordedState = try #require(record["state_text"]?.stringValue)
            #expect(recordedState == stateText, "\(name): think state text")
            if let g = record["group"]?.intValue {
                #expect(
                    recordedGroups[g]["sys_text"]?.stringValue == systemText,
                    "\(name): think \(g) system text")
            } else {
                #expect(sequentialPath, "\(name): a groupless thought outside sequential")
                #expect(
                    SystemText.render(schema.questions, format: schema.format, chunked: false)
                        == systemText,
                    "\(name): sequential think system text")
            }
            let prompt =
                try tokenizer.chatPromptIDs(system: systemText, user: stateText, thinking: true)
                + engine.tokens.thoughtOpen
            let expected = StubBackend.ThinkCall(
                prompt: prompt, budget: try #require(record["budget"]?.intValue),
                stopIDs: engine.tokens.thoughtClose)
            if let index = thinks.firstIndex(of: expected) {
                thinks.remove(at: index)
            } else {
                Issue.record("\(name): no think call for group \(record["group"] ?? .null)")
            }
        }
        #expect(thinks.isEmpty, "\(name): unrecorded think calls")
    }
}
