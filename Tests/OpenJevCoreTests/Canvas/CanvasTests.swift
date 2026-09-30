import OpenJevCore
import Testing

/// Compares ``ReadGrouping``, ``CanvasGeometry`` and ``CanvasBuilder`` with
/// Fixtures/groups-and-canvases/groups_and_canvases.json, driven by ``FixtureTokenizer``, and
/// checks the boundaries with ``WordTokenizer``.
@Suite("Groups and canvases")
struct CanvasTests {
    private static let fixture = "groups-and-canvases/groups_and_canvases.json"

    private static func ints(_ value: JSONValue?) throws -> [Int] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.intValue) }
    }

    private static func strings(_ value: JSONValue?) throws -> [String] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.stringValue) }
    }

    private static func slots(_ value: JSONValue?) throws -> [ResolvedTemplate.Slot] {
        try #require(value?.arrayValue).map { slot in
            ResolvedTemplate.Slot(
                position: try #require(slot["pos"]?.intValue),
                labelIDs: try ints(slot["label_ids"]))
        }
    }

    @Test(
        "Recorded groups, templates, widths and canvases at three canvas sizes",
        .enabled(if: FixtureTokenizer.exists, FixtureTokenizer.missingMessage),
        .enabled(
            if: UpstreamFixtures.exists(fixture, "labels.json"), UpstreamFixtures.missingMessage))
    func recordedGroupsAndCanvases() throws {
        let tokenizer = FixtureTokenizer.shared
        let tokens = try EngineTokens(tokenizer: tokenizer)
        let labels = try LabelDiscovery.choiceLabels(using: tokenizer).labels
        let cases = try UpstreamFixtures.cases(Self.fixture)
        #expect(cases.count > 0)
        var canvases = 0
        for row in cases {
            let canvas = row["settings"]?["canvas"]?.intValue ?? 64
            let step = row["settings"]?["step"]?.intValue ?? 16
            let name = "\(row["name"]?.stringValue ?? "?") at \(canvas)"
            let geometry = try CanvasGeometry(canvas: canvas, step: step)
            let body = try #require(row["request"])
            let formatName = try #require(row["format"]?.stringValue)
            let format = try #require(AnswerFormat(rawValue: formatName))
            let schema = try UpstreamFixtures.schema(for: body, labels: labels)
            #expect(schema.format == format, "\(name): format")

            // The request seed, as #15 derives it.
            let request = try RequestValidator().validate(body)
            let images = try ImageValidation.parts(request.images ?? [])
            let key = SeedDerivation.seedKey(
                state: request.state, questions: request.questions, images: images)
            let seed = try #require(row["seed"]?.intValue)
            #expect(try SeedDerivation.seed(for: key) == UInt64(seed), "\(name): seed")

            let groups = try ReadGrouping.groups(
                schema.questions, format: format, geometry: geometry, scaffold: tokens.scaffold,
                tokenizer: tokenizer)
            let recorded = try #require(row["groups"]?.arrayValue)
            #expect(groups.count == recorded.count, "\(name): group count")
            let resolver = TemplateResolver(tokenizer: tokenizer, tokens: tokens, canvas: canvas)
            for (k, (group, record)) in zip(groups, recorded).enumerated() {
                let label = "\(name) group \(k)"
                #expect(group.map(\.id) == (try Self.strings(record["questions"])), "\(label)")
                let rows = try ReadGrouping.rows(
                    of: group, format: format, scaffold: tokens.scaffold, tokenizer: tokenizer)
                #expect(rows == record["rows"]?.intValue, "\(label): rows")

                let resolved = try resolver.resolve(group, format: format)
                let template = try Self.ints(record["template"])
                #expect(resolved.template == template, "\(label): template")
                #expect(resolved.slots == (try Self.slots(record["slots"])), "\(label): slots")
                let width = geometry.width(templateCount: template.count)
                #expect(width == record["width"]?.intValue, "\(label): width")

                for entry in try #require(record["canvases"]?.arrayValue) {
                    let seed = UInt64(try #require(entry["seed"]?.intValue))
                    let built = CanvasBuilder.build(
                        template: resolved.template, slots: resolved.slots, seed: seed,
                        geometry: geometry)
                    let noise = try Self.ints(entry["noise"])
                    #expect(built.noise == noise, "\(label) seed \(seed): noise")
                    let tokens = try Self.ints(entry["canvas"])
                    #expect(built.tokens == tokens, "\(label) seed \(seed): canvas")
                    #expect(built.width == width, "\(label) seed \(seed): width")
                    canvases += 1
                }
            }
        }
        #expect(canvases > 0)
    }

    @Test("Widths round up to the step and cap at the canvas")
    func widths() throws {
        let geometry = try CanvasGeometry()
        #expect(geometry.canvas == 64 && geometry.step == 16)
        #expect(geometry.width(templateCount: 0) == 16)
        #expect(geometry.width(templateCount: 15) == 16)
        #expect(geometry.width(templateCount: 16) == 32)
        #expect(geometry.width(templateCount: 31) == 32)
        #expect(geometry.width(templateCount: 32) == 48)
        #expect(geometry.width(templateCount: 47) == 48)
        #expect(geometry.width(templateCount: 48) == 64)
        #expect(geometry.width(templateCount: 63) == 64)
        #expect(geometry.width(templateCount: 200) == 64)

        let capped = try CanvasGeometry(canvas: 40)
        #expect(capped.width(templateCount: 31) == 32)
        #expect(capped.width(templateCount: 32) == 40, "48 rounded, capped at 40")
        #expect(capped.width(templateCount: 39) == 40)
    }

    @Test("The geometry refuses a canvas or step below 1")
    func geometryRejectsZero() {
        #expect(throws: CanvasGeometryError.self) { try CanvasGeometry(canvas: 0) }
        #expect(throws: CanvasGeometryError.self) { try CanvasGeometry(step: 0) }
        #expect(throws: CanvasGeometryError.self) { try CanvasGeometry(canvas: -1, step: 16) }
        #expect(throws: Never.self) { try CanvasGeometry(canvas: 1, step: 1) }
    }

    @Test("A template with one token to spare resolves; one more fails")
    func canvasBoundary() throws {
        let tokenizer = WordTokenizer()
        let tokens = try EngineTokens(tokenizer: tokenizer)
        let question = TemplateResolverTests.noul(key: "a", id: "q1")
        let count = try tokenizer.encode("q1: yes", addSpecialTokens: false).count
        let resolver = TemplateResolver(tokenizer: tokenizer, tokens: tokens, canvas: count + 1)

        let fits = try resolver.resolve([question], format: .lines, head: [])
        #expect(fits.template.count + 1 == resolver.canvas)

        let error = #expect(throws: SchemaError.self) {
            try resolver.resolve([question], format: .lines, head: [], lead: "\n")
        }
        #expect(
            error?.message == "answer template is \(count + 1) tokens; the canvas holds \(count)")
        #expect(error?.loc == [.key("body")])
    }

    @Test("Thirty nouls at canvas 32 split into groups that each resolve")
    func manyQuestionsChunk() throws {
        // Upstream's test_many_questions_chunk, with the word tokenizer standing in for the real
        // one, whose encodings of the trial texts at this canvas are not recorded.
        let tokenizer = WordTokenizer()
        let tokens = try EngineTokens(tokenizer: tokenizer)
        let geometry = try CanvasGeometry(canvas: 32)
        let entries = (0..<30).map { "\"n\($0)\": {\"type\": \"noul\"}" }.joined(separator: ", ")
        let body = try JSONParser().parse(
            "{\"state\": \"x\", \"model\": \"jev-latest\", \"questions\": {\(entries)}}")
        let schema = try UpstreamFixtures.schema(for: body, labels: ["A", "B"])
        #expect(schema.format == .indexed)

        let groups = try ReadGrouping.groups(
            schema.questions, format: schema.format, geometry: geometry,
            scaffold: tokens.scaffold, tokenizer: tokenizer)
        #expect(groups.count > 1)
        #expect(groups.flatMap { $0.map(\.id) } == schema.questions.map(\.id))

        let resolver = TemplateResolver(tokenizer: tokenizer, tokens: tokens, canvas: 32)
        for group in groups {
            let rows = try ReadGrouping.rows(
                of: group, format: schema.format, scaffold: tokens.scaffold, tokenizer: tokenizer)
            #expect(rows <= 32)
            let resolved = try resolver.resolve(group, format: schema.format)
            #expect(resolved.slots.count == group.count)
            let width = geometry.width(templateCount: resolved.template.count)
            #expect(width <= 32)
            let built = CanvasBuilder.build(
                template: resolved.template, slots: resolved.slots, seed: 0, geometry: geometry)
            #expect(built.tokens.count == width)
            #expect(built.tokens[resolved.template.count] == EngineTokens.turnClose)
        }
    }

    @Test("No questions give no groups")
    func emptyGroups() throws {
        let tokenizer = WordTokenizer()
        let groups = try ReadGrouping.groups(
            [], format: .lines, geometry: try CanvasGeometry(),
            scaffold: try EngineTokens(tokenizer: tokenizer).scaffold, tokenizer: tokenizer)
        #expect(groups.isEmpty)
    }

    @Test("Canvas noise is one generator's successive draws in slot order")
    func canvasNoise() throws {
        let geometry = try CanvasGeometry()
        let template = [7, 8, 9, 10, 11]
        let slots = [
            ResolvedTemplate.Slot(position: 1, labelIDs: [8, 80]),
            ResolvedTemplate.Slot(position: 4, labelIDs: [11, 110]),
        ]
        let built = CanvasBuilder.build(
            template: template, slots: slots, seed: 42, geometry: geometry)
        var rng = PythonRandom(seed: 42)
        let first = rng.randrange(EngineTokens.vocabularySize)
        let second = rng.randrange(EngineTokens.vocabularySize)
        #expect(built.noise == [first, second])
        #expect(built.tokens.count == 16)
        #expect(Array(built.tokens[0..<6]) == [7, first, 9, 10, second, EngineTokens.turnClose])
        #expect(built.tokens[6...].allSatisfy { $0 == EngineTokens.pad })
    }
}
