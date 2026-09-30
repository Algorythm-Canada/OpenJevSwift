import OpenJevCore
import OpenJevTestSupport
import Testing

/// Compares ``TemplateResolver`` with Fixtures/templates/templates.json and
/// Fixtures/templates/errors.json, driven by ``FixtureTokenizer``, and checks the cache and the
/// mixed indexed schema with ``WordTokenizer``.
@Suite(
    "Template resolution", .enabled(if: FixtureTokenizer.exists, FixtureTokenizer.missingMessage))
struct TemplateResolverTests {
    private let tokenizer = FixtureTokenizer.shared

    /// A resolver over the replay tokenizer at `canvas`.
    private func resolver(canvas: Int = 64) throws -> TemplateResolver {
        TemplateResolver(
            tokenizer: tokenizer, tokens: try EngineTokens(tokenizer: tokenizer), canvas: canvas)
    }

    /// The fixture's `head` as `resolve` takes it: `"scaffold"` is `nil`, `"none"` is empty.
    private static func head(_ value: JSONValue?) throws -> [Int]? {
        let name = try #require(value?.stringValue)
        #expect(name == "scaffold" || name == "none", "unknown head \(name)")
        return name == "none" ? [] : nil
    }

    private static func format(_ value: JSONValue?) throws -> AnswerFormat {
        let name = try #require(value?.stringValue)
        return try #require(AnswerFormat(rawValue: name))
    }

    private static func ints(_ value: JSONValue?) throws -> [Int] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.intValue) }
    }

    private static func strings(_ value: JSONValue?) throws -> [String] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.stringValue) }
    }

    /// The recorded `{template, slots}` as a ``ResolvedTemplate``.
    private static func recorded(_ row: JSONValue) throws -> ResolvedTemplate {
        let slots = try #require(row["slots"]?.arrayValue).map { slot in
            ResolvedTemplate.Slot(
                position: try #require(slot["pos"]?.intValue),
                labelIDs: try ints(slot["label_ids"]))
        }
        return ResolvedTemplate(template: try ints(row["template"]), slots: slots)
    }

    /// A question of errors.json's `internal_questions`, built by hand as upstream's test does.
    private static func internalQuestion(_ row: JSONValue) throws -> ReadQuestion {
        let choices = try #require(row["choices"]?.arrayValue).map { pair in
            (
                name: try #require(pair[0]?.stringValue),
                description: try #require(pair[1]?.stringValue)
            )
        }
        let type = try #require(row["type"]?.stringValue)
        return ReadQuestion(
            key: try #require(row["key"]?.stringValue),
            id: try #require(row["id"]?.stringValue),
            kind: try #require(QuestionKind(rawValue: type)),
            instructions: try #require(row["instructions"]?.stringValue),
            choices: choices,
            labels: try strings(row["labels"]),
            legend: nil)
    }

    @Test(
        "Every recorded group resolves to the recorded template and slots",
        .enabled(
            if: UpstreamFixtures.exists("templates/templates.json", "labels.json"),
            UpstreamFixtures.missingMessage))
    func recordedTemplates() throws {
        let labels = try LabelDiscovery.choiceLabels(using: tokenizer).labels
        let resolver = try resolver()
        let cases = try UpstreamFixtures.cases("templates/templates.json")
        #expect(cases.count > 0)
        var variants = 0
        for row in cases {
            let name = row["name"]?.stringValue ?? "?"
            let format = try Self.format(row["format"])
            let request = try #require(row["request"])
            let schema = try UpstreamFixtures.schema(for: request, labels: labels)
            #expect(schema.format == format, "\(name): format")
            for group in try #require(row["groups"]?.arrayValue) {
                let index = group["group"]?.intValue ?? -1
                let questions = try UpstreamFixtures.questions(group["questions"], in: schema)
                for variant in try #require(group["variants"]?.arrayValue) {
                    let head = try Self.head(variant["head"])
                    let lead = try #require(variant["lead"]?.stringValue)
                    let resolved = try resolver.resolve(
                        questions, format: format, head: head, lead: lead)
                    let label = "\(name) group \(index), head \(String(describing: head)), "
                    #expect(
                        resolved == (try Self.recorded(variant)),
                        "\(label)lead \(lead.debugDescription)")
                    variants += 1
                }
            }
        }
        #expect(variants > 0)
    }

    @Test(
        "Canvas and shared-slot errors match the recorded messages",
        .enabled(
            if: UpstreamFixtures.exists("templates/errors.json", "labels.json"),
            UpstreamFixtures.missingMessage))
    func recordedErrors() throws {
        let labels = try LabelDiscovery.choiceLabels(using: tokenizer).labels
        var checked = 0
        for row in try UpstreamFixtures.cases("templates/errors.json") {
            let name = row["name"]?.stringValue ?? "?"
            if name == "label_ids_over_read_limit" {
                // The 512 label-id limit is checked by `Engine.one_read`, which issue #17 ports.
                continue
            }
            let canvas = row["settings"]?["canvas"]?.intValue ?? 64
            let format = try Self.format(row["format"])
            let head = try Self.head(row["head"])
            let lead = try #require(row["lead"]?.stringValue)
            let questions: [ReadQuestion]
            if let request = row["request"], !request.isNull {
                let schema = try UpstreamFixtures.schema(for: request, labels: labels)
                questions = try UpstreamFixtures.questions(row["questions"], in: schema)
            } else {
                questions = try #require(row["internal_questions"]?.arrayValue)
                    .map(Self.internalQuestion)
            }

            let resolver = try resolver(canvas: canvas)
            do {
                let resolved = try resolver.resolve(
                    questions, format: format, head: head, lead: lead)
                if row["error"] != nil {
                    Issue.record("\(name): resolved, but upstream refused")
                } else {
                    #expect(resolved == (try Self.recorded(row)), "\(name)")
                }
            } catch let error as SchemaError {
                let recorded = try #require(row["error"], "\(name): resolved upstream")
                #expect(error.message == recorded["message"]?.stringValue, "\(name)")
                let loc = try Self.strings(recorded["loc"]).map(LocComponent.key)
                #expect(error.loc == loc, "\(name)")
            }
            checked += 1
        }
        #expect(checked > 0)
    }

    @Test("Twelve mixed questions in the indexed format keep one slot each")
    func indexedMixedTypes() throws {
        // Upstream's test_indexed_format_with_mixed_types: nouls, 10-level scores and 40-option
        // choices past ten questions. The word tokenizer stands in for the real one, whose
        // encodings of these texts are not recorded.
        let tokenizer = WordTokenizer()
        var entries: [String] = []
        for i in 0..<12 {
            switch i % 3 {
            case 0:
                entries.append("\"n\(i)\": {\"type\": \"noul\"}")
            case 1:
                let levels = (0..<10).map { "\"level \($0)\"" }.joined(separator: ", ")
                entries.append("\"s\(i)\": {\"type\": \"score\", \"criteria\": [\(levels)]}")
            default:
                let options = (0..<40).map { "\"opt\($0)\": null" }.joined(separator: ", ")
                entries.append("\"c\(i)\": {\"type\": \"choice\", \"criteria\": {\(options)}}")
            }
        }
        let body = try JSONParser().parse(
            "{\"state\": \"x\", \"model\": \"jev-latest\", \"questions\": {"
                + entries.joined(separator: ", ") + "}}")
        let labels = try LabelDiscovery.choiceLabels(using: tokenizer).labels
        let schema = try UpstreamFixtures.schema(for: body, labels: labels)
        #expect(schema.format == .indexed)
        #expect(schema.questions.count == 12)

        let resolver = TemplateResolver(
            tokenizer: tokenizer, tokens: try EngineTokens(tokenizer: tokenizer))
        let resolved = try resolver.resolve(schema.questions, format: schema.format)
        #expect(resolved.slots.count == 12)
        for (question, slot) in zip(schema.questions, resolved.slots) {
            #expect(slot.labelIDs.count == question.labels.count, "\(question.id)")
            #expect(resolved.template[slot.position] == slot.labelIDs[0], "\(question.id)")
            #expect(Set(slot.labelIDs).count == slot.labelIDs.count, "\(question.id)")
        }
        #expect(resolved.slots.map(\.position) == resolved.slots.map(\.position).sorted())
    }

    @Test("The cache returns the stored template and keys on head and lead")
    func cacheKeys() throws {
        let tokenizer = WordTokenizer()
        let resolver = TemplateResolver(
            tokenizer: tokenizer, tokens: try EngineTokens(tokenizer: tokenizer))
        let question = Self.noul(key: "a", id: "q1")
        let scaffold = resolver.tokens.scaffold

        let plain = try resolver.resolve([question], format: .lines)
        #expect(resolver.cache.count == 1)
        #expect(try resolver.resolve([question], format: .lines) == plain)
        #expect(resolver.cache.count == 1, "a repeated key is a hit")
        #expect(try resolver.resolve([question], format: .lines, head: scaffold) == plain)
        #expect(resolver.cache.count == 1, "nil and the scaffold are the same key")

        let afterThought = try resolver.resolve([question], format: .lines, head: [])
        #expect(afterThought != plain)
        #expect(afterThought.template.count == plain.template.count - scaffold.count)
        #expect(resolver.cache.count == 2, "an empty head is another key")

        let sequential = try resolver.resolve([question], format: .lines, head: [], lead: "\n")
        #expect(sequential != afterThought)
        #expect(sequential.template.count == afterThought.template.count + 1)
        #expect(resolver.cache.count == 3, "a lead is another key")

        _ = try resolver.resolve([question], format: .indexed)
        #expect(resolver.cache.count == 4, "the format is part of the key")
        _ = try resolver.resolve([Self.noul(key: "b", id: "q1")], format: .lines)
        #expect(resolver.cache.count == 4, "the key is not part of the key, the id and labels are")
    }

    @Test("The cache is emptied when it holds more than 4,096 entries")
    func cacheLimit() throws {
        let tokenizer = WordTokenizer()
        let resolver = TemplateResolver(
            tokenizer: tokenizer, tokens: try EngineTokens(tokenizer: tokenizer))
        #expect(resolver.cache.limit == 4096)
        for i in 0..<4097 {
            _ = try resolver.resolve([Self.noul(key: "k", id: "q\(i)")], format: .lines)
        }
        #expect(resolver.cache.count == 4097, "4,096 entries do not trigger the clear")
        let last = try resolver.resolve([Self.noul(key: "k", id: "q4097")], format: .lines)
        #expect(resolver.cache.count == 1, "the 4,098th insert clears the cache first")
        let key = TemplateCache.Key(
            format: .lines, head: resolver.tokens.scaffold, lead: "",
            questions: [.init(id: "q4097", labels: ["yes", "no"])])
        #expect(resolver.cache.template(for: key) == last)
    }

    /// A noul read question built by hand.
    static func noul(key: String, id: String) -> ReadQuestion {
        ReadQuestion(
            key: key, id: id, kind: .noul, instructions: "",
            choices: [(name: "yes", description: ""), (name: "no", description: "")],
            labels: LabelDiscovery.noulLabels, legend: nil)
    }
}
