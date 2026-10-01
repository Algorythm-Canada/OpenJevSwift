import Foundation
import OpenJevCore
import Testing

/// The checks the live suite makes on every `POST /v1/systemone` response, beyond upstream's: the
/// headers and the answer shapes of Jev's contract (docs/02-jev-wire-api.md), which upstream's
/// server and this one both follow.
enum JevContract {
    /// Upstream's `assert GATEWAY or "server-timing" in r.headers`, and the request id both
    /// servers send on every response, errors included: `x-request-id` and
    /// `x-typesafe-request-id`, the same `req_` and 32 lowercase hex characters.
    ///
    /// Behind a gateway (`OPENJEV_LIVE_GATEWAY=1`) only `server-timing` goes unchecked, as
    /// upstream's flag has it. Otherwise the header must time the `model`, `server` and `total`
    /// spans in milliseconds.
    static func checkHeaders(
        _ response: LiveResponse, gateway: Bool,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        if !gateway {
            let timing = response.header("server-timing")
            #expect(
                timing != nil, "no server-timing header in \(response.headers)",
                sourceLocation: sourceLocation)
            if let timing {
                let spans = durations(inServerTiming: timing)
                #expect(
                    ["model", "server", "total"].allSatisfy { (spans[$0] ?? -1) >= 0 },
                    "server-timing is \(timing)", sourceLocation: sourceLocation)
            }
        }
        let id = response.header("x-request-id")
        #expect(
            id.map(isRequestID) == true, "x-request-id is \(id ?? "missing")",
            sourceLocation: sourceLocation)
        let typesafeID = response.header("x-typesafe-request-id")
        #expect(
            typesafeID == id,
            "x-typesafe-request-id is \(typesafeID ?? "missing"), x-request-id \(id ?? "missing")",
            sourceLocation: sourceLocation)
    }

    /// The `name;dur=value` entries of a `server-timing` value, such as
    /// `model;dur=41.2, server;dur=2.8, total;dur=44.0`.
    static func durations(inServerTiming value: String) -> [String: Double] {
        var spans: [String: Double] = [:]
        for entry in value.split(separator: ",") {
            let parts = entry.split(separator: ";").map {
                $0.trimmingCharacters(in: .whitespaces)
            }
            guard let name = parts.first, !name.isEmpty else { continue }
            for parameter in parts.dropFirst() where parameter.hasPrefix("dur=") {
                spans[name] = Double(parameter.dropFirst("dur=".count))
            }
        }
        return spans
    }

    /// `req_` and 32 lowercase hex characters, upstream's `"req_" + secrets.token_hex(16)`.
    static func isRequestID(_ text: String) -> Bool {
        guard text.hasPrefix("req_") else { return false }
        let digits = text.utf8.dropFirst(4)
        return digits.count == 32
            && digits.allSatisfy { (0x30...0x39).contains($0) || (0x61...0x66).contains($0) }
    }

    /// Jev's response to a decision request for `questions` (docs/02-jev-wire-api.md): `model`
    /// is the served version, a name `listed` holds and never the `openjev-latest` alias;
    /// `answers` holds one answer per question in the questions' order, each with its type's
    /// exact keys; and `usage` bills the input tokens, with no output tokens unless the request
    /// asked the model to think.
    static func checkBody(
        _ body: JSONValue, questions: JSONObject, requested model: String, listed: Set<String>,
        thought: Bool = false, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let served = body["model"]?.stringValue
        #expect(
            served.map(listed.contains) == true,
            "the answer's model \(served ?? "missing") is not listed: \(listed.sorted())",
            sourceLocation: sourceLocation)
        if model == "openjev-latest" {
            #expect(served != model, "the answer names the alias", sourceLocation: sourceLocation)
        }
        let answers = body["answers"]?.objectValue
        #expect(
            answers?.keys == questions.keys,
            "answers \(answers?.keys ?? []) for questions \(questions.keys)",
            sourceLocation: sourceLocation)
        for (id, question) in questions {
            guard let answer = answers?[id] else { continue }
            checkAnswer(answer, to: question, id: id, sourceLocation: sourceLocation)
        }
        let input = body["usage"]?["input_tokens"]?.intValue
        let output = body["usage"]?["output_tokens"]?.intValue
        #expect(
            input.map { $0 > 0 } == true, "usage is \(text(body["usage"]))",
            sourceLocation: sourceLocation)
        #expect(
            output.map { thought ? $0 >= 0 : $0 == 0 } == true, "usage is \(text(body["usage"]))",
            sourceLocation: sourceLocation)
    }

    /// One answer's shape: a noul is `{type, noul}`; a choice is `{type, choice, probabilities,
    /// confidence}` over the question's options in their order, its choice one of them; a score
    /// is `{type, score, legend, probabilities, confidence}` keyed `"0"` to `"n-1"`, its legend
    /// the levels. Every probability, noul and confidence is in [0, 1], and a score in
    /// [0, n - 1].
    static func checkAnswer(
        _ answer: JSONValue, to question: JSONValue, id: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let type = question["type"]?.stringValue
        let keys = Set(answer.objectValue?.keys ?? [])
        let shown = "\(id): \(text(answer))"
        #expect(answer["type"]?.stringValue == type, "\(shown)", sourceLocation: sourceLocation)
        switch type {
        case "noul":
            #expect(keys == ["type", "noul"], "\(shown)", sourceLocation: sourceLocation)
            #expect(isProbability(answer["noul"]), "\(shown)", sourceLocation: sourceLocation)
        case "choice":
            let options = question["criteria"]?.objectValue?.keys ?? []
            #expect(
                keys == ["type", "choice", "probabilities", "confidence"], "\(shown)",
                sourceLocation: sourceLocation)
            #expect(
                answer["choice"]?.stringValue.map(options.contains) == true, "\(shown)",
                sourceLocation: sourceLocation)
            checkDistribution(answer, keys: options, shown: shown, sourceLocation: sourceLocation)
        case "score":
            let levels = question["criteria"]?.arrayValue ?? []
            let indices = levels.indices.map(String.init)
            #expect(
                keys == ["type", "score", "legend", "probabilities", "confidence"], "\(shown)",
                sourceLocation: sourceLocation)
            #expect(
                answer["legend"]?.objectValue?.keys == indices, "\(shown)",
                sourceLocation: sourceLocation)
            for (index, level) in levels.enumerated() where level.stringValue != nil {
                #expect(
                    answer["legend"]?[String(index)] == level, "\(shown)",
                    sourceLocation: sourceLocation)
            }
            let score = answer["score"]?.doubleValue ?? -1
            #expect(
                score >= 0 && score <= Double(levels.count - 1) + 1e-9, "\(shown)",
                sourceLocation: sourceLocation)
            checkDistribution(answer, keys: indices, shown: shown, sourceLocation: sourceLocation)
        default:
            Issue.record("\(id) has the question type \(type ?? "none")")
        }
    }

    /// The `probabilities` of a choice or score answer, keyed `keys` in that order, and its
    /// `confidence`.
    private static func checkDistribution(
        _ answer: JSONValue, keys: [String], shown: String, sourceLocation: SourceLocation
    ) {
        let probabilities = answer["probabilities"]?.objectValue
        #expect(probabilities?.keys == keys, "\(shown)", sourceLocation: sourceLocation)
        #expect(
            probabilities?.values.allSatisfy(isProbability) == true, "\(shown)",
            sourceLocation: sourceLocation)
        #expect(isProbability(answer["confidence"]), "\(shown)", sourceLocation: sourceLocation)
    }

    /// Whether `value` is a number in [0, 1].
    static func isProbability(_ value: JSONValue?) -> Bool {
        guard let number = value?.doubleValue else { return false }
        return number >= 0 && number <= 1
    }

    /// The sum of a distribution's values: a running total, which differs from the compensated
    /// sum of upstream's `sum(p.values())` (D-019) far below the suite's tolerances.
    static func total(_ probabilities: JSONObject) -> Double {
        probabilities.values.reduce(0) { $0 + ($1.doubleValue ?? .nan) }
    }

    /// `value` as compact JSON, for failure messages.
    static func text(_ value: JSONValue?) -> String {
        guard let value else { return "missing" }
        return (try? PythonJSONWriter(options: .compact).string(value)) ?? "\(value)"
    }
}
