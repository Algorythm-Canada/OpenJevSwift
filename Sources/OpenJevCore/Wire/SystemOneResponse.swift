// The shapes follow upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`,
// function `to_answer`, and the response built by `openjev/api.py`, route `systemone`.
// Apache-2.0. See THIRD_PARTY.md.

/// The answer to one question, with Jev's exact key sets.
///
/// Computing an answer from label probabilities (the argmax, the expected score and the
/// confidence) belongs to issue #16. This type holds the result and writes it:
///
/// - noul: `{"type", "noul"}`
/// - choice: `{"type", "choice", "probabilities", "confidence"}`
/// - score: `{"type", "score", "legend", "probabilities", "confidence"}`, where `legend` and
///   `probabilities` are objects keyed `"0"` to `"N-1"`.
///
/// Every number is written as a Python float, so `1.0` stays `1.0`.
public enum Answer: Sendable, Hashable, WireEncodable {
    /// The probability that the answer is yes.
    case noul(Double)
    /// The chosen option name, every option's probability in the question's criteria order, and
    /// the confidence.
    case choice(choice: String, probabilities: OrderedMap<Double>, confidence: Double)
    /// The expected level, each level's original description, each level's probability, and the
    /// confidence. `legend` and `probabilities` have one entry per level.
    case score(score: Double, legend: [JSONValue], probabilities: [Double], confidence: Double)

    /// The wire tag: `noul`, `choice` or `score`.
    public var type: String {
        switch self {
        case .noul: return "noul"
        case .choice: return "choice"
        case .score: return "score"
        }
    }

    /// The answer in upstream's key order.
    public var json: JSONValue {
        switch self {
        case .noul(let probability):
            return ["type": "noul", "noul": .float(probability)]
        case .choice(let choice, let probabilities, let confidence):
            let entries = probabilities.map { ($0.key, JSONValue.float($0.value)) }
            return [
                "type": "choice",
                "choice": .string(choice),
                "probabilities": .object(JSONObject(uniqueKeysWithValues: entries)),
                "confidence": .float(confidence),
            ]
        case .score(let score, let legend, let probabilities, let confidence):
            let legendEntries = legend.enumerated().map { (String($0.offset), $0.element) }
            let probabilityEntries = probabilities.enumerated().map {
                (String($0.offset), JSONValue.float($0.element))
            }
            return [
                "type": "score",
                "score": .float(score),
                "legend": .object(JSONObject(uniqueKeysWithValues: legendEntries)),
                "probabilities": .object(JSONObject(uniqueKeysWithValues: probabilityEntries)),
                "confidence": .float(confidence),
            ]
        }
    }

    /// Decodes an answer, requiring its exact key set.
    ///
    /// A score's `legend` and `probabilities` must be keyed `"0"` to `"N-1"` in order, with the
    /// same count.
    public init(json: JSONValue) throws(WireDecodingError) {
        try self.init(json: json, path: [])
    }

    init(json: JSONValue, path: [LocComponent]) throws(WireDecodingError) {
        let object = try json.requireObject(at: path)
        let type = try object.require("type", at: path).requireString(at: path + ["type"])
        switch type {
        case "noul":
            try object.requireKeys(["type", "noul"], at: path)
            self = .noul(try object.require("noul", at: path).requireDouble(at: path + ["noul"]))
        case "choice":
            try object.requireKeys(["type", "choice", "probabilities", "confidence"], at: path)
            let choice = try object.require("choice", at: path).requireString(at: path + ["choice"])
            let probabilitiesPath = path + ["probabilities"]
            let probabilityObject = try object.require("probabilities", at: path)
                .requireObject(at: probabilitiesPath)
            var probabilities = OrderedMap<Double>()
            for (key, value) in probabilityObject {
                let number = try value.requireDouble(at: probabilitiesPath + [.key(key)])
                probabilities.updateValue(number, forKey: key)
            }
            let confidence = try object.require("confidence", at: path)
                .requireDouble(at: path + ["confidence"])
            self = .choice(choice: choice, probabilities: probabilities, confidence: confidence)
        case "score":
            try object.requireKeys(
                ["type", "score", "legend", "probabilities", "confidence"], at: path)
            let score = try object.require("score", at: path).requireDouble(at: path + ["score"])
            let legend = try Self.levels(
                of: object.require("legend", at: path), at: path + ["legend"])
            let probabilityValues = try Self.levels(
                of: object.require("probabilities", at: path), at: path + ["probabilities"])
            guard legend.count == probabilityValues.count else {
                throw WireDecodingError(
                    path: path, reason: "legend and probabilities have different level counts")
            }
            var probabilities: [Double] = []
            for (index, value) in probabilityValues.enumerated() {
                probabilities.append(
                    try value.requireDouble(at: path + ["probabilities", .key(String(index))]))
            }
            let confidence = try object.require("confidence", at: path)
                .requireDouble(at: path + ["confidence"])
            self = .score(
                score: score, legend: legend, probabilities: probabilities, confidence: confidence)
        default:
            throw WireDecodingError(path: path + ["type"], reason: "unknown answer type \(type)")
        }
    }

    /// The values of an object keyed `"0"` to `"N-1"`, in that order.
    private static func levels(
        of value: JSONValue, at path: [LocComponent]
    ) throws(WireDecodingError) -> [JSONValue] {
        let object = try value.requireObject(at: path)
        for (index, key) in object.keys.enumerated() where key != String(index) {
            throw WireDecodingError(path: path + [.key(key)], reason: "expected the key \(index)")
        }
        return object.values
    }
}

/// Token counts for a request.
///
/// `outputTokens` is 0 except after a `think` read, as in Jev's contract. `Codable` is offered
/// for convenience; the wire form is ``json``, written by ``WireEncoder``.
public struct Usage: Sendable, Hashable, Codable, WireEncodable {
    /// The input tokens billed for the reads.
    public var inputTokens: Int
    /// The thought tokens generated before the read.
    public var outputTokens: Int

    /// Creates usage counts.
    public init(inputTokens: Int, outputTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }

    /// `{"input_tokens", "output_tokens"}`.
    public var json: JSONValue {
        ["input_tokens": JSONValue(inputTokens), "output_tokens": JSONValue(outputTokens)]
    }

    /// Decodes usage counts, requiring both keys.
    public init(json: JSONValue) throws(WireDecodingError) {
        try self.init(json: json, path: [])
    }

    init(json: JSONValue, path: [LocComponent]) throws(WireDecodingError) {
        let object = try json.requireObject(at: path)
        try object.requireKeys(["input_tokens", "output_tokens"], at: path)
        inputTokens = try object.require("input_tokens", at: path)
            .requireInt(at: path + ["input_tokens"])
        outputTokens = try object.require("output_tokens", at: path)
            .requireInt(at: path + ["output_tokens"])
    }
}

/// A `POST /v1/systemone` response body.
public struct SystemOneResponse: Sendable, Hashable, WireEncodable {
    /// The served model version, such as `openjev-0.1`, not the alias the request used.
    public var model: String
    /// One answer per question, in the request's question order.
    public var answers: OrderedMap<Answer>
    /// Token counts.
    public var usage: Usage

    /// Creates a response.
    public init(model: String, answers: OrderedMap<Answer>, usage: Usage) {
        self.model = model
        self.answers = answers
        self.usage = usage
    }

    /// `{"model", "answers", "usage"}`.
    public var json: JSONValue {
        let entries = answers.map { ($0.key, $0.value.json) }
        return [
            "model": .string(model),
            "answers": .object(JSONObject(uniqueKeysWithValues: entries)),
            "usage": usage.json,
        ]
    }

    /// Decodes a response, requiring its exact key set and every answer's.
    public init(json: JSONValue) throws(WireDecodingError) {
        let object = try json.requireObject(at: [])
        try object.requireKeys(["model", "answers", "usage"], at: [])
        model = try object.require("model", at: []).requireString(at: ["model"])
        let answerObject = try object.require("answers", at: []).requireObject(at: ["answers"])
        var answers = OrderedMap<Answer>()
        for (key, value) in answerObject {
            answers.updateValue(try Answer(json: value, path: ["answers", .key(key)]), forKey: key)
        }
        self.answers = answers
        usage = try Usage(json: object.require("usage", at: []), path: ["usage"])
    }
}
