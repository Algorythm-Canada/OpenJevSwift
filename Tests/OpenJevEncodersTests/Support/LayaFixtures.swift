import Foundation
import OpenJevCore
import OpenJevEncoders
import OpenJevTestSupport
import Testing

/// Fixtures/encoders/laya.json: Laya's PyTorch reference reads, which Tools/encoders/reference.py
/// wrote through upstream's `LayaEngine.read_batch` and laya 0.3.6's `Agent.system_one`.
///
/// The file is parsed with the core's order-preserving parser, because laya's answers are
/// objects whose key order is the option order. It is committed, so the tests that read it run
/// everywhere, CI included.
enum LayaFixtures {
    /// True when laya.json and corpus.json exist.
    static var available: Bool {
        ["laya.json", "corpus.json"].allSatisfy {
            FileManager.default.fileExists(
                atPath: VerdictFixtures.directory.appendingPathComponent($0).path)
        }
    }

    /// The message shown when they are missing.
    static let missingMessage: Comment =
        "Fixtures/encoders is missing; it is committed, and Tools/encoders/reference.py writes it"

    /// laya.json.
    struct Reference: Sendable {
        var layaRepository: String
        var layaRevision: String
        var maxLength: Int
        var headMaxLength: Int
        var encoderBatch: Int
        var specialTokens: LayaSpecialTokens
        /// The checkpoint's `temperature`, as the file holds it.
        var temperatureRaw: [Double]
        /// The checkpoint's `temperature_by_options`, as the file holds it.
        var temperatureByOptionsRaw: [String: Double]
        /// The temperatures laya applies, clamped.
        var temperature: [Double]
        var temperatureByOptions: [String: Double]
        var clamp: [Double]
        var qtypes: [String: Int]
        /// Each request's state as build_sequence tokenizes it.
        var stateTexts: [String: String]
        var reads: [Read]

        /// The calibration of the file's raw temperatures and lengths.
        var calibration: LayaCalibration {
            LayaCalibration(
                temperatures: temperatureRaw, temperaturesByOptions: temperatureByOptionsRaw,
                maxLength: maxLength, headMaxLength: headMaxLength)
        }
    }

    /// One question's read, in corpus order.
    struct Read: Sendable {
        var request: String
        var key: String
        var kind: QuestionKind
        var options: Int
        /// What upstream passed to laya: `{type, instructions, criteria}`.
        var layaQuestion: JSONValue
        /// The head build_sequence tokenized.
        var head: String
        /// Each option as build_sequence tokenized it, after its marker.
        var optionTexts: [String]
        var ids: [Int]
        var markers: [Int]
        var qtype: Int
        var batch: Int
        var paddedLength: Int
        var stateTokens: Int
        var stateTokensRead: Int
        var truncated: Bool
        var bucket: String
        var temperature: Double
        /// The scorer at the markers, float32 values written as JSON numbers.
        var logits: [Float]
        /// The softmax before laya rounds, float32 values.
        var probabilitiesUnrounded: [Double]
        /// laya's own answer.
        var answer: JSONValue
        /// The distribution upstream publishes, in the caller's option order.
        var probabilities: [Double]
        /// Upstream's `usage.input_tokens` for the request, on its last question only.
        var requestInputTokens: Int?

        /// `request/key`, for messages.
        var name: String { "\(request)/\(key)" }

        /// laya's rounded answer: a choice's or a score's `probabilities` in laya's option
        /// order, or a noul's `[noul]`, its rounded P(true).
        var answerValues: [Double] {
            get throws {
                if kind == .noul {
                    return [try answer.number("noul")]
                }
                guard let byOption = answer["probabilities"]?.objectValue else {
                    throw FixtureError("\(name): the answer has no probabilities")
                }
                return try byOption.values.map { value in
                    guard let number = value.doubleValue else {
                        throw FixtureError("\(name): a probability is not a number")
                    }
                    return number
                }
            }
        }

        /// The recorded scores at the markers and `filler` everywhere else: a row of the
        /// model's output for this sequence.
        func scoreRow(filler: Float = 0) -> [Float] {
            var row = [Float](repeating: filler, count: ids.count)
            for (marker, logit) in zip(markers, logits) {
                row[marker] = logit
            }
            return row
        }

        /// The sequence's pieces before build_sequence joined them: the head's ids, each
        /// option's ids after its marker, and the state's ids as far as this question read them.
        var pieces: (head: [Int], options: [[Int]], state: [Int]) {
            let stateStart = ids.count - 1 - stateTokensRead
            let optionsEnd = stateStart - 1
            let head = Array(ids[1..<(markers[0] - 1)])
            let ends = Array(markers.dropFirst()) + [optionsEnd]
            let options = zip(markers, ends).map { Array(ids[($0 + 1)..<$1]) }
            return (head, options, Array(ids[stateStart..<(ids.count - 1)]))
        }

        init(json: JSONValue) throws {
            request = try json.string("request")
            key = try json.string("key")
            let type = try json.string("type")
            guard let kind = QuestionKind(rawValue: type) else {
                throw FixtureError("a read has the type \(type)")
            }
            self.kind = kind
            options = try json.int("options")
            layaQuestion = try json.value("laya_question")
            let texts = try json.value("texts")
            head = try texts.string("head")
            optionTexts =
                try texts.value("options").arrayValue?.map {
                    try unwrapString($0, "texts.options")
                } ?? []
            ids = try json.ints("ids")
            markers = try json.ints("markers")
            qtype = try json.int("qtype")
            batch = try json.int("batch")
            paddedLength = try json.int("padded_length")
            stateTokens = try json.int("state_tokens")
            stateTokensRead = try json.int("state_tokens_read")
            truncated = try json.value("truncated").boolValue ?? false
            bucket = try json.string("bucket")
            temperature = try json.number("temperature")
            logits = try json.numbers("logits").map { Float($0) }
            probabilitiesUnrounded = try json.numbers("probabilities_unrounded")
            answer = try json.value("answer")
            probabilities = try json.numbers("probabilities")
            requestInputTokens = json["request_input_tokens"]?.intValue
        }
    }

    private static let loadedReference = Result { () throws -> Reference in
        let json = try JSONParser().parse(
            Data(contentsOf: VerdictFixtures.directory.appendingPathComponent("laya.json")))
        let generator = try json.value("generator")
        let special = try json.value("special_tokens")
        let calibration = try json.value("calibration")
        func numbers(_ object: JSONValue) throws -> [String: Double] {
            guard let entries = object.objectValue else {
                throw FixtureError("laya.json: a temperature table is not an object")
            }
            var out: [String: Double] = [:]
            for (key, value) in entries {
                out[key] = try unwrapNumber(value, key)
            }
            return out
        }
        var qtypes: [String: Int] = [:]
        for (name, index) in try json.value("qtypes").objectValue ?? [:] {
            qtypes[name] = index.intValue
        }
        var stateTexts: [String: String] = [:]
        for (name, text) in try json.value("state_texts").objectValue ?? [:] {
            stateTexts[name] = try unwrapString(text, "state_texts")
        }
        return Reference(
            layaRepository: try generator.string("laya_repo"),
            layaRevision: try generator.string("laya_revision"),
            maxLength: try json.int("max_len"),
            headMaxLength: try json.int("head_max_len"),
            encoderBatch: try json.int("encoder_batch"),
            specialTokens: LayaSpecialTokens(
                classToken: try special.int("cls"), separator: try special.int("sep"),
                mask: try special.int("mask"), padding: try special.int("pad")),
            temperatureRaw: try calibration.numbers("temperature_raw"),
            temperatureByOptionsRaw: try numbers(calibration.value("temperature_by_options_raw")),
            temperature: try calibration.numbers("temperature"),
            temperatureByOptions: try numbers(calibration.value("temperature_by_options")),
            clamp: try calibration.numbers("clamp"),
            qtypes: qtypes,
            stateTexts: stateTexts,
            reads: try (json.value("reads").arrayValue ?? []).map(Read.init(json:)))
    }

    /// laya.json, parsed once.
    static func reference() throws -> Reference {
        try loadedReference.get()
    }

    /// The reads of each request, keyed by request name, in corpus order.
    static func readsByRequest() throws -> [String: [Read]] {
        Dictionary(grouping: try reference().reads, by: \.request)
    }

    /// corpus.json's requests in order, each asking `laya-1.0`.
    static func corpus() throws -> [EncoderCorpus.Request] {
        try EncoderCorpus.requests(model: KnownEncoderModels.laya.name)
    }

    /// The read questions of a corpus request, as the engine hands them to Laya's backend.
    static func questions(of request: EncoderCorpus.Request) throws -> [EncoderQuestion] {
        try EncoderCorpus.questions(of: request, maxChoices: 255)
    }

    /// True where the port's float32 arithmetic is laya's bit for bit: on Apple silicon, where
    /// Swift's `exp(Float)` is the libm `expf` that numpy called when laya.json was recorded.
    /// Elsewhere numpy and libm may differ in the last bit, and a rounded probability may then
    /// land one 4-decimal step away.
    static var arithmeticIsLayas: Bool {
        #if arch(arm64)
            return true
        #else
            return false
        #endif
    }

    /// Whether a published distribution is the recorded one: exactly where
    /// ``arithmeticIsLayas``, else within one rounding step.
    static func matchesPublished(_ measured: [Double], _ recorded: [Double]) -> Bool {
        guard measured.count == recorded.count else { return false }
        if arithmeticIsLayas {
            return measured == recorded
        }
        return zip(measured, recorded).allSatisfy { abs($0 - $1) <= 1.5e-4 }
    }
}

/// A tokenizer that hands back the ids laya.json recorded for each text: the head, each option
/// and each request's state, split out of the recorded sequences.
///
/// A recorded piece may already be cut (an option to 48 tokens, a head to its budget, a state to
/// the room it had), but always to a prefix of the text's ids, and every cut ``LayaSequence``
/// makes is a prefix again; keeping the longest piece seen for a text therefore rebuilds every
/// recorded sequence. A text never recorded gives `[-1]`, which ``ReplayModel`` refuses.
struct LayaReplayTokenizer: LayaTokenizing {
    let ids: [String: [Int]]
    let specialTokens: LayaSpecialTokens

    init(_ reference: LayaFixtures.Reference) {
        var ids: [String: [Int]] = [:]
        func keep(_ text: String, _ pieceIDs: [Int]) {
            if pieceIDs.count > ids[text]?.count ?? -1 {
                ids[text] = pieceIDs
            }
        }
        for read in reference.reads {
            let pieces = read.pieces
            keep(read.head, pieces.head)
            for (text, option) in zip(read.optionTexts, pieces.options) {
                keep(text, option)
            }
            if let state = reference.stateTexts[read.request] {
                keep(state, pieces.state)
            }
        }
        self.ids = ids
        self.specialTokens = reference.specialTokens
    }

    func encode(_ text: String) -> [Int] {
        ids[text] ?? [-1]
    }
}

extension ReplayModel {
    /// A model that answers each recorded Laya sequence with a score per position: the recorded
    /// scores at its markers, `filler` everywhere else.
    init(laya reads: [LayaFixtures.Read], filler: Float = 0) {
        self.init(rows: reads.map { ($0.ids, $0.scoreRow(filler: filler)) })
    }
}

/// The files the opt-in Laya tests read from outside the repository: the checkpoint's tokenizer
/// and configuration file, and the converted packages, found as ``EncoderModelFiles`` says.
enum LayaModelFiles {
    /// The tokenizer folder and rl_agent_config.json: `{models}/laya-m18-fp16/tokenizer/`, else
    /// the checkpoint's Hugging Face snapshot, `tokenizer/` and its root.
    static var tokenizer: EncoderTokenizerLocations? {
        EncoderModelFiles.tokenizer(for: .laya)
    }

    /// The tokenizer folder and rl_agent_config.json where the iPhone's set looks for them
    /// (``LayaBackend/load(from:packageSet:)``): `{models}/laya-f18-b1s128-fp16/tokenizer/`,
    /// else the checkpoint's Hugging Face snapshot.
    static var byLengthTokenizer: EncoderTokenizerLocations? {
        EncoderPackageManifest.layaByLength[128].flatMap(EncoderModelFiles.tokenizer(for:))
    }

    /// The Mac's package.
    static var multifunctionPackage: URL? {
        EncoderModelFiles.package(EncoderPackageManifest.laya.package)
    }

    /// The iPhone's packages that exist, by sequence length.
    static var packagesByLength: [Int: URL] {
        var packages: [Int: URL] = [:]
        for (length, manifest) in EncoderPackageManifest.layaByLength {
            packages[length] = EncoderModelFiles.package(manifest.package)
        }
        return packages
    }

    /// The message shown when the tokenizer is missing.
    static let missingTokenizerMessage = Comment(
        rawValue: "OPENJEV_ENCODER_MODELS is unset or lacks laya-m18-fp16/tokenizer/ (and "
            + "laya-f18-b1s128-fp16/tokenizer/ for the iPhone's set), and the Hugging Face "
            + "cache has no convaiinnovations/laya-typed-decisions snapshot at the pinned "
            + "revision; run Tools/encoders/reference.py once, or set OPENJEV_ENCODER_MODELS to "
            + "a folder holding those tokenizer folders")

    /// The message shown when a package or the tokenizer is missing.
    static let missingPackageMessage = Comment(
        rawValue: "\(EncoderModelFiles.modelsDirectory.path) (OPENJEV_ENCODER_MODELS, else the "
            + "converters' cache) lacks laya-m18-fp16.mlpackage or one of "
            + "laya-f18-b1s128-fp16.mlpackage to laya-f18-b1s1024-fp16.mlpackage, or Laya's "
            + "tokenizer is missing; run Tools/encoders/convert_laya.py (with --only for the "
            + "per-length packages) or set OPENJEV_ENCODER_MODELS")
}

// Reading a fixture's JSON values, with an error that names what was missing.
extension JSONValue {
    fileprivate func value(_ key: String) throws -> JSONValue {
        guard let value = self[key] else {
            throw FixtureError("a fixture object lacks \(key)")
        }
        return value
    }

    fileprivate func string(_ key: String) throws -> String {
        try unwrapString(value(key), key)
    }

    fileprivate func int(_ key: String) throws -> Int {
        guard let number = try value(key).intValue else {
            throw FixtureError("\(key) is not an integer")
        }
        return number
    }

    fileprivate func number(_ key: String) throws -> Double {
        try unwrapNumber(value(key), key)
    }

    fileprivate func ints(_ key: String) throws -> [Int] {
        guard let values = try value(key).arrayValue else {
            throw FixtureError("\(key) is not an array")
        }
        return try values.map {
            guard let number = $0.intValue else {
                throw FixtureError("\(key) holds a value that is not an integer")
            }
            return number
        }
    }

    fileprivate func numbers(_ key: String) throws -> [Double] {
        guard let values = try value(key).arrayValue else {
            throw FixtureError("\(key) is not an array")
        }
        return try values.map { try unwrapNumber($0, key) }
    }
}

private func unwrapString(_ value: JSONValue, _ name: String) throws -> String {
    guard let text = value.stringValue else {
        throw FixtureError("\(name) is not a string")
    }
    return text
}

private func unwrapNumber(_ value: JSONValue, _ name: String) throws -> Double {
    guard let number = value.doubleValue else {
        throw FixtureError("\(name) is not a number")
    }
    return number
}
