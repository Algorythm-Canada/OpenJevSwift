import Foundation
import OpenJevCore
import OpenJevEncoders
import OpenJevTestSupport
import Testing

/// Fixtures/encoders: Verdict's PyTorch reference reads (verdict.json) and the requests they were
/// read from (corpus.json), which Tools/encoders/reference.py wrote through upstream's own code.
///
/// Both files are committed, so the tests that read them run everywhere, CI included.
enum VerdictFixtures {
    /// Fixtures/encoders.
    static let directory = UpstreamFixtures.directory.appendingPathComponent("encoders")

    /// True when both files exist.
    static var available: Bool {
        ["verdict.json", "corpus.json"].allSatisfy {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    /// The message shown when they are missing.
    static let missingMessage: Comment =
        "Fixtures/encoders is missing; it is committed, and Tools/encoders/reference.py writes it"

    /// verdict.json.
    struct Reference: Decodable, Sendable {
        var verdictRevision: String
        var maxLength: Int
        var encoderBatch: Int
        var padTokenID: Int
        var classTokenIndex: Int
        var calibrator: VerdictCalibration
        var reads: [Read]

        private enum CodingKeys: String, CodingKey {
            case generator, calibrator, reads
            case maxLength = "max_length"
            case encoderBatch = "encoder_batch"
            case padTokenID = "pad_token_id"
            case classTokenIndex = "class_token_index"
        }

        private struct Generator: Decodable {
            var verdictRevision: String

            private enum CodingKeys: String, CodingKey {
                case verdictRevision = "verdict_revision"
            }
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            verdictRevision = try container.decode(Generator.self, forKey: .generator)
                .verdictRevision
            maxLength = try container.decode(Int.self, forKey: .maxLength)
            encoderBatch = try container.decode(Int.self, forKey: .encoderBatch)
            padTokenID = try container.decode(Int.self, forKey: .padTokenID)
            classTokenIndex = try container.decode(Int.self, forKey: .classTokenIndex)
            calibrator = try container.decode(VerdictCalibration.self, forKey: .calibrator)
            reads = try container.decode([Read].self, forKey: .reads)
        }
    }

    /// One question's read, in corpus order.
    struct Read: Decodable, Sendable {
        var request: String
        var key: String
        var type: String
        var options: Int
        var k: Int
        var prompt: String
        var truncated: Bool
        var inputIDs: [Int]
        var batch: Int
        var paddedLength: Int
        var temperature: Double
        var temperatureSource: String
        /// The model's first k logits, float32 values written as JSON numbers.
        var logits: [Float]
        var probabilities: [Double]
        /// Upstream's `usage.input_tokens` for the request, on its last question only.
        var requestInputTokens: Int?

        /// `request/key`, for messages.
        var name: String { "\(request)/\(key)" }

        private enum CodingKeys: String, CodingKey {
            case request, key, type, options, k, prompt, truncated, batch, temperature, logits,
                probabilities
            case inputIDs = "input_ids"
            case paddedLength = "padded_length"
            case temperatureSource = "temperature_source"
            case requestInputTokens = "request_input_tokens"
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            request = try container.decode(String.self, forKey: .request)
            key = try container.decode(String.self, forKey: .key)
            type = try container.decode(String.self, forKey: .type)
            options = try container.decode(Int.self, forKey: .options)
            k = try container.decode(Int.self, forKey: .k)
            prompt = try container.decode(String.self, forKey: .prompt)
            truncated = try container.decode(Bool.self, forKey: .truncated)
            inputIDs = try container.decode([Int].self, forKey: .inputIDs)
            batch = try container.decode(Int.self, forKey: .batch)
            paddedLength = try container.decode(Int.self, forKey: .paddedLength)
            temperature = try container.decode(Double.self, forKey: .temperature)
            temperatureSource = try container.decode(String.self, forKey: .temperatureSource)
            logits = try container.decode([Double].self, forKey: .logits).map { Float($0) }
            probabilities = try container.decode([Double].self, forKey: .probabilities)
            requestInputTokens = try container.decodeIfPresent(
                Int.self, forKey: .requestInputTokens)
        }
    }

    /// One request of corpus.json, decoded as the API decodes a body.
    struct CorpusRequest: Sendable {
        var name: String
        var request: SystemOneRequest
    }

    private static let loadedReference = Result {
        try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: directory.appendingPathComponent("verdict.json"))
        )
    }

    private static let loadedCorpus = Result { () throws -> [CorpusRequest] in
        let corpus = try JSONParser().parse(
            Data(contentsOf: directory.appendingPathComponent("corpus.json")))
        guard let requests = corpus["requests"]?.arrayValue else {
            throw FixtureError("corpus.json has no requests array")
        }
        return try requests.map { entry in
            guard let name = entry["name"]?.stringValue, let state = entry["state"],
                let questions = entry["questions"]
            else {
                throw FixtureError("a corpus request lacks its name, state or questions")
            }
            let body: JSONValue = .object([
                "model": .string(KnownEncoderModels.verdict.name), "state": state,
                "questions": questions,
            ])
            return CorpusRequest(name: name, request: try SystemOneRequest(json: body))
        }
    }

    /// verdict.json, parsed once.
    static func reference() throws -> Reference {
        try loadedReference.get()
    }

    /// corpus.json's requests in order, parsed once.
    static func corpus() throws -> [CorpusRequest] {
        try loadedCorpus.get()
    }

    /// The reads of each request, keyed by request name, in corpus order.
    static func readsByRequest() throws -> [String: [Read]] {
        Dictionary(grouping: try reference().reads, by: \.request)
    }

    /// The read questions of a corpus request, as the engine hands them to a backend.
    static func questions(of request: CorpusRequest) throws -> [EncoderQuestion] {
        try EncoderQuestionSchemaBuilder(maxChoices: 24).build(request.request.questions)
            .questions
    }
}

/// The files the opt-in tests read from outside the repository: Verdict's tokenizer and
/// calibrator, and the converted package.
///
/// The folder of converted packages is `OPENJEV_ENCODER_MODELS` when set, else
/// ~/Library/Caches/OpenJevSwift/encoders, where Tools/encoders/convert_verdict.py writes. The
/// tokenizer is read from `{that folder}/verdict-m18-fp16/tokenizer/`, else from the Hugging Face
/// cache snapshot at the pinned revision, the rule ``EncoderPackageStore`` applies. In the iOS
/// Simulator, `~` is the Mac's home. Tests skip with a comment naming `OPENJEV_ENCODER_MODELS`
/// when the files are missing, which is the skip CI's check-test-log.sh accepts.
enum VerdictModelFiles {
    /// The process environment, with the Mac's home standing in for the Simulator's.
    static let environment: [String: String] = {
        var environment = ProcessInfo.processInfo.environment
        if let host = environment["SIMULATOR_HOST_HOME"], environment["HF_HUB_CACHE"] == nil,
            environment["HF_HOME"] == nil
        {
            environment["HF_HUB_CACHE"] = host + "/.cache/huggingface/hub"
        }
        return environment
    }()

    /// The user's home, the Mac's in the Simulator.
    static let home = environment["SIMULATOR_HOST_HOME"] ?? NSHomeDirectory()

    /// The folder of converted packages.
    static let modelsDirectory: URL = {
        if let path = environment[EncoderPackageStore.localModelsVariable], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return URL(fileURLWithPath: home, isDirectory: true)
            .appendingPathComponent("Library/Caches/OpenJevSwift/encoders", isDirectory: true)
    }()

    /// The Hugging Face hub cache.
    static let hubDirectory = EncoderPackageStore.huggingFaceHubDirectory(
        environment: environment)

    /// The package, when it has been converted.
    static var packageDirectory: URL? {
        let package = modelsDirectory.appendingPathComponent(
            EncoderPackageManifest.verdict.package + ".mlpackage", isDirectory: true)
        return FileManager.default.fileExists(atPath: package.path) ? package : nil
    }

    /// The first folder that holds tokenizer.json, tokenizer_config.json and calibrator.json.
    static var tokenizerDirectory: URL? {
        let manifest = EncoderPackageManifest.verdict
        let names = (manifest.tokenizerFiles + [manifest.calibrator]).map(\.path)
        let candidates = [
            modelsDirectory.appendingPathComponent(manifest.package, isDirectory: true)
                .appendingPathComponent("tokenizer", isDirectory: true),
            manifest.checkpoint.snapshot(in: hubDirectory),
        ]
        return candidates.first { folder in
            names.allSatisfy {
                FileManager.default.fileExists(atPath: folder.appendingPathComponent($0).path)
            }
        }
    }

    /// The message shown when the tokenizer is missing.
    static let missingTokenizerMessage = Comment(
        rawValue: "OPENJEV_ENCODER_MODELS is unset or lacks verdict-m18-fp16/tokenizer/, and "
            + "the Hugging Face cache has no heman10x/rlcd-modernbert-151m snapshot at the "
            + "pinned revision; run Tools/encoders/reference.py once, or set "
            + "OPENJEV_ENCODER_MODELS to a folder holding verdict-m18-fp16/tokenizer/")

    /// The message shown when the package or the tokenizer is missing.
    static let missingPackageMessage = Comment(
        rawValue: "\(modelsDirectory.path) (OPENJEV_ENCODER_MODELS, else the converters' "
            + "cache) has no verdict-m18-fp16.mlpackage, or Verdict's tokenizer is missing; run "
            + "Tools/encoders/convert_verdict.py or set OPENJEV_ENCODER_MODELS")
}
