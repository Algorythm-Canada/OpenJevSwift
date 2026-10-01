import Foundation
import OpenJevCore
import OpenJevEncoders
import OpenJevTestSupport

/// Fixtures/encoders/corpus.json: the 26 requests both encoders' references were read from,
/// decoded as the API decodes a body.
enum EncoderCorpus {
    /// One request of the corpus.
    struct Request: Sendable {
        var name: String
        var request: SystemOneRequest
    }

    /// Each request's name, state and questions, parsed once.
    private static let loaded = Result {
        () throws -> [(name: String, state: JSONValue, questions: JSONValue)] in
        let corpus = try JSONParser().parse(
            Data(contentsOf: VerdictFixtures.directory.appendingPathComponent("corpus.json")))
        guard let requests = corpus["requests"]?.arrayValue else {
            throw FixtureError("corpus.json has no requests array")
        }
        return try requests.map { entry in
            guard let name = entry["name"]?.stringValue, let state = entry["state"],
                let questions = entry["questions"]
            else {
                throw FixtureError("a corpus request lacks its name, state or questions")
            }
            return (name, state, questions)
        }
    }

    /// The requests in order, each asking `model`.
    static func requests(model: String) throws -> [Request] {
        try loaded.get().map { entry in
            let body: JSONValue = .object([
                "model": .string(model), "state": entry.state, "questions": entry.questions,
            ])
            return Request(name: entry.name, request: try SystemOneRequest(json: body))
        }
    }

    /// The read questions of a request, as the engine hands them to a backend with this option
    /// limit.
    static func questions(of request: Request, maxChoices: Int) throws -> [EncoderQuestion] {
        try EncoderQuestionSchemaBuilder(maxChoices: maxChoices).build(request.request.questions)
            .questions
    }
}

/// Where the opt-in tests find what is not in the repository: the converted packages and the
/// checkpoints' tokenizers.
///
/// The folder of converted packages is `OPENJEV_ENCODER_MODELS` when set, else
/// ~/Library/Caches/OpenJevSwift/encoders, where Tools/encoders' converters write. A tokenizer is
/// read from `{that folder}/{package}/tokenizer/`, else from the Hugging Face cache snapshot at
/// the pinned revision, the rule ``EncoderPackageStore`` applies. In the iOS Simulator, `~` is
/// the Mac's home. Tests skip with a comment naming `OPENJEV_ENCODER_MODELS` when the files are
/// missing, which is the skip CI's check-test-log.sh accepts.
enum EncoderModelFiles {
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

    /// A store that reads the local models and the Hugging Face cache and downloads nothing.
    static var store: EncoderPackageStore {
        EncoderPackageStore(
            directory: FileManager.default.temporaryDirectory.appendingPathComponent(
                "OpenJevEncodersTests-unused"),
            localModelsDirectory: modelsDirectory, huggingFaceHubDirectory: hubDirectory)
    }

    /// A converted package, when it exists.
    static func package(_ name: String) -> URL? {
        let package = modelsDirectory.appendingPathComponent(name + ".mlpackage", isDirectory: true)
        return FileManager.default.fileExists(atPath: package.path) ? package : nil
    }

    /// The first place that holds a manifest's tokenizer files and calibration file:
    /// `{models}/{package}/tokenizer/`, else the checkpoint's Hugging Face snapshot.
    static func tokenizer(for manifest: EncoderPackageManifest) -> EncoderTokenizerLocations? {
        let folder = modelsDirectory.appendingPathComponent(manifest.package, isDirectory: true)
            .appendingPathComponent("tokenizer", isDirectory: true)
        let candidates = [
            EncoderTokenizerLocations(
                tokenizerDirectory: folder,
                calibratorFile: folder.appendingPathComponent(manifest.calibrator.path)),
            manifest.checkpointFiles(in: hubDirectory),
        ]
        return candidates.first { candidate in
            FileManager.default.fileExists(atPath: candidate.calibratorFile.path)
                && manifest.tokenizerFiles.allSatisfy {
                    FileManager.default.fileExists(
                        atPath: candidate.tokenizerDirectory.appendingPathComponent($0.path).path)
                }
        }
    }
}
