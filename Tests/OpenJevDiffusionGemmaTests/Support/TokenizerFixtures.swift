import Darwin
import Foundation
import OpenJevCore
import OpenJevDiffusionGemma
import Testing

/// Locates the pinned tokenizer files and the fixtures the parity tests compare them with.
///
/// The tokenizer directory is `OPENJEV_TEST_TOKENIZER` when set, else the Hugging Face cache
/// snapshot the fixture generator (Tools/fixtures) downloads. Tests are disabled with a message
/// when the directory lacks `tokenizer.json`, `tokenizer_config.json` or `chat_template.jinja`,
/// or when a fixture file is missing. The tokenizer is loaded once per process.
enum TokenizerFixtures {
    /// The Hugging Face cache snapshot of `mlx-community/diffusiongemma-26B-A4B-it-4bit` at the
    /// pinned revision, which `make fixtures` fills.
    static let cachedSnapshot = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent(".cache/huggingface/hub")
        .appendingPathComponent("models--mlx-community--diffusiongemma-26B-A4B-it-4bit")
        .appendingPathComponent("snapshots/a7a81407613811e8ba63af92ac0d852b809e191f")

    /// The environment variable that names another tokenizer directory.
    static let environmentVariable = "OPENJEV_TEST_TOKENIZER"

    /// The tokenizer directory the tests use.
    static let tokenizerDirectory: URL = {
        if let path = ProcessInfo.processInfo.environment[environmentVariable], !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return cachedSnapshot
    }()

    /// The message shown when the tokenizer directory is incomplete.
    static let missingTokenizerMessage = Comment(
        rawValue: "The tokenizer files are missing from \(tokenizerDirectory.path) (set "
            + "\(environmentVariable) or run make fixtures); the parity tests are skipped")

    /// True when the tokenizer directory holds the three required files.
    static var tokenizerAvailable: Bool {
        TokenizerFiles.missingNames(in: tokenizerDirectory).isEmpty
    }

    /// Fixtures/, found relative to this source file.
    static let fixturesDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Support
        .deletingLastPathComponent()  // OpenJevDiffusionGemmaTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repository root
        .appendingPathComponent("Fixtures")

    /// The fixture files the parity tests read, relative to Fixtures/.
    static let fixtureFiles = [
        "tokenizer/corpus.json", "tokenizer/engine_encodings.json",
        "tokenizer/special_tokens.json", "labels.json", "chat-prompts/prompts.json",
    ]

    /// The message shown when a fixture file is missing.
    static let missingFixturesMessage: Comment =
        "A tokenizer fixture is missing; run make upstream, make fixtures-venv and make fixtures"

    /// True when every fixture file exists.
    static var fixturesAvailable: Bool {
        fixtureFiles.allSatisfy {
            FileManager.default.fileExists(
                atPath: fixturesDirectory.appendingPathComponent($0).path)
        }
    }

    /// True when both the tokenizer and the fixtures are available.
    static var available: Bool { tokenizerAvailable && fixturesAvailable }

    /// The message shown when either is missing.
    static var missingMessage: Comment {
        tokenizerAvailable ? missingFixturesMessage : missingTokenizerMessage
    }

    /// The named fixture file, parsed. The path is relative to Fixtures/.
    static func load(_ path: String) throws -> JSONValue {
        try JSONParser().parse(Data(contentsOf: fixturesDirectory.appendingPathComponent(path)))
    }

    /// The `cases` array of the named fixture file.
    static func cases(_ path: String) throws -> [JSONValue] {
        try #require(load(path)["cases"]?.arrayValue)
    }

    /// The integers of a JSON array.
    static func ints(_ value: JSONValue?) throws -> [Int] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.intValue) }
    }

    /// The strings of a JSON array.
    static func strings(_ value: JSONValue?) throws -> [String] {
        let values = try #require(value?.arrayValue)
        return try values.map { try #require($0.stringValue) }
    }

    /// The `files` digests of special_tokens.json, file name to SHA-256.
    static func recordedDigests() throws -> [String: String] {
        let files = try #require(load("tokenizer/special_tokens.json")["files"]?.objectValue)
        var digests: [String: String] = [:]
        for (name, entry) in files {
            digests[name] = try #require(entry["sha256"]?.stringValue)
        }
        return digests
    }

    /// The one load of the real tokenizer this process makes. Loading verifies the files
    /// against the digests the fixtures were generated from, so a parity failure is never a
    /// different tokenizer.
    private static let loading = Task<SwiftTransformersTokenizer, any Error> {
        let files = try TokenizerFiles(directory: tokenizerDirectory)
        try files.verify(digests: try recordedDigests())
        return try await SwiftTransformersTokenizer.load(from: files)
    }

    /// The loaded tokenizer, shared by every test.
    static func tokenizer() async throws -> SwiftTransformersTokenizer {
        try await loading.value
    }
}

/// Writes the figures the spike report needs to files under the temporary directory, so they
/// can be read after a test run without parsing the test log.
///
/// Each test process writes to its own directory, named after its process id and start time,
/// so one directory holds exactly one run and never mixes with an earlier one. Appends are
/// serialized with a lock because Swift Testing runs tests in parallel.
enum SpikeReport {
    /// The directory this process writes to: `openjev-spikes/<pid>-<start time>` under the
    /// temporary directory.
    static let directory: URL = {
        let process = ProcessInfo.processInfo
        let started = Int(Date().timeIntervalSince1970)
        return URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("openjev-spikes")
            .appendingPathComponent("\(process.processIdentifier)-\(started)")
    }()

    /// Serializes appends from parallel tests.
    private static let lock = NSLock()

    /// Appends `text` to `name`.txt in the report directory.
    static func record(_ name: String, _ text: String) {
        lock.lock()
        defer { lock.unlock() }
        do {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let url = directory.appendingPathComponent("\(name).txt")
            if !FileManager.default.fileExists(atPath: url.path) {
                try Data().write(to: url)
            }
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: Data((text + "\n").utf8))
        } catch {
            print("SpikeReport: could not write \(name): \(error)")
        }
    }
}

/// The test process's memory use, read from the kernel, for measuring a load the module does
/// not measure itself (the mlx-swift-lm loader path).
struct ProcessMemory {
    /// Resident memory in bytes, `mach_task_basic_info.resident_size`.
    var residentBytes: Int
    /// Peak resident memory in bytes, `rusage.ru_maxrss`, which macOS reports in bytes.
    var peakResidentBytes: Int

    /// The current figures. A failed kernel call gives zero for its figure.
    static func current() -> ProcessMemory {
        var usage = rusage()
        let peak = getrusage(RUSAGE_SELF, &usage) == 0 ? Int(usage.ru_maxrss) : 0
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        let resident = result == KERN_SUCCESS ? Int(info.resident_size) : 0
        return ProcessMemory(residentBytes: resident, peakResidentBytes: peak)
    }

    /// `bytes` as megabytes with one decimal.
    static func megabytes(_ bytes: Int) -> String {
        String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
    }
}
