import CryptoKit
import Foundation

/// The files of a Hugging Face tokenizer directory that ``SwiftTransformersTokenizer`` loads.
///
/// A DiffusionGemma checkpoint ships `tokenizer.json` (the byte-level BPE vocabulary, merges and
/// pipeline), `tokenizer_config.json` (the special token names) and `chat_template.jinja` (the
/// Gemma 4 chat template, which `tokenizer_config.json` does not embed). `config.json` is the
/// model configuration and is optional here. The directory is validated once, when the value is
/// created, so a missing file is reported by name before any loading starts.
public struct TokenizerFiles: Sendable, Hashable {
    /// The file names a tokenizer directory must hold.
    public static let requiredNames = [
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    ]

    /// The directory the files were found in.
    public var directory: URL
    /// `tokenizer.json`.
    public var tokenizerData: URL
    /// `tokenizer_config.json`.
    public var tokenizerConfig: URL
    /// `chat_template.jinja`.
    public var chatTemplate: URL
    /// `config.json`, when the directory has one.
    public var modelConfig: URL?

    /// Locates the files in `directory`.
    ///
    /// - Throws: ``TokenizerFilesError/missingFiles(_:in:)`` naming every required file that is
    ///   absent. Symbolic links count as present when their target exists, which is how the
    ///   Hugging Face cache lays a snapshot out.
    public init(directory: URL) throws {
        let missing = Self.missingNames(in: directory)
        if !missing.isEmpty {
            throw TokenizerFilesError.missingFiles(missing, in: directory)
        }
        self.directory = directory
        tokenizerData = directory.appendingPathComponent("tokenizer.json")
        tokenizerConfig = directory.appendingPathComponent("tokenizer_config.json")
        chatTemplate = directory.appendingPathComponent("chat_template.jinja")
        let config = directory.appendingPathComponent("config.json")
        modelConfig = FileManager.default.fileExists(atPath: config.path) ? config : nil
    }

    /// The required file names that `directory` lacks, in ``requiredNames`` order. Empty when
    /// the directory is complete.
    public static func missingNames(in directory: URL) -> [String] {
        requiredNames.filter {
            !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    /// Checks the files against known SHA-256 digests.
    ///
    /// `digests` maps a file name to the lower-case hexadecimal digest of its contents, as
    /// Fixtures/tokenizer/special_tokens.json records under `files`. Names the directory does not
    /// hold are skipped, so a caller may pass the fixture's digests even when `config.json` is
    /// absent.
    ///
    /// - Throws: ``TokenizerFilesError/digestMismatch(_:expected:actual:)`` for the first file
    ///   whose digest differs, or the error reading a file gave.
    public func verify(digests: [String: String]) throws {
        for (name, expected) in digests.sorted(by: { $0.key < $1.key }) {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                continue
            }
            let actual = try Self.sha256Hex(of: url)
            if actual != expected.lowercased() {
                throw TokenizerFilesError.digestMismatch(name, expected: expected, actual: actual)
            }
        }
    }

    /// The SHA-256 digest of a file's contents as lower-case hexadecimal.
    public static func sha256Hex(of url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// Why a ``TokenizerFiles`` value could not be made or verified.
public enum TokenizerFilesError: Error, Sendable, Hashable, CustomStringConvertible {
    /// Required files are absent from the directory.
    case missingFiles([String], in: URL)
    /// A file's SHA-256 digest differs from the expected one.
    case digestMismatch(String, expected: String, actual: String)

    /// A readable account of the failure.
    public var description: String {
        switch self {
        case .missingFiles(let names, let directory):
            return "\(directory.path) lacks \(names.joined(separator: ", "))"
        case .digestMismatch(let name, let expected, let actual):
            return "\(name): expected SHA-256 \(expected), found \(actual)"
        }
    }
}
