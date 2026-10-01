import Foundation
import Tokenizers

/// A checkpoint's tokenizer folder, which swift-transformers loads: tokenizer.json and
/// tokenizer_config.json, as Verdict's checkpoint holds them at its root and Laya's under
/// `tokenizer/`.
enum TokenizerFolder {
    /// The files the folder must hold.
    static let fileNames = ["tokenizer.json", "tokenizer_config.json"]

    /// The tokenizer of a folder that holds both files.
    ///
    /// - Throws: ``EncoderLoadError/missingFile(_:)`` for a missing file, and swift-transformers'
    ///   errors for a file it cannot read.
    static func load(_ directory: URL) async throws -> any Tokenizer {
        for name in fileNames {
            let file = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: file.path) else {
                throw EncoderLoadError.missingFile(file)
            }
        }
        return try await AutoTokenizer.from(modelFolder: directory)
    }
}
