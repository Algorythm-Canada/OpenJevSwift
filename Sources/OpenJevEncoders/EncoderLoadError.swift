import Foundation

/// A model, tokenizer or calibrator that cannot be loaded as the backend needs it.
public enum EncoderLoadError: Error, Sendable, Hashable, CustomStringConvertible {
    /// A file the backend reads does not exist.
    case missingFile(URL)
    /// The tokenizer's vocabulary lacks a special token the backend needs.
    case missingToken(String)
    /// The tokenizer and the package disagree, such as on the padding id.
    case mismatch(String)
    /// A setting is out of range, such as a batch larger than the package's largest function.
    case invalidConfiguration(String)
    /// The OS cannot run the model: the multifunction packages need macOS 15 or iOS 18.
    case unsupportedOperatingSystem(String)

    /// What went wrong, naming the file, token or setting.
    public var description: String {
        switch self {
        case .missingFile(let file):
            return "\(file.path) does not exist"
        case .missingToken(let token):
            return "the tokenizer has no \(token) token"
        case .mismatch(let message), .invalidConfiguration(let message),
            .unsupportedOperatingSystem(let message):
            return message
        }
    }
}
