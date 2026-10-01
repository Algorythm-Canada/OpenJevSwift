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
    /// No package this device holds takes a sequence of `length` tokens.
    ///
    /// An iPhone holds Laya's packages one per sequence length and fetches each when it first
    /// needs it (D-011). `package` names the smallest package that takes the sequence, which
    /// ``LayaBackend/prefetch(lengths:)`` downloads, and is `nil` when no package does; `held`
    /// names the packages the device holds. Until the download finishes, the app sends the read
    /// to an OpenJev server (D-011 item 6).
    case noPackage(length: Int, package: String?, held: [String])

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
        case .noPackage(let length, let package, let held):
            let holding = held.isEmpty ? "holds none" : "holds " + held.joined(separator: ", ")
            guard let package else {
                return "no package takes a sequence of \(length) tokens; this device \(holding)"
            }
            return "this device holds no package that takes a sequence of \(length) tokens "
                + "(it \(holding)); download \(package) with prefetch(lengths:), or send the "
                + "read to a server"
        }
    }
}
