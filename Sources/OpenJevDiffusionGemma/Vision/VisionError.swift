/// Why an image could not be turned into model inputs.
///
/// Upstream raises inside PIL or the processor for these cases, which its server turns into a
/// 500; the runtime that wires images in (#47) decides how the wire reports them.
public struct VisionError: Error, Sendable, Hashable, CustomStringConvertible {
    /// What went wrong, in a sentence.
    public let message: String

    /// Creates an error with its message.
    public init(_ message: String) {
        self.message = message
    }

    public var description: String { message }
}
