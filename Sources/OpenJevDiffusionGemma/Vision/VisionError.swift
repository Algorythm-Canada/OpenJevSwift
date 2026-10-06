/// Why an image could not be turned into model inputs.
///
/// Upstream raises inside PIL or the processor for these cases, which its server answers with a
/// bare 500. The DiffusionGemma runtime answers a request's image that fails here with a 400
/// naming the image, `["body", "images", i]` (D-054).
public struct VisionError: Error, Sendable, Hashable, CustomStringConvertible {
    /// What went wrong, in a sentence.
    public let message: String
    /// The position of the image in the request's `images`, when the error is about one image.
    public var imageIndex: Int?

    /// Creates an error with its message.
    public init(_ message: String, imageIndex: Int? = nil) {
        self.message = message
        self.imageIndex = imageIndex
    }

    /// The same error about the image at `index`.
    public func image(_ index: Int) -> VisionError {
        VisionError(message, imageIndex: index)
    }

    /// The message.
    public var description: String { message }
}
