// A port of what upstream's `MlxRuntime._inputs` (razorback16/openjev at dcd2094,
// `openjev/mlx_backend.py:95-114`, Apache-2.0) asks of mlx-vlm 0.6.15 for an image read:
// `Gemma4Processor.apply_chat_template` through `prompt_utils.apply_chat_template`, then
// `Gemma4Processor.__call__`'s placeholder expansion and `mm_token_type_ids` (adapted from
// mlx-vlm, Copyright © 2025 Prince Canuma, MIT). mlx-vlm's `processing_gemma4.py`, which holds
// `Gemma4Processor`, says it is adapted from Hugging Face Transformers (Apache-2.0). See
// THIRD_PARTY.md.

import Foundation
import MLX
import OpenJevCore

/// The prompt half of an image read: the expanded prompt ids and `mm_token_type_ids`, as
/// upstream's MLX backend builds them through mlx-vlm's processor.
///
/// Upstream renders `[system, user]` with the user content `[{"type": "image"}] * n` followed
/// by the state text. mlx-vlm turns every message's content into a list of parts on the way
/// (`prompt_utils.apply_chat_template`), the system message included, and Gemma 4's template
/// writes each system text part as `trim + " "`: an image prompt's system turn ends in a space
/// (token 236743) before `<turn|>`, which a text prompt's does not. On the way mlx-vlm also
/// strips the user's text with Python's `str.strip()` (`extract_text_from_content`), and the
/// template trims it again; the port renders the state as given, and the template's `trim`,
/// which is Python's here (the tokenizer's `templateEnvironment()`, D-054), gives the same text.
/// The template writes one `<|image|>` per image part; the processor then replaces the n-th
/// `<|image|>` of the whole text with `<|image>`, that image's soft tokens of `<|image|>`, and
/// `<image|>`, and tokenizes the result without adding special tokens (the template writes
/// `<bos>`). `mm_token_type_ids` is 1 where an id is the image token, 2 for the video token and 3
/// for the audio token, else 0.
public struct ImagePromptInputs: Sendable, Hashable {
    /// The expanded prompt ids.
    public let ids: [Int]
    /// One entry per id: 1 at the image's soft tokens, 0 elsewhere (2 and 3 for video and
    /// audio tokens, which a text could spell out).
    public let mmTokenTypeIDs: [Int]
    /// The soft tokens of each image, in order.
    public let softTokens: [Int]

    /// The special tokens the expansion uses.
    public struct Tokens: Sendable, Hashable {
        /// `<|image|>`, the placeholder and soft image token (258880).
        public var image = "<|image|>"
        /// `<|image>`, which opens an image block (`boi`, 255999).
        public var beginImage = "<|image>"
        /// `<image|>`, which closes it (`eoi`, 258882).
        public var endImage = "<image|>"
        /// `<|video|>` (258884), marked 2.
        public var video = "<|video|>"
        /// `<|audio|>` (258881), marked 3.
        public var audio = "<|audio|>"

        /// Gemma 4's tokens.
        public init() {}
    }

    /// The chat messages mlx-vlm renders for an image read: the system text as one text part,
    /// then `imageCount` image parts and the state as a text part. mlx-vlm's text parts also
    /// carry the text under `content`, which the template does not read.
    public static func messages(system: String, state: String, imageCount: Int)
        -> [[String: any Sendable]]
    {
        let image: [String: any Sendable] = ["type": "image"]
        let text: [String: any Sendable] = ["type": "text", "text": state, "content": state]
        return [
            [
                "role": "system",
                "content": [["type": "text", "text": system, "content": system]]
                    as [[String: any Sendable]],
            ],
            [
                "role": "user",
                "content": Array(repeating: image, count: imageCount) + [text],
            ],
        ]
    }

    /// The rendered prompt text with each image placeholder expanded, the text the processor
    /// tokenizes.
    ///
    /// - Throws: A ``VisionError`` when the text holds more placeholders than there are images,
    ///   which happens only when the system or state text spells `<|image|>` out; mlx-vlm's
    ///   `re.sub` then runs out of replacements and raises.
    public static func expanded(
        _ text: String, softTokens: [Int], tokens: Tokens = Tokens()
    ) throws(VisionError) -> String {
        var pieces = text.components(separatedBy: tokens.image)
        let placeholders = pieces.count - 1
        guard placeholders <= softTokens.count else {
            throw VisionError(
                "the prompt has \(placeholders) image placeholders for \(softTokens.count) images")
        }
        var out = pieces.removeFirst()
        for (index, piece) in pieces.enumerated() {
            out += tokens.beginImage
            out += String(repeating: tokens.image, count: softTokens[index])
            out += tokens.endImage
            out += piece
        }
        return out
    }

    /// Builds the prompt for a read of `images`, ahead of `state`, under `system`.
    ///
    /// - Parameters:
    ///   - system: The read's system text.
    ///   - state: The state text, which follows the images.
    ///   - softTokens: Each image's soft tokens, from ``Gemma4ImageProcessor``.
    ///   - tokenizer: The checkpoint's tokenizer, whose chat template is rendered.
    ///   - tokens: The special tokens the expansion writes and marks.
    /// - Throws: A ``VisionError`` from the expansion, or the tokenizer's error.
    public init(
        system: String, state: String, softTokens: [Int], tokenizer: SwiftTransformersTokenizer,
        tokens: Tokens = Tokens()
    ) throws {
        let text = try tokenizer.renderChatTemplate(
            messages: Self.messages(system: system, state: state, imageCount: softTokens.count),
            addGenerationPrompt: true, thinking: false)
        let ids = try tokenizer.encode(
            Self.expanded(text, softTokens: softTokens, tokens: tokens), addSpecialTokens: false)
        let marks = [
            (tokenizer.tokenID(of: tokens.image), 1), (tokenizer.tokenID(of: tokens.video), 2),
            (tokenizer.tokenID(of: tokens.audio), 3),
        ]
        var types = [Int: Int]()
        for case (let id?, let mark) in marks {
            types[id] = mark
        }
        self.ids = ids
        self.mmTokenTypeIDs = ids.map { types[$0] ?? 0 }
        self.softTokens = softTokens
    }
}

/// Everything an image read needs besides the canvas: the prompt and the pixels.
public struct ImageReadInputs {
    /// The prompt ids, `mm_token_type_ids` and soft token counts.
    public let prompt: ImagePromptInputs
    /// Each image after the resize and rescale.
    public let images: [Gemma4ImageProcessor.ProcessedImage]
    /// `pixel_values` as mlx-vlm passes them: one `(n, 3, H, W)` array when the images share a
    /// size, else one `(3, H, W)` array per image.
    public let pixelValues: [MLXArray]

    /// Decodes, resizes and rescales each image, upstream's `ImagePrompt.pil` and the
    /// processor's resize, in order.
    ///
    /// - Throws: A ``VisionError`` with the failing image's ``VisionError/imageIndex``.
    public static func process(
        _ parts: [ImagePart], processor: Gemma4ImageProcessor = Gemma4ImageProcessor()
    ) throws(VisionError) -> [Gemma4ImageProcessor.ProcessedImage] {
        var images: [Gemma4ImageProcessor.ProcessedImage] = []
        for (index, part) in parts.enumerated() {
            do throws(VisionError) {
                images.append(try processor.process(RGBImage(decoding: part)))
            } catch {
                throw error.image(index)
            }
        }
        return images
    }

    /// Decodes, resizes and rescales `parts` and builds the prompt that reads them ahead of
    /// `state`, as upstream's `MlxRuntime._inputs` does for an `ImagePrompt`.
    ///
    /// - Throws: A ``VisionError`` with the image's ``VisionError/imageIndex`` for an image that
    ///   does not decode or that the processor cannot size, or the tokenizer's error.
    public init(
        system: String, state: String, parts: [ImagePart], tokenizer: SwiftTransformersTokenizer,
        processor: Gemma4ImageProcessor = Gemma4ImageProcessor()
    ) throws {
        let images = try Self.process(parts, processor: processor)
        self.images = images
        self.prompt = try ImagePromptInputs(
            system: system, state: state, softTokens: images.map(\.softTokens),
            tokenizer: tokenizer)
        self.pixelValues = Gemma4ImageProcessor.pixelValues(images)
    }
}
