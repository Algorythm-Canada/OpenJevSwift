// The streaming detokenizer of generation (issue #51): mlx-vlm 0.6.15's tokenizer_utils.py
// `SPMStreamingDetokenizer` with `trim_space=False`, the class `load_tokenizer` picks for the
// checkpoint's decoder (Replace "▁" by " ", ByteFallback, Fuse), and `StreamingDetokenizer.
// last_segment` (lines 19 to 197), adapted from mlx-vlm, Copyright © 2025 Prince Canuma, MIT.

import Foundation

/// Turns generated ids into text one id at a time, as mlx-vlm streams a reply.
///
/// A token that starts with `▁` flushes the text gathered since the previous one, with every `▁`
/// as a space; any other token is appended to it, and byte-fallback tokens (`<0xNN>`) gather
/// into UTF-8. So text arrives a word at a time: the text of a token is usually emitted with the
/// token after it, and ``finalize()`` flushes the last word. Ids the caller skips are dropped
/// before they enter the buffer; a special token that is not skipped is fused into the word it
/// arrives in, as upstream's chat replies show it.
struct StreamingDetokenizer {
    /// The vocabulary entry of an id, `tokenizer.vocab` inverted.
    let tokenText: (Int) -> String?
    /// The text flushed so far.
    private(set) var text = ""
    /// ``text``'s length in Unicode scalars, Python's `len`.
    private var length = 0
    /// How much of ``text`` ``lastSegment()`` has returned, in Unicode scalars.
    private var offset = 0
    private var unflushed = ""
    private var bytes: [UInt8] = []

    init(tokenText: @escaping (Int) -> String?) {
        self.tokenText = tokenText
    }

    /// `add_token`: drops `token` when `skipping` holds it, gathers it when it is a byte token,
    /// else flushes the gathered word when `token` starts one.
    mutating func add(_ token: Int, skipping: Set<Int>) {
        if skipping.contains(token) {
            return
        }
        let value = tokenText(token) ?? ""
        if let byte = Self.byteValue(of: value) {
            bytes.append(byte)
            return
        }
        flushBytes()
        if value.unicodeScalars.first == "\u{2581}" {
            append(Self.spaced(unflushed))
            unflushed = value
        } else {
            unflushed += value
        }
    }

    /// `finalize`: flushes the gathered bytes and word.
    mutating func finalize() {
        flushBytes()
        append(Self.spaced(unflushed))
        unflushed = ""
    }

    /// `last_segment`: the text flushed since the previous call, or `""` while the text ends with
    /// U+FFFD, a character whose bytes may still be arriving.
    mutating func lastSegment() -> String {
        guard let last = text.unicodeScalars.last, last != "\u{FFFD}" else { return "" }
        let segment = String(String.UnicodeScalarView(text.unicodeScalars.suffix(length - offset)))
        offset = length
        return segment
    }

    private mutating func append(_ piece: String) {
        text += piece
        length += piece.unicodeScalars.count
    }

    /// Every `▁` as a space, by Unicode scalar as Python's `str.replace` works.
    static func spaced(_ piece: String) -> String {
        String(
            String.UnicodeScalarView(piece.unicodeScalars.map { $0 == "\u{2581}" ? " " : $0 }))
    }

    /// `_flush_bytes`: the gathered bytes as UTF-8, with U+FFFD for what does not decode.
    private mutating func flushBytes() {
        guard !bytes.isEmpty else { return }
        unflushed += String(decoding: bytes, as: UTF8.self)
        bytes = []
    }

    /// The byte of a byte-fallback entry, `<0xNN>`: starts with `<0x`, at least six characters,
    /// `>` sixth.
    static func byteValue(of value: String) -> UInt8? {
        let scalars = Array(value.unicodeScalars.prefix(6))
        guard scalars.count == 6, value.hasPrefix("<0x"), scalars[5] == ">" else { return nil }
        return UInt8(String(String.UnicodeScalarView(scalars[3..<5])), radix: 16)
    }
}
