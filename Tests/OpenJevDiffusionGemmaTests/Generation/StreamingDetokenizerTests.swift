import Foundation
import OpenJevCore
import Testing

@testable import OpenJevDiffusionGemma

/// The streaming detokenizer against the text pieces upstream's `MlxRuntime.generate` emitted
/// (Fixtures/generation/generation.json), and on its own rules.
@Suite("The streaming detokenizer")
struct StreamingDetokenizerTests {
    @Test(
        "Replaying a recorded reply's ids gives upstream's emitted pieces, one per token, then the tail",
        .enabled(if: TokenizerFixtures.available, TokenizerFixtures.missingMessage),
        arguments: GenerationOracle.generationCases)
    func replay(_ generation: GenerationOracle.Generation) async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        var detokenizer = StreamingDetokenizer(tokenText: { tokenizer.token(of: $0) })
        let skipped = Set(generation.skipSpecialTokenIDs)
        var pieces: [GenerationOracle.Piece] = []
        for token in generation.generated {
            detokenizer.add(token, skipping: skipped)
            pieces.append(.init(text: detokenizer.lastSegment(), token: token))
        }
        detokenizer.finalize()
        let tail = detokenizer.lastSegment()
        if !tail.isEmpty {
            pieces.append(.init(text: tail, token: nil))
        }
        #expect(pieces == generation.pieces)
        #expect(pieces.map(\.text).joined() == generation.text)
    }

    /// A stand-in vocabulary: SentencePiece words, a byte-fallback run and a special token.
    static let vocabulary: [Int: String] = [
        1: "\u{2581}Hello", 2: ",", 3: "\u{2581}world", 4: "<0xE2>", 5: "<0x82>", 6: "<0xAC>",
        7: "<|channel>", 8: "\u{2581}", 9: "<0x0A>",
    ]

    @Test("A word's text is released when the next word starts, and finalize flushes the last")
    func words() {
        var detokenizer = StreamingDetokenizer(tokenText: { Self.vocabulary[$0] })
        var pieces: [String] = []
        for token in [1, 2, 3] {
            detokenizer.add(token, skipping: [])
            pieces.append(detokenizer.lastSegment())
        }
        detokenizer.finalize()
        pieces.append(detokenizer.lastSegment())
        #expect(pieces == ["", "", " Hello,", " world"])
    }

    @Test("Byte tokens gather into UTF-8, and skipped ids never enter the buffer")
    func bytesAndSkips() {
        var detokenizer = StreamingDetokenizer(tokenText: { Self.vocabulary[$0] })
        for token in [8, 4, 5, 6, 7, 9, 3] {
            detokenizer.add(token, skipping: [7])
        }
        detokenizer.finalize()
        #expect(detokenizer.text == " \u{20AC}\n world")
        // Not skipped, the special token is fused into the word it arrives in.
        var fused = StreamingDetokenizer(tokenText: { Self.vocabulary[$0] })
        for token in [1, 7, 3] {
            fused.add(token, skipping: [])
        }
        fused.finalize()
        #expect(fused.text == " Hello<|channel> world")
    }

    @Test("A text ending in U+FFFD is held back")
    func replacementCharacter() {
        var detokenizer = StreamingDetokenizer(tokenText: { Self.vocabulary[$0] })
        detokenizer.add(1, skipping: [])
        detokenizer.add(4, skipping: [])
        detokenizer.finalize()
        // An unfinished UTF-8 sequence at the end decodes to U+FFFD, which is not released.
        #expect(detokenizer.text == " Hello\u{FFFD}")
        #expect(detokenizer.lastSegment() == "")
    }

    @Test("Byte-fallback entries are recognised as Python recognises them")
    func byteValue() {
        #expect(StreamingDetokenizer.byteValue(of: "<0x0A>") == 10)
        #expect(StreamingDetokenizer.byteValue(of: "<0xE2>") == 0xE2)
        #expect(StreamingDetokenizer.byteValue(of: "<0x0A") == nil)
        #expect(StreamingDetokenizer.byteValue(of: "\u{2581}<0x0A>") == nil)
    }
}
