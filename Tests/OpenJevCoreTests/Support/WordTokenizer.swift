import OpenJevCore

/// A stand-in ``DecisionTokenizer`` for tests whose texts the fixtures never recorded: canvas
/// boundaries, many-question chunking and the cache's size limit.
///
/// It splits a text into runs of letters, runs of digits and single other characters, and maps
/// each piece to a stable id from a hash of its UTF-8 bytes. So `"q1: yes"` is five tokens and
/// `"q10yes"` is three, and every label the schema builder makes (`yes`, `no`, `A` to the
/// two-letter choice labels, and one digit) stays one token, as it does for the real tokenizer.
/// Only the token counts and the one-slot property matter to the tests that use it; the ids
/// themselves are arbitrary.
///
/// `decode` and `chatPromptIDs` are not supported and throw.
struct WordTokenizer: DecisionTokenizer {
    func encode(_ text: String, addSpecialTokens: Bool) throws -> [Int] {
        var pieces: [String] = []
        var current = ""
        var currentKind: Kind? = nil
        for scalar in text.unicodeScalars {
            let kind = Kind(scalar)
            if kind == currentKind, kind != .other {
                current.unicodeScalars.append(scalar)
            } else {
                if !current.isEmpty {
                    pieces.append(current)
                }
                current = String(scalar)
                currentKind = kind
            }
        }
        if !current.isEmpty {
            pieces.append(current)
        }
        return pieces.map(Self.id)
    }

    func decode(_ ids: [Int], skipSpecialTokens: Bool) throws -> String {
        throw TokenizerError("WordTokenizer does not decode")
    }

    func chatPromptIDs(system: String, user: String, thinking: Bool) throws -> [Int] {
        throw TokenizerError("WordTokenizer does not render chat prompts")
    }

    private enum Kind: Equatable {
        case letter, digit, other

        init(_ scalar: Unicode.Scalar) {
            if scalar.properties.isAlphabetic {
                self = .letter
            } else if scalar.properties.numericType == .decimal {
                self = .digit
            } else {
                self = .other
            }
        }
    }

    /// FNV-1a over the UTF-8 bytes, folded to a non-negative id below the vocabulary size.
    private static func id(_ piece: String) -> Int {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in piece.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return Int(hash % UInt64(EngineTokens.vocabularySize))
    }
}
