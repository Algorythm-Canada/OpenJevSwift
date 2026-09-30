import Foundation
import MLXHuggingFace
import MLXLMCommon
import OpenJevCore
import OpenJevDiffusionGemma
import Testing
import Tokenizers

/// Loads the tokenizer through mlx-swift-lm's path, `#huggingFaceTokenizerLoader()` from
/// MLXHuggingFace, and checks it gives what ``SwiftTransformersTokenizer`` gives (spike #20).
///
/// The macro expands to a `TokenizerLoader` whose `load(from:)` calls
/// `Tokenizers.AutoTokenizer.from(modelFolder:)`, the call `SwiftTransformersTokenizer.load`
/// makes, and wraps the result in a private `TokenizerBridge` struct that forwards `encode`,
/// `decode` and `applyChatTemplate`. The bridge hides the upstream object, so the test compares
/// results rather than identity.
@Suite(
    "mlx-swift-lm tokenizer loader path",
    .enabled(if: TokenizerFixtures.available, TokenizerFixtures.missingMessage))
struct MLXTokenizerLoaderTests {
    /// The macro path is loaded and measured before the direct tokenizer is touched, so when
    /// this test runs alone in a fresh process the figures are a cold load, comparable with
    /// `TokenizerParityTests/loads()` run alone. In a full run they are a warm second load.
    @Test("The MLXHuggingFace loader gives the same ids and text as the direct loader")
    func sameResults() async throws {
        let loader = #huggingFaceTokenizerLoader()
        let before = ProcessMemory.current()
        let clock = ContinuousClock()
        let start = clock.now
        let bridged = try await loader.load(from: TokenizerFixtures.tokenizerDirectory)
        let wallTime = clock.now - start
        let after = ProcessMemory.current()
        #expect(String(describing: type(of: bridged)).contains("TokenizerBridge"))
        #expect(wallTime < .seconds(30), "loading took \(wallTime)")
        SpikeReport.record(
            "tokenizer-load",
            """
            mlx-swift-lm #huggingFaceTokenizerLoader() load wall time: \(wallTime)
            resident before: \(ProcessMemory.megabytes(before.residentBytes))
            resident after: \(ProcessMemory.megabytes(after.residentBytes))
            resident added: \(ProcessMemory.megabytes(after.residentBytes - before.residentBytes))
            peak resident (ru_maxrss): \(ProcessMemory.megabytes(after.peakResidentBytes))
            type: \(type(of: bridged))
            """)
        let direct = try await TokenizerFixtures.tokenizer()

        let rows = try TokenizerFixtures.cases("tokenizer/corpus.json")
        var disagreements: [String] = []
        var trailingByteRows = 0
        for row in rows {
            let text = try #require(row["text"]?.stringValue)
            let ids = try TokenizerFixtures.ints(row["ids"])
            if bridged.encode(text: text, addSpecialTokens: false)
                != (try direct.encode(text, addSpecialTokens: false))
            {
                disagreements.append("encode \(text.debugDescription)")
            }
            // The unmodified path drops byte tokens at the end of a sequence
            // (`upstreamDecodeDepartures`); those rows are compared there.
            if let last = ids.last, direct.token(of: last)?.hasPrefix("<0x") == true {
                trailingByteRows += 1
                continue
            }
            if bridged.decode(tokenIds: ids, skipSpecialTokens: false)
                != (try direct.decode(ids, skipSpecialTokens: false))
            {
                disagreements.append("decode \(text.debugDescription)")
            }
        }
        #expect(disagreements.isEmpty, "\(disagreements.prefix(10))")
        #expect(trailingByteRows == 3)
        #expect(bridged.eosToken == "<eos>")
        #expect(bridged.bosToken == "<bos>")
        #expect(bridged.convertTokenToId("<turn|>") == EngineTokens.turnClose)

        let prompts = try TokenizerFixtures.cases("chat-prompts/prompts.json")
        for row in prompts {
            let messages = try #require(row["messages"]?.arrayValue).map { message in
                [
                    "role": try #require(message["role"]?.stringValue),
                    "content": try #require(message["content"]?.stringValue),
                ] as [String: any Sendable]
            }
            for (thinking, key) in [(false, "thinking_off"), (true, "thinking_on")] {
                let expected = try TokenizerFixtures.ints(row[key]?["ids"])
                let ids = try bridged.applyChatTemplate(
                    messages: messages, tools: nil,
                    additionalContext: ["enable_thinking": thinking])
                #expect(ids == expected, "\(row["name"]?.stringValue ?? "") \(key)")
            }
        }
    }

    /// The two decode departures docs/spikes/tokenizer-parity.md records, as swift-transformers
    /// 1.3.4 shows them through mlx-swift-lm's unmodified path. When a swift-transformers update
    /// makes this test fail, the corresponding code in `SwiftTransformersTokenizer` can go.
    @Test("The unmodified swift-transformers path still shows the two decode departures")
    func upstreamDecodeDepartures() async throws {
        let bridged = try await #huggingFaceTokenizerLoader().load(
            from: TokenizerFixtures.tokenizerDirectory)
        // Byte tokens at the end of a sequence are dropped: `<0x00>` decodes to nothing.
        #expect(bridged.decode(tokenIds: [238], skipSpecialTokens: false) == "")
        #expect(bridged.decode(tokenIds: [482, 381, 429, 429], skipSpecialTokens: false) == "")
        // The same bytes followed by an ordinary token are decoded.
        let zero = bridged.encode(text: "zero", addSpecialTokens: false)
        #expect(bridged.decode(tokenIds: [238] + zero, skipSpecialTokens: false) == "\0zero")
        // `clean_up_tokenization_spaces` defaults to true and rewrites the text.
        let spaced = bridged.encode(text: "a , b .", addSpecialTokens: false)
        #expect(bridged.decode(tokenIds: spaced, skipSpecialTokens: false) == "a, b.")
    }
}
