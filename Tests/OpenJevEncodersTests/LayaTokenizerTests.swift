import OpenJevCore
import OpenJevEncoders
import Testing

/// ``LayaTokenizer`` and ``LayaSequence`` against laya's tokenization of every question of the
/// reference corpus. It needs the checkpoint's tokenizer files, which are not committed.
@Suite(
    "Laya tokenizer",
    .enabled(if: LayaModelFiles.tokenizer != nil, LayaModelFiles.missingTokenizerMessage),
    .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
struct LayaTokenizerTests {
    private func tokenizer() async throws -> LayaTokenizer {
        try await LayaTokenizer.load(
            directory: try #require(LayaModelFiles.tokenizer).tokenizerDirectory)
    }

    @Test("The special tokens are laya's: [CLS], [SEP], [MASK] and [PAD] by name")
    func specialTokens() async throws {
        let tokenizer = try await tokenizer()
        #expect(tokenizer.specialTokens == (try LayaFixtures.reference().specialTokens))
        #expect(tokenizer.tokenID("[UNK]") != nil)
        if case .value(let pad) = EncoderPackageSpec.layaMultifunction.padding[0] {
            #expect(Int(pad) == tokenizer.specialTokens.padding)
        }
    }

    @Test("Every sequence and its markers match laya's build_sequence")
    func corpusSequences() async throws {
        let reference = try LayaFixtures.reference()
        let tokenizer = try await tokenizer()
        let byRequest = try LayaFixtures.readsByRequest()
        var mismatches: [String] = []
        var compared = 0
        for corpus in try LayaFixtures.corpus() {
            let reads = try #require(byRequest[corpus.name])
            let stateIDs = tokenizer.encode(LayaPrompt.stateText(corpus.request.state))
            #expect(stateIDs.count == reads.first?.stateTokens, "\(corpus.name): state tokens")
            for (question, read) in zip(try LayaFixtures.questions(of: corpus), reads) {
                let sequence = LayaSequence(
                    prompt: LayaPrompt(question: question), stateIDs: stateIDs,
                    tokenizer: tokenizer, maxLength: reference.maxLength,
                    headMaxLength: reference.headMaxLength)
                if sequence.ids != read.ids || sequence.markers != read.markers {
                    mismatches.append(read.name)
                }
                compared += 1
            }
        }
        #expect(compared == 200)
        #expect(mismatches.isEmpty, "\(mismatches.count) differ: \(mismatches.prefix(10))")
    }

    @Test("Each request bills its rows' lengths, laya's attention-mask sum")
    func billing() async throws {
        let reference = try LayaFixtures.reference()
        let tokenizer = try await tokenizer()
        let byRequest = try LayaFixtures.readsByRequest()
        for corpus in try LayaFixtures.corpus() {
            let stateIDs = tokenizer.encode(LayaPrompt.stateText(corpus.request.state))
            let total = try LayaFixtures.questions(of: corpus).reduce(0) { total, question in
                total
                    + LayaSequence(
                        prompt: LayaPrompt(question: question), stateIDs: stateIDs,
                        tokenizer: tokenizer, maxLength: reference.maxLength,
                        headMaxLength: reference.headMaxLength
                    ).ids.count
            }
            let reads = try #require(byRequest[corpus.name])
            #expect(total == reads.last?.requestInputTokens, "\(corpus.name)")
        }
    }
}
