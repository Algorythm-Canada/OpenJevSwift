import OpenJevCore
import OpenJevEncoders
import Testing

/// ``VerdictTokenizer`` against upstream's tokenization of every prompt of the reference corpus.
/// It needs the checkpoint's tokenizer files, which are not committed.
@Suite(
    "Verdict tokenizer",
    .enabled(
        if: VerdictModelFiles.tokenizerDirectory != nil, VerdictModelFiles.missingTokenizerMessage),
    .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage))
struct VerdictTokenizerTests {
    private func tokenizer() async throws -> VerdictTokenizer {
        try await VerdictTokenizer.load(
            directory: try #require(VerdictModelFiles.tokenizerDirectory))
    }

    @Test("Every prompt tokenizes to upstream's ids, markers and truncation at 512 included")
    func corpusParity() async throws {
        let reference = try VerdictFixtures.reference()
        let tokenizer = try await tokenizer()
        #expect(tokenizer.maxLength == reference.maxLength)
        #expect(tokenizer.tokenID("<<LABEL>>") == reference.classTokenIndex)
        #expect(tokenizer.tokenID("[PAD]") == reference.padTokenID)
        if case .value(let pad) = EncoderPackageSpec.verdict.padding[0] {
            #expect(Int(pad) == reference.padTokenID)
        }
        var mismatches: [String] = []
        for read in reference.reads where tokenizer.inputIDs(for: read.prompt) != read.inputIDs {
            mismatches.append(read.name)
        }
        #expect(reference.reads.count == 200)
        #expect(reference.reads.filter(\.truncated).count == 70)
        #expect(mismatches.isEmpty, "\(mismatches.count) prompts differ: \(mismatches.prefix(10))")
    }

    @Test("Each request bills the rows' lengths, upstream's attention-mask sum")
    func billing() async throws {
        let tokenizer = try await tokenizer()
        let readsByRequest = try VerdictFixtures.readsByRequest()
        for corpus in try VerdictFixtures.corpus() {
            let context = StateText.render(corpus.request.state)
            let rows = try VerdictFixtures.questions(of: corpus).map {
                tokenizer.inputIDs(for: VerdictPrompt(question: $0, context: context).text)
            }
            let reads = try #require(readsByRequest[corpus.name])
            #expect(
                rows.reduce(0) { $0 + $1.count } == reads.last?.requestInputTokens,
                "\(corpus.name)")
        }
    }
}
