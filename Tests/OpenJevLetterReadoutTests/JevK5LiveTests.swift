#if os(macOS)
    import CryptoKit
    import Foundation
    import OpenJevCore
    import OpenJevDiffusionGemma
    import OpenJevLetterReadout
    import Testing

    /// Where the live tests find the converted checkpoint: `OPENJEV_JEVK5_MODEL`, a folder that
    /// `Tools/jevk5/convert.py --bits 4` wrote. Without it the suite skips with a comment naming
    /// the variable, the skip CI's check-test-log.sh accepts.
    enum JevK5LiveModel {
        /// The variable.
        static let variable = "OPENJEV_JEVK5_MODEL"

        /// The folder the variable names, when it holds a checkpoint.
        static let directory: URL? = {
            guard let path = ProcessInfo.processInfo.environment[variable], !path.isEmpty else {
                return nil
            }
            let url = URL(
                fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
            return JevK5ModelFiles.missingFiles(in: url).isEmpty ? url : nil
        }()

        /// Whether the folder is the 4-bit conversion the fixture's logits were recorded with,
        /// judged by its `config.json`, which carries the quantization.
        static let isRecordedConversion: Bool = {
            guard let directory,
                let data = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
            else { return false }
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            return JevK5Checkpoint.fourBit.files.contains {
                $0.name == "config.json" && $0.sha256 == digest
            }
        }()

        static let missingMessage: Comment =
            "set OPENJEV_JEVK5_MODEL to the folder Tools/jevk5/convert.py --bits 4 writes"
        static let otherConversionMessage: Comment =
            "OPENJEV_JEVK5_MODEL is not the 4-bit conversion the fixture was recorded with"

        /// The backend over the folder, loaded once, with MLX's buffer pool capped at 4 GB.
        static let backend = Task {
            MetalLibrary.configure()
            return try await JevK5Backend.load(.directory(directory!), cacheLimitGB: 4)
        }
    }

    /// JevK5 on MLX in Swift against the same 4-bit conversion through mlx-lm in Python
    /// (Fixtures/jevk5/reads.json): the tokenizer's ids for every pass, the letters' logits, and
    /// the corpus through the engine, upstream's billing exactly and its answers within the
    /// bounds below. The measured values are printed.
    ///
    /// Opt-in: it needs the converted checkpoint, `OPENJEV_JEVK5_MODEL`. Long runs cap MLX's
    /// pool, as every MLX run here does.
    @Suite(
        "JevK5 on MLX",
        .serialized,
        .enabled(if: JevK5LiveModel.directory != nil, JevK5LiveModel.missingMessage),
        .enabled(if: JevK5Fixtures.available, "Fixtures/jevk5/reads.json is missing"))
    struct JevK5LiveTests {
        /// The largest difference allowed between a letter logit in Swift and in mlx-lm, and
        /// between a probability of the corpus's answers.
        static let logitBound: Float = 0.25
        static let probabilityBound = 0.02

        @Test("The tokenizer gives transformers' ids for every pass, and vLLM's longest entry")
        func tokenizer() async throws {
            let reference = try JevK5Fixtures.reference()
            let directory = try #require(JevK5LiveModel.directory)
            let tokenizer = try await JevK5Tokenizer.load(directory: directory)
            #expect(tokenizer.maxCharactersPerToken == reference.maxCharactersPerToken)
            #expect(
                JevK5Prompt.letters.map { tokenizer.encode($0) } == reference.letterIDs.map { [$0] }
            )
            var compared = 0
            for request in reference.requests {
                let questions = try EncoderQuestionSchemaBuilder(maxChoices: 255)
                    .build(request.request.questions).questions
                for (question, read) in zip(questions, reference.reads(of: request.name)) {
                    for pass in read.passes {
                        let ids = tokenizer.encode(
                            JevK5Prompt.text(
                                state: request.request.state, criterion: question.rawInstructions,
                                options: pass.texts))
                        #expect(ids.count == pass.tokens, "\(read.request).\(read.key): tokens")
                        #expect(
                            JevK5Fixtures.idsSHA256(ids) == pass.idsSHA256,
                            "\(read.request).\(read.key): ids")
                        compared += 1
                    }
                }
            }
            #expect(compared == reference.passes.count)
        }

        @Test(
            "The letters' logits are mlx-lm's on the same conversion",
            .enabled(if: JevK5LiveModel.isRecordedConversion, JevK5LiveModel.otherConversionMessage)
        )
        func logits() async throws {
            let reference = try JevK5Fixtures.reference()
            let backend = try await JevK5LiveModel.backend.value
            var largest: Float = 0
            var sum: Double = 0
            var count = 0
            var identical = 0
            var topAgrees = 0
            // Each pass's largest difference, with its question and token count.
            var passes: [(label: String, tokens: Int, difference: Float)] = []
            for request in reference.requests {
                let questions = try EncoderQuestionSchemaBuilder(maxChoices: 255)
                    .build(request.request.questions).questions
                for (question, read) in zip(questions, reference.reads(of: request.name)) {
                    for pass in read.passes {
                        let ids = try backend.promptTokens(
                            JevK5Prompt.text(
                                state: request.request.state, criterion: question.rawInstructions,
                                options: pass.texts))
                        let logits = try backend.model.letterLogits(
                            tokens: ids,
                            letterIDs: Array(backend.letterIDs.prefix(pass.texts.count)))
                        let differences = zip(logits, pass.logits).map { abs($0 - $1) }
                        largest = max(largest, differences.max() ?? 0)
                        passes.append(
                            ("\(read.request).\(read.key)", ids.count, differences.max() ?? 0))
                        sum += differences.reduce(0) { $0 + Double($1) }
                        count += differences.count
                        identical += logits == pass.logits ? 1 : 0
                        topAgrees += argmax(logits) == argmax(pass.logits) ? 1 : 0
                    }
                }
            }
            print(
                "JevK5 logits against mlx-lm: \(reference.passes.count) passes, \(identical) "
                    + "identical, top letter agrees on \(topAgrees), largest difference "
                    + "\(largest), mean \(sum / Double(max(count, 1)))")
            let chunked = passes.filter { $0.tokens > 2048 }
            print(
                "JevK5 logits, the largest passes: "
                    + passes.sorted { $0.difference > $1.difference }.prefix(5).map {
                        "\($0.label) (\($0.tokens) tokens) \($0.difference)"
                    }.joined(separator: ", ")
                    + "; the passes over 2,048 tokens: "
                    + chunked.map { "\($0.label) \($0.difference)" }.joined(separator: ", "))
            #expect(largest <= Self.logitBound)
            #expect(sum / Double(max(count, 1)) <= Self.meanLogitBound)
            #expect(topAgrees >= reference.passes.count - Self.topLetterChanges)
        }

        @Test(
            "The corpus through the engine: upstream's billing exactly, its answers within 0.02",
            .enabled(if: JevK5LiveModel.isRecordedConversion, JevK5LiveModel.otherConversionMessage)
        )
        func corpus() async throws {
            let reference = try JevK5Fixtures.reference()
            let backend = try await JevK5LiveModel.backend.value
            let engine = EncoderDecisionEngine(
                backend: backend, configuration: EncoderEngineConfiguration(warmUp: false))
            var largest = 0.0
            var disagreements: [String] = []
            for request in reference.requests {
                let decision = try await engine.decide(request.request)
                #expect(decision.inputTokens == request.inputTokens, "\(request.name)")
                for read in reference.reads(of: request.name) {
                    let answer = try #require(decision.answers[read.key])
                    let probabilities: [Double]
                    switch answer {
                    case .noul(let yes): probabilities = [yes, 1 - yes]
                    case .choice(_, let values, _): probabilities = values.values
                    case .score(_, _, let values, _): probabilities = values
                    }
                    largest = max(
                        largest,
                        zip(probabilities, read.probabilities).map { abs($0 - $1) }.max() ?? 0)
                    if argmax(probabilities) != argmax(read.probabilities) {
                        disagreements.append("\(read.request).\(read.key)")
                    }
                }
            }
            print(
                "JevK5 answers against mlx-lm: \(reference.reads.count) questions, largest "
                    + "probability difference \(largest), top answer differs on "
                    + "\(disagreements.count): \(disagreements)")
            #expect(largest <= Self.probabilityBound)
        }

        /// The index of the largest value, the first on a tie.
        private func argmax<T: Comparable>(_ values: [T]) -> Int {
            values.indices.max { values[$0] < values[$1] || (values[$0] == values[$1] && $0 > $1) }
                ?? 0
        }
    }
#endif
