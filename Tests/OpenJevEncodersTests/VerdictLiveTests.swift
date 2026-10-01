#if os(macOS)
    import Foundation
    import OpenJevCore
    import OpenJevEncoders
    import Testing

    /// Verdict through Core ML against the PyTorch float32 reference: all 200 questions of the
    /// corpus in three settings. The Mac's, on the GPU with up to 16 questions per call through
    /// the batch-16 functions; the GPU with one question per call through the batch-1 functions;
    /// and the iPhone's, on the Neural Engine with one question per call and one function loaded
    /// (D-011). The bounds are spike #56's (the scope notes of docs/spikes/encoder-runtime.md):
    /// the largest probability difference at most 0.02, the mean at most 0.003, and the top
    /// answer unchanged wherever the reference's top two are at least 0.01 apart. The measured
    /// values are printed.
    ///
    /// Opt-in: it needs the converted package, found through `OPENJEV_ENCODER_MODELS` or in
    /// ~/Library/Caches/OpenJevSwift/encoders, and the checkpoint's tokenizer. macOS only: the
    /// iOS Simulator runs Core ML on its CPU, which says nothing about an iPhone.
    @Suite(
        "Verdict on Core ML",
        .serialized,
        .enabled(
            if: VerdictModelFiles.packageDirectory != nil
                && VerdictModelFiles.tokenizerDirectory != nil,
            VerdictModelFiles.missingPackageMessage),
        .enabled(if: VerdictFixtures.available, VerdictFixtures.missingMessage))
    struct VerdictLiveTests {
        /// One setting the parity test reads the corpus with.
        struct Setting: Sendable, CustomTestStringConvertible {
            var computeUnits: EncoderComputeUnits
            var maxBatchRows: Int
            var functionCapacity: Int

            var testDescription: String {
                "\(computeUnits), \(maxBatchRows) per call, \(functionCapacity) loaded"
            }

            /// The Mac's setting, the GPU one question at a time, and the iPhone's setting.
            static let all = [
                Setting(computeUnits: .cpuAndGPU, maxBatchRows: 16, functionCapacity: 2),
                Setting(computeUnits: .cpuAndGPU, maxBatchRows: 1, functionCapacity: 2),
                Setting(computeUnits: .cpuAndNeuralEngine, maxBatchRows: 1, functionCapacity: 1),
            ]
        }

        /// The configuration of the files found, in a setting.
        private func configuration(_ setting: Setting) throws -> VerdictBackend.Configuration {
            let tokenizer = try #require(VerdictModelFiles.tokenizerDirectory)
            return VerdictBackend.Configuration(
                packageDirectory: try #require(VerdictModelFiles.packageDirectory),
                tokenizerDirectory: tokenizer,
                calibratorFile: tokenizer.appendingPathComponent("calibrator.json"),
                computeUnits: setting.computeUnits, maxBatchRows: setting.maxBatchRows,
                functionCapacity: setting.functionCapacity)
        }

        @available(macOS 15, *)
        @Test(
            "All 200 questions stay within the bounds of the PyTorch reference",
            arguments: Setting.all)
        func parity(setting: Setting) async throws {
            let reference = try VerdictFixtures.reference()
            let readsByRequest = try VerdictFixtures.readsByRequest()
            let backend = try await VerdictBackend.load(configuration: configuration(setting))
            let engine = EncoderDecisionEngine(
                backend: backend,
                configuration: EncoderEngineConfiguration(
                    batchSize: reference.encoderBatch, warmUp: false))
            var pairs: [(name: String, measured: [Double], reference: [Double])] = []
            let clock = ContinuousClock()
            let started = clock.now
            for corpus in try VerdictFixtures.corpus() {
                let reads = try #require(readsByRequest[corpus.name])
                let context = StateText.render(corpus.request.state)
                let questions = try VerdictFixtures.questions(of: corpus)
                var measured: [[Double]] = []
                var tokens = 0
                for start in stride(from: 0, to: questions.count, by: reference.encoderBatch) {
                    let batch = Array(
                        questions[start..<min(start + reference.encoderBatch, questions.count)])
                    let result = try await backend.readBatch(
                        state: corpus.request.state, stateText: context, questions: batch)
                    measured += result.probabilities
                    tokens += result.inputTokens
                }
                #expect(tokens == reads.last?.requestInputTokens, "\(corpus.name): input tokens")
                for (read, probabilities) in zip(reads, measured) {
                    pairs.append((read.name, probabilities, read.probabilities))
                }
                // The engine answers from the same reads.
                let decision = try await engine.decide(corpus.request)
                #expect(decision.inputTokens == tokens)
            }
            let elapsed = clock.now - started
            let bounds = ParityBounds(pairs)
            print(
                "Verdict on Core ML, \(setting.testDescription): \(bounds), "
                    + "\(elapsed) for two passes")
            #expect(bounds.questions == 200)
            #expect(bounds.violations.isEmpty, "\(bounds)")
            // At most the capacity stays loaded. One question per call only ever needs the batch-1
            // functions; 16 per call uses a batch-1 function only for a batch of one question.
            let functions = await (backend.model as? CoreMLEncoderModel)?.loadedFunctions ?? []
            #expect(!functions.isEmpty)
            #expect(functions.count <= setting.functionCapacity)
            if setting.maxBatchRows == 1 {
                #expect(functions.allSatisfy { $0.hasPrefix("b1_") }, "\(functions)")
            }
        }

        @available(macOS 15, *)
        @Test("The store's local models folder loads the same backend")
        func loadThroughTheStore() async throws {
            let store = EncoderPackageStore(
                directory: FileManager.default.temporaryDirectory.appendingPathComponent(
                    "OpenJevEncodersTests-unused"),
                localModelsDirectory: VerdictModelFiles.modelsDirectory,
                huggingFaceHubDirectory: VerdictModelFiles.hubDirectory)
            let backend = try await VerdictBackend.load(from: store)
            let engine = EncoderDecisionEngine(backend: backend)
            try await engine.warmUp()
            let decision = try await engine.decide(
                SystemOneRequest(
                    model: "verdict-1.4", state: .string("I was charged twice this month."),
                    questions: [
                        "billing": .noul(
                            instructions: .string("It is about billing"), criteria: nil)
                    ]))
            guard case .noul(let yes) = decision.answers["billing"] else {
                Issue.record("no noul answer: \(decision.answers)")
                return
            }
            #expect(yes > 0.5)
            #expect(decision.inputTokens > 0)
        }
    }
#endif
