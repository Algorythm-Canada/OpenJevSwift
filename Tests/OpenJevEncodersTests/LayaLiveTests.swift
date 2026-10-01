#if os(macOS)
    import Foundation
    import OpenJevCore
    import OpenJevEncoders
    import Testing

    /// Laya through Core ML against the PyTorch float32 reference, in the settings D-011 chose and
    /// the iPhone's packages on the Mac's Neural Engine: the Mac's multifunction package on the GPU
    /// with up to 16 questions per call and with one, and each per-length package with one question
    /// per call on the Neural Engine, reading the questions whose sequences fit it (all 200 for
    /// 1,024 tokens). The bounds are spike #56's (docs/spikes/encoder-runtime.md, "Scope notes"),
    /// on the probabilities before laya rounds them: the largest difference at most 0.02, the mean
    /// at most 0.003, and the top answer unchanged wherever the reference's top two are at least
    /// 0.01 apart. The measured values are printed.
    ///
    /// Opt-in: it needs the converted packages, found through `OPENJEV_ENCODER_MODELS` or in
    /// ~/Library/Caches/OpenJevSwift/encoders, and the checkpoint's tokenizer. macOS only: the
    /// iOS Simulator runs Core ML on its CPU, which says nothing about an iPhone.
    @Suite(
        "Laya on Core ML",
        .serialized,
        .enabled(
            if: LayaModelFiles.multifunctionPackage != nil
                && LayaModelFiles.packagesByLength.count == 4 && LayaModelFiles.tokenizer != nil,
            LayaModelFiles.missingPackageMessage),
        .enabled(if: LayaFixtures.available, LayaFixtures.missingMessage))
    struct LayaLiveTests {
        /// One setting the parity test reads the corpus with.
        struct Setting: Sendable, CustomTestStringConvertible {
            /// 0 for the multifunction package, else the per-length package's length.
            var length: Int
            var computeUnits: EncoderComputeUnits
            var maxBatchRows: Int

            var testDescription: String {
                let package =
                    length == 0
                    ? EncoderPackageSpec.layaMultifunction.name
                    : EncoderPackageSpec.laya(sequenceLength: length).name
                return "\(package), \(computeUnits), \(maxBatchRows) per call"
            }

            /// The Mac's setting, the GPU one question at a time, and the iPhone's packages.
            static let all =
                [
                    Setting(length: 0, computeUnits: .cpuAndGPU, maxBatchRows: 16),
                    Setting(length: 0, computeUnits: .cpuAndGPU, maxBatchRows: 1),
                ]
                + EncoderPackageSpec.layaSequenceLengths.map {
                    Setting(length: $0, computeUnits: .cpuAndNeuralEngine, maxBatchRows: 1)
                }
        }

        /// The configuration of the files found, in a setting.
        private func configuration(_ setting: Setting) throws -> LayaBackend.Configuration {
            let tokenizer = try #require(LayaModelFiles.tokenizer)
            let packages: LayaBackend.Configuration.Packages =
                setting.length == 0
                ? .multifunction(try #require(LayaModelFiles.multifunctionPackage))
                : .byLength([
                    setting.length: try #require(LayaModelFiles.packagesByLength[setting.length])
                ])
            return LayaBackend.Configuration(
                packages: packages, tokenizerDirectory: tokenizer.tokenizerDirectory,
                configurationFile: tokenizer.calibratorFile, computeUnits: setting.computeUnits,
                maxBatchRows: setting.maxBatchRows, functionCapacity: 2)
        }

        @available(macOS 15, *)
        @Test(
            "The reads that fit stay within the PyTorch bounds; a marker off by one does not",
            arguments: Setting.all)
        func parity(setting: Setting) async throws {
            let reference = try LayaFixtures.reference()
            let readsByRequest = try LayaFixtures.readsByRequest()
            let loaded = try await LayaBackend.load(configuration: configuration(setting))
            // The same backend over a model that records every row the package returns, so the
            // probabilities can be compared before laya rounds them.
            let recorder = RecordingModel(loaded.model)
            let backend = LayaBackend(
                model: recorder, tokenizer: loaded.tokenizer, calibration: loaded.calibration,
                maxBatchRows: loaded.maxBatchRows)
            let longest = setting.length == 0 ? reference.maxLength : setting.length
            var unrounded: [(name: String, measured: [Double], reference: [Double])] = []
            var published: [(name: String, measured: [Double], reference: [Double])] = []
            var shifted: [(name: String, measured: [Double], reference: [Double])] = []
            let clock = ContinuousClock()
            let started = clock.now
            for corpus in try LayaFixtures.corpus() {
                let all = try #require(readsByRequest[corpus.name])
                let questions = try LayaFixtures.questions(of: corpus)
                let fitting = zip(questions, all).filter { $0.1.ids.count <= longest }
                guard !fitting.isEmpty else { continue }
                let reads = fitting.map(\.1)
                var distributions: [[Double]] = []
                var tokens = 0
                await recorder.reset()
                for start in stride(from: 0, to: fitting.count, by: reference.encoderBatch) {
                    let batch = fitting[start..<min(start + reference.encoderBatch, fitting.count)]
                        .map(\.0)
                    let result = try await backend.readBatch(
                        state: corpus.request.state,
                        stateText: StateText.render(corpus.request.state), questions: batch)
                    distributions += result.probabilities
                    tokens += result.inputTokens
                }
                #expect(tokens == reads.reduce(0) { $0 + $1.ids.count }, "\(corpus.name)")
                if fitting.count == all.count {
                    #expect(tokens == all.last?.requestInputTokens, "\(corpus.name): billing")
                }
                let rows = await recorder.rows
                try #require(rows.count == reads.count, "\(corpus.name): rows")
                for ((read, row), distribution) in zip(zip(reads, rows), distributions) {
                    let logits = read.markers.map { row[$0] }
                    let p = loaded.calibration.probabilities(logits: logits, kind: read.kind)
                    unrounded.append((read.name, p.map(Double.init), read.probabilitiesUnrounded))
                    published.append((read.name, distribution, read.probabilities))
                    let off = loaded.calibration.probabilities(
                        logits: read.markers.map { row[$0 + 1] }, kind: read.kind)
                    shifted.append((read.name, off.map(Double.init), read.probabilitiesUnrounded))
                }
            }
            let elapsed = clock.now - started
            let bounds = ParityBounds(unrounded)
            let publishedBounds = ParityBounds(published)
            print(
                "Laya on Core ML, \(setting.testDescription): \(bounds), published: "
                    + "largest difference \(publishedBounds.maxDifference), "
                    + "\(elapsed) with the loads")
            #expect(bounds.questions > 0)
            #expect(bounds.violations.isEmpty, "\(bounds)")
            #expect(publishedBounds.violations.isEmpty, "\(publishedBounds)")
            // The bounds catch a read one position after each marker on the model's own scores.
            let planted = ParityBounds(shifted)
            #expect(!planted.violations.isEmpty, "markers off by one: \(planted)")
            if setting.length == 1024 || setting.length == 0 {
                #expect(bounds.questions == 200)
            }
        }

        @available(macOS 15, *)
        @Test("The store's local models load both package sets, and the iPhone's keeps each length")
        func loadThroughTheStore() async throws {
            let store = EncoderModelFiles.store
            let corpus = try #require(try LayaFixtures.corpus().first { $0.name == "quickstart" })
            let reads = try #require(try LayaFixtures.readsByRequest()["quickstart"])

            let mac = try await LayaBackend.load(from: store, packageSet: .multifunction)
            let engine = EncoderDecisionEngine(backend: mac)
            try await engine.warmUp()
            let decision = try await engine.decide(corpus.request)
            #expect(decision.inputTokens == reads.last?.requestInputTokens)
            #expect(answerBounds(decision, reads).violations.isEmpty)

            let phone = try await LayaBackend.load(from: store, packageSet: .byLength)
            let packages = try #require(phone.model as? CoreMLPackagesByLength)
            #expect(await packages.loadedLengths.isEmpty)
            try await phone.prefetch(lengths: [100, 600])
            let phoneEngine = EncoderDecisionEngine(backend: phone)
            try await phoneEngine.warmUp()
            #expect(await packages.loadedLengths == [128])
            let short = try await phoneEngine.decide(corpus.request)
            #expect(answerBounds(short, reads).violations.isEmpty)
            // A 1,024-token state loads the longest package and keeps the shorter one.
            let long = try #require(
                try LayaFixtures.corpus().first { $0.name == "long_conversation" })
            var one = long.request
            let first = try #require(long.request.questions.first)
            one.questions = OrderedMap(uniqueKeysWithValues: [(first.key, first.value)])
            let longDecision = try await phoneEngine.decide(one)
            #expect(longDecision.inputTokens == 1024)
            #expect(await packages.loadedLengths == [128, 1024])
        }

        /// The engine's answers against the published reference.
        private func answerBounds(_ decision: Decision, _ reads: [LayaFixtures.Read])
            -> ParityBounds
        {
            ParityBounds(
                reads.compactMap { read in
                    guard let answer = decision.answers[read.key] else { return nil }
                    let probabilities: [Double]
                    switch answer {
                    case .noul(let yes):
                        probabilities = [yes, 1 - yes]
                    case .choice(_, let byOption, _):
                        probabilities = byOption.map(\.value)
                    case .score(_, _, let byLevel, _):
                        probabilities = byLevel
                    }
                    return (read.name, probabilities, read.probabilities)
                })
        }
    }

    /// A model that passes every call to another and keeps the rows it returns.
    actor RecordingModel: EncoderModelRunner {
        let inner: any EncoderModelRunner
        private(set) var rows: [[Float]] = []

        init(_ inner: any EncoderModelRunner) {
            self.inner = inner
        }

        func reset() {
            rows = []
        }

        func run(_ input: [[[Int32]]]) async throws -> [[Float]] {
            let output = try await inner.run(input)
            rows += output
            return output
        }
    }
#endif
