// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, class
// `LayaEngine`: `model_name`, the `max_choices` of `EncoderEngine`, `load` and `read_batch`, which
// calls laya 0.3.6's `Agent.system_one` (`laya/agent.py`: the sequences, the forward pass read
// at the markers, the calibration and `usage.input_tokens`). Apache-2.0. See THIRD_PARTY.md.

import Foundation
import OpenJevCore

/// Which of Laya's Core ML packages a device runs (D-011).
public enum LayaPackageSet: Sendable, Hashable, CaseIterable {
    /// `laya-m18-fp16`, one function per shape (batch 1 and 16 by 128 to 1,024 tokens): the
    /// Mac's, on the GPU, up to 16 questions per call.
    case multifunction
    /// `laya-f18-b1s128-fp16` to `laya-f18-b1s1024-fp16`, one program for one shape each: the
    /// iPhone's, on the Neural Engine, one question per call. Core ML does not load the
    /// multifunction package for the Neural Engine.
    case byLength

    /// The set D-011 chose for this platform: ``multifunction`` on macOS, ``byLength`` on iOS.
    public static var platformDefault: LayaPackageSet {
        #if os(macOS)
            return .multifunction
        #else
            return .byLength
        #endif
    }
}

/// Laya (`laya-1.0`, Nandakishor M / Convai Innovations' ModernBERT-large decision model) behind
/// the encoder backend contract, upstream's `LayaEngine`.
///
/// For each question of a batch it renders laya's question (``LayaPrompt``), builds its sequence
/// (``LayaSequence``, the state tokenized once for the batch) and refuses the batch when a
/// question's options overflow the head's budget, as upstream does. It runs the rows through the
/// model ``maxBatchRows`` at a time, reads each row's scores at its option markers, and
/// calibrates, rounds and renormalises them as laya and upstream do (``LayaCalibration``). It
/// returns one distribution per question in the question's option order (a noul's is
/// `[P(true), 1 - P(true)]`) and bills the rows' lengths, laya's `usage.input_tokens`.
///
/// ``load(configuration:)`` and ``load(from:packageSet:)`` run the model with Core ML. The
/// initializer takes any ``EncoderModelRunner`` and ``LayaTokenizing``, so tests can replay
/// recorded scores.
public actor LayaBackend: QuestionReadBackend {
    /// The settings ``load(configuration:)`` reads.
    public struct Configuration: Sendable, Hashable {
        /// The converted packages.
        public enum Packages: Sendable, Hashable {
            /// A `laya-m18-fp16.mlpackage` folder, one function per shape. Core ML does not load
            /// it for the Neural Engine (D-011): run it with ``EncoderComputeUnits/cpuAndGPU``.
            case multifunction(URL)
            /// `laya-f18-b1s{length}-fp16.mlpackage` folders by sequence length: the ones the
            /// device holds, among 128, 256, 512 and 1,024.
            case byLength([Int: URL])

            /// Where these packages run by default: the GPU for the multifunction package, the
            /// Neural Engine for the per-length packages, on either platform.
            public var defaultComputeUnits: EncoderComputeUnits {
                switch self {
                case .multifunction: return .cpuAndGPU
                case .byLength: return .cpuAndNeuralEngine
                }
            }

            /// The most questions per call by default: ``Configuration/defaultMaxBatchRows`` for
            /// the multifunction package, 1 for the per-length packages, which hold batch 1.
            public var defaultMaxBatchRows: Int {
                switch self {
                case .multifunction: return Configuration.defaultMaxBatchRows
                case .byLength: return 1
                }
            }
        }

        /// The converted packages.
        public var packages: Packages
        /// A folder holding the checkpoint's tokenizer.json and tokenizer_config.json, its
        /// `tokenizer/` folder.
        public var tokenizerDirectory: URL
        /// The checkpoint's rl_agent_config.json.
        public var configurationFile: URL
        /// Where Core ML may run the model.
        public var computeUnits: EncoderComputeUnits
        /// The most questions one Core ML call reads: up to 16 with the multifunction package,
        /// 1 with the per-length packages.
        public var maxBatchRows: Int
        /// The most functions of the multifunction package loaded at once. The per-length
        /// packages keep every package loaded.
        public var functionCapacity: Int

        /// Creates a configuration with D-011's defaults for the packages: the multifunction
        /// package on the GPU with up to 16 questions per call on macOS and 1 on iOS, and two
        /// functions loaded on macOS; the per-length packages on the Neural Engine, one question
        /// per call.
        public init(
            packages: Packages,
            tokenizerDirectory: URL,
            configurationFile: URL,
            computeUnits: EncoderComputeUnits? = nil,
            maxBatchRows: Int? = nil,
            functionCapacity: Int = Configuration.defaultFunctionCapacity
        ) {
            self.packages = packages
            self.tokenizerDirectory = tokenizerDirectory
            self.configurationFile = configurationFile
            self.computeUnits = computeUnits ?? packages.defaultComputeUnits
            self.maxBatchRows = maxBatchRows ?? packages.defaultMaxBatchRows
            self.functionCapacity = functionCapacity
        }

        /// Creates a configuration of the multifunction package from the files an
        /// ``EncoderPackageStore`` found or downloaded for ``EncoderPackageManifest/laya``, with
        /// its defaults: the GPU, and up to 16 questions per call on macOS.
        public init(
            locations: EncoderPackageLocations,
            computeUnits: EncoderComputeUnits? = nil,
            maxBatchRows: Int? = nil,
            functionCapacity: Int = Configuration.defaultFunctionCapacity
        ) {
            self.init(
                packages: .multifunction(locations.packageDirectory),
                tokenizerDirectory: locations.tokenizerDirectory,
                configurationFile: locations.calibratorFile, computeUnits: computeUnits,
                maxBatchRows: maxBatchRows, functionCapacity: functionCapacity)
        }

        /// The multifunction package's rows per call, as for Verdict: 16 on macOS, where the GPU
        /// reads a batch in one call; 1 on iOS, which reads one question per call.
        public static var defaultMaxBatchRows: Int {
            #if os(macOS)
                return 16
            #else
                return 1
            #endif
        }

        /// 2 on macOS and 1 on iOS, as for Verdict.
        public static var defaultFunctionCapacity: Int {
            #if os(macOS)
                return 2
            #else
                return 1
            #endif
        }
    }

    /// ``KnownEncoderModels/laya``: `laya-1.0` with upstream's description.
    public nonisolated let modelInfo = KnownEncoderModels.laya
    /// 255, upstream's `EncoderEngine.max_choices`. The head's budget refuses fewer in practice:
    /// about 20 options of a few tokens each fit in 256 tokens.
    public nonisolated let maxChoices = 255
    /// `nil`: Laya cuts a long state to fit 1,024 tokens rather than refusing it.
    public nonisolated let maxPromptTokens: Int? = nil

    /// The model the rows run through.
    public nonisolated let model: any EncoderModelRunner
    /// The tokenizer that turns texts into rows.
    public nonisolated let tokenizer: any LayaTokenizing
    /// The checkpoint's configuration and laya's calibration.
    public nonisolated let calibration: LayaCalibration
    /// The most questions one model call reads.
    public nonisolated let maxBatchRows: Int

    /// Creates a backend over a model, a tokenizer and a calibration.
    ///
    /// - Precondition: `maxBatchRows` is at least 1.
    public init(
        model: any EncoderModelRunner,
        tokenizer: any LayaTokenizing,
        calibration: LayaCalibration,
        maxBatchRows: Int = Configuration.defaultMaxBatchRows
    ) {
        precondition(maxBatchRows >= 1, "a model call reads at least one question")
        self.model = model
        self.tokenizer = tokenizer
        self.calibration = calibration
        self.maxBatchRows = maxBatchRows
    }

    /// Reads one batch, upstream's `LayaEngine.read_batch`.
    ///
    /// The state is rendered from its raw value, laya's `serialize_state`
    /// (``LayaPrompt/stateText(_:)``), and tokenized once for every question.
    ///
    /// - Throws: ``SchemaError`` (`"Too many choices for laya-1.0: a question's options must fit
    ///   in 256 tokens."`) when a question's options overflow the head's budget, before anything
    ///   is read; ``EncoderModelError`` when the model returns the wrong number of rows or a row
    ///   too short for a question's markers; ``EncoderLoadError/noPackage(length:package:held:)``
    ///   when no package the device holds takes a sequence; and the model's own errors.
    public func readBatch(
        state: JSONValue, stateText: String, questions: [EncoderQuestion]
    ) async throws -> BatchReadResult {
        guard !questions.isEmpty else {
            return BatchReadResult(probabilities: [], inputTokens: 0)
        }
        let prompts = questions.map(LayaPrompt.init(question:))
        let stateIDs = tokenizer.encode(LayaPrompt.stateText(state))
        var sequences: [LayaSequence] = []
        sequences.reserveCapacity(prompts.count)
        for prompt in prompts {
            let sequence = LayaSequence(
                prompt: prompt, stateIDs: stateIDs, tokenizer: tokenizer,
                maxLength: calibration.maxLength, headMaxLength: calibration.headMaxLength)
            // laya's system_one raises ValueError here, and upstream refuses the request.
            guard sequence.markers.count == prompt.options.count else {
                throw LayaSequence.overflowError(
                    model: modelInfo.name, headMaxLength: calibration.headMaxLength)
            }
            sequences.append(sequence)
        }
        var scores: [[Float]] = []
        scores.reserveCapacity(sequences.count)
        var start = sequences.startIndex
        while start < sequences.endIndex {
            let end = min(start + maxBatchRows, sequences.endIndex)
            let rows = (start..<end).map { Self.row(sequences[$0], prompt: prompts[$0]) }
            let output = try await model.run(rows)
            guard output.count == rows.count else {
                throw EncoderModelError.unexpectedOutput(
                    "\(modelInfo.name) returned \(output.count) rows for \(rows.count)")
            }
            scores += output
            start = end
        }
        var probabilities: [[Double]] = []
        probabilities.reserveCapacity(prompts.count)
        for ((prompt, sequence), row) in zip(zip(prompts, sequences), scores) {
            guard sequence.markers.allSatisfy({ row.indices.contains($0) }) else {
                throw EncoderModelError.unexpectedOutput(
                    "\(modelInfo.name) returned \(row.count) scores for a sequence of "
                        + "\(sequence.ids.count) tokens")
            }
            probabilities.append(
                calibration.distribution(
                    logits: sequence.markers.map { row[$0] }, kind: prompt.kind))
        }
        return BatchReadResult(
            probabilities: probabilities, inputTokens: sequences.reduce(0) { $0 + $1.ids.count })
    }

    /// The model's planes for one sequence: the token ids, the attention mask (all ones) and
    /// the question type at every position.
    static func row(_ sequence: LayaSequence, prompt: LayaPrompt) -> [[Int32]] {
        let ids = sequence.ids.map { Int32($0) }
        return [
            ids, [Int32](repeating: 1, count: ids.count),
            [Int32](repeating: Int32(prompt.questionType), count: ids.count),
        ]
    }

    /// Downloads, checks and compiles the per-length packages that take sequences of these
    /// lengths, so that the reads that need them do not wait for the download (on an iPhone,
    /// 845 MB each) or the compile. A package loads on the first read that needs it; the first
    /// load of one took 33 to 56 s on an A15 (spike #56).
    ///
    /// Does nothing when the model is not ``CoreMLPackagesByLength``: the multifunction package
    /// holds every length.
    ///
    /// - Throws: ``EncoderLoadError/noPackage(length:package:held:)`` for a length beyond 1,024
    ///   tokens, and the store's, the compiler's and Core ML's errors.
    public func prefetch(lengths: [Int]) async throws {
        #if canImport(CoreML)
            if #available(macOS 15, iOS 18, *), let packages = model as? CoreMLPackagesByLength {
                try await packages.prefetch(lengths: lengths)
            }
        #endif
    }
}

#if canImport(CoreML)
    extension LayaBackend {
        /// Loads Laya to run with Core ML: the configuration file, the tokenizer and the
        /// packages, each compiled with `MLModel.compileModel(at:)` unless a compile of it is
        /// kept beside it (``CompiledEncoderModel``).
        ///
        /// The multifunction package's functions load when a read first needs them, and each
        /// per-length package loads when a read first needs its length; ``EncoderDecisionEngine``'s
        /// warm-up read loads the first.
        ///
        /// It can be called from code built for the package's macOS 14 and iOS 17 floors: the
        /// packages need macOS 15 or iOS 18, and an older OS gets
        /// ``EncoderLoadError/unsupportedOperatingSystem(_:)``.
        ///
        /// - Throws: ``EncoderLoadError`` for an older OS, a missing file, a tokenizer that does
        ///   not pad with the packages' padding id, a `max_len` longer than the packages take, a
        ///   batch larger than the packages' or a package length Laya has none for, and the
        ///   tokenizer's, the configuration's and Core ML's own errors.
        public static func load(configuration: Configuration) async throws -> LayaBackend {
            guard #available(macOS 15, iOS 18, *) else {
                throw EncoderLoadError.unsupportedOperatingSystem(
                    "Laya's Core ML packages need macOS 15 or iOS 18")
            }
            let parts = try await loadParts(
                tokenizerDirectory: configuration.tokenizerDirectory,
                configurationFile: configuration.configurationFile)
            let model: any EncoderModelRunner
            switch configuration.packages {
            case .multifunction(let package):
                let spec = EncoderPackageSpec.layaMultifunction
                try checkBatchRows(configuration.maxBatchRows, spec: spec)
                guard configuration.functionCapacity >= 1 else {
                    throw EncoderLoadError.invalidConfiguration(
                        "functionCapacity is \(configuration.functionCapacity); at least one "
                            + "function must stay loaded")
                }
                model = CoreMLEncoderModel(
                    spec: spec, compiledModel: try await CompiledEncoderModel.url(for: package),
                    computeUnits: configuration.computeUnits,
                    capacity: configuration.functionCapacity)
            case .byLength(let packages):
                let specs = EncoderPackageSpec.layaSequenceLengths.map {
                    EncoderPackageSpec.laya(sequenceLength: $0)
                }
                try checkBatchRows(configuration.maxBatchRows, spec: specs[0])
                var folders: [String: URL] = [:]
                for (length, folder) in packages {
                    guard EncoderPackageSpec.layaSequenceLengths.contains(length) else {
                        throw EncoderLoadError.invalidConfiguration(
                            "Laya has no package of \(length) tokens; its lengths are "
                                + EncoderPackageSpec.layaSequenceLengths.map(String.init)
                                .joined(separator: ", "))
                    }
                    folders[EncoderPackageSpec.laya(sequenceLength: length).name] = folder
                }
                model = CoreMLPackagesByLength(
                    specs: specs, computeUnits: configuration.computeUnits,
                    source: .folders(folders))
            }
            return LayaBackend(
                model: model, tokenizer: parts.tokenizer, calibration: parts.calibration,
                maxBatchRows: configuration.maxBatchRows)
        }

        /// Loads Laya from the files a store finds or downloads, with this platform's defaults.
        /// This is the server's backend for `OPENJEV_BACKEND=laya`:
        ///
        /// ```swift
        /// QuestionReadBackendProvider { _ in
        ///     try await LayaBackend.load(from: EncoderPackageStore(environment: environment))
        /// }
        /// ```
        ///
        /// With ``LayaPackageSet/multifunction``, the Mac's set, it gets everything
        /// ``EncoderPackageManifest/laya`` names and runs it on the GPU. With
        /// ``LayaPackageSet/byLength``, the iPhone's, it gets the tokenizer and the configuration
        /// file now and no package, and runs on the Neural Engine: each read uses the smallest
        /// package of ``EncoderPackageManifest/layaByLength`` the device holds that takes its
        /// sequence, and a sequence longer than every one it holds is
        /// ``EncoderLoadError/noPackage(length:package:held:)`` until ``prefetch(lengths:)`` has
        /// fetched one. Until then the warm-up read throws that error too, so an app prefetches
        /// at least the 128-token package before it warms up.
        ///
        /// - Throws: The store's errors (``EncoderPackageError``) and ``load(configuration:)``'s.
        public static func load(
            from store: EncoderPackageStore, packageSet: LayaPackageSet = .platformDefault
        ) async throws -> LayaBackend {
            // Before the store, whose own check would throw EncoderPackageError instead.
            guard #available(macOS 15, iOS 18, *) else {
                throw EncoderLoadError.unsupportedOperatingSystem(
                    "Laya's Core ML packages need macOS 15 or iOS 18")
            }
            switch packageSet {
            case .multifunction:
                // Core ML does not load the multifunction package for the Neural Engine, so it
                // runs on the GPU on an iPhone too, as Configuration's defaults have it.
                return try await load(
                    configuration: Configuration(locations: store.locations(for: .laya)))
            case .byLength:
                return try await loadByLength(from: store)
            }
        }

        /// ``load(from:packageSet:)`` with the per-length packages, on an OS that runs them.
        @available(macOS 15, iOS 18, *)
        private static func loadByLength(from store: EncoderPackageStore) async throws
            -> LayaBackend
        {
            let manifests = EncoderPackageManifest.layaByLength
            guard
                let first = EncoderPackageSpec.layaSequenceLengths.first.flatMap({ manifests[$0] })
            else {
                throw EncoderLoadError.invalidConfiguration("no manifest for Laya's packages")
            }
            let tokenizer = try await store.tokenizerLocations(for: first)
            let parts = try await loadParts(
                tokenizerDirectory: tokenizer.tokenizerDirectory,
                configurationFile: tokenizer.calibratorFile)
            @Sendable func manifest(for spec: EncoderPackageSpec) throws -> EncoderPackageManifest {
                guard let manifest = manifests[spec.sequenceLengths[0]] else {
                    throw EncoderLoadError.invalidConfiguration("no manifest for \(spec.name)")
                }
                return manifest
            }
            let source = CoreMLPackagesByLength.Source(
                held: { spec in try store.heldPackageDirectory(for: manifest(for: spec)) },
                fetch: { spec in
                    try await store.locations(for: manifest(for: spec)).packageDirectory
                })
            let model = CoreMLPackagesByLength(
                specs: EncoderPackageSpec.layaSequenceLengths.map {
                    EncoderPackageSpec.laya(sequenceLength: $0)
                },
                computeUnits: .cpuAndNeuralEngine, source: source)
            return LayaBackend(
                model: model, tokenizer: parts.tokenizer, calibration: parts.calibration,
                maxBatchRows: 1)
        }

        /// The tokenizer and the calibration, checked against the packages: the padding id and
        /// the longest sequence.
        private static func loadParts(tokenizerDirectory: URL, configurationFile: URL)
            async throws -> (tokenizer: LayaTokenizer, calibration: LayaCalibration)
        {
            guard FileManager.default.fileExists(atPath: configurationFile.path) else {
                throw EncoderLoadError.missingFile(configurationFile)
            }
            let calibration = try LayaCalibration(contentsOf: configurationFile)
            let longest = EncoderPackageSpec.layaSequenceLengths.last ?? 0
            guard calibration.maxLength <= longest else {
                throw EncoderLoadError.mismatch(
                    "rl_agent_config.json's max_len is \(calibration.maxLength), but Laya's "
                        + "packages take at most \(longest) tokens")
            }
            let tokenizer = try await LayaTokenizer.load(directory: tokenizerDirectory)
            if case .value(let pad) = EncoderPackageSpec.layaMultifunction.padding.first,
                tokenizer.specialTokens.padding != Int(pad)
            {
                throw EncoderLoadError.mismatch(
                    "the tokenizer's [PAD] is \(tokenizer.specialTokens.padding), but Laya's "
                        + "packages pad with \(pad)")
            }
            return (tokenizer, calibration)
        }

        /// Refuses more rows per call than a package's largest batch.
        private static func checkBatchRows(_ rows: Int, spec: EncoderPackageSpec) throws {
            guard let largest = spec.batchSizes.last, (1...largest).contains(rows) else {
                throw EncoderLoadError.invalidConfiguration(
                    "maxBatchRows is \(rows); \(spec.name) reads 1 to "
                        + "\(spec.batchSizes.last ?? 0) questions per call")
            }
        }
    }
#endif
