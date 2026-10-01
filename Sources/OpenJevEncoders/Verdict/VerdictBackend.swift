// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, class
// `VerdictEngine`: `max_choices`, `load` (the model, the tokenizer and calibrator.json) and
// `read_batch`. Apache-2.0. See THIRD_PARTY.md.

import Foundation
import OpenJevCore

/// Verdict (`verdict-1.4`, Heman10x's ModernBERT-base and GLiClass encoder) behind the encoder
/// backend contract, upstream's `VerdictEngine`.
///
/// For each question of a batch it writes ``VerdictPrompt``, tokenizes it (at most 512 tokens),
/// runs the rows through the model ``maxBatchRows`` at a time and calibrates each row's first
/// `k` logits with ``VerdictCalibration``. It returns one distribution per question in the
/// question's option order (a noul's is `[P(true), P(false)]`, upstream's label order) and bills
/// the rows' lengths, the attention-mask sum upstream reports as `usage.input_tokens`.
///
/// ``load(configuration:)`` runs the model with Core ML. The initializer takes any
/// ``EncoderModelRunner`` and ``VerdictTokenizing``, so tests can replay recorded logits.
public actor VerdictBackend: QuestionReadBackend {
    /// The settings ``load(configuration:)`` reads.
    public struct Configuration: Sendable, Hashable {
        /// The converted package, a `verdict-m18-fp16.mlpackage` folder.
        public var packageDirectory: URL
        /// A folder holding the checkpoint's tokenizer.json and tokenizer_config.json.
        public var tokenizerDirectory: URL
        /// The checkpoint's calibrator.json.
        public var calibratorFile: URL
        /// Where Core ML may run the model.
        public var computeUnits: EncoderComputeUnits
        /// The most questions one Core ML call reads, at most 16.
        public var maxBatchRows: Int
        /// The most package functions loaded at once.
        public var functionCapacity: Int

        /// Creates a configuration with D-011's defaults for this platform: on macOS the GPU,
        /// 16 questions per call and two functions loaded; on iOS the Neural Engine, one
        /// question per call and one function loaded.
        public init(
            packageDirectory: URL,
            tokenizerDirectory: URL,
            calibratorFile: URL,
            computeUnits: EncoderComputeUnits = .platformDefault,
            maxBatchRows: Int = Configuration.defaultMaxBatchRows,
            functionCapacity: Int = Configuration.defaultFunctionCapacity
        ) {
            self.packageDirectory = packageDirectory
            self.tokenizerDirectory = tokenizerDirectory
            self.calibratorFile = calibratorFile
            self.computeUnits = computeUnits
            self.maxBatchRows = maxBatchRows
            self.functionCapacity = functionCapacity
        }

        /// Creates a configuration from the files an ``EncoderPackageStore`` found or
        /// downloaded, with this platform's defaults.
        public init(
            locations: EncoderPackageLocations,
            computeUnits: EncoderComputeUnits = .platformDefault,
            maxBatchRows: Int = Configuration.defaultMaxBatchRows,
            functionCapacity: Int = Configuration.defaultFunctionCapacity
        ) {
            self.init(
                packageDirectory: locations.packageDirectory,
                tokenizerDirectory: locations.tokenizerDirectory,
                calibratorFile: locations.calibratorFile, computeUnits: computeUnits,
                maxBatchRows: maxBatchRows, functionCapacity: functionCapacity)
        }

        /// 16 on macOS, where the GPU reads a batch in one call; 1 on iOS, where the Neural
        /// Engine reads one question per call through the batch-1 functions.
        public static var defaultMaxBatchRows: Int {
            #if os(macOS)
                return 16
            #else
                return 1
            #endif
        }

        /// 2 on macOS and 1 on iOS, as spike #56 measured them.
        public static var defaultFunctionCapacity: Int {
            #if os(macOS)
                return 2
            #else
                return 1
            #endif
        }
    }

    /// ``KnownEncoderModels/verdict``: `verdict-1.4` with upstream's description.
    public nonisolated let modelInfo = KnownEncoderModels.verdict
    /// 24: the head has 25 logits, and the last is kept for the abstention.
    public nonisolated let maxChoices = 24
    /// `nil`: Verdict truncates a long prompt to 512 tokens rather than refusing it.
    public nonisolated let maxPromptTokens: Int? = nil

    /// The model the rows run through.
    public nonisolated let model: any EncoderModelRunner
    /// The tokenizer that turns prompts into rows.
    public nonisolated let tokenizer: any VerdictTokenizing
    /// The checkpoint's calibrator.
    public nonisolated let calibration: VerdictCalibration
    /// The most questions one model call reads.
    public nonisolated let maxBatchRows: Int

    /// Creates a backend over a model, a tokenizer and a calibrator.
    ///
    /// - Precondition: `maxBatchRows` is at least 1.
    public init(
        model: any EncoderModelRunner,
        tokenizer: any VerdictTokenizing,
        calibration: VerdictCalibration,
        maxBatchRows: Int = Configuration.defaultMaxBatchRows
    ) {
        precondition(maxBatchRows >= 1, "a model call reads at least one question")
        self.model = model
        self.tokenizer = tokenizer
        self.calibration = calibration
        self.maxBatchRows = maxBatchRows
    }

    /// Reads one batch, upstream's `VerdictEngine.read_batch`.
    ///
    /// The state is read as `stateText`, upstream's `context`: the string as sent, anything
    /// else as `json.dumps(state, ensure_ascii=False)`.
    ///
    /// - Throws: ``EncoderModelError`` when the model returns the wrong number of rows or a row
    ///   shorter than a question's `k`, and the model's own errors.
    public func readBatch(
        state: JSONValue, stateText: String, questions: [EncoderQuestion]
    ) async throws -> BatchReadResult {
        let prompts = questions.map { VerdictPrompt(question: $0, context: stateText) }
        let rows = prompts.map { tokenizer.inputIDs(for: $0.text) }
        var logits: [[Float]] = []
        logits.reserveCapacity(rows.count)
        var start = rows.startIndex
        while start < rows.endIndex {
            // A Core ML call cannot be interrupted; a cancelled request starts no further one.
            try Task.checkCancellation()
            let end = min(start + maxBatchRows, rows.endIndex)
            let planes = rows[start..<end].map { ids in
                [ids.map { Int32($0) }, [Int32](repeating: 1, count: ids.count)]
            }
            let output = try await model.run(planes)
            guard output.count == planes.count else {
                throw EncoderModelError.unexpectedOutput(
                    "\(modelInfo.name) returned \(output.count) rows for \(planes.count)")
            }
            logits += output
            start = end
        }
        var probabilities: [[Double]] = []
        probabilities.reserveCapacity(prompts.count)
        for (prompt, row) in zip(prompts, logits) {
            guard row.count >= prompt.labelCount else {
                throw EncoderModelError.unexpectedOutput(
                    "\(modelInfo.name) returned \(row.count) logits for a question with "
                        + "\(prompt.labelCount) labels")
            }
            probabilities.append(calibration.probabilities(logits: row, k: prompt.labelCount))
        }
        return BatchReadResult(
            probabilities: probabilities, inputTokens: rows.reduce(0) { $0 + $1.count })
    }
}

extension VerdictBackend: ModelReleasing {
    /// Releases the model's loaded Core ML functions, when the runner adopts ``ModelReleasing``
    /// as ``CoreMLEncoderModel`` does. A later read loads them again.
    public nonisolated func close() async {
        await (model as? any ModelReleasing)?.close()
    }
}

#if canImport(CoreML)
    extension VerdictBackend {
        /// Loads Verdict to run with Core ML: the calibrator, the tokenizer and the package,
        /// which is compiled with `MLModel.compileModel(at:)` unless a compile of it is kept
        /// beside it (``CompiledEncoderModel``).
        ///
        /// Package functions load when a read first needs them; ``EncoderDecisionEngine``'s
        /// warm-up read loads the first one.
        ///
        /// It can be called from code built for the package's macOS 14 and iOS 17 floors: the
        /// multifunction package needs macOS 15 or iOS 18, and an older OS gets
        /// ``EncoderLoadError/unsupportedOperatingSystem(_:)``.
        ///
        /// - Throws: ``EncoderLoadError`` for an older OS, a missing file, a tokenizer that does
        ///   not pad with the package's padding id or a batch larger than the package's, and the
        ///   tokenizer's, the calibrator's and Core ML's own errors.
        public static func load(configuration: Configuration) async throws -> VerdictBackend {
            guard #available(macOS 15, iOS 18, *) else {
                throw EncoderLoadError.unsupportedOperatingSystem(
                    "\(EncoderPackageSpec.verdict.name) needs macOS 15 or iOS 18")
            }
            return try await loadWithCoreML(configuration: configuration)
        }

        /// ``load(configuration:)`` on an OS that runs the package.
        @available(macOS 15, iOS 18, *)
        private static func loadWithCoreML(configuration: Configuration) async throws
            -> VerdictBackend
        {
            let spec = EncoderPackageSpec.verdict
            guard let largest = spec.batchSizes.last,
                (1...largest).contains(configuration.maxBatchRows)
            else {
                throw EncoderLoadError.invalidConfiguration(
                    "maxBatchRows is \(configuration.maxBatchRows); \(spec.name) reads 1 to "
                        + "\(spec.batchSizes.last ?? 0) questions per call")
            }
            guard configuration.functionCapacity >= 1 else {
                throw EncoderLoadError.invalidConfiguration(
                    "functionCapacity is \(configuration.functionCapacity); at least one "
                        + "function must stay loaded")
            }
            guard FileManager.default.fileExists(atPath: configuration.calibratorFile.path) else {
                throw EncoderLoadError.missingFile(configuration.calibratorFile)
            }
            let calibration = try VerdictCalibration(contentsOf: configuration.calibratorFile)
            let tokenizer = try await VerdictTokenizer.load(
                directory: configuration.tokenizerDirectory)
            let padding = tokenizer.tokenID("[PAD]")
            if case .value(let pad) = spec.padding.first, padding != Int(pad) {
                throw EncoderLoadError.mismatch(
                    "the tokenizer's [PAD] is \(padding.map(String.init) ?? "missing"), but "
                        + "\(spec.name) pads with \(pad)")
            }
            let compiled = try await CompiledEncoderModel.url(for: configuration.packageDirectory)
            let model = CoreMLEncoderModel(
                spec: spec, compiledModel: compiled, computeUnits: configuration.computeUnits,
                capacity: configuration.functionCapacity)
            return VerdictBackend(
                model: model, tokenizer: tokenizer, calibration: calibration,
                maxBatchRows: configuration.maxBatchRows)
        }

        /// Loads Verdict from the files a store finds or downloads for
        /// ``EncoderPackageManifest/verdict``, with this platform's defaults. This is the server's
        /// backend for `OPENJEV_BACKEND=verdict`:
        ///
        /// ```swift
        /// QuestionReadBackendProvider { _ in
        ///     try await VerdictBackend.load(from: EncoderPackageStore(environment: environment))
        /// }
        /// ```
        ///
        /// - Throws: The store's errors (``EncoderPackageError``) and ``load(configuration:)``'s.
        public static func load(from store: EncoderPackageStore) async throws -> VerdictBackend {
            // Before the store, whose own check would throw EncoderPackageError instead.
            guard #available(macOS 15, iOS 18, *) else {
                throw EncoderLoadError.unsupportedOperatingSystem(
                    "\(EncoderPackageSpec.verdict.name) needs macOS 15 or iOS 18")
            }
            return try await load(
                configuration: Configuration(locations: store.locations(for: .verdict)))
        }
    }
#endif
