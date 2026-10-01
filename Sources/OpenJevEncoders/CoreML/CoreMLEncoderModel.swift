#if canImport(CoreML)
    import CoreML
    import Dispatch
    import Foundation
    import OpenJevCore

    /// A converted encoder package run with Core ML, one function per input shape (D-011).
    ///
    /// A call runs through the smallest function that holds its rows and its longest row. Each
    /// function is its own `MLModel`, loaded the first time a call needs it. A loaded function
    /// keeps its own copy of the weights once it has run (six Verdict functions took 1.5 GB for a
    /// 306 MB package on a Mac, spike #56), so at most ``capacity`` stay loaded and the least
    /// recently used one is released before another is loaded.
    ///
    /// The model is an actor on a serial queue of its own, so Core ML's blocking loads and
    /// predictions run one at a time and never hold a thread of Swift's cooperative pool.
    @available(macOS 15, iOS 18, *)
    public actor CoreMLEncoderModel: EncoderModelRunner {
        /// The package's shapes, planes and output.
        public nonisolated let spec: EncoderPackageSpec
        /// The compiled package, a `.mlmodelc` folder.
        public nonisolated let compiledModel: URL
        /// Where Core ML may run it.
        public nonisolated let computeUnits: EncoderComputeUnits
        /// The most functions loaded at once.
        public nonisolated let capacity: Int

        private nonisolated let queue: DispatchSerialQueue
        private var functions = LeastRecentlyUsed<String, MLModel>()

        /// Runs the actor on its own serial queue.
        public nonisolated var unownedExecutor: UnownedSerialExecutor {
            queue.asUnownedSerialExecutor()
        }

        /// Creates the model. Nothing is loaded until the first call.
        ///
        /// - Precondition: `capacity` is at least 1.
        public init(
            spec: EncoderPackageSpec, compiledModel: URL, computeUnits: EncoderComputeUnits,
            capacity: Int
        ) {
            precondition(capacity >= 1, "at least one function must stay loaded")
            self.spec = spec
            self.compiledModel = compiledModel
            self.computeUnits = computeUnits
            self.capacity = capacity
            self.queue = DispatchSerialQueue(label: "OpenJevEncoders.\(spec.name)")
        }

        /// The functions loaded now, least recently used first.
        public var loadedFunctions: [String] { functions.keys }

        /// Runs the rows through the smallest function that holds them and returns the output
        /// rows of the rows given.
        ///
        /// - Throws: ``EncoderModelError`` for rows the package cannot take or an output it did
        ///   not expect, and Core ML's errors.
        public func run(_ rows: [[[Int32]]]) throws -> [[Float]] {
            guard !rows.isEmpty else {
                return []
            }
            let longest = try longestRow(of: rows)
            guard let function = spec.function(rows: rows.count, longestRow: longest) else {
                throw EncoderModelError.noFunction(
                    package: spec.name, rows: rows.count, longestRow: longest)
            }
            let model = try model(for: function)
            let input = try MLMultiArray(
                shape: [function.batchSize, spec.planes, function.sequenceLength].map {
                    NSNumber(value: $0)
                },
                dataType: .int32)
            let values = spec.input(rows, for: function)
            input.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, strides in
                var index = 0
                for b in 0..<function.batchSize {
                    for p in 0..<spec.planes {
                        let base = b * strides[0] + p * strides[1]
                        for s in 0..<function.sequenceLength {
                            buffer[base + s * strides[2]] = values[index]
                            index += 1
                        }
                    }
                }
            }
            let features = try MLDictionaryFeatureProvider(dictionary: [
                spec.inputName: MLFeatureValue(multiArray: input)
            ])
            let prediction = try model.prediction(from: features)
            guard let output = prediction.featureValue(for: spec.outputName)?.multiArrayValue
            else {
                throw EncoderModelError.unexpectedOutput(
                    "\(spec.name) \(function) returned no \(spec.outputName) array")
            }
            return try outputRows(output, count: rows.count, function: function)
        }

        /// The length of the rows' planes, which must all agree within a row.
        private func longestRow(of rows: [[[Int32]]]) throws -> Int {
            var longest = 0
            for (index, row) in rows.enumerated() {
                guard row.count == spec.planes, let length = row.first?.count, length > 0,
                    row.allSatisfy({ $0.count == length })
                else {
                    throw EncoderModelError.malformedRows(
                        "row \(index) for \(spec.name) must have \(spec.planes) non-empty planes "
                            + "of one length")
                }
                longest = max(longest, length)
            }
            return longest
        }

        /// The loaded function, loading it after releasing the least recently used ones.
        private func model(for function: EncoderPackageSpec.Function) throws -> MLModel {
            if let model = functions.value(forKey: function.name) {
                return model
            }
            functions.trim(to: capacity - 1)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits.mlComputeUnits
            configuration.functionName = function.name
            let model = try MLModel(contentsOf: compiledModel, configuration: configuration)
            try check(model, function: function)
            functions.insert(model, forKey: function.name)
            return model
        }

        /// Checks that a loaded function takes and returns what the spec says, so that a wrong
        /// package fails with a message rather than wrong numbers.
        private func check(_ model: MLModel, function: EncoderPackageSpec.Function) throws {
            let description = model.modelDescription
            guard
                let constraint = description.inputDescriptionsByName[spec.inputName]?
                    .multiArrayConstraint
            else {
                throw EncoderModelError.unexpectedModel(
                    "\(spec.name) \(function) has no array input named \(spec.inputName)")
            }
            let shape = constraint.shape.map(\.intValue)
            let expected = [function.batchSize, spec.planes, function.sequenceLength]
            guard shape == expected, constraint.dataType == .int32 else {
                throw EncoderModelError.unexpectedModel(
                    "\(spec.name) \(function) takes \(spec.inputName) of shape \(shape), "
                        + "expected int32 \(expected)")
            }
            guard description.outputDescriptionsByName[spec.outputName] != nil else {
                throw EncoderModelError.unexpectedModel(
                    "\(spec.name) \(function) has no output named \(spec.outputName)")
            }
        }

        /// The first `count` rows of a [batch, width] output.
        private func outputRows(
            _ output: MLMultiArray, count: Int, function: EncoderPackageSpec.Function
        ) throws -> [[Float]] {
            let shape = output.shape.map(\.intValue)
            guard shape.count == 2, shape[0] == function.batchSize else {
                throw EncoderModelError.unexpectedOutput(
                    "\(spec.name) \(function) returned \(spec.outputName) of shape \(shape), "
                        + "expected [\(function.batchSize), width]")
            }
            let width = shape[1]
            let strides = output.strides.map(\.intValue)
            if output.dataType == .float32, strides == [width, 1] {
                return output.withUnsafeBufferPointer(ofType: Float.self) { buffer in
                    (0..<count).map { row in Array(buffer[(row * width)..<((row + 1) * width)]) }
                }
            }
            // Another element type or a padded layout: read through NSNumber, which is slower
            // but converts every type Core ML returns.
            return (0..<count).map { row in
                (0..<width).map { column in
                    output[[NSNumber(value: row), NSNumber(value: column)]].floatValue
                }
            }
        }
    }

    @available(macOS 15, iOS 18, *)
    extension CoreMLEncoderModel: ModelReleasing {
        /// Releases every loaded function and the weights it holds. It runs on the model's queue,
        /// after the call in progress, if any. A later call loads its function again.
        public func close() {
            functions = LeastRecentlyUsed()
        }
    }

    extension EncoderComputeUnits {
        /// The Core ML setting.
        var mlComputeUnits: MLComputeUnits {
            switch self {
            case .cpuOnly:
                return .cpuOnly
            case .cpuAndGPU:
                return .cpuAndGPU
            case .cpuAndNeuralEngine:
                return .cpuAndNeuralEngine
            }
        }
    }
#endif
