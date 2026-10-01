#if canImport(CoreML)
    import Foundation
    import OpenJevCore

    /// Encoder packages of one program for one shape each, one per sequence length, run with
    /// Core ML: Laya's iPhone packages (D-011), the only Laya packages Core ML loads for the
    /// Neural Engine.
    ///
    /// Each package takes one row of its own length, so a call runs its rows one at a time, each
    /// through the smallest package the device holds that takes it. A package is compiled and
    /// loaded when a row first needs it and stays loaded: the first load of a Laya package took
    /// 33 to 56 s on an A15, and a second load in the same process 21 to 29 s (spike #56), so
    /// none is released to load another. A row longer than every package the device holds is
    /// ``EncoderLoadError/noPackage(length:package:held:)``, which names the package to fetch
    /// with ``prefetch(lengths:)``.
    ///
    /// Which packages the device holds is asked of ``Source`` whenever a row needs a length no
    /// loaded package takes, so a package that a background ``prefetch(lengths:)`` finished
    /// is used by the next read.
    @available(macOS 15, iOS 18, *)
    public actor CoreMLPackagesByLength: EncoderModelRunner {
        /// Where the packages are.
        public struct Source: Sendable {
            /// The package's folder when this device holds it, else `nil`. It downloads nothing.
            public var held: @Sendable (EncoderPackageSpec) async throws -> URL?
            /// The package's folder, downloaded and checked first when the device lacks it.
            public var fetch: @Sendable (EncoderPackageSpec) async throws -> URL

            /// Creates a source.
            public init(
                held: @escaping @Sendable (EncoderPackageSpec) async throws -> URL?,
                fetch: @escaping @Sendable (EncoderPackageSpec) async throws -> URL
            ) {
                self.held = held
                self.fetch = fetch
            }

            /// Packages at fixed folders: held when the folder exists, and fetched by no one.
            public static func folders(_ folders: [String: URL]) -> Source {
                Source(
                    held: { spec in
                        folders[spec.name].flatMap {
                            FileManager.default.fileExists(atPath: $0.path) ? $0 : nil
                        }
                    },
                    fetch: { spec in
                        guard let folder = folders[spec.name],
                            FileManager.default.fileExists(atPath: folder.path)
                        else {
                            throw EncoderLoadError.missingFile(
                                folders[spec.name]
                                    ?? URL(fileURLWithPath: spec.name + ".mlpackage"))
                        }
                        return folder
                    })
            }
        }

        /// The packages, one per sequence length, shortest first.
        public nonisolated let specs: [EncoderPackageSpec]
        /// Where Core ML may run them.
        public nonisolated let computeUnits: EncoderComputeUnits
        private nonisolated let source: Source
        /// The packages compiled and created so far, by sequence length. Each loads its program
        /// on its first call and keeps it.
        private var models: [Int: CoreMLEncoderModel] = [:]
        /// The fetches running now, by package name, which a second prefetch of the same
        /// package waits for instead of fetching it again.
        private var fetches: [String: Task<URL, any Error>] = [:]
        /// The compiles running now, by package name, which every other caller waits for.
        private var compiles: [String: Task<URL, any Error>] = [:]

        /// Creates the set. Nothing is compiled or loaded until a row needs it.
        ///
        /// - Precondition: Every spec is one program for one shape at batch 1, and no two have
        ///   the same length.
        public init(
            specs: [EncoderPackageSpec], computeUnits: EncoderComputeUnits, source: Source
        ) {
            precondition(
                specs.allSatisfy { $0.layout == .singleShape && $0.batchSizes == [1] },
                "each package holds one row of one length")
            let sorted = specs.sorted { $0.sequenceLengths[0] < $1.sequenceLengths[0] }
            precondition(
                Set(sorted.map { $0.sequenceLengths[0] }).count == sorted.count,
                "one package per length")
            self.specs = sorted
            self.computeUnits = computeUnits
            self.source = source
        }

        /// The sequence lengths whose packages are compiled, loaded or about to load on their
        /// first call.
        public var loadedLengths: [Int] { models.keys.sorted() }

        /// Runs each row through the smallest package this device holds that takes it.
        ///
        /// - Throws: ``EncoderLoadError/noPackage(length:package:held:)`` for a row longer than
        ///   every package the device holds, ``EncoderModelError`` for malformed rows, and the
        ///   source's, the compiler's and Core ML's errors.
        public func run(_ rows: [[[Int32]]]) async throws -> [[Float]] {
            var output: [[Float]] = []
            output.reserveCapacity(rows.count)
            for (index, row) in rows.enumerated() {
                guard let length = row.first?.count, length > 0 else {
                    throw EncoderModelError.malformedRows(
                        "row \(index) has no tokens; each row needs its planes")
                }
                let model = try await model(holding: length)
                let result = try await model.run([row])
                guard result.count == 1 else {
                    throw EncoderModelError.unexpectedOutput(
                        "\(model.spec.name) returned \(result.count) rows for 1")
                }
                output += result
            }
            return output
        }

        /// The package that runs a row of `length` tokens: the smallest the device holds that
        /// takes the row, loaded already or compiled and created now.
        private func model(holding length: Int) async throws -> CoreMLEncoderModel {
            for spec in specs where spec.sequenceLengths[0] >= length {
                let packageLength = spec.sequenceLengths[0]
                if let model = models[packageLength] {
                    return model
                }
                guard let package = try await source.held(spec) else {
                    continue
                }
                let compiled = try await compile(spec, at: package)
                // Another call may have created it while this one compiled.
                if let model = models[packageLength] {
                    return model
                }
                let model = CoreMLEncoderModel(
                    spec: spec, compiledModel: compiled, computeUnits: computeUnits, capacity: 1)
                models[packageLength] = model
                return model
            }
            throw EncoderLoadError.noPackage(
                length: length, package: spec(holding: length)?.name, held: try await heldNames())
        }

        /// The names of the packages this device holds, shortest first.
        private func heldNames() async throws -> [String] {
            var held: [String] = []
            for spec in specs {
                if try await source.held(spec) != nil {
                    held.append(spec.name)
                }
            }
            return held
        }

        /// The smallest package that takes a row of `length` tokens, held or not.
        public nonisolated func spec(holding length: Int) -> EncoderPackageSpec? {
            specs.first { $0.sequenceLengths[0] >= length }
        }

        /// Downloads, checks and compiles the packages that take sequences of these lengths, so
        /// that a read needing one does not wait for the download or the compile. A package
        /// loads on the first read that needs it. A second prefetch of a package that is being
        /// fetched waits for that fetch.
        ///
        /// - Throws: ``EncoderLoadError/noPackage(length:package:held:)`` for a length no
        ///   package takes, and the source's and the compiler's errors.
        public func prefetch(lengths: [Int]) async throws {
            var wanted: [EncoderPackageSpec] = []
            for length in lengths {
                guard let spec = spec(holding: length) else {
                    throw EncoderLoadError.noPackage(
                        length: length, package: nil, held: try await heldNames())
                }
                if !wanted.contains(spec) {
                    wanted.append(spec)
                }
            }
            for spec in wanted {
                _ = try await fetchAndCompile(spec)
            }
        }

        /// Fetches and compiles a package, or waits for the fetch of it that is running.
        private func fetchAndCompile(_ spec: EncoderPackageSpec) async throws -> URL {
            if let running = fetches[spec.name] {
                return try await running.value
            }
            let source = source
            let task = Task {
                try await compile(spec, at: source.fetch(spec))
            }
            fetches[spec.name] = task
            defer { fetches[spec.name] = nil }
            return try await task.value
        }

        /// Compiles a package, or waits for the compile of it that is running: a read that needs
        /// a package a prefetch is compiling shares that compile, so no two compiles of one
        /// package replace each other's output while one of them loads.
        private func compile(_ spec: EncoderPackageSpec, at package: URL) async throws -> URL {
            if let running = compiles[spec.name] {
                return try await running.value
            }
            let task = Task { try await CompiledEncoderModel.url(for: package) }
            compiles[spec.name] = task
            defer { compiles[spec.name] = nil }
            return try await task.value
        }
    }

    @available(macOS 15, iOS 18, *)
    extension CoreMLPackagesByLength: ModelReleasing {
        /// Releases every package loaded so far. A later read loads the one it needs again.
        public func close() async {
            let loaded = models.values
            models = [:]
            for model in loaded {
                await model.close()
            }
        }
    }
#endif
