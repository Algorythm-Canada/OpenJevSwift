#if canImport(CoreML)
    import CoreML
    import Foundation

    /// Compiles a converted package once and keeps the result next to it.
    ///
    /// Core ML runs a compiled model (`.mlmodelc`), and compiling Verdict's package takes about
    /// a second on an iPhone (spike #56). The compiled model is kept beside the package as
    /// `{name}.mlmodelc`, with `{name}.mlmodelc-source.json` recording the size and modification
    /// time of every file of the package it came from, so a package that is downloaded or
    /// converted again is compiled again.
    @available(macOS 15, iOS 18, *)
    public enum CompiledEncoderModel {
        /// The compiled model of a package: the one kept beside it when it was compiled from
        /// these files, else a new compile with `MLModel.compileModel(at:)`, kept there.
        ///
        /// When the package's folder cannot be written, the compiled model stays where Core ML
        /// put it, in the temporary folder, and the next process compiles again.
        ///
        /// - Throws: ``EncoderLoadError/missingFile(_:)`` when the package does not exist, and
        ///   Core ML's compile errors.
        public static func url(for package: URL) async throws -> URL {
            let fileManager = FileManager.default
            guard fileManager.fileExists(atPath: package.path) else {
                throw EncoderLoadError.missingFile(package)
            }
            let name = package.deletingPathExtension().lastPathComponent
            let folder = package.deletingLastPathComponent()
            let compiled = folder.appendingPathComponent(name + ".mlmodelc")
            let stamp = folder.appendingPathComponent(name + ".mlmodelc-source.json")
            let files = try PackageFiles(of: package)
            if fileManager.fileExists(atPath: compiled.path),
                let recorded = try? JSONDecoder().decode(
                    PackageFiles.self, from: Data(contentsOf: stamp)),
                recorded == files
            {
                return compiled
            }
            let temporary = try await MLModel.compileModel(at: package)
            do {
                try? fileManager.removeItem(at: stamp)
                if fileManager.fileExists(atPath: compiled.path) {
                    try fileManager.removeItem(at: compiled)
                }
                try fileManager.moveItem(at: temporary, to: compiled)
                try JSONEncoder().encode(files).write(to: stamp, options: .atomic)
                return compiled
            } catch {
                return fileManager.fileExists(atPath: temporary.path) ? temporary : compiled
            }
        }
    }

    /// Every regular file of a package with its size and modification time, sorted by path.
    /// A symbolic link counts as the file it points to.
    struct PackageFiles: Codable, Hashable {
        struct File: Codable, Hashable {
            var path: String
            var bytes: Int
            var modified: Double
        }

        var files: [File]

        init(of package: URL) throws {
            let fileManager = FileManager.default
            let paths = fileManager.enumerator(atPath: package.path)?.compactMap { $0 as? String }
            var files: [File] = []
            for path in paths ?? [] {
                let attributes = try fileManager.attributesOfItem(
                    atPath: package.appendingPathComponent(path).resolvingSymlinksInPath().path)
                guard attributes[.type] as? FileAttributeType == .typeRegular else {
                    continue
                }
                files.append(
                    File(
                        path: path,
                        bytes: (attributes[.size] as? NSNumber)?.intValue ?? -1,
                        modified: (attributes[.modificationDate] as? Date)?
                            .timeIntervalSince1970 ?? 0))
            }
            self.files = files.sorted { $0.path < $1.path }
        }
    }
#endif
