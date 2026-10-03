import Foundation
import OpenJevDiffusionGemma

/// A conversion of JevK5 v0.2 to MLX, as `Tools/jevk5/convert.py` writes it: the repository it is
/// meant to be published as, its pinned revision once it is, and the size and SHA-256 of every
/// file.
///
/// The conversions are not published yet (D-052): until the maintainer creates the repositories
/// and pins their commits here, ``revision`` is `nil`, a download of the default is refused
/// before any network access, and `OPENJEV_JEVK5_MODEL` names a local folder the script wrote.
/// The digests are the script's output, which two runs reproduced byte for byte, so a local
/// conversion can be checked against them (`convert.py --check`).
public struct JevK5Checkpoint: Sendable, Hashable {
    /// One file of the conversion.
    public struct File: Sendable, Hashable {
        /// Its name in the folder.
        public var name: String
        /// Its size in bytes.
        public var bytes: Int
        /// Its SHA-256, in lowercase hexadecimal.
        public var sha256: String

        /// Creates a file entry.
        public init(name: String, bytes: Int, sha256: String) {
            self.name = name
            self.bytes = bytes
            self.sha256 = sha256
        }
    }

    /// The served model, `jevk5-0.2`.
    public var model: String
    /// The quantization's bits per weight: 4 or 8 (group size 64, affine).
    public var bits: Int
    /// The Hub repository the conversion is published as.
    public var repository: String
    /// The published commit, or `nil` while the conversion is unpublished.
    public var revision: String?
    /// The checkpoint the conversion was made from.
    public var sourceRepository: String
    /// The source's commit: the author's `v0.2` tag.
    public var sourceRevision: String
    /// The conversion's files.
    public var files: [File]

    /// Whether the conversion can be downloaded: its revision is pinned.
    public var isPublished: Bool { revision != nil }

    /// The conversion's Hub source at its pinned revision.
    public var hubSource: ModelSource {
        .hub(repository: repository, revision: revision)
    }

    /// The source checkpoint: `alibiserikbay/JevK5` at its `v0.2` tag. The repository's `main`
    /// has held v0.3, another model with another temperature, since 2026-09-25.
    static let source = (
        repository: "alibiserikbay/JevK5", revision: "ea4804e93a3db07c2250315c400f59683f54db6f"
    )

    /// The files both conversions share: the source's tokenizer, chat template, generation
    /// configuration and calibration, and JevK5's license and notice.
    private static let sharedFiles = [
        File(
            name: "LICENSE", bytes: 11358,
            sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        File(
            name: "NOTICE", bytes: 1401,
            sha256: "c6dc9c346b2b516da42b80902916bb6f07b90139d7aa7543420f0674474f31d7"),
        File(
            name: "chat_template.jinja", bytes: 7756,
            sha256: "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
        File(
            name: "generation_config.json", bytes: 116,
            sha256: "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73"),
        File(
            name: "jevk5_config.json", bytes: 23,
            sha256: "39d6574650b365c77425fc87ffd13ab389ea6508bc1e09ac1289a76faea62419"),
        File(
            name: "tokenizer.json", bytes: 19_989_325,
            sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        File(
            name: "tokenizer_config.json", bytes: 1125,
            sha256: "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d"),
    ]

    /// The 4-bit conversion, 2.37 GB of weights, iOS's ``platformDefault``.
    public static let fourBit = JevK5Checkpoint(
        model: "jevk5-0.2", bits: 4, repository: "Algorythm-Canada/jevk5-0.2-mlx-4bit",
        revision: nil, sourceRepository: source.repository, sourceRevision: source.revision,
        files: sharedFiles + [
            File(
                name: "README.md", bytes: 2190,
                sha256: "281143de695eff0af6408c8560425cfdd88ae1e24540a5817c650821d2dd860b"),
            File(
                name: "config.json", bytes: 2430,
                sha256: "35fb2a84659b33e4d54af1a9d6cc3c179e40c95f3af00b3dc7dc9135d358ca9a"),
            File(
                name: "model.safetensors", bytes: 2_367_223_295,
                sha256: "3fd5171001a72b8879422ed2fe944c3bd8b5edf92b393cb2cdd843e4d409f1f8"),
            File(
                name: "model.safetensors.index.json", bytes: 67148,
                sha256: "f47a8c0ec8aa8d28fde00373f4f6ada5e203ca47c3d73a3133631983d1c37bfc"),
        ])

    /// The 8-bit conversion, 4.47 GB of weights: the server's default (`OPENJEV_JEVK5_MODEL`) and
    /// macOS's ``platformDefault``.
    public static let eightBit = JevK5Checkpoint(
        model: "jevk5-0.2", bits: 8, repository: "Algorythm-Canada/jevk5-0.2-mlx-8bit",
        revision: nil, sourceRepository: source.repository, sourceRevision: source.revision,
        files: sharedFiles + [
            File(
                name: "README.md", bytes: 2190,
                sha256: "42cddae7f1c370442433c5a939f8c4a6f776c91de15c1b2986342a3b42f5df07"),
            File(
                name: "config.json", bytes: 2430,
                sha256: "8ac5be312381d497966eca8877c0fea5d7f3760461a3713e026d838aef050072"),
            File(
                name: "model.safetensors", bytes: 4_469_618_681,
                sha256: "a928abc748a29dd1a853c11d21ae0c0c1fc51f97f11c2a66c43fb0e781285a56"),
            File(
                name: "model.safetensors.index.json", bytes: 67148,
                sha256: "7cc82135d26900005b834c016445b3989789174ff1a841214e2fc4f9c588c72a"),
        ])

    /// Both conversions.
    public static let all = [fourBit, eightBit]

    /// The conversion this platform loads by default (D-052): ``eightBit`` on macOS, which gives
    /// the author's published top answer on 230 of JevBench's 231 items where ``fourBit`` gives it
    /// on 209, and ``fourBit`` on iOS, half the size, where memory is the limit. Neither has run
    /// on an iPhone yet.
    public static var platformDefault: JevK5Checkpoint {
        #if os(macOS)
            return eightBit
        #else
            return fourBit
        #endif
    }

    /// `OPENJEV_JEVK5_MODEL`'s default, the server's: the 8-bit conversion's repository.
    public static let defaultSetting = eightBit.repository
}

/// Where a JevK5 checkpoint comes from: the folder or repository `OPENJEV_JEVK5_MODEL` names,
/// resolved through ``/OpenJevDiffusionGemma/ModelResolver``, the downloader that keeps the
/// Hugging Face cache's layout.
public enum JevK5ModelFiles {
    /// The files a checkpoint folder must hold beside the shards its index names.
    public static let requiredFiles = [
        "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
        "jevk5_config.json",
    ]

    /// The source a setting names, as ``/OpenJevDiffusionGemma/ModelSource/init(setting:)``
    /// reads `OPENJEV_MLX_MODEL`: a folder when it starts with `/`, `~` or `.`, else a repository,
    /// optionally followed by `@revision`. A conversion's repository without a revision takes the
    /// conversion's pinned one.
    public static func source(setting: String) -> ModelSource {
        let source = ModelSource(setting: setting)
        if case .hub(let repository, nil) = source,
            let checkpoint = JevK5Checkpoint.all.first(where: { $0.repository == repository })
        {
            return checkpoint.hubSource
        }
        return source
    }

    /// The required files `directory` lacks, then the shards its index names that are absent.
    public static func missingFiles(in directory: URL) -> [String] {
        let manager = FileManager.default
        let missing = requiredFiles.filter {
            !manager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        return missing.isEmpty ? ModelResolver.missingFiles(in: directory) : missing
    }

    /// The folder `source` names, downloading into `cache` what a Hub source lacks.
    ///
    /// A folder is used as is once it holds ``requiredFiles``. A Hub source is resolved by
    /// ``/OpenJevDiffusionGemma/ModelResolver``, which pins a branch or tag to its commit, checks
    /// every file it downloads and can use a cached snapshot offline. A conversion that is not
    /// published yet (``JevK5Checkpoint/isPublished``) is refused before any request.
    ///
    /// - Throws: ``JevK5LoadError/notPublished(repository:)``,
    ///   ``JevK5LoadError/missingFiles(_:in:)``, and ``/OpenJevDiffusionGemma/ModelResolverError``.
    public static func resolve(
        _ source: ModelSource, cache: HubCacheLocation = .standard, token: String? = nil,
        resolver: ModelResolver = ModelResolver()
    ) async throws -> URL {
        let directory: URL
        switch source {
        case .directory(let url):
            directory = url
        case .hub(let repository, let revision):
            if revision == nil,
                let checkpoint = JevK5Checkpoint.all.first(where: { $0.repository == repository }),
                !checkpoint.isPublished
            {
                throw JevK5LoadError.notPublished(repository: repository)
            }
            directory = try await resolver.resolve(source, cache: cache, token: token)
        }
        let missing = missingFiles(in: directory)
        guard missing.isEmpty else {
            throw JevK5LoadError.missingFiles(missing, in: directory)
        }
        return directory
    }
}
