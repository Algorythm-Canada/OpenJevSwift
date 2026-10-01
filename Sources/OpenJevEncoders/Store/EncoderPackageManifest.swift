import Foundation

/// Where a converted encoder package, its tokenizer and its calibrator are published, with the
/// size and SHA-256 of every file, so that ``EncoderPackageStore`` can check a download before
/// it is used (D-011 item 5, D-033).
///
/// The package is published as its files, not as an archive, so that no unzipping is needed on
/// iOS. The tokenizer and the calibrator are the checkpoint's own files at its pinned revision.
public struct EncoderPackageManifest: Sendable, Hashable, Codable {
    /// One published file.
    public struct File: Sendable, Hashable, Codable {
        /// Where the file goes: inside the `.mlpackage` folder for a package file, inside the
        /// tokenizer folder for a tokenizer file and the calibrator.
        public var path: String
        /// Where it is downloaded from.
        public var url: URL
        /// Its size in bytes.
        public var bytes: Int
        /// Its SHA-256, in lowercase hexadecimal.
        public var sha256: String

        /// Creates a file entry.
        public init(path: String, url: URL, bytes: Int, sha256: String) {
            self.path = path
            self.url = url
            self.bytes = bytes
            self.sha256 = sha256
        }
    }

    /// A Hugging Face checkpoint at a pinned revision.
    public struct Checkpoint: Sendable, Hashable, Codable {
        /// The repository, such as `heman10x/rlcd-modernbert-151m`.
        public var repository: String
        /// The commit.
        public var revision: String

        /// Creates a checkpoint reference.
        public init(repository: String, revision: String) {
            self.repository = repository
            self.revision = revision
        }

        /// The checkpoint's folder in a Hugging Face hub cache, such as
        /// `models--heman10x--rlcd-modernbert-151m/snapshots/{revision}`.
        public func snapshot(in hub: URL) -> URL {
            hub.appendingPathComponent(
                "models--" + repository.replacingOccurrences(of: "/", with: "--"),
                isDirectory: true
            )
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(revision, isDirectory: true)
        }
    }

    /// The first major OS versions that run the package.
    public struct MinimumOS: Sendable, Hashable, Codable {
        /// The first iOS.
        public var iOS: Int
        /// The first macOS.
        public var macOS: Int

        /// Creates the minimum versions.
        public init(iOS: Int, macOS: Int) {
            self.iOS = iOS
            self.macOS = macOS
        }
    }

    /// The served model, such as `verdict-1.4`.
    public var model: String
    /// The package's name, its folder's name without `.mlpackage`.
    public var package: String
    /// The first OS versions that run the package.
    public var minimumOS: MinimumOS
    /// The checkpoint the package was converted from, which publishes the tokenizer and the
    /// calibrator.
    public var checkpoint: Checkpoint
    /// Whether the manifest's remote package files are published and ready to download.
    public var packageDownloadsEnabled: Bool
    /// The package's files, paths relative to the `.mlpackage` folder.
    public var packageFiles: [File]
    /// The tokenizer's files, tokenizer.json and tokenizer_config.json.
    public var tokenizerFiles: [File]
    /// The checkpoint's calibration file, which goes into the tokenizer folder: Verdict's
    /// calibrator.json, Laya's rl_agent_config.json.
    public var calibrator: File
    /// Where the checkpoint keeps the tokenizer's files, relative to its root: empty for Verdict,
    /// whose files are at the root, `tokenizer` for Laya. The calibration file is at the root.
    public var checkpointTokenizerFolder: String

    /// Creates a manifest.
    public init(
        model: String, package: String, minimumOS: MinimumOS, checkpoint: Checkpoint,
        packageDownloadsEnabled: Bool = true,
        packageFiles: [File], tokenizerFiles: [File], calibrator: File,
        checkpointTokenizerFolder: String = ""
    ) {
        self.model = model
        self.package = package
        self.minimumOS = minimumOS
        self.checkpoint = checkpoint
        self.packageDownloadsEnabled = packageDownloadsEnabled
        self.packageFiles = packageFiles
        self.tokenizerFiles = tokenizerFiles
        self.calibrator = calibrator
        self.checkpointTokenizerFolder = checkpointTokenizerFolder
    }

    private enum CodingKeys: String, CodingKey {
        case model, package, minimumOS, checkpoint, packageDownloadsEnabled, packageFiles,
            tokenizerFiles, calibrator, checkpointTokenizerFolder
    }

    /// Decodes a manifest. One encoded before ``checkpointTokenizerFolder`` existed has none,
    /// and keeps the checkpoint's root, as Verdict's did.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        package = try container.decode(String.self, forKey: .package)
        minimumOS = try container.decode(MinimumOS.self, forKey: .minimumOS)
        checkpoint = try container.decode(Checkpoint.self, forKey: .checkpoint)
        packageDownloadsEnabled = try container.decode(Bool.self, forKey: .packageDownloadsEnabled)
        packageFiles = try container.decode([File].self, forKey: .packageFiles)
        tokenizerFiles = try container.decode([File].self, forKey: .tokenizerFiles)
        calibrator = try container.decode(File.self, forKey: .calibrator)
        checkpointTokenizerFolder =
            try container.decodeIfPresent(String.self, forKey: .checkpointTokenizerFolder) ?? ""
    }

    /// Where the checkpoint's snapshot in a Hugging Face hub cache holds the tokenizer's files
    /// and the calibration file.
    public func checkpointFiles(in hub: URL) -> EncoderTokenizerLocations {
        let snapshot = checkpoint.snapshot(in: hub)
        let tokenizer =
            checkpointTokenizerFolder.isEmpty
            ? snapshot
            : snapshot.appendingPathComponent(checkpointTokenizerFolder, isDirectory: true)
        return EncoderTokenizerLocations(
            tokenizerDirectory: tokenizer,
            calibratorFile: snapshot.appendingPathComponent(calibrator.path))
    }
}
