import CryptoKit
import Foundation

/// Where an encoder's tokenizer and calibration file are on this device.
public struct EncoderTokenizerLocations: Sendable, Hashable {
    /// The folder holding tokenizer.json and tokenizer_config.json.
    public var tokenizerDirectory: URL
    /// The calibration file: Verdict's calibrator.json, Laya's rl_agent_config.json.
    public var calibratorFile: URL

    /// Creates the locations.
    public init(tokenizerDirectory: URL, calibratorFile: URL) {
        self.tokenizerDirectory = tokenizerDirectory
        self.calibratorFile = calibratorFile
    }
}

/// Where an encoder's package, tokenizer and calibrator are on this device.
public struct EncoderPackageLocations: Sendable, Hashable {
    /// The `.mlpackage` folder.
    public var packageDirectory: URL
    /// The folder holding tokenizer.json and tokenizer_config.json.
    public var tokenizerDirectory: URL
    /// calibrator.json.
    public var calibratorFile: URL

    /// Creates the locations.
    public init(packageDirectory: URL, tokenizerDirectory: URL, calibratorFile: URL) {
        self.packageDirectory = packageDirectory
        self.tokenizerDirectory = tokenizerDirectory
        self.calibratorFile = calibratorFile
    }

    /// Creates the locations of a package and of its tokenizer and calibration file.
    public init(packageDirectory: URL, tokenizer: EncoderTokenizerLocations) {
        self.init(
            packageDirectory: packageDirectory, tokenizerDirectory: tokenizer.tokenizerDirectory,
            calibratorFile: tokenizer.calibratorFile)
    }

    /// The tokenizer's folder and the calibration file.
    public var tokenizer: EncoderTokenizerLocations {
        EncoderTokenizerLocations(
            tokenizerDirectory: tokenizerDirectory, calibratorFile: calibratorFile)
    }
}

/// Downloads an encoder's files on first use and checks each one's size and SHA-256 against its
/// ``EncoderPackageManifest`` before it is used (D-011 item 5, D-033).
///
/// The files go to `{directory}/{package}/`: the package's files under `{package}.mlpackage/`,
/// and the tokenizer's files and the calibrator under `tokenizer/`. The folder is excluded from
/// backups. A download goes to a temporary file first and is moved into place only when it
/// matches the manifest, and `verified.json` records the digest each file was checked against,
/// so a later call re-downloads only what is missing or what a newer manifest changed.
///
/// When ``localModelsDirectory`` is set (`OPENJEV_ENCODER_MODELS`), nothing is downloaded or
/// checked: the package is `{localModelsDirectory}/{package}.mlpackage`, as the converters in
/// Tools/encoders write it, and the tokenizer and the calibrator are read from
/// `{localModelsDirectory}/{package}/tokenizer/` if it holds them, else from the checkpoint's
/// snapshot in the Hugging Face cache (its root for Verdict; for Laya the tokenizer under
/// `tokenizer/` and rl_agent_config.json at the root,
/// ``EncoderPackageManifest/checkpointTokenizerFolder``).
///
/// ``locations(for:)`` gets everything a manifest names. ``tokenizerLocations(for:)`` gets only
/// the tokenizer and the calibrator, which the checkpoint publishes even while the package is not,
/// and ``heldPackageDirectory(for:)`` says whether the device already holds a package without
/// downloading anything: an iPhone fetches Laya's package for a length only when it needs it.
///
/// Call ``locations(for:)`` once at startup; concurrent calls for the same package may download
/// a file twice.
public struct EncoderPackageStore: Sendable {
    /// The environment variable that names a folder of locally converted packages.
    public static let localModelsVariable = "OPENJEV_ENCODER_MODELS"

    /// Where downloaded packages are kept.
    public var directory: URL
    /// A folder of locally converted packages, used instead of downloading.
    public var localModelsDirectory: URL?
    /// The Hugging Face hub cache, where the local models' tokenizer is looked for.
    public var huggingFaceHubDirectory: URL?
    /// The session that downloads.
    public var session: URLSession

    /// Creates a store.
    public init(
        directory: URL,
        localModelsDirectory: URL? = nil,
        huggingFaceHubDirectory: URL? = nil,
        session: URLSession = .shared
    ) {
        self.directory = directory
        self.localModelsDirectory = localModelsDirectory
        self.huggingFaceHubDirectory = huggingFaceHubDirectory
        self.session = session
    }

    /// The store a server or an app uses: ``defaultDirectory()``, the local models folder that
    /// `OPENJEV_ENCODER_MODELS` names in `environment` (unset or empty means none), and the Hugging
    /// Face hub cache the environment points to.
    ///
    /// The library never reads the process environment; the caller passes it.
    ///
    /// - Throws: The error of creating Application Support.
    public init(environment: [String: String], session: URLSession = .shared) throws {
        let local = environment[Self.localModelsVariable].flatMap { $0.isEmpty ? nil : $0 }
        self.init(
            directory: try Self.defaultDirectory(),
            localModelsDirectory: local.map { URL(fileURLWithPath: $0, isDirectory: true) },
            huggingFaceHubDirectory: Self.huggingFaceHubDirectory(environment: environment),
            session: session)
    }

    /// `Application Support/OpenJevSwift/encoders` in the user's domain.
    ///
    /// - Throws: The error of creating Application Support.
    public static func defaultDirectory() throws -> URL {
        try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil,
            create: true
        )
        .appendingPathComponent("OpenJevSwift", isDirectory: true)
        .appendingPathComponent("encoders", isDirectory: true)
    }

    /// The Hugging Face hub cache as `huggingface_hub` finds it: `HF_HUB_CACHE`, else
    /// `HF_HOME/hub`, else `XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`.
    public static func huggingFaceHubDirectory(environment: [String: String]) -> URL {
        func path(_ name: String) -> String? {
            environment[name].flatMap {
                $0.isEmpty ? nil : NSString(string: $0).expandingTildeInPath
            }
        }
        if let hub = path("HF_HUB_CACHE") {
            return URL(fileURLWithPath: hub, isDirectory: true)
        }
        let home =
            path("HF_HOME").map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? URL(fileURLWithPath: path("XDG_CACHE_HOME") ?? NSHomeDirectory() + "/.cache")
            .appendingPathComponent("huggingface", isDirectory: true)
        return home.appendingPathComponent("hub", isDirectory: true)
    }

    /// The files a manifest names, on this device: the local models' copies when
    /// ``localModelsDirectory`` is set, else the store's, downloading and checking each file
    /// that is missing or unchecked first.
    ///
    /// - Throws: ``EncoderPackageError`` for a download that fails or does not match the
    ///   manifest, an OS the package does not run on, or local models that lack a file; and
    ///   the file system's and URLSession's errors.
    public func locations(for manifest: EncoderPackageManifest) async throws
        -> EncoderPackageLocations
    {
        try Self.checkPaths(of: manifest)
        try Self.checkOperatingSystem(for: manifest)
        if let localModelsDirectory {
            return try localLocations(for: manifest, in: localModelsDirectory)
        }
        guard manifest.packageDownloadsEnabled else {
            throw EncoderPackageError.packageDownloadsUnavailable(manifest.package)
        }
        let root = try storeRoot(for: manifest)
        try await fetch(Self.packageEntries(of: manifest) + Self.tokenizerEntries(of: manifest), into: root)
        return EncoderPackageLocations(
            packageDirectory: Self.packageDirectory(of: manifest, in: root),
            tokenizer: Self.tokenizerLocations(of: manifest, in: root))
    }

    /// The tokenizer's files and the calibration file of a manifest, on this device, without
    /// its package: the local models' copies or the Hugging Face cache's when
    /// ``localModelsDirectory`` is set, else the store's, downloading and checking each file that
    /// is missing or unchecked first.
    ///
    /// The checkpoint publishes these files itself, so they are downloaded even while the
    /// manifest's package is not published (``EncoderPackageManifest/packageDownloadsEnabled``).
    /// A later ``locations(for:)`` of the same manifest finds them checked.
    ///
    /// - Throws: ``EncoderPackageError`` as ``locations(for:)`` does, and
    ///   ``EncoderPackageError/missingLocalTokenizer(files:searched:)`` when the local models
    ///   and the Hugging Face cache lack a file.
    public func tokenizerLocations(for manifest: EncoderPackageManifest) async throws
        -> EncoderTokenizerLocations
    {
        try Self.checkPaths(of: manifest)
        try Self.checkOperatingSystem(for: manifest)
        if let localModelsDirectory {
            return try localTokenizer(for: manifest, in: localModelsDirectory)
        }
        let root = try storeRoot(for: manifest)
        try await fetch(Self.tokenizerEntries(of: manifest), into: root)
        return Self.tokenizerLocations(of: manifest, in: root)
    }

    /// The package's folder when this device holds the whole package, else `nil`; nothing is
    /// downloaded or hashed.
    ///
    /// With ``localModelsDirectory`` set, the package is held when
    /// `{localModelsDirectory}/{package}.mlpackage` exists. Otherwise it is held when the store
    /// has checked every package file at the manifest's digest (`verified.json`) and the file
    /// still has the manifest's size, as ``locations(for:)`` decides what to download again.
    ///
    /// - Throws: ``EncoderPackageError`` for a manifest whose paths leave the store's folder, a
    ///   package for a newer OS, or a symbolic link on a file's way.
    public func heldPackageDirectory(for manifest: EncoderPackageManifest) throws -> URL? {
        try Self.checkPaths(of: manifest)
        try Self.checkOperatingSystem(for: manifest)
        if let localModelsDirectory {
            let package = Self.localPackageDirectory(of: manifest, in: localModelsDirectory)
            return FileManager.default.fileExists(atPath: package.path) ? package : nil
        }
        let root = directory.appendingPathComponent(manifest.package, isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.path) else {
            return nil
        }
        let verified = Self.verifiedDigests(in: root)
        for (path, file) in Self.packageEntries(of: manifest) {
            let destination = try Self.checkDestination(
                root: root, storeDirectory: directory, path: path)
            guard verified[path] == file.sha256.lowercased(),
                (try? Self.size(of: destination)) == file.bytes
            else {
                return nil
            }
        }
        return Self.packageDirectory(of: manifest, in: root)
    }

    /// The store's folder of a manifest's files, created and excluded from backups.
    private func storeRoot(for manifest: EncoderPackageManifest) throws -> URL {
        let root = directory.appendingPathComponent(manifest.package, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Self.checkDestination(root: root, storeDirectory: directory, path: "verified.json")
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var excluded = root
        try excluded.setResourceValues(values)
        return root
    }

    /// Downloads and checks each entry that `verified.json` does not record at the manifest's
    /// digest, or whose file no longer has the manifest's size, and records each one checked.
    private func fetch(_ entries: [(String, EncoderPackageManifest.File)], into root: URL)
        async throws
    {
        let recordFile = root.appendingPathComponent("verified.json")
        var verified = Self.verifiedDigests(in: root)
        for (path, file) in entries {
            let destination = try Self.checkDestination(
                root: root, storeDirectory: directory, path: path)
            if verified[path] == file.sha256.lowercased(),
                (try? Self.size(of: destination)) == file.bytes
            {
                continue
            }
            verified[path] = nil
            try await download(
                file, named: path, to: destination, root: root, storeDirectory: directory)
            verified[path] = file.sha256.lowercased()
            try Self.checkDestination(
                root: root, storeDirectory: directory, path: "verified.json")
            try JSONEncoder().encode(verified).write(to: recordFile, options: .atomic)
        }
    }

    /// The digest `verified.json` records for each path the store has checked.
    private static func verifiedDigests(in root: URL) -> [String: String] {
        (try? JSONDecoder().decode(
            [String: String].self, from: Data(contentsOf: root.appendingPathComponent("verified.json"))))
            ?? [:]
    }

    /// The package's files, each under `{package}.mlpackage/` in the store's folder.
    private static func packageEntries(of manifest: EncoderPackageManifest)
        -> [(String, EncoderPackageManifest.File)]
    {
        manifest.packageFiles.map { (manifest.package + ".mlpackage/" + $0.path, $0) }
    }

    /// The tokenizer's files and the calibration file, each under `tokenizer/` in the store's
    /// folder.
    private static func tokenizerEntries(of manifest: EncoderPackageManifest)
        -> [(String, EncoderPackageManifest.File)]
    {
        (manifest.tokenizerFiles + [manifest.calibrator]).map { ("tokenizer/" + $0.path, $0) }
    }

    /// The package in a folder laid out as the store's.
    private static func packageDirectory(of manifest: EncoderPackageManifest, in root: URL)
        -> URL
    {
        root.appendingPathComponent(manifest.package + ".mlpackage", isDirectory: true)
    }

    /// The tokenizer and the calibration file in a folder laid out as the store's.
    private static func tokenizerLocations(of manifest: EncoderPackageManifest, in root: URL)
        -> EncoderTokenizerLocations
    {
        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        return EncoderTokenizerLocations(
            tokenizerDirectory: tokenizer,
            calibratorFile: tokenizer.appendingPathComponent(manifest.calibrator.path))
    }

    /// Downloads one file to a temporary file, checks it and moves it into place.
    private func download(
        _ file: EncoderPackageManifest.File, named name: String, to destination: URL,
        root: URL, storeDirectory: URL
    ) async throws {
        _ = try Self.checkDestination(root: root, storeDirectory: storeDirectory, path: name)
        let (temporary, response) = try await session.download(from: file.url)
        defer { try? FileManager.default.removeItem(at: temporary) }
        if let response = response as? HTTPURLResponse, !(200..<300).contains(response.statusCode) {
            throw EncoderPackageError.httpStatus(url: file.url, status: response.statusCode)
        }
        try Self.verify(temporary, against: file, named: name)
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        _ = try Self.checkDestination(root: root, storeDirectory: storeDirectory, path: name)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: temporary, to: destination)
    }

    /// The package in the local models folder, and the first place that holds the tokenizer and
    /// the calibrator (``localTokenizer(for:in:)``).
    private func localLocations(for manifest: EncoderPackageManifest, in local: URL) throws
        -> EncoderPackageLocations
    {
        let package = Self.localPackageDirectory(of: manifest, in: local)
        guard FileManager.default.fileExists(atPath: package.path) else {
            throw EncoderPackageError.missingLocalPackage(package)
        }
        return EncoderPackageLocations(
            packageDirectory: package, tokenizer: try localTokenizer(for: manifest, in: local))
    }

    /// The package's folder in the local models folder, as the converters write it.
    private static func localPackageDirectory(of manifest: EncoderPackageManifest, in local: URL)
        -> URL
    {
        local.appendingPathComponent(manifest.package + ".mlpackage", isDirectory: true)
    }

    /// The first place that holds the tokenizer's files and the calibration file:
    /// `{package}/tokenizer/` in the local models folder, else the checkpoint's Hugging Face
    /// snapshot, where Laya's tokenizer is under `tokenizer/` and its calibration file at the
    /// root.
    private func localTokenizer(for manifest: EncoderPackageManifest, in local: URL) throws
        -> EncoderTokenizerLocations
    {
        let fileManager = FileManager.default
        let folder = local.appendingPathComponent(manifest.package, isDirectory: true)
            .appendingPathComponent("tokenizer", isDirectory: true)
        var candidates = [
            EncoderTokenizerLocations(
                tokenizerDirectory: folder,
                calibratorFile: folder.appendingPathComponent(manifest.calibrator.path))
        ]
        if let huggingFaceHubDirectory {
            candidates.append(manifest.checkpointFiles(in: huggingFaceHubDirectory))
        }
        let names = manifest.tokenizerFiles.map(\.path)
        guard
            let found = candidates.first(where: { candidate in
                fileManager.fileExists(atPath: candidate.calibratorFile.path)
                    && names.allSatisfy {
                        fileManager.fileExists(
                            atPath: candidate.tokenizerDirectory.appendingPathComponent($0).path)
                    }
            })
        else {
            throw EncoderPackageError.missingLocalTokenizer(
                files: names + [manifest.calibrator.path],
                searched: candidates.map(\.tokenizerDirectory))
        }
        return found
    }

    /// Refuses a manifest whose package name or file paths would leave the store's folder, or
    /// whose checkpoint tokenizer folder would leave the checkpoint's: an empty or absolute path,
    /// or one with an empty, `.` or `..` component. The tokenizer folder may be empty, for the
    /// checkpoint's root.
    private static func checkPaths(of manifest: EncoderPackageManifest) throws {
        let files = manifest.packageFiles + manifest.tokenizerFiles + [manifest.calibrator]
        let folder = manifest.checkpointTokenizerFolder.isEmpty ? [] : [manifest.checkpointTokenizerFolder]
        for path in [manifest.package] + files.map(\.path) + folder {
            let components = path.split(separator: "/", omittingEmptySubsequences: false)
            guard !path.hasPrefix("/"),
                components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
            else {
                throw EncoderPackageError.invalidPath(path)
            }
        }
        guard !manifest.package.contains("/") else {
            throw EncoderPackageError.invalidPath(manifest.package)
        }
    }

    /// Refuses a package this OS cannot run before anything is downloaded.
    private static func checkOperatingSystem(for manifest: EncoderPackageManifest) throws {
        #if os(macOS)
            let (system, major) = ("macOS", manifest.minimumOS.macOS)
        #else
            let (system, major) = ("iOS", manifest.minimumOS.iOS)
        #endif
        let minimum = OperatingSystemVersion(majorVersion: major, minorVersion: 0, patchVersion: 0)
        guard ProcessInfo.processInfo.isOperatingSystemAtLeast(minimum) else {
            throw EncoderPackageError.unsupportedOperatingSystem(
                package: manifest.package, minimum: "\(system) \(major)")
        }
    }

    /// Rejects symlinks in a destination's ancestors and makes sure its resolved path remains
    /// under the resolved package root, itself under the resolved store directory.
    @discardableResult
    private static func checkDestination(root: URL, storeDirectory: URL, path: String) throws
        -> URL
    {
        let fileManager = FileManager.default
        let resolvedStore = storeDirectory.resolvingSymlinksInPath().standardizedFileURL
        let resolvedRoot = root.resolvingSymlinksInPath().standardizedFileURL
        let rootPath = root.standardizedFileURL.path
        let storePrefix =
            resolvedStore.path.hasSuffix("/") ? resolvedStore.path : resolvedStore.path + "/"
        if (try? fileManager.destinationOfSymbolicLink(atPath: rootPath)) != nil {
            throw EncoderPackageError.symlinkedPath(path)
        }
        guard resolvedRoot.path.hasPrefix(storePrefix) else {
            throw EncoderPackageError.invalidPath(path)
        }

        var current = root.standardizedFileURL
        for component in path.split(separator: "/") {
            current.appendPathComponent(String(component))
            if (try? fileManager.destinationOfSymbolicLink(atPath: current.path)) != nil {
                throw EncoderPackageError.symlinkedPath(path)
            }
        }
        let resolvedDestination = current.resolvingSymlinksInPath().standardizedFileURL
        let rootPrefix =
            resolvedRoot.path.hasSuffix("/") ? resolvedRoot.path : resolvedRoot.path + "/"
        guard resolvedDestination.path.hasPrefix(rootPrefix) else {
            throw EncoderPackageError.invalidPath(path)
        }
        return current
    }

    /// Checks a file against its manifest entry: its size, then its SHA-256.
    ///
    /// - Parameter name: The file's name in messages, such as its path in the package.
    /// - Throws: ``EncoderPackageError/sizeMismatch(file:url:expected:actual:)`` or
    ///   ``EncoderPackageError/digestMismatch(file:url:expected:actual:)``.
    public static func verify(
        _ file: URL, against expected: EncoderPackageManifest.File, named name: String
    ) throws {
        let bytes = try size(of: file)
        guard bytes == expected.bytes else {
            throw EncoderPackageError.sizeMismatch(
                file: name, url: expected.url, expected: expected.bytes, actual: bytes)
        }
        let digest = try sha256(of: file)
        guard digest == expected.sha256.lowercased() else {
            throw EncoderPackageError.digestMismatch(
                file: name, url: expected.url, expected: expected.sha256.lowercased(),
                actual: digest)
        }
    }

    /// A file's SHA-256 in lowercase hexadecimal, read 4 MB at a time.
    public static func sha256(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        let hex = Array("0123456789abcdef".utf8)
        var text: [UInt8] = []
        for byte in hasher.finalize() {
            text.append(hex[Int(byte >> 4)])
            text.append(hex[Int(byte & 0x0F)])
        }
        return String(decoding: text, as: UTF8.self)
    }

    /// A file's size in bytes, through symbolic links: the Hugging Face cache links each
    /// snapshot file to a blob.
    private static func size(of file: URL) throws -> Int {
        let attributes = try FileManager.default.attributesOfItem(
            atPath: file.resolvingSymlinksInPath().path)
        return (attributes[.size] as? NSNumber)?.intValue ?? -1
    }
}

/// A package file that could not be downloaded, did not match its manifest, or is missing from
/// the local models folder.
public enum EncoderPackageError: Error, Sendable, Hashable, CustomStringConvertible {
    /// A file's size is not the manifest's. A download that fails this is not kept.
    case sizeMismatch(file: String, url: URL, expected: Int, actual: Int)
    /// A file's SHA-256 is not the manifest's. A download that fails this is not kept.
    case digestMismatch(file: String, url: URL, expected: String, actual: String)
    /// The server answered a download with an HTTP error.
    case httpStatus(url: URL, status: Int)
    /// The local models folder lacks the package.
    case missingLocalPackage(URL)
    /// No folder holds the tokenizer's files and the calibrator.
    case missingLocalTokenizer(files: [String], searched: [URL])
    /// The package needs a newer OS.
    case unsupportedOperatingSystem(package: String, minimum: String)
    /// A manifest's package name or file path would leave the store's folder.
    case invalidPath(String)
    /// A store destination or one of its ancestors is a symbolic link.
    case symlinkedPath(String)
    /// Remote package assets have not yet been published.
    case packageDownloadsUnavailable(String)

    /// What went wrong, naming the file and where it came from.
    public var description: String {
        switch self {
        case .sizeMismatch(let file, let url, let expected, let actual):
            return "\(file) has \(actual) bytes, but the manifest expects \(expected) for "
                + "\(url.absoluteString); the file was not used"
        case .digestMismatch(let file, let url, let expected, let actual):
            return "\(file) has SHA-256 \(actual), but the manifest expects \(expected) for "
                + "\(url.absoluteString); the file was not used"
        case .httpStatus(let url, let status):
            return "downloading \(url.absoluteString) failed with HTTP status \(status)"
        case .missingLocalPackage(let package):
            return "\(EncoderPackageStore.localModelsVariable) names a folder without "
                + "\(package.lastPathComponent): \(package.path) does not exist"
        case .missingLocalTokenizer(let files, let searched):
            return "no folder holds \(files.joined(separator: ", ")); looked in "
                + searched.map(\.path).joined(separator: " and ")
        case .unsupportedOperatingSystem(let package, let minimum):
            return "\(package) needs \(minimum) or later"
        case .invalidPath(let path):
            return "the manifest names \(path.debugDescription), which is not a relative path "
                + "inside the package's folder"
        case .symlinkedPath(let path):
            return "the store path \(path.debugDescription) traverses a symbolic link"
        case .packageDownloadsUnavailable(let package):
            return "remote files for \(package) are not yet published; set "
                + "\(EncoderPackageStore.localModelsVariable) to a local models folder"
        }
    }
}
