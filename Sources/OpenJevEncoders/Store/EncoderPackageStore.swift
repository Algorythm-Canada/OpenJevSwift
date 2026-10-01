import CryptoKit
import Foundation

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
/// snapshot in the Hugging Face cache.
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
        if let localModelsDirectory {
            return try localLocations(for: manifest, in: localModelsDirectory)
        }
        try Self.checkOperatingSystem(for: manifest)
        let fileManager = FileManager.default
        let root = directory.appendingPathComponent(manifest.package, isDirectory: true)
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var excluded = root
        try excluded.setResourceValues(values)

        let packageFolder = manifest.package + ".mlpackage"
        let entries =
            manifest.packageFiles.map { (packageFolder + "/" + $0.path, $0) }
            + (manifest.tokenizerFiles + [manifest.calibrator]).map { ("tokenizer/" + $0.path, $0) }
        let recordFile = root.appendingPathComponent("verified.json")
        var verified =
            (try? JSONDecoder().decode([String: String].self, from: Data(contentsOf: recordFile)))
            ?? [:]
        for (path, file) in entries {
            let destination = root.appendingPathComponent(path)
            if verified[path] == file.sha256.lowercased(),
                (try? Self.size(of: destination)) == file.bytes
            {
                continue
            }
            verified[path] = nil
            try await download(file, named: path, to: destination)
            verified[path] = file.sha256.lowercased()
            try JSONEncoder().encode(verified).write(to: recordFile, options: .atomic)
        }
        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        return EncoderPackageLocations(
            packageDirectory: root.appendingPathComponent(packageFolder, isDirectory: true),
            tokenizerDirectory: tokenizer,
            calibratorFile: tokenizer.appendingPathComponent(manifest.calibrator.path))
    }

    /// Downloads one file to a temporary file, checks it and moves it into place.
    private func download(
        _ file: EncoderPackageManifest.File, named name: String, to destination: URL
    ) async throws {
        let (temporary, response) = try await session.download(from: file.url)
        defer { try? FileManager.default.removeItem(at: temporary) }
        if let response = response as? HTTPURLResponse, !(200..<300).contains(response.statusCode) {
            throw EncoderPackageError.httpStatus(url: file.url, status: response.statusCode)
        }
        try Self.verify(temporary, against: file, named: name)
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: temporary, to: destination)
    }

    /// The package in the local models folder, and the first folder that holds the tokenizer
    /// and the calibrator: `{package}/tokenizer/` there, else the checkpoint's Hugging Face
    /// snapshot.
    private func localLocations(for manifest: EncoderPackageManifest, in local: URL) throws
        -> EncoderPackageLocations
    {
        let fileManager = FileManager.default
        let package = local.appendingPathComponent(
            manifest.package + ".mlpackage", isDirectory: true)
        guard fileManager.fileExists(atPath: package.path) else {
            throw EncoderPackageError.missingLocalPackage(package)
        }
        var candidates = [
            local.appendingPathComponent(manifest.package, isDirectory: true)
                .appendingPathComponent("tokenizer", isDirectory: true)
        ]
        if let huggingFaceHubDirectory {
            candidates.append(manifest.checkpoint.snapshot(in: huggingFaceHubDirectory))
        }
        let names = (manifest.tokenizerFiles + [manifest.calibrator]).map(\.path)
        guard
            let tokenizer = candidates.first(where: { folder in
                names.allSatisfy {
                    fileManager.fileExists(atPath: folder.appendingPathComponent($0).path)
                }
            })
        else {
            throw EncoderPackageError.missingLocalTokenizer(files: names, searched: candidates)
        }
        return EncoderPackageLocations(
            packageDirectory: package, tokenizerDirectory: tokenizer,
            calibratorFile: tokenizer.appendingPathComponent(manifest.calibrator.path))
    }

    /// Refuses a manifest whose package name or file paths would leave the store's folder: an
    /// empty or absolute path, or one with an empty, `.` or `..` component.
    private static func checkPaths(of manifest: EncoderPackageManifest) throws {
        let files = manifest.packageFiles + manifest.tokenizerFiles + [manifest.calibrator]
        for path in [manifest.package] + files.map(\.path) {
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
        }
    }
}
