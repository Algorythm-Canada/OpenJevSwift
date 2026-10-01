import Foundation
import OpenJevDiffusionGemma
import Testing

/// The real Hub, opt-in through `OPENJEV_TEST_DOWNLOAD=1`: two small files of the pinned
/// checkpoint downloaded into a temporary cache in huggingface_hub's layout.
@Suite(
    "Download from the Hugging Face Hub",
    .enabled(
        if: ProcessInfo.processInfo.environment["OPENJEV_TEST_DOWNLOAD"] == "1",
        "OPENJEV_TEST_DOWNLOAD is unset; set it to 1 to download from the Hugging Face Hub"))
struct HubDownloadTests {
    @Test(
        "tokenizer_config.json and chat_template.jinja of the pinned 4-bit revision match the fixture digests"
    )
    func pinnedTokenizerFiles() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-hub-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = HubCacheLocation(directory: directory)
        guard case .hub(let repository, let revision?) = ModelSource.fourBit else {
            Issue.record("the 4-bit preset is not a pinned Hub source")
            return
        }
        let names: Set<String> = ["tokenizer_config.json", "chat_template.jinja"]
        let token = HubCacheLocation.token(environment: ProcessInfo.processInfo.environment)
        let resolution = try await ModelResolver().resolution(
            of: .fourBit, cache: cache, token: token, files: names)
        #expect(resolution.directory == cache.snapshotDirectory(repository, commit: revision))
        #expect(resolution.downloadedFiles == 2)

        let digests = try Self.fixtureDigests()
        for name in names.sorted() {
            let link = resolution.directory.appendingPathComponent(name)
            let target = try FileManager.default.destinationOfSymbolicLink(atPath: link.path)
            #expect(target.hasPrefix("../../blobs/"))
            #expect(try TokenizerFiles.sha256Hex(of: link) == digests[name])
        }
        // A pinned revision writes no ref.
        #expect(
            !FileManager.default.fileExists(
                atPath: cache.repositoryDirectory(repository).appendingPathComponent("refs").path))
    }

    /// Fixtures/tokenizer/special_tokens.json's `files`, name to SHA-256.
    static func fixtureDigests() throws -> [String: String] {
        struct File: Decodable {
            struct Entry: Decodable { let sha256: String }
            let files: [String: Entry]
        }
        let url = TokenizerFixtures.fixturesDirectory.appendingPathComponent(
            "tokenizer/special_tokens.json")
        return try JSONDecoder().decode(File.self, from: Data(contentsOf: url)).files
            .mapValues(\.sha256)
    }
}
