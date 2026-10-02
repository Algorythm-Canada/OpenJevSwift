// Where a DiffusionGemma checkpoint comes from (issue #30): a local directory or a Hugging Face
// repository at a revision, and the Hugging Face cache it is kept in.

import Foundation

/// A DiffusionGemma checkpoint: a local directory, or a Hugging Face repository at a revision.
public enum ModelSource: Sendable, Hashable, CustomStringConvertible {
    /// A directory holding `config.json`, `model.safetensors.index.json`, the shards and the
    /// tokenizer files. It is used as is, without network access.
    case directory(URL)
    /// A Hub repository such as `mlx-community/diffusiongemma-26B-A4B-it-4bit`, at a commit hash,
    /// a branch or a tag; nil is `main`. A branch or tag is resolved to its commit when loading, so
    /// a moved branch changes what loads; the presets pin a commit for that reason.
    case hub(repository: String, revision: String?)

    /// The 4-bit checkpoint every fixture and oracle was made with, at the revision
    /// THIRD_PARTY.md records: 13 files, 16.58 GB.
    public static let fourBit = ModelSource.hub(
        repository: "mlx-community/diffusiongemma-26B-A4B-it-4bit",
        revision: "a7a81407613811e8ba63af92ac0d852b809e191f")

    /// The 8-bit checkpoint, pinned to its `main` commit of 2026-07-15 (THIRD_PARTY.md). No
    /// fixture covers it.
    public static let eightBit = ModelSource.hub(
        repository: "mlx-community/diffusiongemma-26B-A4B-it-8bit",
        revision: "7b95e3887078ba56283c24f2578d6e5a06b9d7e8")

    /// The bfloat16 checkpoint, pinned to its `main` commit of 2026-07-15 (THIRD_PARTY.md). No
    /// fixture covers it.
    public static let bf16 = ModelSource.hub(
        repository: "mlx-community/diffusiongemma-26B-A4B-it-bf16",
        revision: "2cd36f950eb065c96c80810fb6b859b114cd052d")

    /// The presets, whose repositories a bare repository name resolves to at their pinned
    /// revision.
    public static let presets: [ModelSource] = [.fourBit, .eightBit, .bf16]

    /// The source a setting such as `OPENJEV_MLX_MODEL` names, as upstream hands it to
    /// mlx-vlm's `load`: a path when it starts with `/`, `~` or `.`, else a repository id,
    /// optionally followed by `@revision` (an empty one is `main`). A preset's repository without a revision takes the
    /// preset's pinned revision, so the default setting loads the pinned checkpoint.
    public init(setting: String) {
        let text = setting.trimmingCharacters(in: .whitespaces)
        if text.hasPrefix("/") || text.hasPrefix("~") || text.hasPrefix(".") {
            self = .directory(
                URL(
                    fileURLWithPath: NSString(string: text).expandingTildeInPath,
                    isDirectory: true))
            return
        }
        let parts = text.split(separator: "@", maxSplits: 1).map(String.init)
        let repository = parts.first ?? text
        if parts.count == 2 {
            self = .hub(repository: repository, revision: parts[1].isEmpty ? nil : parts[1])
            return
        }
        for case .hub(let preset, let revision) in Self.presets where preset == repository {
            self = .hub(repository: preset, revision: revision)
            return
        }
        self = .hub(repository: repository, revision: nil)
    }

    /// The directory's path, or the repository with `@revision` when a revision is set.
    public var description: String {
        switch self {
        case .directory(let url):
            return url.path
        case .hub(let repository, let revision):
            return revision.map { "\(repository)@\($0)" } ?? repository
        }
    }
}

/// The Hugging Face hub cache directory, laid out as `huggingface_hub` lays it out, so that
/// upstream OpenJev, mlx-vlm and this library share one copy of a checkpoint:
///
/// ```
/// <directory>/models--{org}--{name}/
///     blobs/<id>                      the file; <id> is its LFS SHA-256 or its git blob SHA-1
///     snapshots/<commit>/<path>       a relative symlink to ../../blobs/<id>
///     refs/<branch or tag>            the commit it resolved to
/// ```
public struct HubCacheLocation: Sendable, Hashable {
    /// The environment variable that names an access token, `HF_TOKEN`.
    public static let tokenVariable = "HF_TOKEN"

    /// The hub cache directory.
    public var directory: URL

    /// A cache at `directory`.
    public init(directory: URL) {
        self.directory = directory
    }

    /// The cache `huggingface_hub` uses for `environment`, which the caller passes; the library
    /// never reads the process environment (D-013): `HF_HUB_CACHE`, else `HF_HOME/hub`, else
    /// `XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`. Empty values count as
    /// unset and `~` is expanded.
    public init(environment: [String: String]) {
        func path(_ name: String) -> String? {
            environment[name].flatMap {
                $0.isEmpty ? nil : NSString(string: $0).expandingTildeInPath
            }
        }
        if let hub = path("HF_HUB_CACHE") {
            directory = URL(fileURLWithPath: hub, isDirectory: true)
            return
        }
        let home =
            path("HF_HOME").map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? URL(fileURLWithPath: path("XDG_CACHE_HOME") ?? NSHomeDirectory() + "/.cache")
            .appendingPathComponent("huggingface", isDirectory: true)
        directory = home.appendingPathComponent("hub", isDirectory: true)
    }

    /// `~/.cache/huggingface/hub`, the default when no variable is set.
    public static var standard: HubCacheLocation { HubCacheLocation(environment: [:]) }

    /// The access token `environment` names, `HF_TOKEN`, with an empty value treated as absent
    /// (upstream's `LayaEngine.load` drops an empty `HF_TOKEN` because laya sent it as
    /// `Authorization: Bearer `, which its HTTP client refused; compose files pass the variable
    /// through even when it is unset).
    public static func token(environment: [String: String]) -> String? {
        guard let token = environment[tokenVariable]?.trimmingCharacters(in: .whitespaces),
            !token.isEmpty
        else { return nil }
        return token
    }

    /// `models--{org}--{name}` for `repository`.
    public func repositoryDirectory(_ repository: String) -> URL {
        directory.appendingPathComponent(
            "models--" + repository.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
    }

    /// The snapshot directory of `repository` at `commit`.
    public func snapshotDirectory(_ repository: String, commit: String) -> URL {
        repositoryDirectory(repository)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(commit, isDirectory: true)
    }
}
