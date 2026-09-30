# Development

How to build, test and format OpenJevSwift. The design lives in
[05-architecture.md](05-architecture.md). This page covers toolchains and day-to-day commands.

## Toolchains

| Build | Minimum | Why |
|---|---|---|
| Linux: `OpenJevCore`, `OpenJevServer`, `openjev` and their tests | Swift 6.2 | The manifest declares `swift-tools-version: 6.2`, the minimum that mlx-swift-lm accepts. |
| macOS: every target | Swift 6.3 (Xcode 26.4 or later) | mlx-swift 0.32.2 declares `swift-tools-version: 6.3`. Older toolchains cannot load its manifest. |

The reference development machine is an Apple silicon Mac with macOS 27.0 and Xcode 27.0, which
ships Swift 6.4.0.

Every target compiles in the Swift 6 language mode, which turns on complete strict concurrency
checking.

### Xcode 27 needs the Metal Toolchain

Swift 6.4 changed the default engine of `swift build` from the native build system to Swift Build.
Swift Build compiles the Metal shaders that mlx-swift ships as sources of its `Cmlx` target, and
that needs Xcode's Metal Toolchain component, which is a separate download. Without it the build
stops with `cannot execute tool 'metal' due to missing Metal Toolchain`. Install it once:

```bash
xcodebuild -downloadComponent MetalToolchain
```

`swift build --build-system native` works without the component because the native build system
skips the shaders. Its products contain no Metal library for MLX's GPU kernels, and Swift 6.4
deprecates that build system. Swift 6.3 (Xcode 26.4 to 26.6) still defaults to it. Whether MLX
kernels run in tests and on hosted runners is the question of issue #8.

## Targets and platforms

The package declares macOS 14 and iOS 17 as minimum deployment targets.

| Target | macOS 14+ | iOS 17+ | Linux |
|---|---|---|---|
| `OpenJevCore` (library product) | yes | yes | yes |
| `OpenJevDiffusionGemma` (library product) | yes, Apple silicon | compiles, but the model does not fit in memory | not declared |
| `OpenJevServer` (internal target) | yes | no | yes |
| `openjev` (executable) | yes | no | yes |

- **Apple-only code.** `Package.swift` declares the MLX packages and the `OpenJevDiffusionGemma`
  targets inside `#if os(macOS)`. SwiftPM evaluates a manifest on the host, and every Apple
  platform build runs on a macOS host, so on Linux those targets do not exist and SwiftPM never
  loads or builds the MLX packages.
- **The server.** `OpenJevServer` is not a library product, so library consumers never link
  Hummingbird (D-009). Its Hummingbird dependency applies only on macOS and Linux, which keeps
  Hummingbird out of iOS builds.

## Dependencies

| Package | Requirement | Resolved | Products used | Declared on |
|---|---|---|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | exactly 0.32.2 | 0.32.2 | `MLX`, `MLXNN` | macOS hosts |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | revision `c043fb3b1ccf00f54ef8882a1e8da45c6e32e6f8` | that revision | `MLXLMCommon`, `MLXVLM` | macOS hosts |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.0 up to the next minor | 1.3.4 | `Tokenizers` | macOS hosts |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | 2.23.0 or later | 2.27.0 | `Hummingbird` | all hosts |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | 1.8.0 or later | 1.8.2 | `ArgumentParser` | all hosts |

`Package.resolved` is committed. It pins these five packages and their 28 transitive dependencies.
`swift-collections` is not a direct dependency. Issue #3 adds it if the JSON model adopts
`OrderedDictionary`. The product names match [05-architecture.md](05-architecture.md).

- **swift-transformers lags the researched commit.** [THIRD_PARTY.md](../THIRD_PARTY.md) records
  `af520cf` from `main`. The 1.3.x requirement resolves to the 1.3.4 tag, which lacks three later
  commits that change `byte_fallback` handling in the BPE and Unigram tokenizers. The tokenizer
  parity spike (#20) should test the resolved version.
- **mlx-swift-lm is pinned by revision.** No release contains `c043fb3`: the newest tag, 3.31.4,
  is 154 commits older and requires mlx-swift 0.31. SwiftPM refuses a revision-pinned dependency
  inside a package that another package requires by version, so on macOS a downstream package can
  depend on OpenJevSwift only by branch, revision or local path. Linux is unaffected because its
  manifest has no MLX packages. Release 0.1.0 (#65) needs an mlx-swift-lm tag at or after the pin.
- **Linux downloads the Apple-only pins.** When `Package.resolved` matches the manifest, SwiftPM
  checks out every pinned package before it computes the graph. A Linux build therefore downloads
  8 Apple-only packages, including mlx-swift and swift-syntax, without loading or building them.
  `Package.resolved` is not modified on Linux.
- **Changing a dependency URL.** SwiftPM keeps the old location in `.build`. Run
  `swift package reset` before `swift package resolve` so the committed file records the new one.

The first resolution on a clean machine clones 33 packages, including swift-syntax, and takes
about a minute.

## Building

On macOS, with the Metal Toolchain installed:

```bash
swift build
```

On Linux, the official `swift` images are enough. From the repository root, with Docker:

```bash
docker run --rm -v "$PWD":/src -w /src swift:6.2-noble swift build --build-tests --scratch-path .build/linux
```

The separate scratch path keeps the Linux build products apart from the macOS ones.

## Format, lint and test

| Command | Runs |
|---|---|
| `make format` | `swift format format --in-place --recursive --parallel Package.swift Sources Tests` |
| `make lint` | `swift format lint --strict --recursive --parallel Package.swift Sources Tests`, which fails on any violation |
| `make test` | `swift test`, which runs every test target that exists on the platform |

swift-format ships with the Swift toolchain. Its configuration, `.swift-format`, is swift-format's
default configuration with four-space indentation and a 100-column line length, and it lists every
rule explicitly. Tests use Swift Testing.

The `swift` images do not include `make`. Inside a container, install it with
`apt-get update && apt-get install -y make`, or run the commands in the table directly.

On macOS without the Metal Toolchain, run the tests with the native build system:

```bash
swift test --build-system native
```

## Upstream reference source

The issues cite upstream OpenJev by file and line (for example `api.py:128-146`). Check the pinned
commit out inside the project so those references resolve without searching:

```bash
make upstream
```

This clones `razorback16/openjev` into `Upstream/openjev` and checks out `dcd2094`, the commit
recorded in [THIRD_PARTY.md](../THIRD_PARTY.md). `Upstream/` is ignored by git; nothing under it
is ever committed. When the pin moves, update `UPSTREAM_OPENJEV_COMMIT` in the `Makefile`,
`THIRD_PARTY.md` and the fixture pins together.

## Fixtures

The golden fixtures in [Fixtures/](../Fixtures/README.md) are generated from upstream's code at
the pinned commit and, for most of them, the pinned DiffusionGemma tokenizer. Regenerate them
from the repository root:

```bash
make upstream
make fixtures-venv
make fixtures
```

| Command | Runs |
|---|---|
| `make fixtures-venv` | Creates `Tools/fixtures/.venv` from `python3.14` (override with `PYTHON=...`) and installs `Tools/fixtures/requirements.txt` |
| `make fixtures` | `python_json_tables.py`, `wire_tables.py` and `upstream_tables.py`, with `PYTHONHASHSEED=0` |

The committed files were written by CPython 3.14.7 with the pinned packages. Another Python
version changes the `python` pin recorded in every file, so use 3.14.7 to reproduce them byte
for byte. The first run downloads the tokenizer files (about 32 MB) into the Hugging Face cache;
no weights are downloaded and nothing outside `Fixtures/` is written. Running `make fixtures`
twice gives no diff. `FixturePinTests` in `OpenJevCoreTests` fails when a file records another
upstream commit or tokenizer revision.

## Continuous integration

Issue #7 adds the workflows. They are intended to run here:

| Job | Runner | Toolchain | Why |
|---|---|---|---|
| Linux | `ubuntu-24.04` with the `swift:6.2-noble` container | Swift 6.2.4 | Holds the declared tools floor for the targets Linux builds. |
| macOS | `macos-26`, Apple silicon, generally available | Xcode 26.6 (Swift 6.3.3), selected with `DEVELOPER_DIR=/Applications/Xcode_26.6.app` | The newest generally available Apple silicon image. Swift 6.3.3 meets mlx-swift's tools 6.3 floor and still defaults to the native build system, so the job needs no Metal Toolchain download. |

The `xcode-27` label is a public preview with Xcode 27.0 (Swift 6.4.0), the reference toolchain.
That image does not include the Metal Toolchain. mlx-swift-lm's own workflow downloads it
(839 MB) before building.
