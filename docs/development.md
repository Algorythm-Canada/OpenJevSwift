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
skips the shaders. Its products contain no Metal library, and MLX cannot run without one: creating
an MLX stream loads the library, so even work on the CPU stops with
`Failed to load the default metallib` (issue #8). A native build is enough only while no code it
runs calls into MLX. Swift 6.4 deprecates that build system. Swift 6.3 (Xcode 26.4 to 26.6) still
defaults to it, which is why the CI macOS job passes `--build-system swiftbuild`.

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

That works while no test runs MLX code, for the reason above.

### MLX in tests

A test that runs MLX code must point MLX at its Metal library before the first MLX call. Swift
Build copies the library into every test bundle that links MLX, as
`Contents/Resources/mlx-swift_Cmlx.bundle`. MLX looks for that bundle through `Bundle` objects,
though, and the Swift Testing runner creates none for the test bundle, so without help the lookup
fails with `Failed to load the default metallib` (issue #8). The helper below sets MLX's
`GPU.metallib` override once; call `MetalLibrary.configure()` at the start of every MLX test.

```swift
import Foundation
import MLX

/// Points MLX at the Metal library that Swift Build copies into this test bundle.
///
/// MLX looks for the library through `Bundle` objects, and the Swift Testing runner creates none
/// for the test bundle. Call `configure()` before the first MLX call in a test.
enum MetalLibrary {
    private final class Token {}

    private static let configured: Void = {
        let url = Bundle(for: Token.self).resourceURL?.appending(
            path: "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        if let url, FileManager.default.fileExists(atPath: url.path) {
            GPU.metallib = url
        }
    }()

    static func configure() {
        configured
    }
}
```

MLX runs on the default device, the GPU, unless a test chooses the CPU with
`Device.withDefaultDevice(.cpu)`. GPU results differ from the CPU's in the last digits, so MLX
tests compare within tolerances (D-014). A synthetic test must fit the hosted runner: 7 GB of
memory, of which Metal recommends at most 4.7 GB, and three CPU cores.

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
no weights are downloaded, and generated fixture outputs are confined to `Fixtures/`. Running `make fixtures`
twice gives no diff. `FixturePinTests` in `OpenJevCoreTests` fails when a file records another
upstream commit or tokenizer revision.

## Continuous integration

Three workflows run on GitHub-hosted runners. The repository is public, so the standard runners
cost nothing. No workflow uses the billed `-xlarge` runners unless asked to.

| Workflow | When it runs | Job | Runner | What it runs |
|---|---|---|---|---|
| [ci.yml](../.github/workflows/ci.yml) | Every pull request and every push to `main` | `Linux` | `ubuntu-24.04` with the `swift:6.2-noble` container | `swift build --build-tests` and `swift test`, both with `--scratch-path .build/linux`, then the test log check |
| | | `macOS` | `macos-26` with Xcode 26.6, selected with `DEVELOPER_DIR` | `swift build --build-tests` and `swift test`, both with `--build-system swiftbuild`, the test log check, then `make lint` |
| [fixtures.yml](../.github/workflows/fixtures.yml) | Pull requests that change `Fixtures/`, `Tools/fixtures/`, `THIRD_PARTY.md`, the `Makefile` or the workflow; manual runs; Mondays at 06:23 UTC | `Regenerate the fixtures` | `macos-26` with CPython 3.14.7 from `actions/setup-python` | `make upstream`, `make fixtures-venv` and `make fixtures`, then fails if `git status --porcelain Fixtures/` lists a file, and prints and uploads the diff |
| [mlx-probe.yml](../.github/workflows/mlx-probe.yml) | Manual runs only | `Probe <label>` | `macos-15`, `macos-26` and `xcode-27`, plus `macos-26-xlarge` when asked | The MLX runner probe of issue #8. Its findings are under R13 in [07-risks-and-unknowns.md](07-risks-and-unknowns.md). |

The runs of 2026-09-30 reported these images and toolchains:

| Job | Image | Toolchain |
|---|---|---|
| `Linux` | `ubuntu-24.04` 20260920.314.1, `swift:6.2-noble` | Swift 6.2.4 (`swift-6.2.4-RELEASE`) |
| `macOS` | `macos-26-arm64` 20260907.0351.1: macOS 26.6.2, Apple M1 (virtual), 3 CPUs, 7 GB | Xcode 26.6 (17F113), Swift 6.3.3, Metal Toolchain installed with the image |
| `Regenerate the fixtures` | `macos-26-arm64` 20260907.0351.1 | CPython 3.14.7 |

And took this long:

| Job | Empty cache | Warm cache |
|---|---|---|
| `Linux` | 3.5 minutes: build 128 s, tests 14 s | 1.5 minutes: build 26 s, tests 20 s |
| `macOS` | 8 minutes: build 270 s with Swift Build (355 s with the native build system), tests 52 s | 3 minutes: cache restore 33 s, build 76 s, tests 37 s |
| `Regenerate the fixtures` | 41 s | 39 s |

- **Why Swift Build on macOS.** Xcode 26.6's `swift build` uses the native build system, which does
  not compile mlx-swift's Metal shaders, and MLX cannot run at all without them. Swift Build
  compiles them with the Metal Toolchain, which the image installs for every Xcode 26; the job
  downloads it (839 MB) if an image ever lacks it. Swift 6.3.3 meets mlx-swift's tools 6.3 floor.
  With the library built, the `OpenJevDiffusionGemma` tests can run MLX on the runner's GPU
  (D-028). Swift Build runs each test target as its own test bundle, so the macOS log holds one
  Swift Testing run per target.
- **Why not `xcode-27`.** That label is a public preview: macOS 27.0 with Xcode 27.0 (Swift 6.4),
  the reference toolchain, but without the Metal Toolchain, which a job must download first (839 MB,
  about 20 s in the probe). The macOS job moves to it when Xcode 27 reaches a generally available
  image.
- **The test log check.** [check-test-log.sh](../.github/scripts/check-test-log.sh) reads the
  saved output of `swift test`. It fails when the log holds no Swift Testing run or a run failed,
  and when a test or suite was skipped for any reason other than an unset `OPENJEV_TEST_MODEL` or
  `OPENJEV_LIVE_URL`. CI has every fixture, so a fixture test that skipped there would stop testing
  without failing. A test that needs the weights names `OPENJEV_TEST_MODEL` in the comment of its
  `.enabled(if:)` trait, and the check lists it as skipped. Both jobs unset the two variables
  before `swift test`.
- **Caches.** Each CI job caches its build directory, `.build/linux` or `.build`, which holds the
  SwiftPM checkouts, the clones they come from and the build products. The key is the compiler's
  build identifier, the macOS build system and the hash of `Package.resolved`, so a new toolchain
  or a dependency change starts from an empty directory. A cache that `main` saved serves every
  pull request; one that a pull request saved serves only that pull request. The fixtures job
  caches pip downloads, keyed on `requirements.txt`, and the tokenizer download, keyed on the
  tokenizer revision.
- **Superseded runs.** A newer push to a pull request cancels the run it replaces. On `main`, a
  run that has started always finishes, so every merged commit keeps its result.
- **Fixture regeneration.** The job runs on Apple silicon because the committed files were written
  there: CPython takes its math functions from the platform's C library, and another library could
  change the last digit of a float. It reproduced every committed file byte for byte.
  `FixturePinTests`, in the CI workflow, checks the pins the files record; this job checks that
  the scripts still write the files.
- **The MLX runner probe.** Run it again when mlx-swift, Xcode or a runner image changes, from the
  repository's Actions tab or with `gh workflow run mlx-probe.yml`. Each job writes a table of
  outcomes and the probe's output to the run summary.

### Running the CI commands locally

The Linux job, from the repository root, with Docker:

```bash
docker run --rm -v "$PWD":/src -w /src swift:6.2-noble bash -o pipefail -c 'swift build --build-tests --scratch-path .build/linux && swift test --scratch-path .build/linux 2>&1 | tee .build/linux/test.log && .github/scripts/check-test-log.sh .build/linux/test.log'
```

The macOS job, with Xcode 26.6 installed:

```bash
export DEVELOPER_DIR=/Applications/Xcode_26.6.app/Contents/Developer
swift build --build-tests --build-system swiftbuild
swift test --build-system swiftbuild 2>&1 | tee .build/test.log
.github/scripts/check-test-log.sh .build/test.log
make lint
```

With Xcode 27, the reference toolchain, leave out `DEVELOPER_DIR` and `--build-system swiftbuild`:
Swift Build is already the default.

The fixture check:

```bash
make upstream
make fixtures-venv
make fixtures
git status --porcelain Fixtures/
```

The last command prints nothing when the fixtures are current.

### Required checks

The workflows do not change repository settings. Protect `main` by requiring the `Linux` and
`macOS` checks, both from the CI workflow. The names carry no toolchain version, so a toolchain
upgrade does not rename them. Do not require `Regenerate the fixtures`: it runs only when a pull
request changes the fixture inputs, and a required check that never reports keeps the pull
request waiting. The probe is manual and is never a required check.
