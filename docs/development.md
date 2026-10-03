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
| `OpenJevEncoders` (library product) | yes; its Core ML types from macOS 15 | yes; its Core ML types from iOS 18 | not declared |
| `OpenJevLetterReadout` (library product) | yes, Apple silicon | compiles; loads the 4-bit conversion (2.37 GB) by default, meant for recent iPhones and iPads, not yet run on one | not declared |
| `OpenJevServer` (internal target) | yes | no | yes |
| `openjev` (executable) | yes, with the encoder and JevK5 backends | no | yes, without the encoder, JevK5 and MLX backends |
| `openjev-stub-server` (executable, not a product) | yes | no | yes |
| `openjev-bench` (executable, not a product) | yes, Apple silicon | no | not declared |

- **Apple-only code.** `Package.swift` declares the MLX packages, swift-transformers and the
  `OpenJevDiffusionGemma`, `OpenJevEncoders` and `OpenJevLetterReadout` targets inside
  `#if os(macOS)`. SwiftPM evaluates
  a manifest on the host, and every Apple platform build runs on a macOS host, so on Linux those
  targets do not exist and SwiftPM never loads or builds the MLX packages.
- **The encoders.** `OpenJevEncoders` runs Verdict and Laya on Core ML (D-011). Its Core
  ML types are `@available(macOS 15, iOS 18, *)`, because the multifunction packages need those
  versions, while the package keeps its macOS 14 and iOS 17 floors. `VerdictBackend.load` and
  `LayaBackend.load` build for those floors and throw `EncoderLoadError.unsupportedOperatingSystem`
  on an older OS, where D-011 sends the reads to a server.
- **JevK5.** `OpenJevLetterReadout` runs JevK5 (`jevk5-0.2`) on MLX through mlx-swift-lm's
  `MLXLLM` Qwen3.5 text model (issue #55). It depends on `OpenJevDiffusionGemma` for
  `ModelResolver`, the downloader that keeps the Hugging Face cache's layout, which links that
  module into an app that uses JevK5 (D-052). Its live tests need the conversion
  `Tools/jevk5/convert.py` writes, named by `OPENJEV_JEVK5_MODEL`.
- **The server.** `OpenJevServer` is not a library product, so library consumers never link
  Hummingbird (D-009). Its Hummingbird dependency applies only on macOS and Linux, which keeps
  Hummingbird out of iOS builds.
- **The CLI.** The `openjev` target is declared for every host, and the `#if os(macOS)` block
  appends its dependency on `OpenJevEncoders`; the code that uses it is behind
  `#if canImport(OpenJevEncoders)`. A Linux build therefore has the CLI without the encoder
  backends, and `OPENJEV_BACKEND=verdict` and `laya` exit with status 3 there, as `mlx` and
  `jevk5` do without MLX.
- **The stub server.** `openjev-stub-server` serves OpenJevTestSupport's stub backends through
  the real application, for the SDK compatibility suite. It is an executable target without a
  product, so `swift build` builds it, `swift build --product openjev-stub-server` builds it alone,
  and nothing ships it.
- **The bench.** `openjev-bench` measures DiffusionGemma reads (issue #32, below). It needs MLX,
  so it is declared inside `#if os(macOS)` beside `OpenJevDiffusionGemma`; it has no product and
  nothing ships it. `OpenJevBenchTests` tests its model-free code through `@testable import`.
- **The iOS scheme.** `.swiftpm/xcode/xcshareddata/xcschemes/OpenJevCore-iOS.xcscheme` is a
  committed Xcode scheme that builds `OpenJevCore`, `OpenJevEncoders` and their test targets and
  runs `OpenJevCoreTests` and `OpenJevEncodersTests`. The CI iOS job builds and tests it on a
  simulator. Everything the scheme builds must compile without Hummingbird and without MLX-only
  APIs: server code stays out of the scheme, or behind `#if canImport(Hummingbird)`. Add every
  later iOS-capable target to the same scheme, except `OpenJevLiveTests`, which tests a running
  server rather than the iOS build (D-043), and the MLX targets, `OpenJevDiffusionGemma` and
  `OpenJevLetterReadout`, which the iOS job builds for the simulator in steps of their own. A package opens in Xcode with autogenerated
  schemes, which are per-user and are never written to disk, so this one is committed.
  `.gitignore` allows the scheme files in that one directory, which is where Xcode writes a scheme
  marked Shared, and keeps ignoring the rest of `.swiftpm`, `xcuserdata` and the per-user scheme
  management plist included.

## Dependencies

| Package | Requirement | Resolved | Products used | Declared on |
|---|---|---|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | exactly 0.32.2 | 0.32.2 | `MLX`, `MLXNN` | macOS hosts |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | revision `c043fb3b1ccf00f54ef8882a1e8da45c6e32e6f8` | that revision | `MLXLMCommon`, `MLXVLM` | macOS hosts |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.0 up to the next minor | 1.3.4 | `Tokenizers` | macOS hosts |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | 2.4.2 or later | 2.5.1 | `Jinja` | macOS hosts |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | 2.23.0 or later | 2.27.0 | `Hummingbird`, `HummingbirdCore` (server), `HummingbirdTesting` (server and CLI tests) | all hosts |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | 1.8.0 or later | 1.8.2 | `ArgumentParser` | all hosts |
| [swift-http-types](https://github.com/apple/swift-http-types) | 1.8.0 or later | 1.8.0 | `HTTPTypes` (server, server and CLI tests) | all hosts |
| [swift-log](https://github.com/apple/swift-log) | 1.15.1 or later | 1.15.1 | `Logging` (server, CLI, server and CLI tests) | all hosts |
| [swift-nio](https://github.com/apple/swift-nio) | 2.103.0 or later | 2.103.0 | `NIOCore` (server and server tests), `NIOPosix` and `NIOHTTP1` (server), `NIOEmbedded` (server tests) | all hosts |
| [swift-service-lifecycle](https://github.com/swift-server/swift-service-lifecycle) | 2.12.0 or later | 2.12.0 | `ServiceLifecycle` (server, CLI, stub server, server tests), `UnixSignals` (CLI, stub server) | all hosts |
| [async-http-client](https://github.com/swift-server/async-http-client) | 1.36.2 or later | 1.36.2 | `AsyncHTTPClient` (server and server tests) | all hosts |
| [swift-docc-plugin](https://github.com/swiftlang/swift-docc-plugin) | 1.5.0 or later | 1.5.0 | none: its `generate-documentation` and `preview-documentation` commands build the DocC catalogs | all hosts |

`Package.resolved` is committed. It pins these twelve packages and their 23 transitive
dependencies. swift-http-types, swift-log, swift-nio, swift-service-lifecycle and
async-http-client are Hummingbird's own dependencies, declared at the versions it already
resolved, so declaring them changed no pin. AsyncHTTPClient forwards a routed model's request
(D-040); it brings swift-nio-ssl and its BoringSSL into the `openjev` binary.
`swift-collections` is not a direct dependency. Issue #3 adds it if the JSON model adopts
`OrderedDictionary`. swift-docc-plugin is a command plugin, with swift-docc-symbolkit as its one
dependency; no target links either ([API documentation](#api-documentation)). The product names
match [05-architecture.md](05-architecture.md).

- **swift-transformers lags the researched commit.** [THIRD_PARTY.md](../THIRD_PARTY.md) records
  `af520cf` from `main`. The 1.3.x requirement resolves to the 1.3.4 tag, which lacks three later
  commits that change `byte_fallback` handling in the BPE and Unigram tokenizers. The tokenizer
  parity spike (#20) should test the resolved version.
- **mlx-swift-lm is pinned by revision.** No release contained `c043fb3` when it was pinned: the
  newest tag then, 3.31.4, was 154 commits older and required mlx-swift 0.31. Release 3.32.3
  (2026-09-30) is the first that contains it, five commits later, and requires mlx-swift 0.32.3;
  moving to it is #119 ([upstream-log.md](upstream-log.md)). Until then, SwiftPM refuses a
  revision-pinned dependency inside a package that another package requires by version, so on
  macOS a downstream package can depend on OpenJevSwift only by branch, revision or local path.
  Linux is unaffected because its manifest has no MLX packages. Release 0.1.0 (#65) needs an
  mlx-swift-lm tag at or after the pin, which 3.32.3 now is.
- **Linux downloads the Apple-only pins.** When `Package.resolved` matches the manifest, SwiftPM
  checks out every pinned package before it computes the graph. A Linux build therefore downloads
  8 Apple-only packages, including mlx-swift and swift-syntax, without loading or building them.
  `Package.resolved` is not modified on Linux.
- **Changing a dependency URL.** SwiftPM keeps the old location in `.build`. Run
  `swift package reset` before `swift package resolve` so the committed file records the new one.

The first resolution on a clean machine clones 35 packages, including swift-syntax, and takes
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

### The CLI tests

`OpenJevCLITests` imports the `openjev` executable's module to parse command lines and to run the
commands in-process with stub backends, which its `CommandContext` registers. It also runs the
built binary as a child process: `swift build --build-tests` and `swift test` build it into the
folder that holds the test bundles, where the tests look for it, and a test fails when it is not
there. The smoke test serves Verdict from the binary and sends it Jev's quickstart request; it
needs the converted package and the tokenizer, like the encoder tests, and skips naming
`OPENJEV_ENCODER_MODELS` without them.

### The SDK compatibility suite

`Tools/sdk-compat` runs TypeSafe's Python and TypeScript SDKs, at the versions it pins, against
`openjev-stub-server` ([Tools/sdk-compat/README.md](../Tools/sdk-compat/README.md)). It needs
CPython 3.12 and Node.js 20 or later. From the repository root, once:

```bash
make sdk-compat-venv
```

Then, which builds the stub server first:

```bash
make sdk-compat
```

`make sdk-compat SDK_COMPAT_ARGS=--swift-sdk` also builds and runs NSStudent's JevSwiftSDK, as CI
does. A failed check prints every HTTP exchange it made, and every exchange and the servers' logs
are written to `Tools/sdk-compat/exchanges/`.

### The live suite

`OpenJevLiveTests` (issue #41, D-043) is upstream's `tests/test_live.py` in Swift: it sends HTTP
requests to a running server and checks the answers, the headers and Jev's shapes. It links no
server and no backend, so it runs on macOS and Linux against any implementation. Without
`OPENJEV_LIVE_URL` the live tests skip naming it, which is how `swift test` and CI run them; the
tests of the suite's own settings, listing decoding and cancellation run everywhere.

| Variable | Meaning |
|---|---|
| `OPENJEV_LIVE_URL` | The server, such as `http://127.0.0.1:8080`. Unset or empty: every test skips. |
| `OPENJEV_LIVE_KEY`, else `OPENJEV_API_KEY` | Sent as `Authorization: Bearer <key>`, for a server with `OPENJEV_API_KEY` set. |
| `OPENJEV_ORIGIN_SECRET` | Sent as `X-Origin-Secret`. |
| `OPENJEV_LIVE_GATEWAY=1` | A gateway in front strips `server-timing`, so the suite does not require it. |

The server's `/v1/models` listing decides what runs: the DiffusionGemma tests when it lists
`openjev-latest`, `test_encoder` for each encoder model it lists, and the unknown-model test
always. Start the Swift server with the backend to test, Verdict here. `laya` and `mlx` work the
same way: the encoder packages download on first use unless `OPENJEV_ENCODER_MODELS` names a
folder of converted ones, and `mlx` loads the 16 GB checkpoint that `OPENJEV_MLX_MODEL` names, a
directory or a Hub repository in the Hugging Face cache, downloaded the first time.

```bash
swift build -c release --product openjev
OPENJEV_BACKEND=verdict OPENJEV_HOST=127.0.0.1 OPENJEV_PORT=8080 "$(swift build -c release --show-bin-path)/openjev" serve
```

Then, from another shell:

```bash
OPENJEV_LIVE_URL=http://127.0.0.1:8080 swift test --filter OpenJevLiveTests
```

Upstream's Python server takes the same suite, from the pinned checkout (`make upstream`). Its
encoder backends run in `Tools/jevbench/.venv` (see [The JevBench harness](#the-jevbench-harness)),
on a Mac on the CPU in float32:

```bash
PYTHONPATH=Upstream/openjev OPENJEV_BACKEND=verdict OPENJEV_HOST=127.0.0.1 OPENJEV_PORT=8081 Tools/jevbench/.venv/bin/python -m openjev
```

Its `mlx` backend needs mlx-vlm, which `Tools/oracle/.venv` has (CPython 3.14); start it the same
way with `OPENJEV_BACKEND=mlx` and that interpreter:

```bash
python3.14 -m venv Tools/oracle/.venv
Tools/oracle/.venv/bin/python -m pip install -r Tools/oracle/requirements.txt
```

Without `OPENJEV_VERDICT_MODEL`, `OPENJEV_LAYA_MODEL` or `OPENJEV_MLX_MODEL`, upstream loads the
Hub's newest revision of its checkpoint; `upstream_server` in `Tools/jevbench/servers.py` shows
the environment that reads the pinned snapshot instead, as the recorded runs did.

Upstream's own file runs against either server with pytest and httpx, in a virtual environment
of its own under the ignored `Upstream/` folder; `Tools/jevbench/.venv` keeps to its lock:

```bash
python3.12 -m venv Upstream/.venv
Upstream/.venv/bin/python -m pip install pytest httpx==0.28.1
OPENJEV_LIVE_URL=http://127.0.0.1:8080 Upstream/.venv/bin/python -m pytest Upstream/openjev/tests/test_live.py -v
```

Against the Swift server's `mlx` backend, its image, `think`, chat and stream tests fail until
#48, #52 and #53 land, while the Swift suite skips them. The chat tests run there rather than
skip because the listing already names `diffusiongemma-26b`, as upstream's does.

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

## API documentation

Each library module has a DocC catalog, `Sources/<module>/Documentation.docc`, whose root page
curates the module's symbols, with these articles:

| Module | Articles |
|---|---|
| `OpenJevCore` | Making decisions in an app; Requests and answers; Implementing a backend |
| `OpenJevServer` | Running the server; Configuration reference |
| `OpenJevDiffusionGemma` | Reading with DiffusionGemma |
| `OpenJevEncoders` | Reading Verdict and Laya |
| `OpenJevLetterReadout` | none; its root page shows a read |

The swift-docc-plugin dependency adds `swift package generate-documentation` and
`swift package preview-documentation`. Build the published site, on a Mac:

```bash
make docs
```

`make docs` runs [Tools/docs/build-site.sh](../Tools/docs/build-site.sh): one
`generate-documentation` call over the four targets with
`--enable-experimental-combined-documentation`, which builds each archive with
`--transform-for-static-hosting --hosting-base-path OpenJevSwift` and merges them into one site
with a shared sidebar, then [Tools/docs/index.html](../Tools/docs/index.html) as the front page. It
passes `--exclude-extended-types` too (see Links below), and every DocC warning is an error. The
site, about 27 MB, is in `.build/docs-site`; the script takes another folder as its argument.
Serve it at the path GitHub Pages serves it under, <http://localhost:8000/OpenJevSwift/>:

```bash
make docs-preview
```

While writing one module's pages, the plugin's live preview rebuilds on every save and serves
<http://localhost:8080/documentation/openjevcore>:

```bash
swift package --disable-sandbox preview-documentation --target OpenJevCore
```

- **Links.** A link inside a module is ``` ``Symbol`` ```. A link to another module's symbol or
  article is absolute, with the module first, ``` ``/OpenJevCore/DecisionEngine`` ``` or
  `<doc:/OpenJevCore/GettingStarted>`, and resolves only when the module's archive is built with
  its dependencies', as the combined build does; a single-target build or preview warns about it.
  `OpenJevServer` extends two core types, and the page DocC makes for those extensions is named
  `OpenJevCore` too: Swift 6.2's DocC resolved every such link against that page and failed, so the
  build leaves extended types out, and the two `init(_:)` the server adds to
  `EngineConfiguration` and `EncoderEngineConfiguration` are documented in the source alone.
  `OpenJevCore` depends on no other module, so it names the backends' types in code voice.
- **Coverage.** Every public symbol declared in the four modules has a doc comment, and a new one
  needs one too. `generate-documentation` with `--experimental-documentation-coverage
  --coverage-summary-level detailed` reports coverage per symbol, though it also counts the
  members the compiler synthesizes, which no comment can document.
- **The configuration reference.** `ConfigurationReferenceTests` in `OpenJevServerTests` reads the
  reference's table: its variables must be exactly the ones `ServerSettings(environment:)` reads,
  found in its source, plus `OPENJEV_ENCODER_MODELS`; each documented default must be the code's;
  and the settings table of [deployment.md](deployment.md) must agree with it. A new setting
  changes the three together. A change to either Markdown file alone still runs CI, which reads
  them.
- **Linux.** `OpenJevDiffusionGemma`, `OpenJevEncoders` and `OpenJevLetterReadout` do not exist
  there, so the site needs a Mac. The other two build on Linux too, as they did with Swift 6.2 in the `swift:6.2-noble`
  container:

  ```bash
  swift package generate-documentation --target OpenJevCore --target OpenJevServer --enable-experimental-combined-documentation --exclude-extended-types
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

## The upstream review

`Tools/upstream/review.py` (issue #66) compares the pins of [THIRD_PARTY.md](../THIRD_PARTY.md)
it follows, six GitHub projects and six Hugging Face checkpoints, with their projects as they are
now. The other rows of THIRD_PARTY.md, such as TypeSafe's SDKs and vLLM, are not reviewed. [upstream-log.md](upstream-log.md) says what a review checks and
what moving a pin takes, and holds each review's note. From the repository root:

```bash
make upstream
python3 Tools/upstream/review.py
```

`make upstream` fetches upstream OpenJev into `Upstream/openjev`, and the script reads upstream's
history there with git. Without that clone, or when it lacks the head of `main`, the script reads
the REST API instead. The other projects come from the REST API through `--gh`, `ghp` by default:
pass `--gh gh` to use the GitHub CLI itself. The checkpoints come from the Hub's `/api/models`.
The script writes nothing. A review makes about 50 API requests and takes from half a minute to
three minutes, depending on the network.

| Option | Effect |
|---|---|
| `--json` | Prints the review as JSON, the input of `tracking_issue.py` |
| `--from FILE` | Renders a review that `--json` saved, without reading anything else |
| `--project NAME` | Reviews only this project, such as `razorback16/openjev` or `mlx-community/diffusiongemma-26B-A4B-it-4bit` (repeatable) |
| `--clone OWNER/NAME=PATH` | Reads a project's history from another local clone |
| `--max-commits N` | Lists each project's newest N commits (30 by default); commits that touch a watched path are always listed |

Its test needs no network and no account:

```bash
python3 Tools/upstream/test_review.py
```

`tracking_issue.py` is the step of the Upstream review workflow that opens or updates the
tracking issue. With `--dry-run` it lists the issues and prints what it would do, changing nothing:

```bash
python3 Tools/upstream/review.py --json > "$TMPDIR/review.json"
python3 Tools/upstream/tracking_issue.py "$TMPDIR/review.json" --repo Algorythm-Canada/OpenJevSwift --dry-run
```

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
| `make fixtures` | `python_json_tables.py`, `wire_tables.py`, `upstream_tables.py` and `checkpoint_tables.py`, with `PYTHONHASHSEED=0` |

The committed files were written by CPython 3.14.7 with the pinned packages. Another Python
version changes the `python` pin recorded in every file, so use 3.14.7 to reproduce them byte
for byte. The first run downloads the tokenizer files (about 32 MB) into the Hugging Face cache;
no weights are downloaded, and generated fixture outputs are confined to `Fixtures/`. Running `make fixtures`
twice gives no diff. `FixturePinTests` in `OpenJevCoreTests` fails when a file records another
upstream commit or tokenizer revision.

## The JevBench harness

`Tools/jevbench` (issue #61) runs the JevBench v1 public items and SemIf's TypeSafe subset against a
`/v1/systemone` server and compares runs; [quality.md](quality.md) holds its tables and
[Tools/jevbench/README.md](../Tools/jevbench/README.md) every command. The harness needs only the
standard library, so its test runs anywhere and in the macOS CI job:

```bash
python3 Tools/jevbench/smoke_test.py
```

Running upstream's server for the comparison needs a virtual environment of its own,
`Tools/jevbench/.venv` (CPython 3.12, about 1 GB, ignored by git like every `.venv`), and the
checked-out upstream (`make upstream`):

```bash
/usr/local/bin/python3.12 -m venv Tools/jevbench/.venv
Tools/jevbench/.venv/bin/python -m pip install -r Tools/jevbench/requirements-upstream.txt
```

The datasets are downloaded into `~/Library/Caches/OpenJevSwift/jevbench`, outside the
repository, by `python3 Tools/jevbench/harness.py fetch`.

## The read benchmark

`openjev-bench` (issue #32) measures DiffusionGemma reads; [benchmarks.md](benchmarks.md) holds the
tables it produced and what they mean. Time a release build: a debug build compiles MLX's C++
without optimisation. From the repository root:

```bash
swift build -c release
.build/release/openjev-bench reads
.build/release/openjev-bench concurrency
.build/release/openjev-bench memory
.build/release/openjev-bench memory --cache-limit-gb 4
.build/release/openjev-bench prefill
```

From Xcode, give a scheme of your own a Release run configuration that builds `openjev-bench` and
`openjev`, and run the binaries from `Build/Products/Release` under its derived data. Each mode
loads the checkpoint from `--model`, else `OPENJEV_TEST_MODEL`, else the Hugging Face cache snapshot
of the pinned 4-bit revision, as the live tests do, and prints markdown tables headed by the
machine, its memory, the macOS version and whether it is on AC power. `--json` also appends the
run to `Tools/bench/results/<date>-<machine>.json` (`--output-dir` to change it). `--runs` (50) and
`--warmup` (5) set the timed and untimed requests per row; every request has a state no earlier one
sent, so each prefills.

`reads` and `concurrency` also take `--url` and time any `/v1/systemone` server over HTTP,
recording its `server-timing` `model` figure beside the HTTP latency, so the Swift and upstream
servers are measured the same way. `Tools/jevbench/servers.py --command` starts either server
(release `openjev serve`, or upstream's `python -m openjev` from `Tools/jevbench/.venv`, above)
on a free port and runs a command against it with `{url}` replaced:

```bash
python3 Tools/jevbench/servers.py --server upstream --backend mlx --command \
    .build/release/openjev-bench reads --url {url} --server upstream --json
python3 Tools/jevbench/servers.py --server swift --backend mlx --binary .build/release/openjev \
    --command .build/release/openjev-bench reads --url {url} --server swift --json
```

Before a timing run, check that nothing else is building or serving a model
(`pgrep -fl 'swift-build|xcodebuild|openjev|python'`), and keep the Mac on AC power. `openjev-bench
profile` splits a read into its stages on the GPU (`--state-tokens` for a long state, `--metallib` for
another Metal library). For the host side, `xctrace record --template 'Time Profiler' --launch --
.build/release/openjev-bench reads --runs 20` would be the tool, but xctrace 27.0 (27A266a) stops
on an assertion for every template on macOS 27.0.1 (D-044); `sample <pid> 30` on a running bench
gives a call tree meanwhile. Traces stay out of git.

## Continuous integration

Five workflows run on GitHub-hosted runners. The repository is public, so the standard runners
cost nothing. No workflow uses the billed `-xlarge` runners unless asked to.

| Workflow | When it runs | Job | Runner | What it runs |
|---|---|---|---|---|
| [ci.yml](../.github/workflows/ci.yml) | Every pull request and every push to `main`, except changes that touch only Markdown files no test reads | `Linux` | `ubuntu-24.04` with the `swift:6.2-noble` container | `swift build --build-tests` and `swift test`, both with `--scratch-path .build/linux`, then the test log check |
| | | `macOS` | `macos-26` with Xcode 26.6, selected with `DEVELOPER_DIR` | `swift build --build-tests` and `swift test`, both with `--build-system swiftbuild`, the test log check, `make lint`, then the JevBench harness's smoke test and the upstream review's test with the image's `python3` |
| | | `iOS` | `macos-26` with Xcode 26.6, selected with `DEVELOPER_DIR` | `xcodebuild test` of the `OpenJevCore-iOS` scheme on an iPhone 17 Pro simulator, the test log check, then a build of `OpenJevDiffusionGemma` for the iOS Simulator |
| | | `SDK compatibility` | `ubuntu-24.04` with the `swift:6.2-noble` container | `swift build --product openjev-stub-server` with `--scratch-path .build/linux`, Ubuntu's CPython 3.12 and Node.js 20 from `actions/setup-node`, the pinned SDKs, then `Tools/sdk-compat/run.py --swift-sdk`; the exchanges are uploaded when it fails |
| [fixtures.yml](../.github/workflows/fixtures.yml) | Pull requests that change `Fixtures/`, `Tools/fixtures/`, `THIRD_PARTY.md`, the `Makefile` or the workflow; manual runs; Mondays at 06:23 UTC | `Regenerate the fixtures` | `macos-26` with CPython 3.14.7 from `actions/setup-python` | `make upstream`, `make fixtures-venv` and `make fixtures`, then fails if `git status --porcelain Fixtures/` lists a file, and prints and uploads the diff |
| [docs.yml](../.github/workflows/docs.yml) | Pull requests and pushes to `main` that change `Sources/`, `Package.swift`, `Package.resolved`, `Tools/docs/` or the workflow; manual runs | `Build the documentation` | `macos-26` with Xcode 26.6, selected with `DEVELOPER_DIR` | `Tools/docs/build-site.sh`, then the upload of the site as the Pages artifact |
| | Pushes to `main` and manual runs on `main` | `Check GitHub Pages` | `ubuntu-24.04` | Whether Pages publishes from GitHub Actions, through the Pages API with the workflow's token |
| | The same, when it does | `Deploy to GitHub Pages` | `ubuntu-24.04` | `actions/deploy-pages`, to <https://algorythm-canada.github.io/OpenJevSwift/> |
| [upstream-review.yml](../.github/workflows/upstream-review.yml) | The first day of each month at 06:37 UTC; manual runs, with a dry-run option; pull requests that change `Tools/upstream/`, `THIRD_PARTY.md` or the workflow, as a dry run | `Review the pins` | `ubuntu-24.04` | The upstream review's test, `make upstream`, `Tools/upstream/review.py --gh gh --json` with the workflow's token and the report in the run summary, then `tracking_issue.py`, which opens or updates the tracking issue (`issues: write`) or, in a dry run, says what it would do |
| [mlx-probe.yml](../.github/workflows/mlx-probe.yml) | Manual runs only | `Probe <label>` | `macos-15`, `macos-26` and `xcode-27`, plus `macos-26-xlarge` when asked | The MLX runner probe of issue #8. Its findings are under R13 in [07-risks-and-unknowns.md](07-risks-and-unknowns.md). |

The runs of 2026-09-30 reported these images and toolchains:

| Job | Image | Toolchain |
|---|---|---|
| `Linux` | `ubuntu-24.04` 20260920.314.1, `swift:6.2-noble` | Swift 6.2.4 (`swift-6.2.4-RELEASE`) |
| `macOS` | `macos-26-arm64` 20260907.0351.1: macOS 26.6.2, Apple M1 (virtual), 3 CPUs, 7 GB | Xcode 26.6 (17F113), Swift 6.3.3, Metal Toolchain installed with the image |
| `iOS` | `macos-26-arm64` 20260907.0351.1, the same as the macOS job | Xcode 26.6 (17F113), Swift 6.3.3, the iPhone 17 Pro of the iOS 26.5 runtime |
| `Regenerate the fixtures` | `macos-26-arm64` 20260907.0351.1 | CPython 3.14.7 |

And took this long:

| Job | Empty cache | Warm cache |
|---|---|---|
| `Linux` | 3.5 minutes: build 128 s, tests 14 s | 1.5 minutes: build 26 s, tests 20 s |
| `macOS` | 8 minutes: build 270 s with Swift Build (355 s with the native build system), tests 52 s | 3 minutes: cache restore 33 s, build 76 s, tests 37 s |
| `iOS` | 10 minutes: resolve 100 s, tests 129 s, `OpenJevDiffusionGemma` 327 s | 10 minutes: cache restore 13 s, resolve 32 s, tests 163 s, `OpenJevDiffusionGemma` 382 s |
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
- **The iOS job.** It builds the `OpenJevCore-iOS` scheme rather than running `swift test`:
  SwiftPM builds and tests for the host, and only `xcodebuild` reaches a simulator. The
  destination names a device, `platform=iOS Simulator,name=iPhone 17 Pro,OS=latest`. Every iOS
  runtime on the image ships that device, and `OS=latest` picks the newest runtime, so the name
  resolves to one simulator; a step checks that before the tests and fails with a readable message
  when it stops holding, because a device name that matches none or several produces a wall of
  `xcodebuild` output. The job writes an `.xcresult` bundle and uploads it when a step failed. The
  tests read `Fixtures/` through a path derived from `#filePath`, which works unchanged in the
  simulator: the simulator runs on the host and reads the checkout in place, so the tests of both
  targets run there with no resource copying (223 and 89 on 2026-10-01). The encoder tests that
  need Verdict's or Laya's tokenizer or converted packages look for them in the Mac's home, which
  the simulator names in `SIMULATOR_HOST_HOME`, and skip on CI, which has neither.
- **Why the iOS job builds `OpenJevDiffusionGemma`.** The target declares iOS 17 like the rest of
  the package, so a build for the iOS Simulator keeps it honest. mlx-swift compiles for the
  simulator, arm64 and x86_64 both. Only the build runs: the tests need weights that no iOS
  device holds. The step runs even when the tests failed, so one push reports both.
- **The SDK compatibility job.** It restores the Linux job's build directory without saving it,
  builds only the stub server, and runs the official SDKs against it (decision D-040). CPython
  3.12 is Ubuntu 24.04's own `python3`, installed with apt inside the Swift container and checked
  by version. A failure prints every HTTP exchange of the failed checks in the log, and the
  artifact `sdk-compat-exchanges` holds every exchange and the three servers' logs for 14 days.
- **The documentation job.** It builds the four modules' DocC catalogs into one site with
  `Tools/docs/build-site.sh`, every DocC warning an error, so a link that does not resolve or a
  parameter documented under the wrong name fails the pull request
  ([API documentation](#api-documentation)). It keeps Xcode 26.6's default build system, the
  native one, since symbol graphs need no Metal library, and it has no cache, so each run
  compiles the dependencies. The site is uploaded with `actions/upload-pages-artifact`, so a pull
  request's run also offers it for download. On `main`, `Check GitHub Pages` asks the Pages API
  whether Pages publishes from GitHub Actions, and `Deploy to GitHub Pages` runs only when it does;
  until then that job is skipped and the run carries a notice. Enabling Pages is a repository
  setting (Settings, Pages, Build and deployment, Source: GitHub Actions), which no workflow
  changes.
- **The test log check.** [check-test-log.sh](../.github/scripts/check-test-log.sh) reads the
  saved output of `swift test`. It fails when the log holds no Swift Testing run or a run failed,
  and when a test or suite was skipped or cancelled for any reason other than an unset
  `OPENJEV_TEST_MODEL`, `OPENJEV_ENCODER_MODELS`, `OPENJEV_JEVK5_MODEL`, `OPENJEV_LIVE_URL` or
  `OPENJEV_TEST_DOWNLOAD`;
  a skip without a comment fails too. CI has every fixture, so a fixture test that skipped there would stop testing without
  failing. A test that needs the weights names `OPENJEV_TEST_MODEL` in the comment of its
  `.enabled(if:)` trait, one that needs a converted encoder package or its tokenizer names
  `OPENJEV_ENCODER_MODELS`, and one that needs the converted JevK5 checkpoint names
  `OPENJEV_JEVK5_MODEL`; the check lists them as skipped. The one test that downloads from the
  Hugging Face Hub names `OPENJEV_TEST_DOWNLOAD` and runs only when it is `1`. Every job unsets the
  first four variables before it runs the tests; no job sets the fifth. The iOS job saves the output of `xcodebuild test`, which prints the
  same Swift Testing lines, and checks it with the same script.
- **Caches.** The Linux and macOS jobs cache their build directory, `.build/linux` or `.build`,
  which holds the SwiftPM checkouts, the clones they come from and the build products. The iOS
  job caches only the checkouts, which `xcodebuild` keeps under `.build/ios/SourcePackages`. The
  key is the compiler's build identifier, the macOS build system and the hash of
  `Package.resolved`, so a new toolchain or a dependency change starts from an empty directory.
  The iOS cache therefore saves the clone, not the compiling, and the job takes about ten minutes
  either way. Caching its build products would mean caching gigabytes of MLX intermediates. A
  cache that `main` saved serves every pull request; one that a pull request saved serves only
  that pull request. The fixtures job caches pip downloads, keyed on `requirements.txt`, and the
  tokenizer download, keyed on the tokenizer revision. Its scheduled runs skip both caches: `huggingface_hub` serves a cached
  revision without asking the Hub, so only a fresh download shows that the pins can still be
  fetched and installed.
- **Superseded runs.** A newer push to a pull request cancels the run it replaces. Every commit on
  `main` runs in a concurrency group of its own, so no merged commit's run is cancelled.
- **Documentation-only changes.** Both triggers list every path, then `!**.md`, then the three
  Markdown files tests read: `THIRD_PARTY.md` (`FixturePinTests`, the JevBench smoke test and the
  upstream review's test), the configuration reference and `docs/deployment.md`
  (`ConfigurationReferenceTests`). A pull request or push that changes only other Markdown files
  starts no run; one that changes one of the three, or anything else, runs as usual. The
  `Protect main` ruleset requires a review, not a status check, so such a pull request is still
  mergeable. If
  a required status check is ever added, replace the path filters with a job that detects the
  documentation-only case and reports success, or GitHub will wait for a check that never runs. The
  DocC catalogs are Markdown under `Sources/`, so a pull request that changes only them starts no CI
  run unless it changes the configuration reference, but it starts the Documentation workflow, which
  builds them.
- **Fixture regeneration.** The job runs on Apple silicon because the committed files were written
  there: CPython takes its math functions from the platform's C library, and another library could
  change the last digit of a float. It reproduced every committed file byte for byte.
  `FixturePinTests`, in the CI workflow, checks the pins the files record; this job checks that
  the scripts still write the files. It cannot notice a committed file that no script writes any
  more, because the scripts only write; delete such a file when the script that wrote it stops.
- **The upstream review.** It reads other repositories through the REST API with the workflow's own
  token, which needs no permission for public repositories, and `issues: write` is for the tracking
  issue alone. A pull request that changes the review or the pins runs it as a dry run, so its
  summary shows the report and what the issue step would do. Each run uploads the review as the
  `upstream-review` artifact for 90 days. GitHub disables a public repository's schedule after 60
  days without activity; enable it again from the workflow's page.
  [upstream-log.md](upstream-log.md) describes the tracking issue and the review a maintainer then
  does.
- **The MLX runner probe.** Run it again when mlx-swift, Xcode or a runner image changes, from the
  repository's Actions tab or with `gh workflow run mlx-probe.yml`. Each job writes a table of
  outcomes and the probe's output to the run summary.

### Running the CI commands locally

The SDK compatibility job runs the same checks as `make sdk-compat-venv` and
`make sdk-compat SDK_COMPAT_ARGS=--swift-sdk` on a Mac, against a Linux build of the stub server;
its steps in [ci.yml](../.github/workflows/ci.yml) are the commands.

The Linux job, from the repository root, with Docker:

```bash
docker run --rm -v "$PWD":/src -w /src swift:6.2-noble bash -o pipefail -c 'swift build --build-tests --scratch-path .build/linux && swift test --scratch-path .build/linux 2>&1 | tee .build/linux/test.log && .github/scripts/check-test-log.sh .build/linux/test.log'
```

SwiftPM in the container resolves the Linux graph again and rewrites `Package.resolved` without
the eight Apple-only pins, keeping every other version. Restore the committed file afterwards
with `git checkout Package.resolved`.

The macOS job, with Xcode 26.6 installed:

```bash
export DEVELOPER_DIR=/Applications/Xcode_26.6.app/Contents/Developer
swift build --build-tests --build-system swiftbuild
swift test --build-system swiftbuild 2>&1 | tee .build/test.log
.github/scripts/check-test-log.sh .build/test.log
make lint
python3 Tools/jevbench/smoke_test.py
python3 Tools/upstream/test_review.py
```

With Xcode 27, the reference toolchain, leave out `DEVELOPER_DIR` and `--build-system swiftbuild`:
Swift Build is already the default.

The documentation job, with Xcode 26.6 or 27:

```bash
make docs
```

The iOS job, from the repository root:

```bash
mkdir -p .build/ios
xcodebuild test -scheme OpenJevCore-iOS -destination 'platform=iOS Simulator,name=iPhone 17 Pro,OS=latest' -derivedDataPath .build/ios 2>&1 | tee .build/ios/test.log
.github/scripts/check-test-log.sh .build/ios/test.log
xcodebuild build -scheme OpenJevDiffusionGemma -destination 'generic/platform=iOS Simulator' -derivedDataPath .build/ios
```

`xcodebuild -list` shows the scheme. A Mac with several simulators of one name, which a developer
machine often has, fails with `Unable to find a device matching the provided destination
specifier`; pass `id=<identifier>` from `xcrun simctl list devices available` instead of the name.

The fixture check:

```bash
make upstream
make fixtures-venv
make fixtures
git status --porcelain Fixtures/
```

The last command prints nothing when the fixtures are current.

### Required checks

The workflows do not change repository settings. The `Protect main` ruleset requires a review
today, not a status check. Requiring one means handling the documentation-only case first, as the
bullet above says. The checks worth requiring then are `Linux`, `macOS`, `iOS` and
`SDK compatibility`, all four from the CI workflow. Their names carry no toolchain version, so a
toolchain upgrade does not rename them. Do not require `Regenerate the fixtures`: it runs only when
a pull request changes the fixture inputs, and a required check that never reports keeps the pull
request waiting. `Build the documentation` runs only when a pull request changes the sources or the
site's files, and `Review the pins` only when one changes the upstream review or the pins, so the
same holds for both. The probe is manual and is never a required check.
