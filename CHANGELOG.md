# Changelog

Every release of OpenJevSwift, newest first.

OpenJevSwift follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Its public
interface is the library products' public API, the `openjev` tool's commands, flags and exit
statuses, the `OPENJEV_*` settings, and the server's answers as
[docs/compatibility.md](docs/compatibility.md) records them.

- **Before 1.0, a minor release may break that interface.** 0.2.0 may change what 0.1.x did, and
  its entry here says what broke and what to change. A patch release, such as 0.1.1, never breaks
  it.
- **From 1.0, only a major release breaks it.**

SwiftPM's `from: "0.1.0"` accepts every release below 1.0.0, minor ones included; an app that
wants only 0.1 patches depends with `.upToNextMinor(from: "0.1.0")`.
[docs/development.md](docs/development.md#versioning-and-releases) has the policy in full.

## [0.1.0] - 2026-10-06

The first release: a native Swift implementation of upstream OpenJev, the Jev-compatible
"System One" decision server, compatible with
[razorback16/openjev](https://github.com/razorback16/openjev) 0.5.0 (`dcd2094`). One `openjev`
binary serves a model on a Mac with upstream's wire API, and the same libraries answer requests
inside an iPhone or Mac app. [docs/release-notes/0.1.0.md](docs/release-notes/0.1.0.md) has the
release notes.

### Backends

- `mlx`: DiffusionGemma 26B-A4B, served as `openjev-0.1`, on MLX on an Apple silicon Mac, from
  mlx-community's 4-bit checkpoint at a pinned revision, downloaded into the Hugging Face cache on
  first use ([deployment.md](docs/deployment.md#diffusiongemma),
  [credits.md](docs/credits.md#the-models-this-port-serves)).
- `verdict`: Verdict 1.4, served as `verdict-1.4`, on Core ML on an Apple silicon Mac with macOS 15
  or later, from a converted package published as a release of Algorythm-Canada/openjev-models,
  downloaded on first use and checked against the SHA-256 the library embeds
  ([deployment.md](docs/deployment.md#the-models-files),
  [credits.md](docs/credits.md#where-the-weights-come-from)).
- `laya`: Laya 1.0, served as `laya-1.0`, on Core ML in the same way
  ([deployment.md](docs/deployment.md#the-models-files)).
- `jevk5`: JevK5 0.2, served as `jevk5-0.2`, on MLX on an Apple silicon Mac, from an 8-bit
  conversion published on the Hugging Face Hub at a pinned commit
  ([deployment.md](docs/deployment.md#jevk5)).

### Endpoints

- `POST /v1/systemone`: Jev's decision request, with upstream's validation, error bodies,
  headers and response bytes ([compatibility.md](docs/compatibility.md#identical-to-upstream),
  [Requests and answers](Sources/OpenJevCore/Documentation.docc/RequestsAndAnswers.md)).
- `GET /v1/models`: the served model's listing, with the SDK aliases `jev-latest` and
  `jev-preview` ([compatibility.md](docs/compatibility.md#identical-to-upstream)).
- `GET /health`, without authentication ([deployment.md](docs/deployment.md#health-check)).
- `POST /v1/chat/completions` on the `mlx` backend: upstream's OpenAI-compatible text generation,
  whole or streamed, with JSON mode and thinking
  ([deployment.md](docs/deployment.md#text-generation)).
- Model routes: `OPENJEV_MODEL_ROUTES` passes a request for another model to the OpenJev server
  that serves it, so one address serves several models
  ([deployment.md](docs/deployment.md#one-origin-for-several-models)).
- `OPENJEV_API_KEY` and `OPENJEV_ORIGIN_SECRET` guard every `/v1/` route, as upstream's do
  ([deployment.md](docs/deployment.md#settings), [SECURITY.md](SECURITY.md)).

### Request options

- `noul`, `choice` and `score` questions, with upstream's labels, limits and forced answers
  ([Requests and answers](Sources/OpenJevCore/Documentation.docc/RequestsAndAnswers.md#questions)).
- `images` on `mlx`: up to `OPENJEV_MAX_IMAGES` (8) JPEG, PNG, WebP or GIF images per request,
  read as upstream reads them ([deployment.md](docs/deployment.md#diffusiongemma)).
- `steps`, `samples` and `sequential` on `mlx`, verified end to end on the checkpoint
  ([compatibility.md](docs/compatibility.md#what-runs-where)).
- `think` on `mlx`: a thought of up to the given budget before the read, billed as output tokens
  ([compatibility.md](docs/compatibility.md#what-runs-where)).
- `verdict`, `laya` and `jevk5` refuse images, `steps` or `samples` above 1, a `think` budget and
  `sequential` with upstream's messages, as upstream's encoder engines do
  ([Requests and answers](Sources/OpenJevCore/Documentation.docc/RequestsAndAnswers.md#the-request)).

### The `openjev` tool

- `openjev serve`, `openjev decide` and `openjev models`
  ([deployment.md](docs/deployment.md#openjev-decide),
  [Running the server](Sources/OpenJevServer/Documentation.docc/RunningTheServer.md)).
- Upstream's `OPENJEV_*` settings, with upstream's defaults and startup checks
  ([configuration reference](Sources/OpenJevServer/Documentation.docc/Configuration.md)).
- A graceful shutdown on SIGINT and SIGTERM, exit statuses for scripts and launchd, and a log
  that never holds a request body or a header value and shows the keys only as set or unset
  ([deployment.md](docs/deployment.md#graceful-shutdown),
  [deployment.md](docs/deployment.md#logs)).

### Apps on iOS and macOS

- Four library products answer requests in an app without a server: `OpenJevCore`,
  `OpenJevEncoders`, `OpenJevDiffusionGemma` and `OpenJevLetterReadout`
  ([Making decisions in an app](Sources/OpenJevCore/Documentation.docc/GettingStarted.md)).
- iOS: `OpenJevCore` from iOS 17, and Verdict and Laya on Core ML from iOS 18, on the Neural
  Engine, Laya with one package per sequence length that the app downloads
  ([Reading Verdict and Laya](Sources/OpenJevEncoders/Documentation.docc/ReadingVerdictAndLaya.md#on-an-iphone)).
  JevK5 builds for iOS but has not run on an iPhone yet, DiffusionGemma compiles for iOS but no
  iPhone holds it, and the server does not run on iOS
  ([compatibility.md](docs/compatibility.md#what-runs-where)).
- macOS: macOS 14 or later; the MLX modules need Apple silicon, and Verdict and Laya macOS 15
  ([development.md](docs/development.md#targets-and-platforms)).
- Linux builds `OpenJevCore`, the server and the `openjev` tool, without a backend
  ([compatibility.md](docs/compatibility.md#what-runs-where)).

### Compatibility

- Everything up to the model's probabilities is upstream's, byte for byte, and the probabilities
  agree within measured bounds; each difference has a decision record
  ([compatibility.md](docs/compatibility.md)).
- TypeSafe's Python SDK 0.7.2 and TypeScript SDK 0.6.0 work against the server unchanged, checked
  in CI ([compatibility.md](docs/compatibility.md#identical-to-upstream)).
- There is no `vllm` backend: the default backend is `mlx` (D-030,
  [compatibility.md](docs/compatibility.md#different-and-why)).

### Documentation

- DocC documentation of the five modules, published at
  <https://algorythm-canada.github.io/OpenJevSwift/>
  ([development.md](docs/development.md#api-documentation)).
- The deployment guide ([docs/deployment.md](docs/deployment.md)), the compatibility matrix
  ([docs/compatibility.md](docs/compatibility.md)), the models' credits and licenses
  ([docs/credits.md](docs/credits.md)) and the security notes ([SECURITY.md](SECURITY.md)).

### Known gaps

- CLM (`clm-v0.1`), upstream's other model, is not served on any platform; a server can still pass
  its requests to an upstream server through `OPENJEV_MODEL_ROUTES`
  ([#59](https://github.com/Algorythm-Canada/OpenJevSwift/issues/59),
  [credits.md](docs/credits.md#models-upstream-serves-that-this-port-does-not-yet)).
- DiffusionGemma's read performance has open work: the experts' matrix multiplications
  ([#100](https://github.com/Algorythm-Canada/OpenJevSwift/issues/100)), the prefill of long
  prompts ([#101](https://github.com/Algorythm-Canada/OpenJevSwift/issues/101)), and batching the
  decoder passes of concurrent reads, which run one at a time today
  ([#102](https://github.com/Algorythm-Canada/OpenJevSwift/issues/102)).

[0.1.0]: https://github.com/Algorythm-Canada/OpenJevSwift/releases/tag/0.1.0
