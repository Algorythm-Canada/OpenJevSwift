# Security

## Reporting a vulnerability

Report a vulnerability privately, through GitHub's private vulnerability reporting: the
repository's **Security** tab, then **Report a vulnerability**
(<https://github.com/Algorythm-Canada/OpenJevSwift/security/advisories/new>). Do not open a public
issue, pull request or discussion for it.

Say which version or commit is affected, which backend and platform, how to reproduce the problem
and what an attacker gains. The report and the conversation about it stay private between you and
the maintainers until a fix is released; the advisory published with the fix can credit you.

A vulnerability in a dependency, such as Hummingbird, swift-nio, MLX or swift-transformers, or in
upstream OpenJev, belongs to that project. Report it there, and here too when OpenJevSwift's use of
it is affected.

## Supported versions

| Version | Security fixes |
|---|---|
| 0.1.x | yes |
| `main` | yes, released in the next version |

Before 1.0, security fixes go to the newest minor release only ([CHANGELOG.md](CHANGELOG.md) has
the versioning policy).

## What the code connects to

This section describes the code in `Sources/`: the library products and the `openjev` tool. It was
written from a search of the sources for every way to open a connection (`URLSession`,
`URLRequest`, `URL(string:)`, the Hugging Face Hub's API, swift-nio and AsyncHTTPClient clients and
bootstraps, the Network framework, sockets and child processes), and from reading each match.

### The server's listener

`openjev serve` listens on `OPENJEV_HOST` and `OPENJEV_PORT`, `127.0.0.1:8080` by default, so a
default server is reachable only from the same machine (`OpenJevApplication`, `ServerSettings`).
It speaks plain HTTP/1.1 and has no TLS. To serve other machines, set `OPENJEV_API_KEY` and put a
proxy that terminates TLS in front of it; `OPENJEV_ORIGIN_SECRET` makes the server answer only
requests that carry the proxy's secret ([docs/deployment.md](docs/deployment.md#settings)).

| Route | Authentication |
|---|---|
| `GET /health` | never |
| `GET /v1/models`, `POST /v1/systemone` | `OPENJEV_ORIGIN_SECRET` and `OPENJEV_API_KEY`, each when set |
| `POST /v1/chat/completions`, on the `mlx` backend only | the same |

With neither variable set, every route is open to whoever can reach the port. The size of a request
and the work clients can queue are bounded by `OPENJEV_MAX_BODY_BYTES` (64 MiB),
`OPENJEV_MAX_QUEUE`, `OPENJEV_MAX_QUESTIONS`, `OPENJEV_MAX_IMAGES`, `OPENJEV_MAX_IMAGE_BYTES` and
the `OPENJEV_GEN_*` settings, which the
[configuration reference](Sources/OpenJevServer/Documentation.docc/Configuration.md) describes.

### Outgoing connections

The shipped code opens three kinds of outgoing connection, each only when a setting or a call asks
for it.

1. **Encoder packages** (`OpenJevEncoders`, `EncoderPackageStore`). Loading Verdict or Laya
   downloads, with `URLSession.shared`, each file the embedded manifests name
   (`EncoderPackageManifest+Verdict.swift`, `EncoderPackageManifest+Laya.swift`):
   - the converted Core ML package's files, from
     `https://github.com/Algorythm-Canada/openjev-models/releases/download/<tag>/<asset>`, which
     GitHub redirects to its release storage;
   - the checkpoint's tokenizer and calibration files, from
     `https://huggingface.co/<repository>/resolve/<pinned commit>/<file>`.

   No token is sent. A file is checked against the size and SHA-256 the manifest embeds before it
   is used, and a file that does not match is not kept. Only files that are missing or unchecked
   are downloaded, so a complete store makes no request at all. With `OPENJEV_ENCODER_MODELS` set,
   or a store over a local folder in an app, nothing is downloaded. On an iPhone,
   `LayaBackend.prefetch(lengths:)` downloads Laya's per-length packages the same way, when the app
   calls it.
2. **Hugging Face checkpoints** (`OpenJevDiffusionGemma`, `ModelResolver`). The `mlx` backend
   (`DiffusionGemmaRuntime.load`) and the `jevk5` backend (`JevK5ModelFiles.resolve`) resolve a Hub
   repository with an ephemeral `URLSession` to `https://huggingface.co`:
   - `GET /api/models/<repository>/revision/<revision>`, only when the revision is a branch or a
     tag;
   - `GET /api/models/<repository>/tree/<commit>?recursive=true`, the file list, at every load of a
     Hub source, also when the cache already holds every file; a next page on another host is
     refused;
   - `GET /<repository>/resolve/<commit>/<path>` for each file the cache lacks, resumed after an
     interruption and checked against the size and SHA-256 (or git blob SHA-1) the Hub lists for
     that commit before it is kept.

   The repositories this port knows load at pinned commits: by default `OPENJEV_MLX_MODEL` is
   `mlx-community/diffusiongemma-26B-A4B-it-4bit` at `a7a81407` and `OPENJEV_JEVK5_MODEL` is
   `Algorythm-Canada/jevk5-0.2-mlx-8bit` at `d19a6f09`, and the other DiffusionGemma and JevK5
   conversions are pinned the same way. Any other repository named without `@<revision>` loads its
   `main` as it is at each start. A setting that starts with `/`, `~` or `.` names a local folder
   and makes no request. When the Hub cannot be reached, a complete cached snapshot of the commit
   loads without it.
3. **Model routes** (`OpenJevServer`, `ModelRouter`). With `OPENJEV_MODEL_ROUTES` set, a
   `POST /v1/systemone` for a routed model that this server does not serve is sent with
   AsyncHTTPClient to the route's `<url>/v1/systemone`, over `http` or `https`. It carries the
   client's body and its `authorization`, `x-origin-secret` and `content-type` headers, so the
   client's key reaches the routed server, and a user name and password in the route's URL go as
   HTTP Basic authorization in its place. Redirects are not followed and proxy variables are
   ignored. The routed server's answer is read whole and decompressed without a size bound, so
   route only to servers you run. Without the setting, the server opens no connection of its own.

Nothing else in `Sources/` opens a connection:

- **`OpenJevCore`** uses Foundation alone.
- **Images** arrive inside the request, as `data:` URLs or `{content_type, base64}` objects; the
  server never fetches a URL a request names.
- **Tokenizers** load from local folders through swift-transformers
  (`AutoTokenizer.from(modelFolder:)`, `LanguageModelConfigurationFromHub(modelFolder:)`), which
  read files only. Creating swift-transformers' shared `HubApi` for them starts an `NWPathMonitor`,
  which watches the device's network state and sends nothing. No download function of
  swift-transformers or of mlx-swift-lm is called.
- **Not shipped:** `openjev-bench` sends requests to the server its `--url` names,
  `openjev-stub-server` listens for the SDK compatibility suite, the live tests send requests to
  the server `OPENJEV_LIVE_URL` names, one test downloads from the Hub when
  `OPENJEV_TEST_DOWNLOAD=1`, and the scripts in `Tools/` download fixtures, datasets and checkpoints
  as their READMEs describe. None of them is part of a product.

### No telemetry

OpenJevSwift collects no analytics, usage data or crash reports, and contains no code that sends
any. The `openjev` tool writes its log to standard error through swift-log's `StreamLogHandler`,
and mlx-swift writes its own messages, if any, to the system's unified log; neither leaves the
machine. Hummingbird's metrics and tracing middleware are not installed, and nothing in `Sources/`
bootstraps swift-metrics or swift-distributed-tracing, so any metric or span a dependency records
goes to their no-op handlers.

## Keys and secrets

| Variable | What it holds | Where it goes |
|---|---|---|
| `OPENJEV_API_KEY` | The key clients send as `Authorization: Bearer <key>` | Compared in constant time on every `/v1/` request. A routed request carries the client's header on to the routed server. |
| `OPENJEV_ORIGIN_SECRET` | The `X-Origin-Secret` a front proxy sends | The same |
| `HF_TOKEN` | A Hugging Face token, for a gated or private repository | Sent as `Authorization: Bearer` with the requests to `https://huggingface.co`; a download the Hub redirects to another host, its CDN, goes there without it. The encoder downloads never send it. |
| `OPENJEV_MODEL_ROUTES` | A route's URL may hold a user name and password | Sent to that route's server as HTTP Basic authorization |

None of them is logged:

- The settings line `serve` writes first says `api_key=set` or `unset` and `origin_secret=set` or
  `unset`, and names the routes without their URLs; the line that follows for each route shows its
  URL without the user name and password.
- The request log has the method, the path without its query string, the status, the time and
  the request id. A body, a state, instructions, a header value or a query string is never written
  ([docs/deployment.md](docs/deployment.md#logs)).
- A download error names the repository, the revision and the file, and says whether a token was
  sent, never the token.

One exception: an `OPENJEV_MODEL_ROUTES` entry that is not `name=url`, or has an empty name or
URL, stops the server at startup with upstream's message, which quotes the entry whole, so a user
name and password in that entry reach standard error.

The library modules never read the process environment for a key: `ServerSettings(environment:)`,
`HubCacheLocation.token(environment:)` and `EncoderPackageStore(environment:)` read the dictionary
their caller passes, which for `openjev` is its own environment. A launchd job keeps the key in its
plist, which [docs/deployment.md](docs/deployment.md#launchd) makes readable by root alone.

## Downloaded files

Verdict's and Laya's files go to `Application Support/OpenJevSwift/encoders/` of the user or app,
excluded from backups. DiffusionGemma's checkpoint and JevK5's conversion go to the Hugging Face
cache (`HF_HUB_CACHE`, else `HF_HOME/hub`, else `XDG_CACHE_HOME/huggingface/hub`, else
`~/.cache/huggingface/hub`), in `huggingface_hub`'s layout, which upstream OpenJev and mlx-vlm
share. The files are model weights, Core ML packages, which Core ML compiles and loads, and
tokenizer and configuration files, DiffusionGemma's chat template among them, which swift-jinja
renders. What protects them is the pinned digests and commits above.
