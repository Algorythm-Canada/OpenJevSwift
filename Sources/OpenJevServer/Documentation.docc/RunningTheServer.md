# Running the server

Serve a model over Jev's wire API with `openjev serve`, and answer single requests with
`openjev decide`.

## Overview

`openjev` is the command line tool this package builds. It loads the backend that
`OPENJEV_BACKEND` names and serves it with ``DecisionServer`` until SIGINT or SIGTERM. One server
serves one backend, as upstream's does; several servers can share one address through model
routes.

| Backend | Model | Needs |
|---|---|---|
| `verdict` | `verdict-1.4`, 151M parameters, Core ML | An Apple silicon Mac with macOS 15 or later; 1.6 to 2.8 GB of memory |
| `laya` | `laya-1.0`, 421M parameters, Core ML | An Apple silicon Mac with macOS 15 or later; 4.7 to 8.9 GB, 2.1 GB with `OPENJEV_ENCODER_FUNCTIONS=2` |
| `mlx` | `openjev-0.1`, DiffusionGemma 26B-A4B 4-bit, MLX | An Apple silicon Mac; about 16 GB to load, 32 GB or more recommended |

The memory figures were measured on an M3 Max and depend on the shapes the requests take; the
repository's `docs/deployment.md` explains them.

### Build

With Xcode 27 and its Metal Toolchain component (`xcodebuild -downloadComponent MetalToolchain`
installs it once):

```bash
swift build -c release --product openjev
```

The binary is `.build/release/openjev`. Xcode 26.4 to 26.6 build it too, but their default build
system leaves out MLX's Metal shaders, without which the `mlx` backend cannot run; pass
`--build-system swiftbuild` there. A Linux build has the tool without any backend, since MLX and
Core ML are Apple's: every backend there exits with status 3. The commands below call the binary
`openjev`, as if it were installed on the `PATH`; the deployment guide installs it in
`/usr/local/bin`.

### Serve

```bash
openjev serve --backend verdict
```

The first start downloads Verdict's package, tokenizer and calibrator into Application Support
(about 310 MB) and checks every file's SHA-256; later starts reuse them. The server binds only
once the model has loaded and warmed up, as upstream's does, and logs `serving on
127.0.0.1:8080`. From another terminal:

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/models
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' \
    -d '{"model":"jev-latest","state":"The deploy failed twice and the site is down.","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}'
```

`serve` takes six flags. The first five override their variable, and a flag's value is checked as
the variable's is:

| Flag | Variable |
|---|---|
| `--backend <name>` | `OPENJEV_BACKEND` |
| `--host <address>` | `OPENJEV_HOST` |
| `--port <port>` | `OPENJEV_PORT` |
| `--log-level <level>` | `OPENJEV_LOG_LEVEL` |
| `--no-warmup` | `OPENJEV_WARMUP=0` |
| `--shutdown-timeout <seconds>` | none: the time requests in flight get after SIGTERM, 0 to 86,400, 30 by default |

### Answer one request without a server

`openjev decide` reads a request from `--request <file>`, or from standard input, answers it
through the same code as `POST /v1/systemone` and prints the server's response body, without the
warm-up read:

```bash
echo '{"model":"jev-latest","state":"The deploy failed twice and the site is down.","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}' | openjev decide --backend verdict
```

A refused request prints the error body to standard error and exits 4, so a script can tell a
request to fix from a deployment to fix. `openjev models --backend verdict` prints the listing
`GET /v1/models` would answer, without loading a model.

### Several models at one address

Run each backend as its own server and give one of them routes to the others:

```bash
OPENJEV_BACKEND=laya OPENJEV_PORT=8081 openjev serve
OPENJEV_BACKEND=verdict OPENJEV_MODEL_ROUTES=laya-1.0=http://127.0.0.1:8081 openjev serve
```

Clients send both models to port 8080. A request for `laya-1.0` is passed to the Laya server as
the client sent it, with the client's `authorization`, `x-origin-secret` and `content-type`, and
its answer comes back unchanged. `GET /v1/models` lists both, without asking the Laya server.

### Logs, shutdown and exit statuses

Everything goes to standard error: the settings, without the API key or the origin secret, the
phases, and one line per request with its method, path, status, milliseconds and request id. A
body, a header value or a query string is never written. On SIGINT or SIGTERM the server stops
accepting connections, lets the requests in flight finish within `--shutdown-timeout`, releases
the model and exits 0. The exit statuses:

| Status | Meaning |
|---|---|
| 0 | Success; for `serve`, a clean shutdown |
| 1 | Any other failure, such as an address in use or a shutdown that cancelled requests |
| 2 | Invalid settings or command line; the message names the variable |
| 3 | The backend cannot run: not in this build, or it failed to load |
| 4 | `decide` only: the request was refused |

The repository's
[deployment guide](https://github.com/Algorythm-Canada/OpenJevSwift/blob/main/docs/deployment.md)
covers a launchd job that starts the server at boot, the log lines and the health check.

## See Also

- <doc:Configuration>
